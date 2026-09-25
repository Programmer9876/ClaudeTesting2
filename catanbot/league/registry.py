"""Champion registry (``league/champions.json``) and materialisation of champions' exact code.

A champion is ``{name, commit, spec, weights, created, notes, gate}``:

* ``commit`` - the git commit whose code the champion plays with (full sha).
* ``spec`` - the bot spec understood by *that commit's* ``selfplay.make_bot``.
* ``weights`` - ``null`` or ``{"path", "sha256"}``: a value-net file the spec
  loads (``model=`` in the spec is replaced by the materialised copy).
* ``gate`` - the evidence that promoted it (summary of the gate's verdict and
  the path + sha256 of its game records), ``null`` for the seed champion.
* optional ``env`` (extra environment variables for its bot servers), ``shim``
  (a compatibility shim file for state-format skew, see ``serve.py``) and
  ``external`` (recorded external benchmark results, e.g. Catanatron win rate).

``materialize`` gives a champion its own source tree under
``<league home>/<name>/tree`` - by default a detached ``git worktree`` of its
commit - builds its C++ extension there and copies its weights next to it
(``<league home>/<name>/weights/``), verifying the sha256.  It is idempotent:
a finished materialisation leaves ``materialized.json`` and is reused.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "league" / "champions.json"
DEFAULT_GATES_DIR = REPO_ROOT / "league" / "gates"
DEFAULT_LEAGUE_HOME = Path(os.environ.get("CATANBOT_LEAGUE_HOME", "/home/user/league"))
SCHEMA = 1


class LeagueError(RuntimeError):
    """A league operation cannot proceed (bad registry, failed materialisation, ...)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def parse_spec(spec: str) -> Tuple[str, List[Tuple[str, str]]]:
    """``"search:depth=1,model=x"`` -> ``("search", [("depth", "1"), ("model", "x")])`` (order kept)."""
    name, _, rest = spec.partition(":")
    kv: List[Tuple[str, str]] = []
    for part in rest.split(","):
        if part:
            k, _, v = part.partition("=")
            kv.append((k.strip(), v.strip()))
    return name.strip(), kv


def format_spec(name: str, kv: List[Tuple[str, str]]) -> str:
    return name + (":" + ",".join(f"{k}={v}" for k, v in kv) if kv else "")


def spec_model(spec: str) -> Optional[str]:
    """The ``model=`` path of a spec (``None`` when absent or the heuristic evaluator)."""
    _, kv = parse_spec(spec)
    for k, v in kv:
        if k == "model" and v and v not in ("heuristic", "none"):
            return v
    return None


def spec_with_model(spec: str, path: Optional[str]) -> str:
    """Replace (or add) the spec's ``model=`` by ``path``; ``None`` leaves the spec unchanged."""
    if not path:
        return spec
    name, kv = parse_spec(spec)
    out = [(k, v) for k, v in kv if k != "model"]
    out.append(("model", str(path)))
    return format_spec(name, out)


# ---------------------------------------------------------------------------
# git (read-only queries, plus ``worktree add`` in materialize)
# ---------------------------------------------------------------------------
def _git(repo: os.PathLike, *args: str, check: bool = True) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise LeagueError(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def resolve_commit(repo: os.PathLike, rev: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")


def tree_commit(tree: os.PathLike) -> Tuple[Optional[str], Optional[bool]]:
    """``(HEAD sha, dirty)`` of a git checkout, ``(None, None)`` when ``tree`` is not one."""
    tree = Path(tree)
    if not (tree / ".git").exists():
        return None, None
    try:
        head = _git(tree, "rev-parse", "HEAD")
        dirty = bool(_git(tree, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"))
        return head, dirty
    except LeagueError:
        return None, None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclass
class Champion:
    name: str
    commit: str
    spec: str
    weights: Optional[Dict[str, str]] = None
    created: str = ""
    notes: str = ""
    gate: Optional[Dict[str, Any]] = None
    env: Dict[str, str] = field(default_factory=dict)
    shim: Optional[str] = None
    external: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "commit": self.commit, "spec": self.spec, "weights": self.weights,
                             "created": self.created, "notes": self.notes, "gate": self.gate}
        if self.env:
            d["env"] = dict(self.env)
        if self.shim:
            d["shim"] = self.shim
        if self.external:
            d["external"] = self.external
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Champion":
        missing = [k for k in ("name", "commit", "spec") if not d.get(k)]
        if missing:
            raise LeagueError(f"champion entry {d!r} lacks {missing}")
        return Champion(name=d["name"], commit=d["commit"], spec=d["spec"], weights=d.get("weights"),
                        created=d.get("created", ""), notes=d.get("notes", ""), gate=d.get("gate"),
                        env=dict(d.get("env") or {}), shim=d.get("shim"), external=dict(d.get("external") or {}))


@dataclass
class Registry:
    path: Path
    champions: List[Champion]
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: os.PathLike = DEFAULT_REGISTRY) -> "Registry":
        path = Path(path)
        if not path.exists():
            raise LeagueError(f"no champion registry at {path}")
        with open(path) as f:
            d = json.load(f)
        if d.get("schema") != SCHEMA:
            raise LeagueError(f"{path}: unsupported registry schema {d.get('schema')!r} (expected {SCHEMA})")
        champs = [Champion.from_dict(c) for c in d.get("champions", [])]
        names = [c.name for c in champs]
        if len(set(names)) != len(names):
            raise LeagueError(f"{path}: duplicate champion names {names}")
        meta = {k: v for k, v in d.items() if k not in ("champions",)}
        return Registry(path, champs, meta)

    def save(self) -> None:
        d = dict(self.meta)
        d["schema"] = SCHEMA
        d["champions"] = [c.to_dict() for c in self.champions]
        atomic_write_json(self.path, d)

    def get(self, name: str) -> Champion:
        for c in self.champions:
            if c.name == name:
                return c
        raise LeagueError(f"no champion named {name!r} in {self.path} (have {[c.name for c in self.champions]})")

    @property
    def current(self) -> Champion:
        if not self.champions:
            raise LeagueError(f"{self.path}: the registry has no champion")
        return self.champions[-1]

    def next_name(self) -> str:
        return f"champion-{len(self.champions)}"

    def resolve_path(self, p: str) -> Path:
        """Registry-relative paths (weights, shims) are relative to the repository holding the registry."""
        q = Path(p)
        return q if q.is_absolute() else (self.path.parent.parent / q)


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------
@dataclass
class Materialized:
    name: str
    root: Path            # <league home>/<name>
    tree: Path            # its source tree (PYTHONPATH of its bot servers)
    commit: str
    spec: str             # spec with model= pointing at the materialised weights
    weights: Optional[Path]
    info: Dict[str, Any]


_COPY_IGNORE_ANY = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
_COPY_IGNORE_ROOT = {"models", "data", "build", "league", "tests", "docs"}


def copy_tree(src: os.PathLike, dst: os.PathLike) -> None:
    """Copy a source tree without VCS data and caches, and without the root's models/, data/, build/, league/,
    tests/ and docs/ (for tests and non-git trees; the built extension in ``catanbot/`` is kept)."""
    src_root = os.path.abspath(src)

    def ignore(d, names):
        out = set(_COPY_IGNORE_ANY(d, names))
        if os.path.abspath(d) == src_root:
            out |= _COPY_IGNORE_ROOT & set(names)
        return out

    shutil.copytree(src, dst, ignore=ignore, symlinks=True)


def _export_archive(repo: os.PathLike, commit: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    p = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", commit], stdout=subprocess.PIPE,
                         env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    assert p.stdout is not None
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(dst, filter="data")
        else:
            tf.extractall(dst)
    if p.wait() != 0:
        raise LeagueError(f"git archive {commit} failed")


def build_extension(tree: Path, log_path: Path, python: str = sys.executable) -> Dict[str, Any]:
    """Run the tree's own ``scripts/build_cpp.sh`` (single process); returns build info."""
    script = tree / "scripts" / "build_cpp.sh"
    if not script.exists():
        return {"built": False, "reason": "no scripts/build_cpp.sh at this commit"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("CATANBOT_") and k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(tree), PYTHON=python, MAKEFLAGS="-j1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        r = subprocess.run(["bash", str(script)], cwd=str(tree), env=env, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = log_path.read_text()[-2000:]
        raise LeagueError(f"C++ build failed in {tree} (log {log_path}):\n{tail}")
    sos = sorted((tree / "catanbot").glob("catanbot_core*.so"))
    return {"built": True, "so": str(sos[0]) if sos else None, "so_sha256": sha256_file(sos[0]) if sos else None,
            "log": str(log_path)}


def materialize(champ: Champion, league_home: os.PathLike = DEFAULT_LEAGUE_HOME, repo: os.PathLike = REPO_ROOT,
                method: str = "worktree", source: Optional[os.PathLike] = None, build: bool = True,
                registry: Optional[Registry] = None, python: str = sys.executable,
                log=print) -> Materialized:
    """Give ``champ`` its own tree under ``<league_home>/<name>/`` (idempotent).

    ``method``: ``worktree`` (``git worktree add --detach``, the default),
    ``archive`` (``git archive`` export of the commit: no worktree metadata,
    repository untouched) or ``copy`` (copy ``source``, e.g. a temporary tree;
    the commit is then taken on trust and recorded as such).
    """
    root = Path(league_home) / champ.name
    tree = root / "tree"
    marker = root / "materialized.json"
    if marker.exists() and tree.exists():
        info = json.loads(marker.read_text())
        if info.get("commit") != champ.commit:
            raise LeagueError(f"{root} holds commit {info.get('commit')} but {champ.name} is {champ.commit}; "
                              f"remove {root} (and `git worktree prune`) to re-materialise")
        weights = _materialize_weights(champ, root, registry, check_only=True)
        return Materialized(champ.name, root, tree, champ.commit, spec_with_model(champ.spec, str(weights) if weights else None),
                            weights, info)
    root.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"name": champ.name, "commit": champ.commit, "method": method, "started": now_iso()}
    if tree.exists() and any(tree.iterdir()):
        head, _ = tree_commit(tree)
        if method == "worktree" and head == champ.commit:
            log(f"[materialize] reusing existing worktree {tree} at {head[:10]}")
        else:
            raise LeagueError(f"{tree} exists but is not a finished materialisation of {champ.commit}; remove it first")
    elif method == "worktree":
        log(f"[materialize] git worktree add --detach {tree} {champ.commit[:10]}")
        _git(repo, "worktree", "add", "--detach", str(tree), champ.commit)
    elif method == "archive":
        log(f"[materialize] git archive {champ.commit[:10]} -> {tree}")
        if tree.exists():
            tree.rmdir()
        _export_archive(repo, champ.commit, tree)
    elif method == "copy":
        if source is None:
            raise LeagueError("method 'copy' needs a source tree")
        log(f"[materialize] copying {source} -> {tree} (commit {champ.commit[:10]} taken on trust)")
        if tree.exists():
            tree.rmdir()
        copy_tree(source, tree)
        info["source"] = str(source)
        info["commit_verified"] = False
    else:
        raise LeagueError(f"unknown materialisation method {method!r}")
    if method in ("worktree", "archive"):
        info["commit_verified"] = True
    if build:
        log(f"[materialize] building the C++ extension of {champ.name} (single process, log {root / 'build.log'})")
        info["build"] = build_extension(tree, root / "build.log", python)
    else:
        sos = sorted((tree / "catanbot").glob("catanbot_core*.so"))
        info["build"] = {"built": False, "reason": "--no-build", "so": str(sos[0]) if sos else None}
    weights = _materialize_weights(champ, root, registry)
    info["weights"] = str(weights) if weights else None
    info["finished"] = now_iso()
    atomic_write_json(marker, info)
    return Materialized(champ.name, root, tree, champ.commit, spec_with_model(champ.spec, str(weights) if weights else None),
                        weights, info)


def _materialize_weights(champ: Champion, root: Path, registry: Optional[Registry], check_only: bool = False
                         ) -> Optional[Path]:
    if not champ.weights:
        return None
    src = Path(champ.weights["path"])
    if registry is not None and not src.is_absolute():
        src = registry.resolve_path(champ.weights["path"])
    dst = root / "weights" / src.name
    want = champ.weights.get("sha256")
    if not dst.exists():
        if check_only:
            raise LeagueError(f"{dst} missing: re-materialise {champ.name}")
        if not src.exists():
            raise LeagueError(f"weights of {champ.name} not found at {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    got = sha256_file(dst)
    if want and got != want:
        raise LeagueError(f"weights of {champ.name}: sha256 {got} != registered {want} ({dst})")
    return dst


def seed_registry(path: os.PathLike, commit: str) -> Registry:
    """Create the registry with champion-0 (used once to bootstrap; refuses to overwrite)."""
    path = Path(path)
    if path.exists():
        raise LeagueError(f"{path} exists")
    reg = Registry(path, [Champion(
        name="champion-0", commit=commit, spec="search:depth=1,beam=4,expand=8,evaluator=heuristic", weights=None,
        created=now_iso()[:10], notes="default bot at the strength proof",
        gate={"kind": "seed", "note": "seed champion: the bot frozen for the pre-registered strength proof; no gate"})],
        {"league": "catanbot", "description": "Champion ladder; append-only, see docs/LEAGUE.md"})
    reg.save()
    return reg
