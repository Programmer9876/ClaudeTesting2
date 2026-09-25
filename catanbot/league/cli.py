"""Command line of the champion league (``scripts/league.py``); see ``docs/LEAGUE.md``.

    league.py status
    league.py materialize champion-0 [--method worktree|archive|copy --source DIR] [--no-build]
    league.py gate --candidate-spec SPEC [--candidate-commit HEAD|SHA | --candidate-tree DIR] [--weights PATH] ...
    league.py gate --resume GATE_ID
    league.py evaluate GATE_ID
    league.py promote GATE_ID [--name NAME] [--notes TEXT]
    league.py external-baseline champion-0 --cmd TEMPLATE [--key catanatron]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import gate as G
from .match import ServerError
from .registry import (DEFAULT_GATES_DIR, DEFAULT_LEAGUE_HOME, DEFAULT_REGISTRY, REPO_ROOT, Champion, LeagueError,
                       Registry, materialize, resolve_commit, sha256_file, spec_model, spec_with_model, tree_commit)


def engine_info() -> Dict[str, Any]:
    import catanbot
    tree = Path(catanbot.__file__).resolve().parents[1]
    commit, dirty = tree_commit(tree)
    return {"tree": str(tree), "commit": commit, "dirty": dirty, "python": sys.version.split()[0],
            "hash_seed": os.environ.get("PYTHONHASHSEED")}


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:n]


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------
def prepare_champions(reg: Registry, args, gate_dir: Path, log) -> List[Dict[str, Any]]:
    out = []
    for i, c in enumerate(reversed(reg.champions)):      # current first
        m = materialize(c, args.league_home, args.repo, method=args.method, source=args.source,
                        build=not args.no_build, registry=reg, log=log)
        ent: Dict[str, Any] = {"name": c.name, "commit": c.commit, "spec": c.spec, "spec_resolved": m.spec,
                               "tree": str(m.tree), "env": dict(c.env), "weights": c.weights,
                               "role": "current" if i == 0 else "earlier",
                               "materialized": {k: m.info.get(k) for k in ("method", "commit_verified", "build")}}
        if c.shim:
            src = reg.resolve_path(c.shim)
            dst = gate_dir / "shims" / f"{c.name}.py"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ent["shim_copy"] = f"shims/{c.name}.py"
        out.append(ent)
    return out


def prepare_candidate(reg: Registry, args, gate_dir: Path, log) -> Dict[str, Any]:
    spec = args.candidate_spec
    if args.candidate_tree and args.candidate_commit:
        raise LeagueError("give either --candidate-commit (materialised) or --candidate-tree (used as is), not both")
    if args.candidate_tree:
        tree = Path(args.candidate_tree).resolve()
        commit, dirty = tree_commit(tree)
        if commit is None:
            log(f"[gate] candidate tree {tree} is not a git checkout: its commit is unknown (promotion will need "
                f"--allow-uncommitted)")
    else:
        rev = args.candidate_commit or "HEAD"
        commit = resolve_commit(args.repo, rev)
        if rev == "HEAD":
            _, repo_dirty = tree_commit(args.repo)
            if repo_dirty:
                log(f"[gate] note: the working tree has uncommitted changes; the candidate is commit {commit[:10]} "
                    f"only (commit your changes first if they are part of the candidate)")
        m = materialize(Champion(name=f"candidate-{commit[:12]}", commit=commit, spec=spec),
                        Path(args.league_home) / "candidates", args.repo, method=args.method, source=args.source,
                        build=not args.no_build, log=log)
        tree, dirty = m.tree, False
    weights = None
    src = args.weights or spec_model(spec)
    spec_resolved = spec
    if src:
        p = Path(src)
        if not p.is_absolute():
            p = p if p.exists() else (tree / p)
        if not p.exists():
            raise LeagueError(f"candidate weights {src} not found")
        sha = sha256_file(p)
        dst = gate_dir / "weights" / f"{sha[:12]}-{p.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)
        weights = {"path": str(dst), "sha256": sha, "source": str(p)}
        spec_resolved = spec_with_model(spec, str(dst))
    return {"name": args.name or reg.next_name(), "spec": spec, "spec_resolved": spec_resolved, "commit": commit,
            "dirty": dirty, "tree": str(tree), "weights": weights, "env": dict(kv.split("=", 1) for kv in args.env)}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_materialize(args) -> int:
    reg = Registry.load(args.registry)
    names = [c.name for c in reg.champions] if args.all else [args.name]
    for name in names:
        m = materialize(reg.get(name), args.league_home, args.repo, method=args.method, source=args.source,
                        build=not args.no_build, registry=reg)
        print(f"{name}: tree {m.tree} (commit {m.commit[:10]}), spec {m.spec}, build {m.info.get('build')}")
    return 0


def cmd_gate(args) -> int:
    log = print
    if args.resume:
        gate_dir = Path(args.gates_dir) / args.resume
        cfg = G.GateConfig.from_dict(json.loads((gate_dir / "gate.json").read_text()))
        eng = engine_info()
        if eng["commit"] != cfg.engine.get("commit") or eng["dirty"] != cfg.engine.get("dirty"):
            log(f"[gate] WARNING: engine tree changed since the gate started ({cfg.engine.get('commit')} "
                f"dirty={cfg.engine.get('dirty')} -> {eng['commit']} dirty={eng['dirty']}); every game record "
                f"keeps the engine it ran on")
        if args.workers:
            cfg.workers = args.workers
    else:
        reg = Registry.load(args.registry)
        if not args.candidate_spec:
            raise LeagueError("--candidate-spec is required (or --resume GATE_ID)")
        gid = args.gate_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(args.name or reg.next_name())}-{_slug(args.candidate_spec, 30)}"
        gate_dir = Path(args.gates_dir) / gid
        if (gate_dir / "gate.json").exists():
            raise LeagueError(f"gate {gid} exists: use --resume {gid}")
        gate_dir.mkdir(parents=True, exist_ok=True)
        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
        for f in formats:
            fm = G.parse_format(f)
            if args.batch_games % len(fm.arrangements):
                log(f"[gate] note: batch of {args.batch_games} games is not a multiple of the {len(fm.arrangements)} "
                    f"seat arrangements of {f}: a look is then not seat-balanced")
        rule = G.GateRule(alpha=args.alpha, alpha_fail=args.alpha_fail, ci=args.ci, spending=args.spending,
                          futility_margin=None if args.futility == "none" else float(args.futility),
                          futility_min_t=args.futility_min_t, holm=not args.no_holm, max_errors=args.max_errors,
                          max_voids=args.max_voids)
        external = None
        if args.external_cmd:
            external = {"cmd": args.external_cmd, "key": args.external_key, "alpha": args.external_alpha,
                        "tolerance": args.external_tolerance,
                        "baseline": reg.current.external.get(args.external_key)}
        cfg = G.GateConfig(gate_id=gid, candidate=prepare_candidate(reg, args, gate_dir, log),
                           champions=prepare_champions(reg, args, gate_dir, log), formats=formats,
                           batch_games=args.batch_games, max_games=args.max_games, max_turns=args.max_turns,
                           seed_base=args.seed_base, rule=rule, external=external, engine=engine_info(),
                           created=G.now_iso(), registry=str(Path(args.registry).resolve()), workers=args.workers or 1,
                           check_roundtrip=not args.no_roundtrip_check, decide_timeout=args.decide_timeout,
                           require_accel=not args.allow_no_accel, notes=args.notes)
        log(f"[gate] {gid}: candidate {cfg.candidate['spec']} @ {str(cfg.candidate['commit'])[:10]} "
            f"({cfg.candidate['tree']}) vs {[c['name'] for c in cfg.champions]}, formats {formats}, "
            f"batch {cfg.batch_games}, max {cfg.max_games} per comparison, {cfg.num_looks} looks")
    v = G.run_gate(cfg, gate_dir, log=log, stop_after=args.stop_after)
    if v.get("interrupted"):
        log(f"[gate] interrupted after --stop-after; resume with: scripts/league.py gate --resume {cfg.gate_id}")
        return 3
    return 0 if v["verdict"] == "PASS" else 1


def cmd_evaluate(args) -> int:
    gate_dir = Path(args.gates_dir) / args.gate_id
    cfg = G.GateConfig.from_dict(json.loads((gate_dir / "gate.json").read_text()))
    ev = G.evaluate(cfg, G.load_records(gate_dir))
    for L in ev["looks"]:
        G._log_look(L, print)
    v = G.verdict(cfg, ev)
    if v is None:
        print(f"gate {args.gate_id}: running / incomplete ({ev['complete_looks']}/{cfg.num_looks} looks complete)")
        for k, s in ev["summaries"].items():
            print(f"  {k}: {s['wins']}/{s['decisive']} decisive, p(better) {s['p_greater']:.3g}, p(worse) {s['p_less']:.3g}")
        return 2
    ext_path = gate_dir / "external.json"
    if ext_path.exists():
        v = G.verdict(cfg, ev, json.loads(ext_path.read_text()))
    print(G.format_verdict(v))
    return 0


def cmd_promote(args) -> int:
    reg = Registry.load(args.registry)
    champ = G.promote(Path(args.gates_dir) / args.gate_id, reg, args.name, args.notes, args.allow_uncommitted)
    print(f"promoted {champ.name}: commit {champ.commit[:10]} spec {champ.spec} (registry {reg.path})")
    return 0


def cmd_external_baseline(args) -> int:
    reg = Registry.load(args.registry)
    champ = reg.get(args.name)
    m = materialize(champ, args.league_home, args.repo, method=args.method, source=args.source,
                    build=not args.no_build, registry=reg)
    out = Path(args.league_home) / champ.name / "external" / f"{args.key}.json"
    res = G.run_external_command(args.cmd, str(m.tree), m.spec, str(m.weights) if m.weights else None, out,
                                 out.with_suffix(".log"), champ.commit, champ.env)
    champ.external[args.key] = res
    reg.save()
    print(f"{champ.name} {args.key}: {res['wins']}/{res['games']} = {res['win_rate']:.3f} recorded in {reg.path}")
    return 0


def _fmt_h(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return f"{x:.1f} h" if x < 48 else f"{x / 24:.1f} d"


def cmd_status(args) -> int:
    reg = Registry.load(args.registry)
    print(f"Champions ({reg.path}):")
    for i, c in enumerate(reg.champions):
        tag = " <- current" if i == len(reg.champions) - 1 else ""
        g = c.gate or {}
        ev = "seed (no gate)" if g.get("kind") == "seed" else (
            f"gate {g.get('gate_id')}: " + ", ".join(f"{k} {s['wins']}/{s['decisive']}={s['share']:.3f}"
                                                      for k, s in (g.get("comparisons") or {}).items()))
        mat = Path(args.league_home) / c.name / "materialized.json"
        print(f"  #{i} {c.name:<14} {c.commit[:10]}  {c.spec}{tag}")
        print(f"      {c.created}  {c.notes}  [{ev}]  materialized: {'yes' if mat.exists() else 'no'}"
              + (f"  weights {c.weights['path']}" if c.weights else ""))
    gd = Path(args.gates_dir)
    gates = sorted(p.parent for p in gd.glob("*/gate.json")) if gd.exists() else []
    print(f"Gates ({gd}):" if gates else f"Gates ({gd}): none")
    for d in gates:
        try:
            st = G.gate_status(d)
        except Exception as exc:
            print(f"  {d.name}: unreadable ({exc!r})")
            continue
        cand = st["candidate"]
        head = f"  {st['gate_id']}: {cand['spec']} @ {str(cand.get('commit'))[:10]} vs {', '.join(st['champions'])}"
        if st["state"] == "done":
            v = st["verdict"]
            print(head + f"  -> {v['verdict']} (look {v['stop']['look']}, {v['stop']['reason']})")
            for k, s in v["comparisons"].items():
                share = "-" if s["share"] is None else f"{s['share']:.3f}"
                print(f"      {k}: {s['wins']}/{s['decisive']} = {share}  CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
                      f"  p(better) {s['p_greater']:.3g}  p(worse) {s['p_less']:.3g}")
            for r in v["reasons"]:
                print(f"      - {r}")
        else:
            p = st["progress"]
            print(head + f"  -> {st['state'].upper()} (look {st['complete_looks'] + 1}/{st['num_looks']}, "
                         f"{p.get('games_done', 0)} games, {p.get('games_per_hour') or 0:.1f} games/h, "
                         f"ETA look {_fmt_h(p.get('eta_hours_look'))}, ETA max {_fmt_h(p.get('eta_hours_max'))})")
            for k, s in st["summaries"].items():
                share = "-" if s["share"] is None else f"{s['share']:.3f}"
                print(f"      {k}: {s['wins']}/{s['decisive']} = {share}  p(better) {s['p_greater']:.3g}")
            if str(p.get("state", "")).startswith("error"):
                print(f"      stopped by {p['state']}")
            if st["state"] == "stopped":
                print(f"      resume: scripts/league.py gate --resume {st['gate_id']}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="league.py", description="Champion league and promotion gate (docs/LEAGUE.md)")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="champion registry (default league/champions.json)")
    ap.add_argument("--gates-dir", default=str(DEFAULT_GATES_DIR), help="gate records (default league/gates)")
    ap.add_argument("--league-home", default=str(DEFAULT_LEAGUE_HOME),
                    help="materialised trees (default $CATANBOT_LEAGUE_HOME or /home/user/league)")
    ap.add_argument("--repo", default=str(REPO_ROOT), help="git repository of the champions' commits")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def mat_opts(p):
        p.add_argument("--method", choices=["worktree", "archive", "copy"], default="worktree",
                       help="worktree: git worktree add --detach (default); archive: git archive export; "
                            "copy: copy --source (tests)")
        p.add_argument("--source", default=None, help="source tree for --method copy")
        p.add_argument("--no-build", action="store_true", help="skip the C++ build")

    p = sub.add_parser("status", help="print the ladder and every gate")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("materialize", help="create a champion's tree (idempotent)")
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    mat_opts(p)
    p.set_defaults(fn=cmd_materialize)

    p = sub.add_parser("gate", help="run (or resume) a promotion gate")
    p.add_argument("--candidate-spec")
    p.add_argument("--candidate-commit", default=None, help="commit to gate (default HEAD; materialised as a worktree)")
    p.add_argument("--candidate-tree", default=None, help="use this source tree as is instead of a commit")
    p.add_argument("--weights", default=None, help="value-net file (copied into the gate, hashed; overrides model=)")
    p.add_argument("--name", default=None, help="champion name if promoted (default champion-<N>)")
    p.add_argument("--notes", default="")
    p.add_argument("--gate-id", default=None)
    p.add_argument("--resume", default=None, metavar="GATE_ID")
    p.add_argument("--formats", default="4p2v2", help="comma list, first = primary (e.g. 4p2v2,3p1v2)")
    p.add_argument("--batch-games", type=int, default=198, help="games per comparison per look (multiple of 6)")
    p.add_argument("--max-games", type=int, default=1188, help="maximum games per comparison")
    p.add_argument("--max-turns", type=int, default=400)
    p.add_argument("--seed-base", type=int, default=G.DEFAULT_SEED_BASE)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--alpha-fail", type=float, default=0.05)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--spending", choices=["obf", "pocock"], default="obf")
    p.add_argument("--futility", default="0.0", help="stop when primary share <= null + this (after "
                                                      "--futility-min-t of the games); 'none' disables")
    p.add_argument("--futility-min-t", type=float, default=0.5)
    p.add_argument("--no-holm", action="store_true", help="criterion (b) without Holm (stricter for the candidate)")
    p.add_argument("--max-errors", type=int, default=0)
    p.add_argument("--max-voids", type=int, default=3)
    p.add_argument("--workers", type=int, default=None, help="parallel games (threads, each with its own servers)")
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for the candidate's servers")
    p.add_argument("--decide-timeout", type=float, default=600.0)
    p.add_argument("--allow-no-accel", action="store_true", help="allow seats without the C++ extension")
    p.add_argument("--no-roundtrip-check", action="store_true", help="skip the per-state version-skew check")
    p.add_argument("--stop-after", type=int, default=None, help="return after N games (resume later)")
    p.add_argument("--external-cmd", default=None,
                   help="external benchmark template; placeholders {spec} {tree} {weights} {out} {commit} {python}; "
                        "must write JSON with wins and games to {out}")
    p.add_argument("--external-key", default="catanatron")
    p.add_argument("--external-alpha", type=float, default=0.05)
    p.add_argument("--external-tolerance", type=float, default=None)
    mat_opts(p)
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("evaluate", help="recompute a gate's looks / verdict from its records (no games)")
    p.add_argument("gate_id")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("promote", help="append a PASSed gate's candidate to the ladder")
    p.add_argument("gate_id")
    p.add_argument("--name", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--allow-uncommitted", action="store_true")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("external-baseline", help="run the external benchmark for a champion and record it")
    p.add_argument("name")
    p.add_argument("--cmd", required=True)
    p.add_argument("--key", default="catanatron")
    mat_opts(p)
    p.set_defaults(fn=cmd_external_baseline)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (LeagueError, ServerError) as exc:
        print(f"league: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
