"""Copy the per-game results of the test queue (docs/QUEUE.md) into the repository.

The queue writes one file per row, `<dir>/<row>.jsonl`, in ablate_catanatron's format:
- a `run` record per run (arms, overrides, seeds, code fingerprint);
- a `game` record per game (seed, seat, winner, VPs, turns, decisions, trades, mechanics);
- a `stop` record when the row is finished.
The ledger and the verdicts are copied to docs/queue by hand at each check-in. This script adds the games:
each finished row (last record `stop`) goes to docs/queue/games/<row>.jsonl.gz, gzip with no timestamp,
so an unchanged row gives a byte-identical file and git stores it once.  `--all` also copies rows still
running, cut at the last complete line (before a session ends).

It also writes:
- index.json: per row, its status, number of games, code fingerprints and the sha256 of the uncompressed file;
- epochs.json: per code epoch, the git commit whose files match the snapshot the games ran on;
- MANIFEST.sha256: `sha256sum -c` of the .gz files, run from docs/queue/games.

    python3 scripts/archive_queue.py --dir /home/user/queue_runs/queue1
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "queue", "games")


def complete_lines(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n")
    return data[:cut + 1] if cut >= 0 else b""


def records(data: bytes) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def kind(r: Dict[str, Any]) -> str:
    return r.get("kind") or ("game" if "arm" in r else "?")


def gz(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout


def _tree(commit: str, files: List[str]) -> Dict[str, str]:
    tree = {}
    for line in git("ls-tree", "-r", commit, "--", *files).splitlines():
        meta, name = line.split("\t", 1)
        tree[name] = meta.split()[2]
    return tree


def epoch_commit(path: str, max_commits: int = 300) -> Dict[str, Any]:
    """The commit that introduced the snapshot's code: going back from HEAD, the oldest commit of the first run
    of commits whose tracked files equal the snapshot's (compiled and cache files aside).  Later commits that
    leave these files alone do not move it."""
    files = []
    for base, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        files += [os.path.relpath(os.path.join(base, n), path) for n in names
                  if not n.endswith((".so", ".pyc"))]
    files.sort()
    blobs = dict(zip(files, git("hash-object", *[os.path.join(path, f) for f in files]).split()))
    found = None
    for commit in git("rev-list", "--first-parent", "-n", str(max_commits), "HEAD").split():
        tree = _tree(commit, files)
        if all(tree.get(f) == b for f, b in blobs.items()):
            found = commit
        elif found:
            break
    if found:
        return {"commit": found, "files": len(files)}
    tree = _tree(git("rev-list", "-n", "1", "HEAD").strip(), files)
    return {"commit": None, "files": len(files),
            "differs_from_head": sorted(f for f, b in blobs.items() if tree.get(f) != b)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the queue directory (run_queue.py --dir)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--all", action="store_true", help="also copy rows that are still running")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    idx_path = os.path.join(args.out, "index.json")
    index: Dict[str, Any] = {}
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
    changed = 0
    for name in sorted(os.listdir(args.dir)):
        if not name.endswith(".jsonl") or name == "ledger.jsonl":
            continue
        row = name[:-len(".jsonl")]
        data = complete_lines(os.path.join(args.dir, name))
        recs = records(data)
        finished = bool(recs) and kind(recs[-1]) == "stop"
        if not (finished or args.all):
            continue
        sha = hashlib.sha256(data).hexdigest()
        dest = os.path.join(args.out, name + ".gz")
        if index.get(row, {}).get("sha256") == sha and os.path.exists(dest):
            continue
        with open(dest + ".tmp", "wb") as f:
            f.write(gz(data))
        os.replace(dest + ".tmp", dest)
        index[row] = {
            "status": "finished" if finished else "partial",
            "games": sum(1 for r in recs if kind(r) == "game"),
            "runs": sum(1 for r in recs if kind(r) == "run"),
            "code": sorted({r["code"] for r in recs if kind(r) == "game" and r.get("code")}),
            "sha256": sha,
            "bytes": len(data),
        }
        changed += 1
        print(f"{row}: {index[row]['status']}, {index[row]['games']} games")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
        f.write("\n")
    epochs = {}
    with open(os.path.join(args.dir, "ledger.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "epoch" and os.path.isdir(r["path"]):
                epochs[r["id"]] = {"areas": r["areas"], "fingerprint": r["fingerprint"],
                                   "files_sha": r["files_sha"], **epoch_commit(r["path"])}
    with open(os.path.join(args.out, "epochs.json"), "w") as f:
        json.dump(epochs, f, indent=1, sort_keys=True)
        f.write("\n")
    lines = []
    for name in sorted(os.listdir(args.out)):
        if name.endswith(".gz"):
            with open(os.path.join(args.out, name), "rb") as f:
                lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {name}\n")
    with open(os.path.join(args.out, "MANIFEST.sha256"), "w") as f:
        f.writelines(lines)
    games = sum(v["games"] for v in index.values())
    print(f"{changed} row file(s) written; {len(index)} rows, {games} games in {args.out}")
    for eid, e in epochs.items():
        print(f"epoch {eid}: commit {e['commit'] or 'none matches: ' + ', '.join(e['differs_from_head'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
