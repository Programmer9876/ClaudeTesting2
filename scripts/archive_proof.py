#!/usr/bin/env python3
"""Archive a finished strength-proof run into the repository (``proof/``), sorted and checksummed.

The proof (``scripts/run_proof.sh``) writes into a scratch directory::

    RUN/json/<ID>/g<a>-<b>.json           bench results of one chunk of games [a, b)
    RUN/logs/<ID>/*_g<a>-<b>.jsonl.gz     action logs of that chunk, one game per line in completion order
    RUN/out/<ID>_g<a>-<b>.txt             console output of the chunk
    RUN/replay_check_<ID>.txt             replay --check of the test's logs
    RUN/run_proof.log, proof.txt, proof.json, PROOF.md   run log and analysis

This copies them to::

    DEST/<ID>/results/<ID>_g<a>-<b>.json        unchanged
    DEST/<ID>/logs/<ID>_<name>.jsonl.gz         re-written with the games sorted by game number
    DEST/<ID>/console/<ID>_g<a>-<b>.txt         unchanged
    DEST/<ID>/replay_check_<ID>.txt             unchanged
    DEST/run/                                   run_proof log(s), proof.txt / proof.json / PROOF.md
    DEST/MANIFEST.sha256                        sha256 of every archived file

and checks, per test, that the results cover exactly games 0..N-1 once each and that the logs hold
exactly the same games.  Re-written logs are byte-deterministic (gzip mtime 0).  Existing files of
the same tests are replaced; other tests already in DEST are kept (the manifest covers everything).

    python3 scripts/archive_proof.py --run RUN --dest proof T1 T2 T3 T4 T5 T6 R1 R2
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
from typing import List


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def archive_test(run: str, dest: str, tid: str) -> str:
    rdir, ldir = os.path.join(run, "json", tid), os.path.join(run, "logs", tid)
    results = sorted(glob.glob(os.path.join(rdir, "*.json")))
    logs = sorted(glob.glob(os.path.join(ldir, "*.jsonl.gz")))
    if not results:
        raise SystemExit(f"{tid}: no results in {rdir}")
    tdir = os.path.join(dest, tid)
    for sub in ("results", "logs", "console"):
        shutil.rmtree(os.path.join(tdir, sub), ignore_errors=True)
    games: List[int] = []
    for p in results:
        d = json.load(open(p))
        games += [r["game"] for r in d["results"]]
        _copy(p, os.path.join(tdir, "results", f"{tid}_{os.path.basename(p)}"))
    n = len(games)
    if sorted(games) != list(range(n)):
        raise SystemExit(f"{tid}: results do not cover games 0..{n - 1} exactly once")
    logged: List[int] = []
    for p in logs:
        with gzip.open(p, "rt") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        entries.sort(key=lambda g: g["game"])
        logged += [g["game"] for g in entries]
        out = os.path.join(tdir, "logs", f"{tid}_{os.path.basename(p)}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            for g in entries:
                gz.write((json.dumps(g, separators=(",", ":")) + "\n").encode())
    if sorted(logged) != sorted(games):
        raise SystemExit(f"{tid}: logged games differ from result games ({len(logged)} logged, {n} results)")
    for p in sorted(glob.glob(os.path.join(run, "out", f"{tid}_*.txt"))):
        _copy(p, os.path.join(tdir, "console", os.path.basename(p)))
    chk = os.path.join(run, f"replay_check_{tid}.txt")
    if os.path.exists(chk):
        _copy(chk, os.path.join(tdir, os.path.basename(chk)))
    return f"{tid}: {n} games, {len(results)} result chunks, {len(logs)} log chunks"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the proof output directory (PROOF_OUT)")
    ap.add_argument("--dest", default="proof", help="archive root in the repository (default: proof)")
    ap.add_argument("tests", nargs="+", metavar="ID")
    args = ap.parse_args(argv)
    for tid in args.tests:
        print(archive_test(args.run, args.dest, tid))
    rundir = os.path.join(args.dest, "run")
    for name in ("run_proof.log", "proof.txt", "proof.json", "PROOF.md"):
        p = os.path.join(args.run, name)
        if os.path.exists(p):
            _copy(p, os.path.join(rundir, name))
    lines = []
    for root, _, files in os.walk(args.dest):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, args.dest)
            if rel == "MANIFEST.sha256":
                continue
            lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {rel}")
    with open(os.path.join(args.dest, "MANIFEST.sha256"), "w") as fh:
        fh.write("\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1])) + "\n")
    print(f"manifest: {len(lines)} files in {os.path.join(args.dest, 'MANIFEST.sha256')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
