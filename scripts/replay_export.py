#!/usr/bin/env python3
"""Replay archive exporter: every logged game as compact shards plus one index, with "what our bot knew".

Implements docs/designs/replay_archive.md (revision 2).  The spec calls this script
``scripts/export_replay_archive.py``; it lives here as ``scripts/replay_export.py``.

    # sample bundle (three tests), default --out / --work under the session scratchpad
    python3 scripts/replay_export.py --tests T2,T8,H1
    # everything (18 tests, 9,600 games)
    python3 scripts/replay_export.py
    # re-merge the index from parts already exported (e.g. tests exported by separate runs)
    python3 scripts/replay_export.py --merge-only --tests T2,T8,H1,R1

The orchestrator is stdlib-only and runs under any interpreter.  It starts one worker per
100-game shard, as ``nice -n 10 <interpreter> scripts/replay_export.py --worker T --shard k ...``,
with the interpreter of the catanatron generation that played the test (3.3.* -> ``--py33``,
3.2.* -> ``--py32``) and at most ``--workers`` (<= 2) at once.  For the tests where our bot had
public information only (T7, T8, T9, T11) it also runs a head-digest job per shard
(``--digest T --shard k``) that re-runs the same tracker from a frozen copy of the repository's
code and compares the per-position belief digests with the snapshot's.

Per shard the worker (catanatron imported lazily under its code root):

* replays every game exactly (``AD.replay_log_action(..., check=True)``) and checks the final
  VPs, winner, turn count and ``AD.state_fingerprint`` against the log's ``final``;
* encodes it (board code, step tuples ``[op, v, h, x]``; spec 3.2-3.4);
* for counted games runs the proof's own ``PublicInfoTracker`` (``/home/user/proof_snapshot2``,
  commit 9599eed) with the run's settings in lockstep with the replay, plus an audited copy and a
  ``reveal_hidden`` twin, and writes the sparse bot-view layer ``bv`` (spec 5);
* writes ``shards/<T>_<kk>.json.gz`` (bundle) and ``parts/``, ``digests/``, ``truth/``, ``logs/``
  (work directory, never published).

Work-file formats (not published):

* ``parts/<T>_<kk>.rows.json``: ``{"rows": [index row with the deal key in column d, ...]}``
* ``parts/<T>_<kk>.checks.json``: ``{"geo", "code", "games": [{"id", "t", "maxH", "stats", "audit",
  "twin", "same", "lrOther", "turnsFormula", "bvBytes", ...}], "shardBytes", "shardBytesNoBv"}``
* ``digests/<T>_<kk>.snap.json`` / ``.head.json``: ``{"<id>": [16-hex digest per position 1..N]}``
* ``truth/<T>_<kk>.steps.json.gz``: ``{"<id>": {"pos": [truth_k, k = 0..N], "bv": [belief_k | null]}}``
  with ``truth_k = {"turn", "cur": [turn seat, player seat], "prompt", "robber", "bank", "deck",
  "P": [[vp, public_vp, hand[5], held[5], played[5], settlements, cities, road edge indices,
  road length, has_road, has_army] per seat]}`` and ``belief_k = {"nh", "pool", "opp": {"<j>":
  {"exact", "own", "E", "u", "p", "top", "calp", "vpLaw"}}}`` (counted games only).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import glob
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRATCH = "/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad"
DEFAULT_OUT = os.path.join(SCRATCH, "replay_archive")
DEFAULT_WORK = os.path.join(SCRATCH, "replay_archive_work")
DEFAULT_PY33 = "/home/user/venv_cat33/bin/python"
DEFAULT_PY32 = "python3"
SNAPSHOT2 = "/home/user/proof_snapshot2"
PAGE_TEMPLATE = os.path.join(HERE, "replay_archive_template.html")   # page template (spec 2.1)
PAGE_CORE = os.path.join(HERE, "replay_core.js")                     # inlined into the page, never published
VALIDATOR = os.path.join(HERE, "check_replay_bundle.mjs")            # node validator (spec 9.1)
MAX_WORKERS = 2

FORMAT = 2
SHARD_SIZE = 100
SUITES = ("proof", "bench_1v1", "bench_multi")
TEST_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "R1", "R2",
              "H1", "H2", "M1", "M2", "M3")
COUNTED_TESTS = ("T7", "T8", "T9", "T11")
SNAPSHOT_TESTS = ("T7", "T8", "T9", "T10", "T11")
SNAPSHOT_SHA_PREFIX = {"catanbot/bench/public_info.py": "703e3fa8967837f3",
                       "catanbot/counting.py": "718422ad13cc3e7f"}
BELIEF_FILES = ("catanbot/bench/public_info.py", "catanbot/counting.py", "catanbot/bench/catanatron_adapter.py")
COUNTED_INFO = {"mode": "counted", "samples": 4, "discards_public": False}
TRUTH_SEED = "20260926"
DEV_BELIEF_TEXT = "uniform over the unseen pool, conditioned on no opponent having already won"

RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]                     # AD.CB_TO_RESOURCE order
DEV = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
KINDS = ["BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_CITY", "ROLL", "END_TURN", "BUY_DEVELOPMENT_CARD",
         "MOVE_ROBBER", "MARITIME_TRADE", "DISCARD_RESOURCE", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY",
         "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING", "DISCARD"]
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
RES_INDEX = {r: i for i, r in enumerate(RES)}
DEV_INDEX = {d: i for i, d in enumerate(DEV)}
PLAY_DEV = {"PLAY_KNIGHT_CARD": 0, "PLAY_ROAD_BUILDING": 2, "PLAY_YEAR_OF_PLENTY": 3, "PLAY_MONOPOLY": 4}
DISCARD_KINDS = ("DISCARD_RESOURCE", "DISCARD")
DEV_DECK = [14, 5, 2, 2, 2]
BANK_PER_RESOURCE = 19
COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]
RES_CH = {"WOOD": "W", "BRICK": "B", "SHEEP": "S", "WHEAT": "G", "ORE": "O", None: "D"}
CH_RES = {v: k for k, v in RES_CH.items()}
NUM_CH = {n: "23456789ABC"[n - 2] for n in range(2, 13)}
CH_NUM = {v: k for k, v in NUM_CH.items()}
PORT_CH = {None: "3", "WOOD": "W", "BRICK": "B", "SHEEP": "S", "WHEAT": "G", "ORE": "O"}
CH_PORT = {v: k for k, v in PORT_CH.items()}
LINEUP_CH = {"value": "v", "alphabeta": "a", "sameturn": "s", "vf": "f", "ab": "b"}
INDEX_COLS = ["t", "g", "seed", "d", "lineup", "win", "vps", "pvps", "turns", "rolls", "acts", "hvp", "hv", "dev",
              "lr", "la", "def", "defp", "bk"]
STATS6 = ("events", "branching_events", "max_hypotheses", "pruned_mass", "contradictions", "resets")
HIDDEN3 = ("hidden_steals", "hidden_discards", "hidden_dev_draws")

G_FULL = "Proof · all hands seen"
G_PUB = "Proof · public information only"
G_REP = "Replication · catanatron 3.2.1"
G_1V1 = "Heads-up 1v1"
G_MULTI = "Several of our seats"

#: static labels, cross-checked against each test's results metadata (opponent, format, info, catanatron)
TEST_LABELS: Dict[str, Dict[str, str]] = {
    "T1": {"label": "3× ValueFunction", "group": G_FULL, "fmt": "1v3", "opp": "value", "info": "full", "cat": "3.3"},
    "T2": {"label": "3× AlphaBeta", "group": G_FULL, "fmt": "1v3", "opp": "alphabeta", "info": "full", "cat": "3.3"},
    "T3": {"label": "3× SameTurnAlphaBeta", "group": G_FULL, "fmt": "1v3", "opp": "sameturn", "info": "full",
           "cat": "3.3"},
    "T4": {"label": "2 of ours vs 2× ValueFunction", "group": G_FULL, "fmt": "2v2", "opp": "value", "info": "full",
           "cat": "3.3"},
    "T5": {"label": "2 of ours vs 2× AlphaBeta", "group": G_FULL, "fmt": "2v2", "opp": "alphabeta", "info": "full",
           "cat": "3.3"},
    "T6": {"label": "2 of ours vs 2× SameTurnAlphaBeta", "group": G_FULL, "fmt": "2v2", "opp": "sameturn",
           "info": "full", "cat": "3.3"},
    "T7": {"label": "3× ValueFunction", "group": G_PUB, "fmt": "1v3", "opp": "value", "info": "counted",
           "cat": "3.3"},
    "T8": {"label": "3× AlphaBeta", "group": G_PUB, "fmt": "1v3", "opp": "alphabeta", "info": "counted",
           "cat": "3.3"},
    "T9": {"label": "3× SameTurnAlphaBeta", "group": G_PUB, "fmt": "1v3", "opp": "sameturn", "info": "counted",
           "cat": "3.3"},
    "T10": {"label": "ValueFunction, AlphaBeta, SameTurnAlphaBeta", "group": G_FULL, "fmt": "1v3-mixed",
            "opp": "value,alphabeta,sameturn", "info": "full", "cat": "3.3"},
    "T11": {"label": "ValueFunction, AlphaBeta, SameTurnAlphaBeta", "group": G_PUB, "fmt": "1v3-mixed",
            "opp": "value,alphabeta,sameturn", "info": "counted", "cat": "3.3"},
    "R1": {"label": "3× ValueFunction stand-in", "group": G_REP, "fmt": "1v3", "opp": "vf", "info": "full",
           "cat": "3.2"},
    "R2": {"label": "3× AlphaBeta stand-in", "group": G_REP, "fmt": "1v3", "opp": "ab", "info": "full", "cat": "3.2"},
    "H1": {"label": "Heads-up vs AlphaBeta", "group": G_1V1, "fmt": "1v1", "opp": "alphabeta", "info": "full",
           "cat": "3.3"},
    "H2": {"label": "Heads-up vs ValueFunction", "group": G_1V1, "fmt": "1v1", "opp": "value", "info": "full",
           "cat": "3.3"},
    "M1": {"label": "3 of ours vs AlphaBeta", "group": G_MULTI, "fmt": "3v1", "opp": "alphabeta", "info": "full",
           "cat": "3.3"},
    "M2": {"label": "3 of ours vs ValueFunction", "group": G_MULTI, "fmt": "3v1", "opp": "value", "info": "full",
           "cat": "3.3"},
    "M3": {"label": "2 of ours vs ValueFunction + AlphaBeta", "group": G_MULTI, "fmt": "2v2-mixed",
           "opp": "value,alphabeta", "info": "full", "cat": "3.3"},
}

_P1 = {"played": "9984181", "commit": "9984181", "worktree": "/home/user/proof_snapshot"}
_P2 = {"played": "9599eed + 4 tooling files from 855bfd0 (PROOF_PROTOCOL amendment 5)", "commit": "9599eed",
       "worktree": SNAPSHOT2}
_PH = {"played": "35ef224", "commit": "35ef224", "worktree": None}      # docs/BENCH_1V1.md: frozen worktree
_PM = {"played": "332775b", "commit": "332775b", "worktree": None}      # docs/BENCH_MULTI.md: frozen worktree
#: the code that played each test (docs/PROOF.md "What was played", SCRUTINY Q20, BENCH_1V1.md, BENCH_MULTI.md)
CODE_PLAYED: Dict[str, Dict[str, Optional[str]]] = {
    **{t: _P1 for t in ("T1", "T2", "T3", "T4", "T5", "T6", "R1", "R2")},
    **{t: _P2 for t in ("T7", "T8", "T9", "T10", "T11")},
    **{t: _PH for t in ("H1", "H2")},
    **{t: _PM for t in ("M1", "M2", "M3")},
}


class ExportError(RuntimeError):
    """A check of the export failed (the shard / the export fails)."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ExportError(msg)


# ---------------------------------------------------------------------------
# Paths, logs, results (stdlib only)
# ---------------------------------------------------------------------------
def shard_path(t: str, g: int, size: int = SHARD_SIZE) -> str:
    """Published path of the shard holding game ``g`` of test ``t`` (``ReplayCore.shardPath``)."""
    return f"shards/{t}_{int(g) // int(size):02d}.json.gz"


def shard_name(t: str, k: int) -> str:
    return f"{t}_{int(k):02d}"


def test_dir(t: str, repo: str = REPO) -> Tuple[str, str]:
    """``(suite, directory)`` of test ``t`` (``proof/<T>``, ``bench_1v1/<T>`` or ``bench_multi/<T>``)."""
    for suite in SUITES:
        d = os.path.join(repo, suite, t)
        if os.path.isdir(os.path.join(d, "logs")):
            return suite, d
    raise ValueError(f"no logs for test {t!r} under {', '.join(SUITES)}")


def interpreter_for(version: str, py33: str = DEFAULT_PY33, py32: str = DEFAULT_PY32) -> str:
    """The interpreter that replays logs written by catanatron ``version``."""
    v = str(version or "")
    if v.startswith("3.3."):
        return py33
    if v.startswith("3.2."):
        return py32
    raise ValueError(f"no interpreter for logs of catanatron {version!r}: 3.3.* replays with --py33, "
                     f"3.2.* with --py32")


def code_root_for(t: str, snapshot2: str = SNAPSHOT2, code_head: Optional[str] = None) -> str:
    """T7-T11 are replayed with the code that played them; every other test with the frozen head copy."""
    if t in SNAPSHOT_TESTS:
        return snapshot2
    if code_head is None:
        raise ValueError("code_head is required for tests outside T7-T11")
    return code_head


LOG_NAME = re.compile(r"_g(\d+)-(\d+)\.jsonl\.gz$")


def log_files(t: str, repo: str = REPO) -> List[Dict[str, Any]]:
    """The test's log files with the ``[from, to)`` game range of their ``..._g<a>-<b>`` name."""
    suite, d = test_dir(t, repo)
    out = []
    for p in glob.glob(os.path.join(d, "logs", "*.jsonl.gz")):
        m = LOG_NAME.search(p)
        if not m:
            raise ValueError(f"log file name without a game range: {p}")
        out.append({"path": p, "rel": os.path.relpath(p, repo), "suiteRel": os.path.relpath(p, os.path.join(repo, suite)),
                    "from": int(m.group(1)), "to": int(m.group(2))})
    out.sort(key=lambda f: f["from"])
    nxt = 0
    for f in out:
        if f["from"] != nxt or f["to"] <= f["from"]:
            raise ValueError(f"{t}: log files do not tile the games (at {f['rel']})")
        nxt = f["to"]
    if not out:
        raise ValueError(f"{t}: no log files")
    return out


def read_games(files: Sequence[Dict[str, Any]], lo: int, hi: int) -> List[Dict[str, Any]]:
    """Records of games ``lo .. hi-1``: game g is line ``g - from`` (0-based) of its file (asserted)."""
    recs: List[Dict[str, Any]] = []
    for f in files:
        a, b = max(lo, f["from"]), min(hi, f["to"])
        if a >= b:
            continue
        with gzip.open(f["path"], "rt") as fh:
            for idx, line in enumerate(fh):
                g = f["from"] + idx
                if g >= b:
                    break
                if g < a:
                    continue
                rec = json.loads(line)
                if int(rec.get("game", -1)) != g:
                    raise ValueError(f"{f['rel']}: line {idx + 1} holds game {rec.get('game')}, expected {g}")
                recs.append(rec)
    got = [int(r["game"]) for r in recs]
    if got != list(range(lo, hi)):
        raise ValueError(f"games {lo}..{hi - 1} not all found (got {len(got)})")
    return recs


def iter_all_games(files: Sequence[Dict[str, Any]]):
    for f in files:
        with gzip.open(f["path"], "rt") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_results(t: str, repo: str = REPO) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[str]]:
    """``(metadata of every results file, per-game results by game, results file paths)``."""
    _suite, d = test_dir(t, repo)
    files = sorted(glob.glob(os.path.join(d, "results", "*.json")))
    if not files:
        raise ValueError(f"{t}: no results files")
    metas, by_game = [], {}
    for p in files:
        with open(p) as fh:
            doc = json.load(fh)
        metas.append({k: v for k, v in doc.items() if k != "results"})
        for r in doc.get("results", []):
            g = int(r["game"])
            if g in by_game:
                raise ValueError(f"{t}: game {g} in two results files")
            by_game[g] = r
    return metas, by_game, files


def test_info(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The run's information mode; runs without the field predate the switch and were full information."""
    info = meta.get("info")
    return dict(info) if info else {"mode": "full", "logged": False}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(suite_dir: str) -> Dict[str, str]:
    out = {}
    p = os.path.join(suite_dir, "MANIFEST.sha256")
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    out[parts[1].strip()] = parts[0]
    return out


def board_code(rec: Dict[str, Any]) -> str:
    """48-character board code: 19 land tiles (resource, number) + ``|`` + 9 port resources (spec 3.2)."""
    land, ports = [], []
    for t in rec["board"]["tiles"]:
        if t["t"] == "land":
            land.append(RES_CH[t["resource"]] + (NUM_CH[int(t["number"])] if t["number"] is not None else "-"))
        elif t["t"] == "port":
            ports.append(PORT_CH[t["resource"]])
    return "".join(land) + "|" + "".join(ports)


def parse_board_code(code: str) -> Dict[str, List[Any]]:
    """Inverse of :func:`board_code`: ``{"land": [(resource | None, number | None)], "ports": [resource | None]}``."""
    land_s, port_s = code.split("|")
    land = [(CH_RES[land_s[i]], CH_NUM.get(land_s[i + 1])) for i in range(0, len(land_s), 2)]
    return {"land": land, "ports": [CH_PORT[c] for c in port_s]}


def lineup_of(rec: Dict[str, Any]) -> str:
    """One char per seat: ``c`` our bot, else the opponent preset's letter."""
    by_seat = {int(p["seat"]): p for p in rec["players"]}
    out = []
    for i in range(len(rec["colors"])):
        p = by_seat[i]
        if p.get("kind") == "catanbot":
            out.append("c")
        else:
            out.append(LINEUP_CH[p["preset"]])
    return "".join(out)


def deal_key(rec: Dict[str, Any]) -> str:
    """Games with equal board code and development deck share a deal."""
    text = board_code(rec) + "|" + ",".join(rec["dev_deck"])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Belief mathematics (pure)
# ---------------------------------------------------------------------------
def sub_multisets(hand: Sequence[int], k: int) -> List[Tuple[Tuple[int, ...], float]]:
    """Every ``d <= hand`` with ``sum(d) == k`` and its multivariate hypergeometric probability."""
    hand = [int(x) for x in hand]
    n = sum(hand)
    if k < 0 or k > n:
        return []
    denom = math.comb(n, k)
    last = len(hand) - 1
    suffix = [sum(hand[r:]) for r in range(len(hand) + 1)]
    out: List[Tuple[Tuple[int, ...], float]] = []
    cur = [0] * len(hand)

    def rec(r: int, rem: int, num: int) -> None:
        if r == last:
            if rem <= hand[r]:
                cur[r] = rem
                out.append((tuple(cur), num * math.comb(hand[r], rem) / denom))
            return
        for x in range(max(0, rem - suffix[r + 1]), min(hand[r], rem) + 1):
            cur[r] = x
            rec(r + 1, rem - x, num * math.comb(hand[r], x))
        cur[r] = 0

    rec(0, k, 1)
    return out


def hand_marginal(hyps: Dict[Tuple, float], j: int, pend: int = 0) -> Dict[Tuple[int, ...], float]:
    """``M_j``: distribution of seat ``j``'s current hand.  With ``pend`` hidden cards discarded and not yet
    resolved, each hypothesis hand is replaced by the hands left after a uniformly random ``pend``-card discard."""
    m: Dict[Tuple[int, ...], float] = {}
    for joint, w in hyps.items():
        h = joint[j]
        m[h] = m.get(h, 0.0) + w
    if not pend:
        return m
    out: Dict[Tuple[int, ...], float] = {}
    for h, w in m.items():
        for d, p in sub_multisets(h, pend):
            key = tuple(h[r] - d[r] for r in range(len(h)))
            out[key] = out.get(key, 0.0) + w * p
    return out


def p_true_hand(marg: Dict[Tuple[int, ...], float], hand: Sequence[int]) -> float:
    return marg.get(tuple(int(x) for x in hand), 0.0)


def largest_remainder(ps: Sequence[float], total: int = 1000) -> List[int]:
    """Integers summing to ``total`` in proportion to ``ps`` (largest remainder, ties by index)."""
    s = sum(ps)
    raw = [p * total / s for p in ps] if s > 0 else [0.0] * len(ps)
    out = [int(math.floor(x)) for x in raw]
    order = sorted(range(len(ps)), key=lambda i: (-(raw[i] - out[i]), i))
    i = 0
    while sum(out) < total and order:
        out[order[i % len(order)]] += 1
        i += 1
    return out


def dev_vp_law(pool: Sequence[int], holders: Sequence[int], counts: Sequence[int], pvps: Sequence[int],
               vps_to_win: int = 10, tries: int = 20) -> Dict[int, List[float]]:
    """Law of the number of VP cards among each unknown holder's development cards under the proof tracker's
    ``_deal_devs`` sampler (spec 5.4): deals from the public pool, re-dealt up to ``tries`` times while some
    holder's public VP plus its dealt VP cards reaches ``vps_to_win``; after the last failure VP cards are swapped
    for non-VP cards left in the pool, holder by holder in seat order.  ``pvps[i]`` belongs to ``holders[i]``."""
    m = len(holders)
    if m == 0:
        return {}
    T, V = int(sum(pool)), int(pool[1])
    counts = [int(c) for c in counts]
    C = sum(counts)
    if C > T:
        raise ExportError(f"dev_vp_law: {C} unknown cards but a pool of {T}")
    denom = math.comb(T, V)
    joint = []
    for v in itertools.product(*[range(min(c, V) + 1) for c in counts]):
        sv = sum(v)
        if sv > V or C - sv > T - V:
            continue
        p = math.prod(math.comb(c, x) for c, x in zip(counts, v)) * math.comb(T - C, V - sv) / denom
        if p > 0:
            joint.append((v, p))

    def accept(v):
        return all(pvps[i] + v[i] < vps_to_win for i in range(m))

    q = sum(p for v, p in joint if accept(v))
    rej = sum(p for v, p in joint if not accept(v))
    fail = (1.0 - q) ** tries if q < 1.0 else 0.0
    out = [[0.0] * (c + 1) for c in counts]
    if q > 0:
        for v, p in joint:
            if accept(v):
                for i in range(m):
                    out[i][v[i]] += (1.0 - fail) * p / q
    if rej > 0 and fail > 0:
        for v, p in joint:
            if accept(v):
                continue
            left_nonvp = (T - V) - (C - sum(v))
            v2 = list(v)
            for i in range(m):
                while pvps[i] + v2[i] >= vps_to_win and v2[i] > 0 and left_nonvp > 0:
                    v2[i] -= 1
                    left_nonvp -= 1
            for i in range(m):
                out[i][v2[i]] += fail * p / rej
    return {holders[i]: out[i] for i in range(m)}


def hyps_fingerprint(hyps: Dict[Tuple, float]) -> int:
    return hash(frozenset(hyps.items()))


def belief_digest(tr) -> str:
    """16-hex digest of a tracker's belief at one position; reads only names both codes have (spec 5.1)."""
    c = tr.counter
    doc = (sorted(c.hyps.items()), sorted(tr.pending_discards.items()), list(tr.dev_pool()),
           [bool(tr.is_exact(j)) for j in range(tr.n)], [list(tr.expected_hand(j)) for j in range(tr.n)],
           [c.stats[k] for k in STATS6], [tr.stats[k] for k in HIDDEN3])
    return hashlib.sha256(repr(doc).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Input audit (spec 5.8 c)
# ---------------------------------------------------------------------------
def make_audited_tracker(base):
    """``AuditedTracker(base)``: records every counter mutator call and every chance-result read with the log
    entry that caused it; behaviour unchanged (every recorder delegates)."""

    class AuditedTracker(base):
        def _init_from(self, st):
            super()._init_from(st)
            self.audit_calls: List[Tuple] = []
            self.audit_flags: List[Tuple[int, str]] = []
            self._aud_i = -1
            self._aud_depth = 0
            self._aud_updates = 0
            c = self.counter
            names = {n for n in dir(type(c)) if (n.startswith("observe_") or n in ("sync", "set_exact"))
                     and callable(getattr(c, n, None))}
            for name in sorted(names | {"_update", "_reset"}):
                setattr(c, name, self._aud_wrap(name, getattr(c, name)))
            self._aud_fp = hyps_fingerprint(c.hyps)

        def _aud_wrap(self, name, orig):
            public = not name.startswith("_")

            def recorder(*a, **k):
                if public:
                    self.audit_calls.append((self._aud_i, name, copy.deepcopy(a), copy.deepcopy(k), self._aud_depth))
                else:
                    self.audit_calls.append((self._aud_i, name, (), {}, self._aud_depth))
                if name == "_update":
                    self._aud_updates += 1
                if public:
                    self._aud_depth += 1
                try:
                    return orig(*a, **k)
                finally:
                    if public:
                        self._aud_depth -= 1
            return recorder

        def _observe_entry(self, entry, pre, post):
            self._aud_i += 1
            c = self.counter
            fp0 = hyps_fingerprint(c.hyps)
            if fp0 != self._aud_fp:
                self.audit_flags.append((self._aud_i, "hypotheses changed between entries"))
            u0 = self._aud_updates
            super()._observe_entry(entry, pre, post)
            fp1 = hyps_fingerprint(self.counter.hyps)
            if fp1 != fp0 and self._aud_updates == u0:
                self.audit_flags.append((self._aud_i, "hypotheses changed without an audited _update"))
            self._aud_fp = fp1

        def _revealed_result(self, entry, index=None):
            self.audit_calls.append((self._aud_i, "_revealed_result", (index,), {}, self._aud_depth))
            return super()._revealed_result(entry, index)

    AuditedTracker.__name__ = "AuditedTracker"
    return AuditedTracker


def _arg(a: Sequence, k: Dict[str, Any], idx: int, name: str, default=None):
    if len(a) > idx:
        return a[idx]
    return k.get(name, default)


def _discard_runs(items: Sequence[Sequence], hands: Sequence[Sequence[Sequence[int]]], seat_of: Dict[str, int],
                  me: int, discards_public: bool) -> Tuple[List[Dict[int, int]], List[Dict[int, int]]]:
    """Public per-seat hidden-discard counts of the current 7, before and after each entry."""
    before, after = [], []
    run: Dict[int, int] = {}
    for i, it in enumerate(items):
        before.append(dict(run))
        if it[1] in DISCARD_KINDS:
            a = seat_of[it[0]]
            drop = sum(hands[i][a]) - sum(hands[i + 1][a])
            if drop > 0 and a != me and not discards_public:
                run[a] = run.get(a, 0) + drop
        else:
            run = {}
        after.append(dict(run))
    return before, after


def audit_violations(calls: Sequence[Tuple], flags: Sequence[Tuple[int, str]], fields: Sequence[Tuple],
                     items: Sequence[Sequence], hands: Sequence[Sequence[Sequence[int]]],
                     banks: Sequence[Sequence[int]], colors: Sequence[str], me: int,
                     discards_public: bool = False) -> List[str]:
    """Every recorded call that breaks the public-information rule of its entry (spec 5.8 c table), every
    structural flag, and every direct field write that is not public.  ``hands[k]`` / ``banks[k]`` are the
    true hands / bank at position k (the exporter's own replay); ``fields[i]`` = ``(counter.last_bank,
    counter.size)`` after entry i."""
    seat_of = {c: i for i, c in enumerate(colors)}
    n = len(colors)
    run_before, run_after = _discard_runs(items, hands, seat_of, me, discards_public)
    bad = [f"entry {i}: {msg}" for i, msg in flags]
    for (i, name, a, k, depth) in calls:
        if i < 0 or i >= len(items):
            bad.append(f"{name} called outside any log entry")
            continue
        color, kind, value, result = items[i]
        actor = seat_of[color]
        victim = None
        if kind == "MOVE_ROBBER" and value is not None and len(value) > 1 and value[1] is not None:
            victim = seat_of[value[1]]
        pre, post = hands[i], hands[i + 1]
        ok = False
        if name == "observe_hand":
            j, hand = _arg(a, k, 0, "i"), _arg(a, k, 1, "resources")
            ok = j == me and hand is not None and [int(x) for x in hand] == list(post[me])
        elif name == "observe_hand_size":
            j, s = _arg(a, k, 0, "i"), _arg(a, k, 1, "size")
            ok = j is not None and 0 <= j < n and int(s) == sum(post[j])
        elif name == "observe_bank":
            bank = _arg(a, k, 0, "bank")
            ok = kind not in DISCARD_KINDS and [int(x) for x in bank] == list(banks[i + 1])
        elif name == "observe_delta":
            j, d = _arg(a, k, 0, "i"), _arg(a, k, 1, "counts")
            ok = (kind not in ("MOVE_ROBBER", "PLAY_MONOPOLY") + DISCARD_KINDS and j is not None and 0 <= j < n
                  and [int(x) for x in d] == [post[j][r] - pre[j][r] for r in range(5)])
        elif name == "observe_steal":
            v2, a2, res = _arg(a, k, 0, "victim"), _arg(a, k, 1, "thief"), _arg(a, k, 2, "res")
            ok = kind == "MOVE_ROBBER" and v2 == victim and a2 == actor
            if ok and res is not None:
                stolen = result if result is not None else (value[2] if len(value) > 2 else None)
                ok = me in (actor, victim) and stolen is not None and res == RES_INDEX[stolen]
        elif name == "observe_discard":
            j, counts, nn = _arg(a, k, 0, "i"), _arg(a, k, 1, "counts"), _arg(a, k, 2, "n")
            ok = (kind in DISCARD_KINDS and j == actor and (j == me or discards_public) and counts is not None
                  and nn is None and [int(x) for x in counts] == [pre[j][r] - post[j][r] for r in range(5)])
        elif name == "observe_discards":
            pend, bank = _arg(a, k, 0, "hidden"), _arg(a, k, 1, "bank")
            ok = (kind not in DISCARD_KINDS and bank is not None and [int(x) for x in bank] == list(banks[i])
                  and {int(x): int(y) for x, y in dict(pend).items() if int(y) > 0} == run_before[i]
                  and bool(run_before[i]))
        elif name == "observe_monopoly":
            j, res = _arg(a, k, 0, "i"), _arg(a, k, 1, "res")
            taken_by, taken = k.get("taken_by", _arg(a, k, 3, "taken_by")), _arg(a, k, 2, "taken")
            want = {x: sum(pre[x]) - sum(post[x]) for x in range(n) if x != actor}
            ok = (kind == "PLAY_MONOPOLY" and j == actor and res == RES_INDEX[value] and taken is None
                  and taken_by is not None and {int(x): int(y) for x, y in dict(taken_by).items()} == want)
        elif name == "_revealed_result":
            ok = kind == "MOVE_ROBBER" and me in (actor, victim)
        elif name == "_update":
            ok = depth >= 1
        else:        # _reset and every other counter mutator are never allowed
            ok = False
        if not ok:
            bad.append(f"entry {i} ({kind} by seat {actor}): {name}{tuple(a)}{k or ''} depth {depth}")
    for i, (last_bank, size) in enumerate(fields):
        if last_bank is None or [int(x) for x in last_bank] not in (list(banks[i]), list(banks[i + 1])):
            bad.append(f"entry {i}: counter.last_bank {last_bank} is not the public bank")
        want = [sum(hands[i + 1][j]) + run_after[i].get(j, 0) for j in range(n)]
        if [int(x) for x in size] != want:
            bad.append(f"entry {i}: counter.size {list(size)} != public sizes {want}")
    return bad


# ---------------------------------------------------------------------------
# Engine side (imported lazily, under the worker's code root)
# ---------------------------------------------------------------------------
class _Env:
    AD = None
    PIT = None
    root: Optional[str] = None
    info: Dict[str, Any] = {}


_ENV = _Env()


def load_engine(root: Optional[str] = None, expect_snapshot: bool = False) -> Dict[str, Any]:
    """Import ``catanbot`` (adapter + tracker) from ``root`` (``None``: whatever is on ``sys.path``).  With
    ``expect_snapshot`` assert it is the proof snapshot (file location and sha256 prefixes)."""
    if root is not None:
        root = os.path.abspath(root)
        for p in (os.path.join(root, "scripts"), root):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        if "catanbot" in sys.modules:
            loaded = os.path.abspath(sys.modules["catanbot"].__file__)
            if not loaded.startswith(root + os.sep):
                raise ExportError(f"catanbot already imported from {loaded}, cannot switch to {root}")
    import catanbot  # noqa: F401
    from catanbot.bench import catanatron_adapter as AD
    from catanbot.bench.public_info import PublicInfoTracker
    cb_file = os.path.abspath(catanbot.__file__)
    cb_root = os.path.dirname(os.path.dirname(cb_file))
    if root is not None and cb_root != root:
        raise ExportError(f"catanbot imported from {cb_file}, expected under {root}")
    shas = {}
    for rel in BELIEF_FILES:
        p = os.path.join(cb_root, rel)
        shas[rel] = sha256_file(p) if os.path.exists(p) else None
    if expect_snapshot:
        for rel, prefix in SNAPSHOT_SHA_PREFIX.items():
            if not (shas.get(rel) or "").startswith(prefix):
                raise ExportError(f"{rel} under {cb_root} has sha256 {shas.get(rel)}, expected {prefix}...: "
                                  f"the proof snapshot changed")
    _ENV.AD, _ENV.PIT, _ENV.root = AD, PublicInfoTracker, cb_root
    _ENV.info = {"catanbot": cb_file, "root": cb_root, "sha256": shas, "catanatron": AD.CATANATRON_VERSION,
                 "api33": bool(AD.API_33), "python": sys.executable}
    return _ENV.info


CORNER_DEG = {"NORTH": -90, "NORTHEAST": -30, "SOUTHEAST": 30, "SOUTH": 90, "SOUTHWEST": 150, "NORTHWEST": 210}


def _center(c: Sequence[int]) -> Tuple[float, float]:
    x, _, z = c
    return math.sqrt(3) * (x + z / 2), 1.5 * z


def global_geometry(game) -> Dict[str, Any]:
    """``geo`` (spec 3.1): node positions (id order), sorted edges, land tiles and ports in catanatron order.
    Same formulas as ``scripts/game_viewer.py geometry`` (unit circumradius, pointy-top, y down)."""
    from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
    nodes: Dict[int, List[float]] = {}
    edges = set()
    tiles, ports = [], []
    for coord, tile in game.state.board.map.tiles.items():
        cx, cy = _center(coord)
        for ref, nid in (getattr(tile, "nodes", None) or {}).items():
            a = math.radians(CORNER_DEG[ref.value])
            p = [cx + math.cos(a), cy + math.sin(a)]
            if nid in nodes:
                _need(math.dist(nodes[nid], p) <= 1e-6, f"node {nid}: positions disagree")
            else:
                nodes[int(nid)] = p
        for e in (getattr(tile, "edges", None) or {}).values():
            edges.add(tuple(sorted(int(x) for x in e)))
        kind = type(tile).__name__
        if kind == "LandTile":
            tiles.append([[int(x) for x in coord], round(cx, 4), round(cy, 4)])
        elif kind == "Port":
            ra, rb = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            ports.append([round(cx, 4), round(cy, 4), [int(tile.nodes[ra]), int(tile.nodes[rb])]])
    ids = sorted(nodes)
    _need(ids == list(range(len(ids))), "node ids are not 0..n-1")
    return {"nodes": [[round(nodes[i][0], 4), round(nodes[i][1], 4)] for i in ids],
            "edges": [list(e) for e in sorted(edges)], "tiles": tiles, "ports": ports}


def geo_indexes(geo: Dict[str, Any]) -> Tuple[Dict[Tuple[int, ...], int], Dict[Tuple[int, int], int]]:
    return ({tuple(t[0]): i for i, t in enumerate(geo["tiles"])},
            {(int(e[0]), int(e[1])): i for i, e in enumerate(geo["edges"])})


def seat_rows(st, n: int) -> List[Dict[str, Any]]:
    """Per seat: hand, actual VP, public VP, longest-road length, titles (read from ``player_state``)."""
    ps = st.player_state
    out = []
    for i in range(n):
        k = f"P{i}"
        out.append({"h": [int(ps[f"{k}_{r}_IN_HAND"]) for r in RES],
                    "vp": int(ps[f"{k}_ACTUAL_VICTORY_POINTS"]), "pvp": int(ps[f"{k}_VICTORY_POINTS"]),
                    "lr": int(ps[f"{k}_LONGEST_ROAD_LENGTH"]), "LR": bool(ps[f"{k}_HAS_ROAD"]),
                    "LA": bool(ps[f"{k}_HAS_ARMY"])})
    return out


def dev_rows(st, n: int) -> Tuple[List[List[int]], List[List[int]]]:
    """``(held, played)`` development cards per seat."""
    ps = st.player_state
    held = [[int(ps[f"P{i}_{d}_IN_HAND"]) for d in DEV] for i in range(n)]
    played = [[int(ps.get(f"P{i}_PLAYED_{d}", 0)) for d in DEV] for i in range(n)]
    return held, played


def public_vp_from_pieces(st, n: int) -> List[int]:
    """Settlements + 2 cities + 2 [Longest Road] + 2 [Largest Army] (the proof code's ``GameState.public_vp``)."""
    ps = st.player_state
    out = []
    for i in range(n):
        k = f"P{i}"
        s = 5 - int(ps[f"{k}_SETTLEMENTS_AVAILABLE"])
        c = 4 - int(ps[f"{k}_CITIES_AVAILABLE"])
        out.append(s + 2 * c + 2 * int(bool(ps[f"{k}_HAS_ROAD"])) + 2 * int(bool(ps[f"{k}_HAS_ARMY"])))
    return out


def encode_value(kind: str, value, result, seat_of: Dict[str, int], tile_index: Dict[Tuple[int, ...], int],
                 edge_index: Dict[Tuple[int, int], int]):
    """The ``v`` part of a step (spec 3.4 table)."""
    if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        return int(value)
    if kind == "BUILD_ROAD":
        a, b = sorted(int(x) for x in value)
        return edge_index[(a, b)]
    if kind == "ROLL":
        d = result if result is not None else value
        return 10 * int(d[0]) + int(d[1])
    if kind == "BUY_DEVELOPMENT_CARD":
        return DEV_INDEX[result if result is not None else value]
    if kind == "MOVE_ROBBER":
        coord, who = value[0], value[1] if len(value) > 1 else None
        stolen = result if result is not None else (value[2] if len(value) > 2 else None)
        return (tile_index[tuple(int(x) for x in coord)] * 100 + ((seat_of[who] + 1) if who is not None else 0) * 10
                + ((RES_INDEX[stolen] + 1) if stolen is not None else 0))
    if kind == "MARITIME_TRADE":
        give = [x for x in value[:4] if x is not None]
        return RES_INDEX[give[0]] * 100 + len(give) * 10 + RES_INDEX[value[4]]
    if kind == "DISCARD_RESOURCE":
        return RES_INDEX[result if result is not None else value]
    if kind == "DISCARD":
        c = [0] * 5
        for x in (result if result is not None else value):
            c[RES_INDEX[x]] += 1
        return c
    if kind == "PLAY_MONOPOLY":
        return RES_INDEX[value]
    if kind == "PLAY_YEAR_OF_PLENTY":
        cards = [value] if isinstance(value, str) else list(value)
        return [RES_INDEX[x] for x in cards]
    if kind in ("END_TURN", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"):
        return None
    raise ExportError(f"unexpected action kind {kind}")


def step_tuple(op: int, v, h: List[int], x: List[int]) -> List[Any]:
    """``[op, v, h, x]`` with trailing empty parts dropped (missing middle parts: ``null`` / ``0``)."""
    row: List[Any] = [op]
    if v is not None or h or x:
        row.append(v)
    if h or x:
        row.append(h or 0)
    if x:
        row.append(x)
    return row


def truth_summary(st, edge_index: Dict[Tuple[int, int], int]) -> Dict[str, Any]:
    """``AD.state_summary`` reduced to lists (roads as ``geo.edges`` indices), with ``num_turns``."""
    s = _ENV.AD.state_summary(st)
    P = []
    for p in s["players"]:
        P.append([p["vp"], p["public_vp"], [p["resources"][r] for r in RES], [p["dev_cards"][d] for d in DEV],
                  [p["dev_played"].get(d, 0) for d in DEV], list(p["settlements"]), list(p["cities"]),
                  sorted(edge_index[(int(e[0]), int(e[1]))] for e in p["roads"]), p["longest_road_length"],
                  int(p["has_longest_road"]), int(p["has_largest_army"])])
    return {"turn": s["turn"], "cur": [s["current_turn_seat"], s["current_player_seat"]], "prompt": s["prompt"],
            "robber": s["robber"], "bank": [s["bank"][r] for r in RES], "deck": s["dev_deck_left"], "P": P}


def _lg(x: float) -> int:
    """``round(1000 * -log10 x)`` (x in (0, 1])."""
    v = int(round(-1000.0 * math.log10(x)))
    return 0 if v == 0 else v


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def _sig(x: Optional[float], digits: int = 6) -> Optional[float]:
    """``x`` to ``digits`` significant digits (for values that can be tiny, such as the minimum chance)."""
    return None if x is None else float(f"{float(x):.{digits}g}")


class BotView:
    """The bot-view lockstep of one counted game (spec 5.2-5.8): the displayed tracker, an audited copy and a
    ``reveal_hidden`` twin, all built as the proof player built its tracker, fed ``final_state``'s log."""

    def __init__(self, rec: Dict[str, Any], final_state, me: int, info: Dict[str, Any], vps_to_win: int,
                 want_truth: bool, want_digests: bool):
        AD, PIT = _ENV.AD, _ENV.PIT
        self.me = me
        self.n = len(rec["colors"])
        self.colors = list(rec["colors"])
        self.vps_to_win = int(vps_to_win)
        self.discards_public = bool(info["discards_public"])
        color = AD.Color(rec["colors"][me])
        kw = dict(discards_public=self.discards_public, vps_to_win=self.vps_to_win)
        self.plain = PIT(color, **kw)
        self.audit = make_audited_tracker(PIT)(color, **kw)
        self.twin = PIT(color, reveal_hidden=True, **kw)
        self.gens = [t.replay_log(final_state) for t in (self.plain, self.audit, self.twin)]
        self.opps = [j for j in range(self.n) if j != me]
        self.want_truth, self.want_digests = want_truth, want_digests
        self.q: List[List[Any]] = []
        self.nh: List[int] = []
        self.dp: List[List[int]] = []
        self.dv: List[List[int]] = []
        self.last_row = {j: [j, 1, 0, 0, 0, 0, 0] for j in self.opps}
        self.last_nh = 1
        self.last_pool = list(DEV_DECK)
        self.last_law = {j: [1000] for j in self.opps}
        self.cache: Dict[int, Tuple[Any, int, Dict[str, Any]]] = {}
        self.law_key = None
        self.law: Dict[int, List[float]] = {}
        self.fields: List[Tuple] = []
        self.digests: List[str] = []
        self.beliefs: List[Optional[Dict[str, Any]]] = [self.position0()] if want_truth else []
        self.twin_ok = True
        self.same_ok = True
        self.stats_at: Optional[Dict[str, int]] = None
        # summary accumulators
        self.positions = 0
        self.all_exact = 0
        self.p_sum = self.calp_sum = 0.0
        self.pu_sum = self.calpu_sum = 0.0
        self.pairs = self.nu = 0
        self.p_min = 1.0

    def position0(self) -> Dict[str, Any]:
        return {"nh": 1, "pool": list(DEV_DECK),
                "opp": {str(j): {"exact": True, "own": [0] * 5, "E": [0.0] * 5, "u": 0, "p": 1.0, "top": 1.0,
                                 "calp": 1.0, "vpLaw": [1.0]} for j in self.opps}}

    def _derived(self, j: int, hyps, pend: int) -> Dict[str, Any]:
        hit = self.cache.get(j)
        if hit is not None and hit[0] is hyps and hit[1] == pend:
            return hit[2]
        M = hand_marginal(hyps, j, pend)
        hands = list(M)
        u = 0
        for r in range(5):
            if len({h[r] for h in hands}) > 1:
                u |= 1 << r
        mean = [sum(w * h[r] for h, w in M.items()) for r in range(5)]
        d = {"M": M, "u": u, "top": max(M.values()), "calp": sum(w * w for w in M.values()), "mean": mean,
             "size": sum(hands[0]), "single": len(hands) == 1, "own": list(hands[0]) if len(hands) == 1 else None}
        self.cache[j] = (hyps, pend, d)
        return d

    def step(self, k: int, item: Sequence, hands_k: List[List[int]], game) -> None:
        """Advance the three trackers by logged entry ``k - 1`` and record position ``k`` (truth: ``game``)."""
        AD = _ENV.AD
        entry, _ = next(self.gens[0])
        next(self.gens[1])
        next(self.gens[2])
        _need(AD.encode_log_entry(entry) == list(item), f"lockstep broken at entry {k - 1}")
        plain, me, n = self.plain, self.me, self.n
        c = plain.counter
        st = game.state
        # --- structural asserts (spec 5.5) ---
        _need(c.stats["resets"] == 0 and c.stats["contradictions"] == 0 and c.stats["pruned_mass"] == 0,
              f"position {k}: counter stats {dict(c.stats)}")
        _need(plain.stats.get("resyncs", 0) == 0, f"position {k}: tracker resynced")
        _need(plain.pool_is_consistent(), f"position {k}: development pool inconsistent")
        for j in self.opps:
            _need(plain.known_dev[j] is None, f"position {k}: known_dev set for seat {j}")
        held, played = dev_rows(st, n)
        pool = [int(x) for x in plain.dev_pool()]
        want_pool = [DEV_DECK[t] - sum(played[s][t] for s in range(n)) - held[me][t] for t in range(5)]
        _need(pool == want_pool, f"position {k}: pool {pool} != {want_pool}")
        nh = len(c.hyps)
        _need(nh <= plain.max_hypotheses, f"position {k}: {nh} hypotheses")
        rows = seat_rows(st, n)
        pvps = [r["pvp"] for r in rows]
        _need(pvps == public_vp_from_pieces(st, n), f"position {k}: public VP differs from the pieces")
        # --- per opponent belief (spec 5.3) ---
        pend = plain.pending_discards
        bel_opp = {}
        every_exact = True
        for j in self.opps:
            truth = hands_k[j]
            d = self._derived(j, c.hyps, int(pend.get(j, 0)))
            exact = bool(plain.is_exact(j))
            E = [float(x) for x in plain.expected_hand(j)]
            _need(exact == d["single"], f"position {k} seat {j}: is_exact {exact} but support {len(d['M'])}")
            _need(d["size"] == sum(truth), f"position {k} seat {j}: belief size {d['size']} != {sum(truth)}")
            _need(abs(sum(E) - sum(truth)) <= 1e-6, f"position {k} seat {j}: sum E {sum(E)} != {sum(truth)}")
            _need(all(abs(E[r] - d["mean"][r]) <= 1e-9 for r in range(5)),
                  f"position {k} seat {j}: expected_hand {E} != mean of M {d['mean']}")
            _need((d["u"] == 0) == exact, f"position {k} seat {j}: u {d['u']} vs exact {exact}")
            p = p_true_hand(d["M"], truth)
            _need(p > 0, f"position {k} seat {j}: the real hand {truth} is excluded")
            _need(d["top"] >= p - 1e-12, f"position {k} seat {j}: top < p")
            if exact:
                _need(d["own"] == list(truth), f"position {k} seat {j}: exact belief {d['own']} != real {truth}")
                row = [j, 1] + d["own"]
            else:
                row = [j, 0, _lg(p), _lg(d["top"]), _lg(d["calp"]), d["u"]] + [int(round(100 * x)) for x in E]
            if row != self.last_row[j]:
                self.q.append([k] + row)
                self.last_row[j] = row
            every_exact = every_exact and exact
            self.pairs += 1
            self.p_sum += p
            self.calp_sum += d["calp"]
            self.p_min = min(self.p_min, p)
            if not exact:
                self.nu += 1
                self.pu_sum += p
                self.calpu_sum += d["calp"]
            if self.want_truth:
                bel_opp[str(j)] = {"exact": exact, "own": d["own"] if exact else None, "E": [round(x, 9) for x in E],
                                   "u": d["u"], "p": p, "top": d["top"], "calp": d["calp"]}
        self.positions += 1
        self.all_exact += every_exact
        if nh != self.last_nh:
            self.nh += [k, nh]
            self.last_nh = nh
        if pool != self.last_pool:
            self.dp.append([k] + pool)
            self.last_pool = pool
        # --- VP-card law (spec 5.4) ---
        holders = list(plain.unknown_dev_holders())
        counts = [int(plain.dev_count[j]) for j in holders]
        for j, cnt in zip(holders, counts):
            _need(cnt == sum(held[j]), f"position {k}: tracker dev count {cnt} != {sum(held[j])} (seat {j})")
        key = (tuple(pool), tuple(counts), tuple(pvps[j] for j in holders))
        if key != self.law_key:
            self.law = dev_vp_law(pool, holders, counts, [pvps[j] for j in holders], self.vps_to_win)
            self.law_key = key
        for j in self.opps:
            law = self.law.get(j, [1.0])
            _need(abs(sum(law) - 1.0) <= 1e-9, f"position {k} seat {j}: VP law sums to {sum(law)}")
            enc = largest_remainder(law, 1000)
            if enc != self.last_law[j]:
                self.dv.append([k, j] + enc)
                self.last_law[j] = enc
            if self.want_truth:
                bel_opp[str(j)]["vpLaw"] = law
        if self.want_truth:
            self.beliefs.append({"nh": nh, "pool": pool, "opp": bel_opp})
        # --- twin (5.8 d) and audited copy (5.8 c) ---
        tw = self.twin
        if (tw.pending_discards or len(tw.counter.hyps) != 1
                or [list(h) for h in next(iter(tw.counter.hyps))] != [list(h) for h in hands_k]):
            self.twin_ok = False
        au = self.audit
        if au.counter.hyps != c.hyps or au.pending_discards != plain.pending_discards:
            self.same_ok = False
        self.fields.append((list(au.counter.last_bank) if au.counter.last_bank is not None else None,
                            list(au.counter.size)))
        if self.want_digests:
            self.digests.append(belief_digest(plain))

    def record_stats(self) -> None:
        """The tracker's statistics at this position (the state of the bot's tracker at its last decide)."""
        c = self.plain.counter
        self.stats_at = {"hidden_steals": int(self.plain.stats["hidden_steals"]),
                         "hidden_discards": int(self.plain.stats["hidden_discards"]),
                         "hidden_dev_draws": int(self.plain.stats["hidden_dev_draws"]),
                         "info_resets": int(c.stats["resets"]), "info_max_hypotheses": int(c.stats["max_hypotheses"])}

    def finish(self) -> None:
        for gen in self.gens:
            _need(next(gen, None) is None, "a tracker has entries left after the last logged action")

    def summary(self) -> Dict[str, Any]:
        c = self.plain.counter
        return {"allExact": _r(self.all_exact / max(1, self.positions)),
                # minP can be ~1e-9 (T11-186): relative precision, never rounded to a fixed number of decimals
                "meanP": _r(self.p_sum / max(1, self.pairs)), "minP": _sig(self.p_min),
                "calP": _r(self.calp_sum / max(1, self.pairs)),
                "meanPu": _r(self.pu_sum / self.nu) if self.nu else None,
                "calPu": _r(self.calpu_sum / self.nu) if self.nu else None, "nu": self.nu,
                "maxH": int(c.stats["max_hypotheses"]),
                "hidden": [int(self.plain.stats[k]) for k in HIDDEN3]}


def encode_game(rec: Dict[str, Any], ctx: Dict[str, Any], result_row: Optional[Dict[str, Any]] = None,
                want_truth: bool = False, want_digests: bool = False) -> Dict[str, Any]:
    """Replay one logged game exactly and encode it.  Returns ``{"record", "row", "checks", "truth",
    "digests"}``: the shard's game record, the index row (deal key in column ``d``), per-game check results,
    per-position truth (``want_truth``) and per-position belief digests (counted games, ``want_digests``)."""
    AD = _ENV.AD
    t0 = time.perf_counter()
    t, g = ctx["test"], int(rec["game"])
    gid = f"{t}-{g}"
    if AD.CATANATRON_VERSION != rec.get("catanatron"):
        raise ValueError(f"{gid} was played with catanatron {rec.get('catanatron')} but this interpreter "
                         f"({sys.executable}) runs catanatron {AD.CATANATRON_VERSION}: use the interpreter of "
                         f"that generation (--py33 / --py32)")
    colors = list(rec["colors"])
    n = len(colors)
    _need(colors == COLORS[:n], f"{gid}: seat colours {colors}")
    _need(int(rec["vps_to_win"]) == ctx["vps_to_win"], f"{gid}: vps_to_win {rec['vps_to_win']}")
    seat_of = {c: i for i, c in enumerate(colors)}
    ours = [int(x) for x in rec["our_seats"]]
    lineup = lineup_of(rec)
    _need([i for i, ch in enumerate(lineup) if ch == "c"] == ours, f"{gid}: our seats {ours} vs lineup {lineup}")
    actions = rec["actions"]
    N = len(actions)
    fin = rec["final"]
    geo, tile_index, edge_index = ctx["geo"], ctx["tile_index"], ctx["edge_index"]
    counted = bool(ctx["counted"])

    game = AD.rebuild_game(rec)            # raises ValueError for the other API generation
    _need(global_geometry(game) == geo, f"{gid}: geometry differs from the global geometry")
    land = [t_ for t_ in rec["board"]["tiles"] if t_["t"] == "land"]
    _need([list(t_["c"]) for t_ in land] == [t_[0] for t_ in geo["tiles"]], f"{gid}: land tile order differs")
    desert = [list(t_["c"]) for t_ in land if t_["resource"] is None]
    _need(desert == [list(rec["board"]["robber"])], f"{gid}: the robber does not start on the desert")
    bcode = board_code(rec)

    bot = None
    if counted:
        _need(len(ours) == 1, f"{gid}: a counted game must have one catanbot seat")
        me = ours[0]
        pinfo = next(p for p in rec["players"] if int(p["seat"]) == me).get("info")
        _need(pinfo == ctx["info"] == COUNTED_INFO, f"{gid}: info {pinfo} / results {ctx['info']}")
        final = AD.rebuild_game(rec)
        for item in actions:
            AD.replay_log_action(final, item, check=True)   # replay_log needs the complete engine log
        _check_final(gid, final, fin, N)
        bot = BotView(rec, final.state, me, ctx["info"], ctx["vps_to_win"], want_truth, want_digests)
        our_last = max(i for i, a in enumerate(actions) if seat_of[a[0]] == me)

    prev = seat_rows(game.state, n)
    hands = [[list(r["h"]) for r in prev]]
    banks = [[int(x) for x in game.state.resource_freqdeck]]
    truth = [truth_summary(game.state, edge_index)] if want_truth else None
    prev_lr = next((i for i, r in enumerate(prev) if r["LR"]), -1)
    prev_la = next((i for i, r in enumerate(prev) if r["LA"]), -1)
    steps: List[List[Any]] = []
    rolls = dev_bought = 0
    last_end = -1
    lr_other = False
    deficits: List[Tuple[int, int]] = [(0, 0)]
    opp_seats = [i for i in range(n) if i not in ours]
    for k in range(1, N + 1):
        item = actions[k - 1]
        color, kind, value, result = item
        a = seat_of[color]
        AD.replay_log_action(game, item, check=True)
        cur = seat_rows(game.state, n)
        v = encode_value(kind, value, result, seat_of, tile_index, edge_index)
        h: List[int] = []
        for s in range(n):
            for r in range(5):
                dlt = cur[s]["h"][r] - prev[s]["h"][r]
                if dlt:
                    h += [s * 5 + r, dlt]
        x: List[int] = []
        for s in range(n):
            for f, key in enumerate(("vp", "pvp", "lr")):
                if cur[s][key] != prev[s][key]:
                    x += [f * 4 + s, cur[s][key]]
        lr_holder = next((i for i, r in enumerate(cur) if r["LR"]), -1)
        la_holder = next((i for i, r in enumerate(cur) if r["LA"]), -1)
        _need(sum(r["LR"] for r in cur) <= 1 and sum(r["LA"] for r in cur) <= 1, f"{gid}: two title holders")
        if lr_holder != prev_lr:
            x += [12, lr_holder]
            if lr_holder != a:
                lr_other = True
        if la_holder != prev_la:
            x += [13, la_holder]
        prev_lr, prev_la = lr_holder, la_holder
        steps.append(step_tuple(KIND_CODE[kind] * 4 + a, v, h, x))
        if kind == "ROLL":
            rolls += 1
        elif kind == "END_TURN":
            last_end = k - 1
        elif kind == "BUY_DEVELOPMENT_CARD" and a in ours:
            dev_bought += 1
        hands.append([list(r["h"]) for r in cur])
        banks.append([int(x_) for x_ in game.state.resource_freqdeck])
        best_opp_vp = max(cur[s]["vp"] for s in opp_seats)
        best_opp_pvp = max(cur[s]["pvp"] for s in opp_seats)
        deficits.append((best_opp_vp - max(cur[s]["vp"] for s in ours),
                         best_opp_pvp - max(cur[s]["pvp"] for s in ours)))
        if bot is not None:
            bot.step(k, item, hands[k], game)
            if k == our_last:
                bot.record_stats()
        if want_truth:
            truth.append(truth_summary(game.state, edge_index))
        prev = cur
    if bot is not None:
        bot.finish()
    _check_final(gid, game, fin, N)
    winner = int(fin["winner_seat"])
    _need(winner >= 0, f"{gid}: no winner")
    vps = [r["vp"] for r in prev]
    pvps = [r["pvp"] for r in prev]
    upto = last_end + 1 if last_end >= 0 else N
    dmax = max(0, max(dv for dv, _ in deficits[:upto + 1]))
    dpmax = max(0, max(dp for _, dp in deficits[:upto + 1]))
    lr_end = next((i for i, r in enumerate(prev) if r["LR"]), -1)
    la_end = next((i for i, r in enumerate(prev) if r["LA"]), -1)
    record: Dict[str, Any] = {"id": gid, "g": g, "b": bcode, "n": n, "o": ours, "s": steps}
    checks: Dict[str, Any] = {"id": gid, "actions": N, "lrOther": lr_other,
                              "turnsFormula": int(fin["turns"]) == sum(1 for a_ in actions if a_[1] == "END_TURN")
                              + 2 * n - 2}
    bk = None
    digests = None
    if bot is not None:
        summ = bot.summary()
        logged = (result_row or {}).get("stats") or {}
        want = {k_: logged.get(k_) for k_ in ("hidden_steals", "hidden_discards", "hidden_dev_draws", "info_resets",
                                              "info_max_hypotheses")}
        stats_ok = bot.stats_at is not None and bot.stats_at == want
        viol = audit_violations(bot.audit.audit_calls, bot.audit.audit_flags, bot.fields, actions, hands, banks,
                                colors, bot.me, bot.discards_public)
        chk = {"code": CODE_PLAYED[t]["commit"], "stats": int(stats_ok),
               "audit": [len(bot.audit.audit_calls), len(viol)], "twin": int(bot.twin_ok), "same": int(bot.same_ok),
               "head": None}
        record["bv"] = {"q": bot.q, "nh": bot.nh, "dp": bot.dp, "dv": bot.dv, "sum": summ, "chk": chk}
        uncertain = logged.get("info_uncertain")
        bk = [_r4(summ["allExact"]), _r4(summ["meanP"]), _r4(summ["minP"]), _r4(summ["calP"]), _r4(summ["meanPu"]),
              _r4(summ["calPu"]), summ["maxH"], uncertain] + summ["hidden"]
        pl = bot.plain
        checks.update({"tracker": {"class": f"{type(pl).__module__}.{type(pl).__name__}", "me": bot.me,
                                   "discards_public": pl.discards_public, "reveal_hidden": pl.reveal_hidden,
                                   "vps_to_win": pl.vps_to_win, "max_hypotheses": pl.max_hypotheses},
                       "maxH": summ["maxH"], "stats": {"ours": bot.stats_at, "logged": want, "equal": stats_ok},
                       "audit": [len(bot.audit.audit_calls), len(viol)], "violations": viol[:20],
                       "twin": bot.twin_ok, "same": bot.same_ok,
                       "bvBytes": len(json.dumps(record["bv"], separators=(",", ":")))})
        _need(stats_ok, f"{gid}: tracker statistics {bot.stats_at} != logged {want}")
        _need(not viol, f"{gid}: {len(viol)} audit violations, first: {viol[:3]}")
        _need(bot.twin_ok, f"{gid}: the reveal_hidden twin is not exact / not the real hands")
        _need(bot.same_ok, f"{gid}: the audited tracker's belief differs from the plain tracker's")
        if want_digests:
            digests = bot.digests
        if want_truth:
            truth = {"pos": truth, "bv": bot.beliefs}
    elif want_truth:
        truth = {"pos": truth, "bv": None}
    row = [t, g, int(rec["seed"]), deal_key(rec), lineup, winner, vps, pvps, int(fin["turns"]), rolls, N,
           int(pvps[winner] < ctx["vps_to_win"]), vps[winner] - pvps[winner], dev_bought, lr_end, la_end, dmax, dpmax,
           bk]
    if result_row is not None:
        _need(int(result_row.get("winner_seat", -2)) == winner, f"{gid}: results winner {result_row.get('winner_seat')}")
        _need(list(result_row.get("vps", vps)) == vps, f"{gid}: results VPs {result_row.get('vps')}")
    checks["t"] = round(time.perf_counter() - t0, 3)
    return {"record": record, "row": row, "checks": checks, "truth": truth, "digests": digests}


def _r4(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(f"{x:.4g}")


def _check_final(gid: str, game, fin: Dict[str, Any], N: int) -> None:
    AD = _ENV.AD
    st = game.state
    w = game.winning_color()
    got = {"winner": w.value if w is not None else None,
           "vps": [int(st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]) for i in range(len(st.colors))],
           "turns": int(st.num_turns), "num_actions": len(AD.action_log(st)),
           "fingerprint": AD.state_fingerprint(st)}
    bad = [f"{k}: logged {fin.get(k)!r}, replayed {v!r}" for k, v in got.items() if fin.get(k) != v]
    _need(not bad and got["num_actions"] == N, f"{gid}: replay differs from the log's final state: {bad}")


def audit_game(rec: Dict[str, Any], info: Dict[str, Any], vps_to_win: int = 10,
               tracker_cls=None) -> Tuple[int, List[str]]:
    """Run an audited tracker (default ``make_audited_tracker(PublicInfoTracker)``) over one counted game in
    lockstep with an exact replay; ``(recorded calls, violations)``.  Used by the tests (a leaking subclass
    must be caught)."""
    AD = _ENV.AD
    cls = tracker_cls or make_audited_tracker(_ENV.PIT)
    colors = list(rec["colors"])
    me = int(rec["our_seats"][0])
    final = AD.rebuild_game(rec)
    for item in rec["actions"]:
        AD.replay_log_action(final, item, check=True)
    game = AD.rebuild_game(rec)
    tr = cls(AD.Color(colors[me]), discards_public=bool(info["discards_public"]), vps_to_win=int(vps_to_win))
    gen = tr.replay_log(final.state)
    n = len(colors)
    hands = [[list(r["h"]) for r in seat_rows(game.state, n)]]
    banks = [[int(x) for x in game.state.resource_freqdeck]]
    fields = []
    for item in rec["actions"]:
        AD.replay_log_action(game, item, check=True)
        next(gen)
        hands.append([list(r["h"]) for r in seat_rows(game.state, n)])
        banks.append([int(x) for x in game.state.resource_freqdeck])
        fields.append((list(tr.counter.last_bank) if tr.counter.last_bank is not None else None,
                       list(tr.counter.size)))
    viol = audit_violations(tr.audit_calls, tr.audit_flags, fields, rec["actions"], hands, banks, colors, me,
                            bool(info["discards_public"]))
    return len(tr.audit_calls), viol


# ---------------------------------------------------------------------------
# Worker and head-digest jobs
# ---------------------------------------------------------------------------
def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _gz(obj: Any) -> bytes:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _read_gz_json(path: str) -> Any:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)


def _ints(text: Optional[str]) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def worker_main(args) -> int:
    t, k = args.worker, int(args.shard)
    info = load_engine(args.code_root, expect_snapshot=t in SNAPSHOT_TESTS)
    metas, by_game, _ = load_results(t)
    meta = metas[0]
    tinfo = test_info(meta)
    for m in metas:
        _need(test_info(m) == tinfo, f"{t}: results files disagree on the information mode")
    counted = tinfo.get("mode") == "counted"
    _need(counted == (t in COUNTED_TESTS), f"{t}: information mode {tinfo} is not what the spec expects")
    files = log_files(t)
    n_games = files[-1]["to"]
    size = int(args.shard_size)
    lo, hi = k * size, min(n_games, (k + 1) * size)
    _need(lo < hi, f"{t}: shard {k} is empty")
    recs = read_games(files, lo, hi)
    only = set(_ints(args.games))
    truth_games = set(_ints(args.truth))
    geo = global_geometry(_ENV.AD.rebuild_game(recs[0]))
    tile_index, edge_index = geo_indexes(geo)
    ctx = {"test": t, "geo": geo, "tile_index": tile_index, "edge_index": edge_index, "info": tinfo,
           "counted": counted, "vps_to_win": int(meta.get("vps_to_win", 10))}
    name = shard_name(t, k)
    work, out = args.work, args.out
    records, rows, checks, truth, digests = [], [], [], {}, {}
    t_start = time.perf_counter()
    log_path = os.path.join(work, "logs", f"{name}.log" if not args.truth_only else f"{name}.truth.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    progress = open(log_path, "w")
    for rec in recs:
        g = int(rec["game"])
        if only and g not in only:
            continue
        res = encode_game(rec, ctx, by_game.get(g), want_truth=g in truth_games, want_digests=counted)
        records.append(res["record"])
        rows.append(res["row"])
        checks.append(res["checks"])
        if res["truth"] is not None:
            truth[res["record"]["id"]] = res["truth"]
        if res["digests"] is not None:
            digests[res["record"]["id"]] = res["digests"]
        if len(records) % 10 == 0 or g == hi - 1:
            progress.write(f"{res['record']['id']} {res['checks']['t']:.2f}s maxH {res['checks'].get('maxH', '-')} "
                           f"elapsed {time.perf_counter() - t_start:.1f}s\n")
            progress.flush()
    progress.close()
    if args.truth_only:
        _merge_truth(os.path.join(work, "truth", f"{name}.steps.json.gz"), truth)
        print(json.dumps({"shard": name, "truth": sorted(truth)}))
        return 0
    shard = {"v": FORMAT, "t": t, "from": lo, "to": hi, "games": records}
    shard_bytes = _gz(shard)
    no_bv = dict(shard, games=[{k_: v for k_, v in r.items() if k_ != "bv"} for r in records])
    nobv_bytes = len(_gz(no_bv)) if counted else len(shard_bytes)
    _atomic_write(os.path.join(out, shard_path(t, lo, size)), shard_bytes)
    _atomic_write(os.path.join(work, "parts", f"{name}.rows.json"), _json_bytes({"rows": rows}))
    _atomic_write(os.path.join(work, "parts", f"{name}.checks.json"),
                  _json_bytes({"shard": name, "test": t, "from": lo, "to": hi, "geo": geo, "code": info,
                               "info": tinfo, "games": checks, "shardBytes": len(shard_bytes),
                               "shardBytesNoBv": nobv_bytes, "truthGames": sorted(truth),
                               "seconds": round(time.perf_counter() - t_start, 2)}))
    if counted:
        _atomic_write(os.path.join(work, "digests", f"{name}.snap.json"), _json_bytes(digests))
    if truth:
        _atomic_write(os.path.join(work, "truth", f"{name}.steps.json.gz"), _gz(truth))
    print(json.dumps({"shard": name, "games": len(records), "bytes": len(shard_bytes),
                      "seconds": round(time.perf_counter() - t_start, 2)}))
    return 0


def _merge_truth(path: str, add: Dict[str, Any]) -> None:
    cur = _read_gz_json(path) if os.path.exists(path) else {}
    cur.update(add)
    _atomic_write(path, _gz(cur))


def digest_main(args) -> int:
    """Head cross-check: the same tracker from ``--code-root`` (the frozen head copy) over the same logs."""
    t, k = args.digest, int(args.shard)
    load_engine(args.code_root)
    AD, PIT = _ENV.AD, _ENV.PIT
    metas, _, _ = load_results(t)
    tinfo = test_info(metas[0])
    _need(tinfo.get("mode") == "counted", f"{t} is not a counted test")
    vps_to_win = int(metas[0].get("vps_to_win", 10))
    files = log_files(t)
    size = int(args.shard_size)
    lo, hi = k * size, min(files[-1]["to"], (k + 1) * size)
    only = set(_ints(args.games))
    out = {}
    t0 = time.perf_counter()
    for rec in read_games(files, lo, hi):
        if only and int(rec["game"]) not in only:
            continue
        me = int(rec["our_seats"][0])
        final = AD.rebuild_game(rec)
        for item in rec["actions"]:
            AD.replay_log_action(final, item, check=True)
        tr = PIT(AD.Color(rec["colors"][me]), discards_public=bool(tinfo["discards_public"]), vps_to_win=vps_to_win)
        out[f"{t}-{rec['game']}"] = [belief_digest(tr) for _ in tr.replay_log(final.state)]
    _atomic_write(os.path.join(args.work, "digests", f"{shard_name(t, k)}.head.json"), _json_bytes(out))
    print(json.dumps({"digest": shard_name(t, k), "games": len(out), "seconds": round(time.perf_counter() - t0, 2)}))
    return 0


# ---------------------------------------------------------------------------
# Orchestrator (stdlib only)
# ---------------------------------------------------------------------------
def _tree_files(root: str) -> List[str]:
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc") or f == "SHA256SUMS":
                continue
            out.append(os.path.relpath(os.path.join(base, f), root))
    return sorted(out)


def _sums_text(root: str) -> str:
    return "".join(f"{sha256_file(os.path.join(root, rel))}  {rel}\n" for rel in _tree_files(root))


def prepare_code_head(work: str, fresh: bool, interps: Sequence[str], repo: str = REPO) -> Dict[str, Any]:
    """Frozen copy of the repository's ``catanbot/`` and ``scripts/replay_catanatron.py`` (spec 6.2)."""
    head = os.path.join(work, "code_head")
    sums_path = os.path.join(head, "SHA256SUMS")
    reuse = False
    if os.path.exists(sums_path) and not fresh:
        with open(sums_path) as fh:
            recorded = fh.read()
        reuse = recorded == _sums_text(head)
        if not reuse:
            print(f"code_head: existing copy does not match its SHA256SUMS; re-copying", flush=True)
    if not reuse:
        if os.path.exists(head):
            shutil.rmtree(head)
        shutil.copytree(os.path.join(repo, "catanbot"), os.path.join(head, "catanbot"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        os.makedirs(os.path.join(head, "scripts"), exist_ok=True)
        shutil.copy2(os.path.join(repo, "scripts", "replay_catanatron.py"), os.path.join(head, "scripts"))
        text = _sums_text(head)
        with open(sums_path, "w") as fh:
            fh.write(text)
    with open(sums_path) as fh:
        text = fh.read()
    tree = hashlib.sha256(text.encode()).hexdigest()
    for py in sorted(set(interps)):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import catanbot; "
                "from catanbot.bench import catanatron_adapter as AD; "
                "from catanbot.bench.public_info import PublicInfoTracker; "
                "import os; f = os.path.abspath(catanbot.__file__); "
                "assert f.startswith(os.path.abspath(sys.argv[1]) + os.sep), f; print(AD.CATANATRON_VERSION)")
        r = subprocess.run([py, "-c", code, head], capture_output=True, text=True, cwd="/")
        if r.returncode != 0:
            raise ExportError(f"code_head does not import under {py}: {r.stderr.strip()[-800:]}")
    return {"path": head, "tree": tree, "reused": reuse, "files": len(text.splitlines())}


def _first_record(t: str) -> Dict[str, Any]:
    f = log_files(t)[0]
    with gzip.open(f["path"], "rt") as fh:
        return json.loads(fh.readline())


def plan_test(t: str) -> Dict[str, Any]:
    suite, d = test_dir(t)
    files = log_files(t)
    rec0 = _first_record(t)
    metas, by_game, rfiles = load_results(t)
    return {"t": t, "suite": suite, "dir": d, "files": files, "games": files[-1]["to"],
            "catanatron": rec0["catanatron"], "metas": metas, "by_game": by_game, "resultFiles": rfiles,
            "info": test_info(metas[0])}


def truth_targets(plan: Dict[str, Any], per_test: int) -> Tuple[Dict[int, List[str]], List[int]]:
    """Games whose per-position truth is written (spec 6.5), as ``({g: [reasons]}, games)``: the random sample
    and the log-only targeted scans (first match by g).  The Longest Road scan needs the engine and is done
    after the workers (from their ``lrOther`` flags)."""
    t, N = plan["t"], plan["games"]
    chosen: Dict[int, List[str]] = {}

    def add(g: int, why: str) -> None:
        chosen.setdefault(int(g), []).append(why)

    for g in random.Random(f"{t}-{TRUTH_SEED}").sample(range(N), min(per_test, N)):
        add(g, "random")
    if t in COUNTED_TESTS:
        maxh = sorted(((int((r.get("stats") or {}).get("info_max_hypotheses", 0)), g)
                       for g, r in plan["by_game"].items()), reverse=True)
        if maxh:
            add(maxh[0][1], "heaviest hypotheses")
        for h, g in maxh:
            if h >= 1500:
                add(g, f"{h} hypotheses")
    want_scan = t in COUNTED_TESTS or t in ("R1", "R2")
    if want_scan:
        found = {}
        for rec in iter_all_games(plan["files"]):
            g = int(rec["game"])
            seat_of = {c: i for i, c in enumerate(rec["colors"])}
            me = int(rec["our_seats"][0])
            acts = rec["actions"]
            if t in ("R1", "R2") and "big discard" not in found:
                if any(a[1] == "DISCARD" and len(a[3] or a[2]) >= 9 for a in acts):
                    found["big discard"] = g
            if t in COUNTED_TESTS:
                run: set = set()
                turn_hidden = turn_mono = False
                thief = victim_me = False
                for a in acts:
                    who = seat_of[a[0]]
                    if a[1] in DISCARD_KINDS:
                        if who != me:
                            run.add(who)
                            turn_hidden = True
                        continue
                    if a[1] == "MOVE_ROBBER":
                        v = a[2][1] if len(a[2]) > 1 else None
                        stole = a[3] is not None or (len(a[2]) > 2 and a[2][2] is not None)
                        if run and v is not None and seat_of[v] in run and stole and "steal from discarder" not in found:
                            found["steal from discarder"] = g
                        if stole and who == me:
                            thief = True
                        if stole and v is not None and seat_of[v] == me:
                            victim_me = True
                    if a[1] == "PLAY_MONOPOLY":
                        turn_mono = True
                    if turn_hidden and turn_mono and "monopoly with hidden discards" not in found:
                        found["monopoly with hidden discards"] = g
                    if a[1] == "END_TURN":
                        turn_hidden = turn_mono = False
                    run = set()
                if thief and victim_me and "our bot thief and victim" not in found:
                    found["our bot thief and victim"] = g
        for why, g in found.items():
            add(g, why)
    return chosen, sorted(chosen)


class Job:
    def __init__(self, name: str, cmd: List[str], log: str):
        self.name, self.cmd, self.log = name, cmd, log
        self.proc = None
        self.t0 = 0.0
        self.seconds = 0.0
        self.rc: Optional[int] = None
        self.out = ""


def run_jobs(jobs: Sequence[Job], workers: int, export_log: str) -> List[Job]:
    """Run ``jobs`` with at most ``workers`` at once; one line per finished job in ``export_log``."""
    pending = list(jobs)
    running: List[Job] = []
    done: List[Job] = []
    with open(export_log, "a") as lg:
        while pending or running:
            while pending and len(running) < workers:
                j = pending.pop(0)
                os.makedirs(os.path.dirname(j.log), exist_ok=True)
                j.fh = open(j.log, "w")
                j.t0 = time.perf_counter()
                j.proc = subprocess.Popen(j.cmd, stdout=j.fh, stderr=subprocess.STDOUT, cwd=REPO)
                running.append(j)
            time.sleep(0.2)
            for j in list(running):
                rc = j.proc.poll()
                if rc is None:
                    continue
                j.fh.close()
                j.rc = rc
                j.seconds = time.perf_counter() - j.t0
                with open(j.log) as fh:
                    j.out = fh.read()
                line = (f"{_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')} {j.name} "
                        f"{'OK' if rc == 0 else 'FAILED rc=' + str(rc)} {j.seconds:.1f}s")
                lg.write(line + "\n")
                lg.flush()
                print(line, flush=True)
                if rc != 0:
                    print("   " + "\n   ".join(j.out.strip().splitlines()[-8:]), flush=True)
                running.remove(j)
                done.append(j)
    return done


def _worker_cmd(args, py: str, t: str, k: int, root: str, extra: Sequence[str] = ()) -> List[str]:
    return ["nice", "-n", "10", py, os.path.abspath(__file__), "--worker", t, "--shard", str(k), "--code-root", root,
            "--out", args.out, "--work", args.work, "--shard-size", str(args.shard_size), *extra]


def _load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def orchestrate(args) -> int:
    t_all = time.perf_counter()
    tests = [t.strip() for t in args.tests.split(",")] if args.tests else list(TEST_ORDER)
    unknown = [t for t in tests if t not in TEST_ORDER]
    if unknown:
        raise SystemExit(f"unknown tests {unknown}")
    tests = [t for t in TEST_ORDER if t in tests]
    if args.workers > MAX_WORKERS or args.workers < 1:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} on this machine")
    for d in (args.out, args.work):
        os.makedirs(d, exist_ok=True)
    plans = {t: plan_test(t) for t in tests}
    interps = {t: interpreter_for(plans[t]["catanatron"], args.py33, args.py32) for t in tests}
    export_log = os.path.join(args.work, "export.log")
    head = prepare_code_head(args.work, args.fresh_code, [args.py33, args.py32])   # spec 6.2: both interpreters
    print(f"code_head: {head['path']} ({head['files']} files, tree {head['tree'][:16]}, "
          f"{'reused' if head['reused'] else 'copied'})", flush=True)
    with open(export_log, "a") as lg:
        lg.write(f"--- export {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} tests "
                 f"{','.join(tests)} workers {args.workers} code_head {head['tree'][:16]}\n")
    failures: List[str] = []
    truth_why: Dict[str, Dict[int, List[str]]] = {t: truth_targets(plans[t], args.truth_per_test)[0] for t in tests}
    if not args.merge_only:
        jobs: List[Job] = []
        for t in tests:
            p = plans[t]
            chosen = truth_why[t]
            root = code_root_for(t, args.snapshot2, head["path"])
            nshards = math.ceil(p["games"] / args.shard_size)
            for k in range(nshards):
                name = shard_name(t, k)
                lo, hi = k * args.shard_size, (k + 1) * args.shard_size
                tg = [g for g in sorted(chosen) if lo <= g < hi]
                extra = ["--truth", ",".join(map(str, tg))] if tg else []
                if args.resume and _shard_done(args, t, k):
                    continue
                jobs.append(Job(f"worker {name}", _worker_cmd(args, interps[t], t, k, root, extra),
                                os.path.join(args.work, "logs", f"{name}.worker.out")))
                if t in COUNTED_TESTS:
                    jobs.append(Job(f"digest {name}", ["nice", "-n", "10", interps[t], os.path.abspath(__file__),
                                                       "--digest", t, "--shard", str(k), "--code-root", head["path"],
                                                       "--work", args.work, "--shard-size", str(args.shard_size)],
                                    os.path.join(args.work, "logs", f"{name}.digest.out")))
        # heavy counted shards first so the tail does not come last
        jobs.sort(key=lambda j: (0 if any(f" {t}_" in j.name for t in COUNTED_TESTS) else 1))
        done = run_jobs(jobs, args.workers, export_log)
        failures += [f"{j.name}: rc {j.rc}: {j.out.strip().splitlines()[-1] if j.out.strip() else ''}"
                     for j in done if j.rc != 0]
    # Longest Road changing hands on another player's turn: first game by g (test order), from the workers' flags
    if not failures:
        lr_game = None
        for t in tests:
            for k in range(math.ceil(plans[t]["games"] / args.shard_size)):
                ck = _load_json(os.path.join(args.work, "parts", f"{shard_name(t, k)}.checks.json"))
                hit = next((c for c in ck["games"] if c.get("lrOther")), None)
                if hit:
                    lr_game = (t, int(hit["id"].split("-")[1]), k)
                    break
            if lr_game:
                break
        if lr_game:
            t, g, k = lr_game
            truth_why[t].setdefault(g, []).append("Longest Road changes on another's turn")
            tpath = os.path.join(args.work, "truth", f"{shard_name(t, k)}.steps.json.gz")
            have = set(_read_gz_json(tpath)) if os.path.exists(tpath) else set()
            if f"{t}-{g}" not in have:
                if args.merge_only:
                    failures.append(f"truth file lacks {t}-{g} (run without --merge-only)")
                else:
                    j = Job(f"truth {t}-{g}", _worker_cmd(args, interps[t], t, k, code_root_for(t, args.snapshot2,
                                                                                               head["path"]),
                                                          ["--games", str(g), "--truth", str(g), "--truth-only"]),
                            os.path.join(args.work, "logs", f"{shard_name(t, k)}.truth.out"))
                    done2 = run_jobs([j], 1, export_log)
                    failures += [f"{x.name}: rc {x.rc}" for x in done2 if x.rc != 0]
    if failures:
        print("FAILED jobs:\n  " + "\n  ".join(failures), flush=True)
        return 1
    ok = merge(args, tests, plans, head, truth_why)
    print(f"total wall time {time.perf_counter() - t_all:.1f}s", flush=True)
    if ok and args.validate:
        ok = run_validator(args)
    return 0 if ok else 1


def _shard_done(args, t: str, k: int) -> bool:
    name = shard_name(t, k)
    need = [os.path.join(args.out, shard_path(t, k * args.shard_size, args.shard_size)),
            os.path.join(args.work, "parts", f"{name}.rows.json"), os.path.join(args.work, "parts", f"{name}.checks.json")]
    if t in COUNTED_TESTS:
        need += [os.path.join(args.work, "digests", f"{name}.snap.json"),
                 os.path.join(args.work, "digests", f"{name}.head.json")]
    return all(os.path.exists(p) for p in need)


def merge(args, tests: Sequence[str], plans: Dict[str, Dict[str, Any]], head: Dict[str, Any],
          truth_why: Dict[str, Dict[int, List[str]]]) -> bool:
    """Compare head digests, merge the index parts, check them against the results files and manifests, write
    ``index.json.gz`` (and ``index.html`` when the page template exists); print the per-test table."""
    errors: List[str] = []
    geo = None
    rows: List[List[Any]] = []
    tests_meta: Dict[str, Any] = {}
    table = []
    manifests = {s: read_manifest(os.path.join(REPO, s)) for s in SUITES}
    for t in tests:
        p = plans[t]
        nshards = math.ceil(p["games"] / args.shard_size)
        t_rows, t_checks, shard_bytes, nobv_bytes, code_info = [], [], 0, 0, None
        truth_games: List[str] = []
        head_same = [0, 0]
        drift = []
        for k in range(nshards):
            name = shard_name(t, k)
            ck = _load_json(os.path.join(args.work, "parts", f"{name}.checks.json"))
            rw = _load_json(os.path.join(args.work, "parts", f"{name}.rows.json"))["rows"]
            if geo is None:
                geo = ck["geo"]
            elif ck["geo"] != geo:
                errors.append(f"{name}: geometry differs from the other shards")
            code_info = code_info or ck["code"]
            if ck["code"]["sha256"] != code_info["sha256"]:
                errors.append(f"{name}: code root differs between shards")
            t_rows += rw
            t_checks += ck["games"]
            truth_games += ck.get("truthGames", [])
            nobv_bytes += ck["shardBytesNoBv"]
            spath = os.path.join(args.out, shard_path(t, k * args.shard_size, args.shard_size))
            if t in COUNTED_TESTS:
                snap = _load_json(os.path.join(args.work, "digests", f"{name}.snap.json"))
                hd = _load_json(os.path.join(args.work, "digests", f"{name}.head.json"))
                shard = _read_gz_json(spath)
                changed = False
                for rec in shard["games"]:
                    a, b = snap.get(rec["id"]), hd.get(rec["id"])
                    same = a is not None and a == b
                    head_same[1] += 1
                    head_same[0] += int(same)
                    if not same:
                        first = next((i + 1 for i, (x, y) in enumerate(zip(a or [], b or [])) if x != y),
                                     min(len(a or []), len(b or [])) + 1)
                        drift.append(f"{rec['id']} (first differing position {first})")
                    if rec["bv"]["chk"].get("head") != int(same):
                        rec["bv"]["chk"]["head"] = int(same)
                        changed = True
                if changed:
                    _atomic_write(spath, _gz(shard))
            shard_bytes += os.path.getsize(spath)
        if drift and not args.allow_head_drift:
            errors.append(f"{t}: head tracker differs from the snapshot in {len(drift)} games: {drift[:5]}")
        # completeness and results
        gs = sorted(r[1] for r in t_rows)
        if gs != list(range(p["games"])):
            errors.append(f"{t}: index has {len(gs)} rows, games 0..{p['games'] - 1} expected exactly once")
        pub_games = sum(int(m.get("games", 0)) for m in p["metas"])
        pub_wins = sum(int(m.get("wins", 0)) for m in p["metas"])
        wins = 0
        for r in t_rows:
            res = p["by_game"].get(r[1])
            our_win = r[4][r[5]] == "c"
            wins += our_win
            if res is None:
                errors.append(f"{t}-{r[1]}: no per-game result")
            elif int(res["winner_seat"]) != r[5] or bool(res.get("won")) != our_win:
                errors.append(f"{t}-{r[1]}: winner {r[5]} / results {res['winner_seat']} won={res.get('won')}")
        if pub_games != p["games"] or pub_wins != wins or len(p["by_game"]) != p["games"]:
            errors.append(f"{t}: published {pub_games} games / {pub_wins} wins, index {len(t_rows)} / {wins}")
        # labels vs metadata
        lab = TEST_LABELS[t]
        meta = p["metas"][0]
        if (meta.get("opponent") != lab["opp"] or meta.get("format") != lab["fmt"]
                or p["info"].get("mode") != lab["info"] or not str(meta.get("catanatron")).startswith(lab["cat"])):
            errors.append(f"{t}: TEST_LABELS {lab} disagree with the metadata "
                          f"({meta.get('opponent')}, {meta.get('format')}, {p['info']}, {meta.get('catanatron')})")
        files = []
        for f in p["files"]:
            sha = sha256_file(f["path"])
            listed = manifests[p["suite"]].get(f["suiteRel"])
            if listed != sha:
                errors.append(f"{f['rel']}: sha256 {sha[:16]} vs MANIFEST {listed}")
            files.append({"path": f["rel"], "from": f["from"], "to": f["to"], "sha256": sha, "manifest": listed == sha})
        root = code_root_for(t, args.snapshot2, head["path"])
        entry = {"suite": p["suite"], "label": lab["label"], "group": lab["group"], "fmt": lab["fmt"],
                 "opp": meta.get("opponent"), "catanatron": p["catanatron"], "info": p["info"], "spec": meta.get("spec"),
                 "seed": meta.get("seed"), "games": p["games"], "shards": nshards,
                 "published": {"games": pub_games, "wins": pub_wins, "resultFiles": len(p["resultFiles"])},
                 "code": {"played": CODE_PLAYED[t]["played"], "commit": CODE_PLAYED[t]["commit"],
                          "worktree": CODE_PLAYED[t]["worktree"],
                          "exportRoot": root if root == args.snapshot2 else "code_head (frozen copy of the repository)",
                          "exportTree": None if root == args.snapshot2 else head["tree"],
                          "sha256": code_info["sha256"], "catanatron": code_info["catanatron"],
                          "headSame": head_same if t in COUNTED_TESTS else None},
                 "files": files,
                 "hvp": {"needed": sum(r[11] for r in t_rows),
                         "ours": sum(r[11] for r in t_rows if r[4][r[5]] == "c")},
                 "truthSample": {str(g): why for g, why in sorted(truth_why.get(t, {}).items())}}
        missing = [g for g in truth_why.get(t, {}) if f"{t}-{g}" not in set(truth_games)]
        tpaths = glob.glob(os.path.join(args.work, "truth", f"{t}_*.steps.json.gz"))
        have = set().union(*[set(_read_gz_json(x)) for x in tpaths]) if tpaths else set()
        missing = [g for g in missing if f"{t}-{g}" not in have]
        if missing:
            errors.append(f"{t}: per-position truth missing for sampled games {missing}")
        if t in COUNTED_TESTS:
            bks = [r[18] for r in t_rows]
            pu = [b[4] for b in bks if b[4] is not None]
            cu = [b[5] for b in bks if b[5] is not None]
            entry["counted"] = {"allExact": _r4(_mean([b[0] for b in bks])), "meanP": _r4(_mean([b[1] for b in bks])),
                                "calP": _r4(_mean([b[3] for b in bks])), "meanPu": _r4(_mean(pu)),
                                "calPu": _r4(_mean(cu)), "devBelief": DEV_BELIEF_TEXT,
                                "checks": _check_totals(t_checks)}
            bad = [c["id"] for c in t_checks if not (c["stats"]["equal"] and c["audit"][1] == 0 and c["twin"] and c["same"])]
            if bad:
                errors.append(f"{t}: bot-view checks failed in {bad[:5]}")
        tests_meta[t] = entry
        rows += sorted(t_rows, key=lambda r: r[1])
        slow = max(t_checks, key=lambda c: c["t"])
        table.append((t, p["games"], shard_bytes, shard_bytes - nobv_bytes if t in COUNTED_TESTS else 0,
                      _check_totals(t_checks) if t in COUNTED_TESTS else None,
                      head_same if t in COUNTED_TESTS else None, f"{slow['id']} {slow['t']:.2f}s",
                      sum(c["t"] for c in t_checks)))
    # deal ids
    deal_ids: Dict[str, int] = {}
    seed_deal: Dict[int, int] = {}
    exceptions = []
    for r in rows:
        key = r[3]
        if key not in deal_ids:
            deal_ids[key] = len(deal_ids)
        r[3] = deal_ids[key]
        if r[2] in seed_deal and seed_deal[r[2]] != r[3]:
            exceptions.append(f"{r[0]}-{r[1]}")
        seed_deal.setdefault(r[2], r[3])
    if exceptions:
        print(f"NOTE: same seed, different deal in {len(exceptions)} games: {exceptions[:10]}", flush=True)
    index = {"v": FORMAT, "built": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "shardSize": args.shard_size, "geo": geo, "tests": tests_meta, "cols": INDEX_COLS, "rows": rows,
             "dealExceptions": exceptions}
    if set(tests) == set(TEST_ORDER) and len(rows) != 9600:
        errors.append(f"index has {len(rows)} rows, 9,600 expected")
    index_bytes = _gz(index)
    _atomic_write(os.path.join(args.out, "index.json.gz"), index_bytes)
    page = write_page(args.out)
    # table
    print(f"\n{'test':<5} {'games':>6} {'shard KB':>9} {'bot KB':>7} {'checks (stats/audit calls/viol/twin/same)':<44} "
          f"{'headSame':>10} {'CPU s':>7}  slowest")
    for t, games, sb, bb, ct, hs, slow, cpu in table:
        cts = (f"{ct['stats']}/{ct['auditCalls']}/{ct['violations']}/{ct['twin']}/{ct['same']}" if ct else "-")
        hss = f"{hs[0]}/{hs[1]}" if hs else "-"
        print(f"{t:<5} {games:>6} {sb / 1024:>9.1f} {bb / 1024:>7.1f} {cts:<44} {hss:>10} {cpu:>7.1f}  {slow}")
    total = sum(os.path.getsize(os.path.join(args.out, shard_path(t, k * args.shard_size, args.shard_size)))
                for t in tests for k in range(math.ceil(plans[t]["games"] / args.shard_size)))
    print(f"index.json.gz {len(index_bytes) / 1024:.1f} KB ({len(rows)} rows); shards {total / 1024 / 1024:.2f} MB; "
          f"index.html {'written' if page else 'not written (page template missing)'}")
    if errors:
        print("ERRORS:\n  " + "\n  ".join(errors[:40]), flush=True)
        return False
    return True


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _check_totals(checks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {"games": len(checks), "stats": sum(int(c["stats"]["equal"]) for c in checks),
            "auditCalls": sum(c["audit"][0] for c in checks), "violations": sum(c["audit"][1] for c in checks),
            "twin": sum(int(c["twin"]) for c in checks), "same": sum(int(c["same"]) for c in checks)}


def write_page(out: str) -> bool:
    """``index.html`` = the page template with ``replay_core.js`` inlined between the markers (spec 2.1)."""
    tpl, core = PAGE_TEMPLATE, PAGE_CORE
    if not (os.path.exists(tpl) and os.path.exists(core)):
        return False
    with open(tpl, encoding="utf-8") as fh:
        html = fh.read()
    with open(core, encoding="utf-8") as fh:
        js = fh.read()
    begin, end = "/* BEGIN replay_core.js */", "/* END replay_core.js */"
    a, b = html.find(begin), html.find(end)
    if a < 0 or b < a:
        raise ExportError("page template lacks the replay_core.js markers")
    html = html[:a + len(begin)] + js + html[b:]
    _atomic_write(os.path.join(out, "index.html"), html.encode("utf-8"))
    return True


def run_validator(args) -> bool:
    js = VALIDATOR
    if not os.path.exists(js):
        print(f"--validate: {js} does not exist", flush=True)
        return False
    cmd = ["node", js, "--bundle", args.out, "--repo", REPO, "--truth", os.path.join(args.work, "truth"),
           "--core", PAGE_CORE, "--template", PAGE_TEMPLATE]
    print("validate: " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tests", default=None, help="comma-separated tests (default: all 18)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="bundle directory (published)")
    ap.add_argument("--work", default=DEFAULT_WORK, help="work directory (never published)")
    ap.add_argument("--workers", type=int, default=2, help=f"parallel jobs (1..{MAX_WORKERS})")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--truth-per-test", type=int, default=4, help="random games per test with per-position truth")
    ap.add_argument("--py33", default=DEFAULT_PY33)
    ap.add_argument("--py32", default=DEFAULT_PY32)
    ap.add_argument("--snapshot2", default=SNAPSHOT2)
    ap.add_argument("--fresh-code", action="store_true", help="re-copy the frozen head code")
    ap.add_argument("--allow-head-drift", action="store_true", help="record head/snapshot digest mismatches")
    ap.add_argument("--validate", action="store_true", help="run the node validator afterwards")
    ap.add_argument("--merge-only", action="store_true", help="skip the jobs; merge the existing parts")
    ap.add_argument("--resume", action="store_true", help="skip shards whose outputs already exist")
    # internal
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--digest", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--code-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--games", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--truth", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--truth-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.out = os.path.abspath(args.out)
    args.work = os.path.abspath(args.work)
    if args.worker:
        return worker_main(args)
    if args.digest:
        return digest_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
