"""The promotion gate: candidate vs every champion, sequential, resumable, with a written verdict.

Design (details and rationale in ``docs/LEAGUE.md``):

* **Comparisons.**  One per (champion, format); the *primary* comparison is
  the current champion in the first format (default ``4p2v2``: two candidate
  seats and two champion seats at a 4-player table).  Format ``NpKvM`` puts
  ``K`` candidate seats among ``N``; the ``C(N, K)`` seat arrangements (turn
  orders) are rotated game by game, so every block of ``C(N, K)`` games shares
  one board + dice seed and gives every seat to each side equally often.  The
  null share of candidate wins is ``K / N`` (0.5 in 2v2, 1/3 in 3-player 1v2).
* **Seeds.**  Fixed per (champion, format, block) from ``seed_base``, not per
  candidate: every gate against champion X is played on the same boards and
  dice, so gates are reproducible and comparable.  A seat's bot seed depends
  only on the block and the seat - a candidate identical to the champion wins
  *exactly* half of every 2v2 block (a built-in harness self-test).
* **Batches / looks.**  ``batch_games`` games per comparison per look, up to
  ``max_games``.  After each look: the exact alpha-spending boundaries of
  ``sequential.py`` decide ``success`` (primary crosses the upper boundary at
  ``alpha``), ``fail`` (any comparison crosses its lower boundary at
  ``alpha_fail / #comparisons``: clearly worse), ``futility`` (optional,
  non-binding: primary share <= null + margin after half the games) or
  continue; the last look spends all the alpha.
* **Verdict** (at the stop): PASS iff (a) the primary crossed its success
  boundary *and* its one-sided exact p < ``alpha`` *and* the lower end of its
  ``ci`` Clopper-Pearson interval > the null share; (b) no other comparison
  (earlier champions, other formats) is significantly worse - exact one-sided
  "worse" p-values, Holm-adjusted across those comparisons, all > ``alpha_fail``
  (and no interim failure); (c) the external non-regression hook, when
  configured, passes; and the gate is valid (errors <= ``max_errors``, voided
  games <= ``max_voids``).
* **Records.**  ``games.jsonl`` (one line per game), ``looks.jsonl`` (the
  evaluation after each look), ``verdict.json``, ``progress.json``, plus
  ``gate.json`` (the frozen configuration).  Everything is recomputed from the
  game records, so an interrupted gate resumes exactly where it stopped.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import sequential as Q
from .match import BotServer, GameVoid, ServerSeat, Seat, play_match_game, server_env
from .registry import (Champion, LeagueError, Registry, atomic_write_json, now_iso, sha256_file, spec_with_model,
                       tree_commit)
from .stats import (binom_test_greater, binom_test_less, binom_test_two_sided, clopper_pearson, fisher_less, holm)

DEFAULT_SEED_BASE = 20260925


# ---------------------------------------------------------------------------
# Formats, arrangements, seeds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Format:
    name: str
    n: int       # players
    k: int       # candidate seats

    @property
    def arrangements(self) -> List[Tuple[int, ...]]:
        return list(itertools.combinations(range(self.n), self.k))

    @property
    def p0(self) -> float:
        return self.k / self.n

    def pattern(self, cand_seats: Sequence[int]) -> str:
        return "".join("C" if s in cand_seats else "H" for s in range(self.n))


def parse_format(name: str) -> Format:
    m = re.fullmatch(r"(\d)p(\d)v(\d)", name.strip())
    if not m:
        raise ValueError(f"bad format {name!r} (expected e.g. 4p2v2 or 3p1v2)")
    n, k, h = map(int, m.groups())
    if k + h != n or k < 1 or h < 1 or not 2 <= n <= 4:
        raise ValueError(f"bad format {name!r}: need K + M = N players (2-4), both sides seated")
    return Format(name.strip(), n, k)


def _h32(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0x7FFFFFFF


def block_seed(seed_base: int, champion: str, fmt: str, block: int) -> int:
    """Board + dice seed of a block: fixed per (champion, format, block), independent of the candidate."""
    return _h32(f"catanbot-league:{seed_base}:{champion}:{fmt}:{block}")


def bot_seed(engine_seed: int, seat: int) -> int:
    """A seat's private bot rng seed: the same whichever side sits there (mirror property)."""
    return _h32(f"catanbot-league-bot:{engine_seed}:{seat}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GateRule:
    alpha: float = 0.01            # (a) one-sided level vs the current champion (sequentially spent)
    alpha_fail: float = 0.05       # (b) "significantly worse" level (Holm family / interim Bonferroni)
    ci: float = 0.95               # (a) Clopper-Pearson level of the lower bound
    spending: str = "obf"          # obf | pocock
    futility_margin: Optional[float] = 0.0   # stop when primary share <= null + margin (None: off)
    futility_min_t: float = 0.5    # ... but only once this information fraction is reached
    holm: bool = True
    max_errors: int = 0            # illegal actions + bot exceptions, both sides, whole gate
    max_voids: int = 3             # games voided by server crashes / timeouts


@dataclass
class GateConfig:
    gate_id: str
    candidate: Dict[str, Any]              # name, spec, spec_resolved, commit, dirty, tree, weights
    champions: List[Dict[str, Any]]        # current first: name, commit, spec, spec_resolved, tree, env, shim, role
    formats: List[str] = field(default_factory=lambda: ["4p2v2"])
    batch_games: int = 198
    max_games: int = 1188
    max_turns: int = 400
    seed_base: int = DEFAULT_SEED_BASE
    rule: GateRule = field(default_factory=GateRule)
    external: Optional[Dict[str, Any]] = None
    engine: Dict[str, Any] = field(default_factory=dict)
    created: str = ""
    registry: str = ""
    workers: int = 1
    check_roundtrip: bool = True
    decide_timeout: float = 600.0
    require_accel: bool = True
    notes: str = ""

    # --- (de)serialisation -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema"] = 1
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GateConfig":
        d = dict(d)
        d.pop("schema", None)
        rule = GateRule(**d.pop("rule", {}))
        return GateConfig(rule=rule, **d)

    # --- schedule ----------------------------------------------------------------------------
    def comparisons(self) -> List[Tuple[str, str]]:
        """``(champion, format)`` pairs; the first is the primary (current champion, first format)."""
        return [(c["name"], f) for c in self.champions for f in self.formats]

    @property
    def num_looks(self) -> int:
        return max(1, math.ceil(self.max_games / self.batch_games))

    def look_end(self, k: int) -> int:
        return min(k * self.batch_games, self.max_games)

    def champion(self, name: str) -> Dict[str, Any]:
        for c in self.champions:
            if c["name"] == name:
                return c
        raise KeyError(name)


def comp_key(c: Tuple[str, str]) -> str:
    return f"{c[0]}/{c[1]}"


def game_plan(cfg: GateConfig, comp: Tuple[str, str], index: int) -> Dict[str, Any]:
    """Seats and seeds of game ``index`` of a comparison (pure function of the configuration)."""
    fmt = parse_format(comp[1])
    arrs = fmt.arrangements
    block, arr = divmod(index, len(arrs))
    cand = arrs[arr]
    es = block_seed(cfg.seed_base, comp[0], comp[1], block)
    return {"block": block, "arrangement": arr, "cand_seats": list(cand), "pattern": fmt.pattern(cand),
            "seed": es, "bot_seeds": [bot_seed(es, s) for s in range(fmt.n)], "n": fmt.n}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def load_records(gate_dir: Path) -> Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]:
    """``{(champion, format): {index: record}}``; a non-void record supersedes a void one (retried game)."""
    out: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    p = gate_dir / "games.jsonl"
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue   # a line cut by an interruption
            if r.get("type") != "game":
                continue
            key = (r["champion"], r["format"])
            prev = out.setdefault(key, {}).get(r["index"])
            if prev is None or prev.get("void") or not r.get("void"):
                out[key][r["index"]] = r
    return out


def summarize(recs: Sequence[Dict[str, Any]], fmt: Format, ci: float = 0.95) -> Dict[str, Any]:
    valid = [r for r in recs if not r.get("void")]
    decisive = [r for r in valid if r["winner"] >= 0]
    n = len(decisive)
    wins = sum(1 for r in decisive if r["cand_win"])
    diffs = [r["vp_diff"] for r in valid]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    sd = math.sqrt(sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    lo, hi = clopper_pearson(wins, n, ci)
    arr: Dict[str, Dict[str, int]] = {}
    for r in valid:
        a = arr.setdefault(r["pattern"], {"games": 0, "decisive": 0, "wins": 0})
        a["games"] += 1
        if r["winner"] >= 0:
            a["decisive"] += 1
            a["wins"] += int(bool(r["cand_win"]))
    side_err = {"C": {"illegal": 0, "errors": 0, "observe_errors": 0}, "H": {"illegal": 0, "errors": 0, "observe_errors": 0}}
    ms = {"C": [0.0, 0], "H": [0.0, 0]}
    ovh = {"C": [0.0, 0], "H": [0.0, 0]}
    for r in valid:
        for s, side in zip(r["seat_stats"], r["sides"]):
            for k in side_err[side]:
                side_err[side][k] += s.get(k, 0)
            ms[side][0] += s["decide_ms_nontrivial"] * s["nontrivial"]
            ms[side][1] += s["nontrivial"]
            ovh[side][0] += s["overhead_ms"] * s["decisions"]
            ovh[side][1] += s["decisions"]
    return {
        "games": len(valid), "decisive": n, "draws": len(valid) - n, "voids": len(recs) - len(valid),
        "wins": wins, "share": wins / n if n else None, "p0": fmt.p0,
        "ci": [lo, hi], "ci_level": ci,
        "p_greater": binom_test_greater(wins, n, fmt.p0), "p_less": binom_test_less(wins, n, fmt.p0),
        "p_two_sided": binom_test_two_sided(wins, n, fmt.p0),
        "vp_diff": mean_diff, "vp_diff_se": sd / math.sqrt(len(diffs)) if diffs else 0.0,
        "arrangements": dict(sorted(arr.items())),
        "errors": side_err,
        "decide_ms": {k: (v[0] / v[1] if v[1] else None) for k, v in ms.items()},
        "overhead_ms": {k: (v[0] / v[1] if v[1] else None) for k, v in ovh.items()},
        "turns": (sum(r["turns"] for r in valid) / len(valid)) if valid else None,
    }


# ---------------------------------------------------------------------------
# Sequential evaluation and verdict
# ---------------------------------------------------------------------------
def evaluate(cfg: GateConfig, recs: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Replay the look-by-look decisions from the game records (deterministic)."""
    comps = cfg.comparisons()
    m = len(comps)
    rule = cfg.rule
    prim = comps[0]
    ns: Dict[Tuple[str, str], List[int]] = {c: [] for c in comps}
    looks: List[Dict[str, Any]] = []
    stop: Optional[Dict[str, Any]] = None
    summaries: Dict[str, Dict[str, Any]] = {}
    for k in range(1, cfg.num_looks + 1):
        end = cfg.look_end(k)
        if not all(all(i in recs.get(c, {}) for i in range(end)) for c in comps):
            break
        final = end >= cfg.max_games
        finals = [False] * (k - 1) + [final]
        summaries = {}
        crossed_low: List[str] = []
        entry: Dict[str, Any] = {"look": k, "games_per_comparison": end, "final": final, "comparisons": {}}
        for c in comps:
            fmt = parse_format(c[1])
            s = summarize([recs[c][i] for i in range(end)], fmt, rule.ci)
            ns[c].append(s["decisive"])
            low = Q.lower_boundary(ns[c], cfg.max_games, rule.alpha_fail / m, fmt.p0, rule.spending, finals)
            losses = s["decisive"] - s["wins"]
            s["fail_if_wins_le"] = s["decisive"] - low.crit[-1]
            s["crossed_fail"] = losses >= low.crit[-1]
            if s["crossed_fail"]:
                crossed_low.append(comp_key(c))
            summaries[comp_key(c)] = s
            entry["comparisons"][comp_key(c)] = {kk: s[kk] for kk in ("games", "decisive", "wins", "share", "p_greater",
                                                                        "p_less", "vp_diff", "fail_if_wins_le",
                                                                        "crossed_fail")}
        pfmt = parse_format(prim[1])
        up = Q.upper_boundary(ns[prim], cfg.max_games, rule.alpha, pfmt.p0, rule.spending, finals)
        ps = summaries[comp_key(prim)]
        ps["success_if_wins_ge"] = up.crit[-1]
        ps["crossed_success"] = ps["wins"] >= up.crit[-1]
        ps["alpha_spent"] = up.spent[-1]
        ps["t"] = up.ts[-1]
        entry["comparisons"][comp_key(prim)].update(success_if_wins_ge=up.crit[-1], crossed_success=ps["crossed_success"],
                                                    alpha_spent=up.spent[-1], t=up.ts[-1])
        if crossed_low:
            decision = "fail"
        elif ps["crossed_success"]:
            decision = "success"
        elif (not final and rule.futility_margin is not None and up.ts[-1] >= rule.futility_min_t
              and ps["share"] is not None and ps["share"] <= pfmt.p0 + rule.futility_margin):
            decision = "futility"
        elif final:
            decision = "max_games"
        else:
            decision = "continue"
        entry["decision"] = decision
        if crossed_low:
            entry["crossed_fail"] = crossed_low
        looks.append(entry)
        if decision != "continue":
            stop = {"look": k, "reason": decision, "crossed_fail": crossed_low}
            break
    return {"looks": looks, "complete_looks": len(looks), "stopped": stop is not None, "stop": stop,
            "summaries": summaries, "primary": comp_key(prim)}


def verdict(cfg: GateConfig, ev: Dict[str, Any], external: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The gate's verdict once it stopped (``None`` while it runs)."""
    if not ev["stopped"]:
        return None
    rule = cfg.rule
    S = ev["summaries"]
    prim = ev["primary"]
    ps = S[prim]
    reasons: List[str] = []
    # validity
    errors = sum(s["errors"][side][k] for s in S.values() for side in ("C", "H") for k in ("illegal", "errors"))
    voids = sum(s["voids"] for s in S.values())
    valid = errors <= rule.max_errors and voids <= rule.max_voids
    if errors > rule.max_errors:
        reasons.append(f"{errors} illegal actions / bot exceptions > max_errors {rule.max_errors}")
    if voids > rule.max_voids:
        reasons.append(f"{voids} voided games > max_voids {rule.max_voids}")
    # (a)
    a_ok = bool(ps.get("crossed_success")) and ps["p_greater"] < rule.alpha and ps["ci"][0] > ps["p0"]
    a = {"passed": a_ok, "comparison": prim, "wins": ps["wins"], "decisive": ps["decisive"], "share": ps["share"],
         "p_greater": ps["p_greater"], "ci": ps["ci"], "crossed_success": bool(ps.get("crossed_success")),
         "success_if_wins_ge": ps.get("success_if_wins_ge"), "alpha_spent": ps.get("alpha_spent")}
    if not a_ok:
        if ev["stop"]["reason"] == "futility":
            reasons.append(f"(a) futility stop: share {ps['share']:.3f} <= null {ps['p0']:.3f} + margin after "
                           f"{ps['games']} games vs {prim}")
        else:
            reasons.append(f"(a) not significantly better than the current champion ({prim}: {ps['wins']}/{ps['decisive']}"
                           f" = {ps['share'] if ps['share'] is None else round(ps['share'], 4)}, one-sided p "
                           f"{ps['p_greater']:.3g}, needed {ps.get('success_if_wins_ge')} wins)")
    # (b)
    fam = {k: s["p_less"] for k, s in S.items() if k != prim}
    if fam:
        if rule.holm:
            h = holm(fam, rule.alpha_fail)
        else:
            h = {k: {"p": p, "p_adj": p, "reject": p <= rule.alpha_fail} for k, p in fam.items()}
    else:
        h = {}
    worse = sorted(k for k, v in h.items() if v["reject"])
    interim = list(ev["stop"].get("crossed_fail") or [])
    b_ok = not worse and not interim
    if worse:
        reasons.append("(b) significantly worse than " + ", ".join(
            f"{k} (share {S[k]['share']:.3f}, p_adj {h[k]['p_adj']:.3g})" for k in worse))
    if interim:
        reasons.append("interim failure boundary crossed (clearly worse) vs " + ", ".join(interim))
    b = {"passed": b_ok, "holm": rule.holm, "family": h, "interim_fail": interim}
    # for information: which comparisons the candidate wins, Holm-adjusted over all of them
    better = holm({k: s["p_greater"] for k, s in S.items()}, rule.alpha)
    # (c)
    if cfg.external:
        c = external or {"status": "pending"}
        c_ok = c.get("status") == "pass"
        if c.get("status") == "fail":
            reasons.append(f"(c) external non-regression: {c.get('detail')}")
        elif c.get("status") not in ("pass", "pending"):
            reasons.append(f"(c) external hook: {c.get('status')}: {c.get('detail')}")
    else:
        c, c_ok = {"status": "not configured"}, True
    if not valid:
        v = "INVALID"
    elif a_ok and b_ok and c_ok:
        v = "PASS"
    elif a_ok and b_ok and c.get("status") == "pending":
        v = "PENDING_EXTERNAL"
    else:
        v = "FAIL"
    return {"verdict": v, "reasons": reasons, "stop": ev["stop"], "a": a, "b": b, "c": c, "valid": valid,
            "better_holm": {k: {"p": x["p"], "p_adj": x["p_adj"]} for k, x in better.items()},
            "errors": errors, "voids": voids, "decided": now_iso(), "gate_id": cfg.gate_id,
            "candidate": cfg.candidate, "champions": [c_["name"] for c_ in cfg.champions],
            "comparisons": {k: {kk: s.get(kk) for kk in ("games", "decisive", "draws", "wins", "share", "p0", "ci", "ci_level",
                                                           "p_greater", "p_less", "p_two_sided", "vp_diff", "vp_diff_se",
                                                           "arrangements", "errors", "decide_ms", "overhead_ms", "turns")}
                            for k, s in S.items()}}


# ---------------------------------------------------------------------------
# External non-regression hook
# ---------------------------------------------------------------------------
def run_external_command(template: str, tree: str, spec: str, weights: Optional[str], out: Path, log_path: Path,
                         commit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run an external benchmark command template and read its JSON result (``wins`` + ``games``)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = template.format(spec=shlex.quote(spec), tree=shlex.quote(str(tree)), weights=shlex.quote(weights or ""),
                          out=shlex.quote(str(out)), commit=shlex.quote(commit or ""), python=shlex.quote(sys.executable))
    with open(log_path, "w") as log:
        log.write(f"$ {cmd}\n")
        log.flush()
        r = subprocess.run(cmd, shell=True, cwd=str(tree), env=server_env(tree, env), stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise LeagueError(f"external command failed (exit {r.returncode}), log {log_path}")
    res = json.loads(out.read_text())
    return normalize_external(res, cmd)


def normalize_external(res: Dict[str, Any], cmd: str = "") -> Dict[str, Any]:
    games = int(res.get("games") or res.get("n") or 0)
    if "wins" in res:
        wins = int(res["wins"])
    elif "win_rate" in res and games:
        wins = int(round(float(res["win_rate"]) * games))
    else:
        raise LeagueError(f"external result needs 'wins' and 'games' (or 'win_rate' and 'games'): {res}")
    return {"wins": wins, "games": games, "win_rate": wins / games if games else None, "command": cmd,
            "recorded": now_iso()}


def compare_external(cand: Dict[str, Any], base: Optional[Dict[str, Any]], alpha: float = 0.05,
                     tolerance: Optional[float] = None) -> Dict[str, Any]:
    if not base:
        return {"status": "no baseline", "detail": "the current champion has no recorded result for this benchmark: "
                                                   "run `scripts/league.py external-baseline`", "candidate": cand}
    p = fisher_less(cand["wins"], cand["games"], base["wins"], base["games"])
    ok = p > alpha
    detail = (f"candidate {cand['wins']}/{cand['games']} = {cand['win_rate']:.3f} vs champion {base['wins']}/"
              f"{base['games']} = {base['wins'] / base['games']:.3f}, one-sided Fisher p(worse) = {p:.3g}")
    if tolerance is not None and cand["win_rate"] < base["wins"] / base["games"] - tolerance:
        ok = False
        detail += f"; below the champion's rate by more than {tolerance}"
    return {"status": "pass" if ok else "fail", "p_worse": p, "alpha": alpha, "tolerance": tolerance,
            "candidate": cand, "baseline": base, "detail": detail}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
SeatFactory = Callable[[str, int], Seat]


class SeatPool:
    """Seats of one worker: the candidate's servers persist, champion servers are kept for one champion at a time."""

    def __init__(self, cfg: GateConfig, gate_dir: Path, worker: int, factory: Optional[SeatFactory] = None,
                 on_start: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.cfg = cfg
        self.gate_dir = gate_dir
        self.worker = worker
        self.factory = factory
        self.on_start = on_start
        self.seats: Dict[Tuple[str, int], Seat] = {}

    def _make(self, who: str, slot: int) -> Seat:
        if self.factory is not None:
            return self.factory(who, slot)
        if who == "candidate":
            ent = self.cfg.candidate
        else:
            ent = self.cfg.champion(who)
        shim = str(self.gate_dir / ent["shim_copy"]) if ent.get("shim_copy") else None
        server = BotServer(ent["tree"], ent["spec_resolved"], label=f"{who}#{slot}",
                           serve_script=self.gate_dir / "serve.py", env=ent.get("env") or {}, shim=shim,
                           log_path=self.gate_dir / "servers" / f"w{self.worker}-{who}-{slot}.log",
                           timeout=self.cfg.decide_timeout, require_accel=self.cfg.require_accel)
        return ServerSeat(server)

    def seat(self, who: str, slot: int) -> Seat:
        key = (who, slot)
        s = self.seats.get(key)
        if s is None:
            if who != "candidate":
                for k in [k for k in self.seats if k[0] not in ("candidate", who)]:
                    self.seats.pop(k).close()
            s = self._make(who, slot)
            self.seats[key] = s
        return s

    def report_started(self) -> None:
        if self.on_start is None:
            return
        for (who, slot), s in self.seats.items():
            if slot == 0:
                self.on_start(who, s.info())

    def close(self) -> None:
        for s in self.seats.values():
            try:
                s.close()
            except Exception:
                pass
        self.seats.clear()


def play_gate_game(cfg: GateConfig, comp: Tuple[str, str], index: int, pool: SeatPool,
                   engine_id: Optional[str] = None) -> Dict[str, Any]:
    plan = game_plan(cfg, comp, index)
    n = plan["n"]
    seats: List[Seat] = []
    sides: List[str] = []
    ci = hi = 0
    for s in range(n):
        if s in plan["cand_seats"]:
            seats.append(pool.seat("candidate", ci))
            ci += 1
            sides.append("C")
        else:
            seats.append(pool.seat(comp[0], hi))
            hi += 1
            sides.append("H")
    res = None
    void = None
    for attempt in range(2):
        try:
            res = play_match_game(seats, plan["seed"], plan["bot_seeds"], n, cfg.max_turns, cfg.check_roundtrip)
            void = None
            break
        except GameVoid as exc:
            void = str(exc)
            for s in seats:     # restart whatever died; the retry replays the same seeds
                srv = getattr(s, "server", None)
                if srv is not None and not srv.alive:
                    srv.restart()
    pool.report_started()
    rec: Dict[str, Any] = {"type": "game", "gate": cfg.gate_id, "champion": comp[0], "format": comp[1],
                           "index": index, **{k: plan[k] for k in ("block", "arrangement", "pattern", "cand_seats",
                                                                   "seed", "bot_seeds")},
                           "sides": sides, "specs": [s.spec for s in seats], "labels": [s.label for s in seats]}
    if res is None:
        rec.update(void=void, winner=-1, cand_win=None, vps=[], vp_diff=0.0, turns=0, actions=0, seat_stats=[])
    else:
        w = res["winner"]
        cv = [res["vps"][s] for s in range(n) if sides[s] == "C"]
        hv = [res["vps"][s] for s in range(n) if sides[s] == "H"]
        rec.update(void=None, winner=w, winner_side=(sides[w] if w >= 0 else None),
                   cand_win=(sides[w] == "C") if w >= 0 else None, vps=res["vps"],
                   vp_diff=sum(cv) / len(cv) - sum(hv) / len(hv), turns=res["turns"], actions=res["actions"],
                   capped=res["capped"], seat_stats=res["seats"], error_log=res["error_log"],
                   serialize_ms=res["serialize_ms"], duration_s=res["duration_s"])
    rec["engine"] = engine_id
    rec["finished"] = now_iso()
    return rec


def current_engine_id() -> str:
    """``<sha12>[+dirty]`` of the tree whose engine referees this process's games."""
    import catanbot
    commit, dirty = tree_commit(Path(catanbot.__file__).resolve().parents[1])
    return f"{(commit or 'unknown')[:12]}{'+dirty' if dirty else ''}"


class _Progress:
    def __init__(self, gate_dir: Path, cfg: GateConfig, done: int):
        self.path = gate_dir / "progress.json"
        self.cfg = cfg
        self.started = time.time()
        self.run_games = 0
        self.done = done
        self.lock = threading.Lock()

    def update(self, look: int, look_remaining: int, final: bool = False, state: str = "running") -> None:
        el = time.time() - self.started
        gph = self.run_games / el * 3600 if el > 0 and self.run_games else None
        m = len(self.cfg.comparisons())
        remaining_max = max(0, self.cfg.max_games * m - self.done)
        d = {"pid": os.getpid(), "host": socket.gethostname(), "state": state, "started": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)), "updated": now_iso(), "updated_ts": time.time(),
             "games_done": self.done, "games_this_run": self.run_games, "seconds_this_run": round(el, 1),
             "games_per_hour": gph, "look": look, "num_looks": self.cfg.num_looks,
             "look_games_remaining": look_remaining, "games_remaining_to_max": remaining_max,
             "eta_hours_look": (look_remaining / gph) if gph else None,
             "eta_hours_max": (remaining_max / gph) if gph else None}
        atomic_write_json(self.path, d)


def run_gate(cfg: GateConfig, gate_dir: Path, seat_factory: Optional[SeatFactory] = None, log=print,
             stop_after: Optional[int] = None, external_runner: Optional[Callable[[GateConfig, Path], Dict[str, Any]]] = None
             ) -> Dict[str, Any]:
    """Play (or resume) the gate until it stops; writes and returns the verdict.

    ``stop_after``: return after that many new games (an interruption, for tests
    and time-boxed runs); the result then has ``{"interrupted": True}``.
    """
    gate_dir = Path(gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = gate_dir / "gate.json"
    if not cfg_path.exists():
        atomic_write_json(cfg_path, cfg.to_dict())
    if seat_factory is None and not (gate_dir / "serve.py").exists():
        shutil.copy2(Path(__file__).with_name("serve.py"), gate_dir / "serve.py")
    games_path = gate_dir / "games.jsonl"
    if games_path.exists() and games_path.stat().st_size:
        with open(games_path, "rb+") as f:      # a line cut by an interruption must not swallow the next record
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
    recs = load_records(gate_dir)
    done = sum(1 for d in recs.values() for r in d.values() if not r.get("void"))
    progress = _Progress(gate_dir, cfg, done)
    looks_path = gate_dir / "looks.jsonl"
    logged_looks = set()
    if looks_path.exists():
        for line in looks_path.read_text().splitlines():
            try:
                logged_looks.add(json.loads(line)["look"])
            except (ValueError, KeyError):
                pass
    servers_info: Dict[str, Any] = {}
    info_lock = threading.Lock()

    def on_start(who: str, info: Dict[str, Any]) -> None:
        with info_lock:
            if who not in servers_info:
                servers_info[who] = info
                atomic_write_json(gate_dir / "servers.json", servers_info)

    pools = [SeatPool(cfg, gate_dir, w, seat_factory, on_start) for w in range(max(1, cfg.workers))]
    write_lock = threading.Lock()
    comps = cfg.comparisons()
    try:
        while True:
            ev = evaluate(cfg, recs)
            for L in ev["looks"]:
                if L["look"] not in logged_looks:
                    with open(looks_path, "a") as f:
                        f.write(json.dumps({"type": "look", "gate": cfg.gate_id, "evaluated": now_iso(), **L}) + "\n")
                    logged_looks.add(L["look"])
                    _log_look(L, log)
            if ev["stopped"]:
                break
            k = ev["complete_looks"] + 1
            end = cfg.look_end(k)
            start = cfg.look_end(k - 1) if k > 1 else 0
            # missing games of this look (and, defensively, of earlier ones); a voided game of this
            # not-yet-evaluated look is retried (voids never depend on the outcome)
            jobs = [(c, i) for c in comps for i in range(end)
                    if i not in recs.get(c, {}) or (i >= start and recs[c][i].get("void"))]
            if stop_after is not None and progress.run_games >= stop_after:
                progress.update(k, len(jobs), state="interrupted")
                return {"interrupted": True, "games_done": progress.done}
            log(f"[gate {cfg.gate_id}] look {k}/{cfg.num_looks}: {len(jobs)} games to play "
                f"({len(comps)} comparison(s) x up to {end} games)")
            budget = None if stop_after is None else stop_after - progress.run_games
            if budget is not None:
                jobs = jobs[:budget]
            _run_jobs(cfg, jobs, pools, recs, gate_dir, write_lock, progress, k, log)
            if stop_after is not None and progress.run_games >= stop_after:
                ev2 = evaluate(cfg, recs)
                if not ev2["stopped"]:
                    remaining = sum(1 for c in comps for i in range(cfg.look_end(ev2["complete_looks"] + 1))
                                    if i not in recs.get(c, {}))
                    progress.update(ev2["complete_looks"] + 1, remaining, state="interrupted")
                    return {"interrupted": True, "games_done": progress.done}
    finally:
        for p in pools:
            p.close()
    external = None
    v = verdict(cfg, ev)
    if cfg.external and v is not None and v["a"]["passed"] and v["b"]["passed"] and v["valid"]:
        external = _external(cfg, gate_dir, external_runner, log)
        v = verdict(cfg, ev, external)
    atomic_write_json(gate_dir / "verdict.json", v)
    progress.update(ev["complete_looks"], 0, state="done")
    log(format_verdict(v))
    return v


def _external(cfg: GateConfig, gate_dir: Path, runner, log) -> Dict[str, Any]:
    ext = cfg.external or {}
    path = gate_dir / "external.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        if runner is not None:
            cand = runner(cfg, gate_dir)
        else:
            key = ext.get("key", "external")
            log(f"[gate {cfg.gate_id}] running the external benchmark hook ({key})")
            cand = run_external_command(ext["cmd"], cfg.candidate["tree"], cfg.candidate["spec_resolved"],
                                        (cfg.candidate.get("weights") or {}).get("path"),
                                        gate_dir / "external" / f"{key}.json", gate_dir / "external" / f"{key}.log",
                                        cfg.candidate.get("commit"))
        res = compare_external(cand, ext.get("baseline"), float(ext.get("alpha", 0.05)), ext.get("tolerance"))
    except Exception as exc:
        res = {"status": "error", "detail": repr(exc)}
    atomic_write_json(path, res)
    return res


def _run_jobs(cfg, jobs, pools, recs, gate_dir, write_lock, progress, look, log) -> None:
    if not jobs:
        return
    engine_id = current_engine_id()
    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    errors: List[BaseException] = []
    remaining = [len(jobs)]

    def worker(pool: SeatPool) -> None:
        while not errors:
            try:
                comp, i = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = play_gate_game(cfg, comp, i, pool, engine_id)
            except BaseException as exc:   # version skew, server start failures: abort the gate loudly
                errors.append(exc)
                return
            with write_lock:
                with open(gate_dir / "games.jsonl", "a") as f:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                recs.setdefault(comp, {})
                if i not in recs[comp] or recs[comp][i].get("void") or not rec.get("void"):
                    recs[comp][i] = rec
                remaining[0] -= 1
                if not rec.get("void"):
                    progress.done += 1
                progress.run_games += 1
                progress.update(look, remaining[0])
                _log_game(rec, progress, log)
                voids = sum(1 for d in recs.values() for r in d.values() if r.get("void"))
                if voids > cfg.rule.max_voids:
                    errors.append(LeagueError(f"{voids} voided games > max_voids {cfg.rule.max_voids}: "
                                              f"see {gate_dir / 'servers'} (last: {rec.get('void')})"))

    if len(pools) == 1:
        worker(pools[0])
    else:
        threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in pools]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]


def _log_game(rec: Dict[str, Any], progress: _Progress, log) -> None:
    if rec.get("void"):
        log(f"  {rec['champion']}/{rec['format']} #{rec['index']} VOID: {rec['void'][:200]}")
        return
    gph = progress.run_games / max(1e-9, time.time() - progress.started) * 3600
    w = rec["winner_side"] or "-"
    log(f"  {rec['champion']}/{rec['format']} #{rec['index']} {rec['pattern']} seed {rec['seed']}: winner {w} "
        f"vps {rec['vps']} turns {rec['turns']} {rec['duration_s']:.1f}s ({gph:.1f} games/h)"
        + (f" ERRORS {rec['error_log'][:2]}" if rec.get("error_log") else ""))


def _log_look(L: Dict[str, Any], log) -> None:
    parts = []
    for k, s in L["comparisons"].items():
        share = "-" if s["share"] is None else f"{s['share']:.3f}"
        extra = f", success at >= {s['success_if_wins_ge']}" if "success_if_wins_ge" in s else ""
        parts.append(f"{k}: {s['wins']}/{s['decisive']} = {share}{extra}, fail at <= {s['fail_if_wins_le']}")
    log(f"[look {L['look']}] " + "; ".join(parts) + f" -> {L['decision']}")


def format_verdict(v: Dict[str, Any]) -> str:
    L = [f"VERDICT {v['verdict']} (gate {v['gate_id']}, stopped at look {v['stop']['look']}: {v['stop']['reason']})"]
    for k, s in v["comparisons"].items():
        share = "-" if s["share"] is None else f"{s['share']:.3f}"
        L.append(f"  {k}: {s['wins']}/{s['decisive']} decisive = {share} (null {s['p0']:.3f}), "
                 f"{round(100 * s.get('ci_level', 0.95))}% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}], p(better) {s['p_greater']:.3g}, "
                 f"p(worse) {s['p_less']:.3g}, VP diff {s['vp_diff']:+.2f} +- {s['vp_diff_se']:.2f}, "
                 f"draws {s['draws']}")
    for r in v["reasons"]:
        L.append(f"  - {r}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
def promote(gate_dir: Path, registry: Registry, name: Optional[str] = None, notes: str = "",
            allow_uncommitted: bool = False, weights_dir: Optional[Path] = None) -> Champion:
    """Append the gate's candidate as the next champion - only if its verdict is PASS and the ladder is unchanged."""
    gate_dir = Path(gate_dir)
    vpath = gate_dir / "verdict.json"
    if not vpath.exists():
        raise LeagueError(f"{gate_dir}: no verdict yet (the gate has not stopped)")
    v = json.loads(vpath.read_text())
    cfg = GateConfig.from_dict(json.loads((gate_dir / "gate.json").read_text()))
    if v.get("verdict") != "PASS":
        raise LeagueError(f"gate {cfg.gate_id} verdict is {v.get('verdict')}: not promotable ({'; '.join(v.get('reasons', []))})")
    ladder = [c.name for c in registry.champions]
    gated = [c["name"] for c in cfg.champions]
    if sorted(ladder) != sorted(gated) or cfg.champions[0]["name"] != registry.current.name:
        raise LeagueError(f"the ladder changed since gate {cfg.gate_id} (gated against {gated}, registry has {ladder}): "
                          f"re-run the gate against the current champions")
    for c in cfg.champions:
        if registry.get(c["name"]).commit != c["commit"]:
            raise LeagueError(f"champion {c['name']} changed commit since the gate")
    cand = cfg.candidate
    if not cand.get("commit") or cand.get("dirty"):
        if not allow_uncommitted:
            raise LeagueError("the candidate is not a clean commit (dirty or copied tree): commit it and gate that "
                              "commit, or pass --allow-uncommitted (the champion could then not be materialised exactly)")
    new_name = name or cand.get("name") or registry.next_name()
    if any(c.name == new_name for c in registry.champions):
        raise LeagueError(f"a champion named {new_name} exists")
    weights = None
    spec = cand["spec"]
    if cand.get("weights"):
        src = Path(cand["weights"]["path"])
        wdir = weights_dir or (registry.path.parent / "weights")
        wdir.mkdir(parents=True, exist_ok=True)
        dst = wdir / f"{new_name}{src.suffix}"
        shutil.copy2(src, dst)
        sha = sha256_file(dst)
        if sha != cand["weights"]["sha256"]:
            raise LeagueError(f"weights copy {dst} does not match the gated sha256")
        try:
            rel = str(dst.resolve().relative_to(registry.path.parent.parent.resolve()))
        except ValueError:
            rel = str(dst)
        weights = {"path": rel, "sha256": sha}
        spec = spec_with_model(spec, rel)
    games = gate_dir / "games.jsonl"
    try:
        gate_rel = str(gate_dir.resolve().relative_to(registry.path.parent.parent.resolve()))
    except ValueError:
        gate_rel = str(gate_dir)
    evidence = {"kind": "gate", "gate_id": cfg.gate_id, "verdict": v["verdict"], "decided": v["decided"],
                "stop": v["stop"], "rule": asdict(cfg.rule), "formats": cfg.formats, "batch_games": cfg.batch_games,
                "max_games": cfg.max_games, "seed_base": cfg.seed_base, "against": gated,
                "comparisons": {k: {kk: s[kk] for kk in ("games", "decisive", "wins", "share", "ci", "p_greater",
                                                            "p_less", "vp_diff")}
                                for k, s in v["comparisons"].items()},
                "a": {k: v["a"][k] for k in ("share", "p_greater", "ci", "crossed_success")},
                "b": {"holm": v["b"]["holm"], "p_adj": {k: x["p_adj"] for k, x in v["b"]["family"].items()}},
                "c": v.get("c"), "engine": cfg.engine, "records": f"{gate_rel}/games.jsonl",
                "records_sha256": sha256_file(games) if games.exists() else None}
    champ = Champion(name=new_name, commit=cand.get("commit") or "", spec=spec, weights=weights,
                     created=now_iso()[:10], notes=notes or cfg.notes, gate=evidence,
                     env=dict(cand.get("env") or {}))
    ext = v.get("c") or {}
    if ext.get("status") == "pass" and cfg.external:
        champ.external = {cfg.external.get("key", "external"): ext["candidate"]}
    registry.champions.append(champ)
    registry.save()
    return champ


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def gate_status(gate_dir: Path) -> Dict[str, Any]:
    gate_dir = Path(gate_dir)
    cfg = GateConfig.from_dict(json.loads((gate_dir / "gate.json").read_text()))
    out: Dict[str, Any] = {"gate_id": cfg.gate_id, "dir": str(gate_dir), "candidate": cfg.candidate,
                           "champions": [c["name"] for c in cfg.champions], "formats": cfg.formats}
    vp = gate_dir / "verdict.json"
    pp = gate_dir / "progress.json"
    prog = json.loads(pp.read_text()) if pp.exists() else {}
    out["progress"] = prog
    if vp.exists():
        out["state"] = "done"
        out["verdict"] = json.loads(vp.read_text())
        return out
    alive = False
    if prog.get("pid") and prog.get("host") == socket.gethostname():
        try:
            os.kill(int(prog["pid"]), 0)
            alive = prog.get("state") == "running" and time.time() - float(prog.get("updated_ts", 0)) < 3 * 3600
        except (OSError, ValueError):
            alive = False
    out["state"] = "running" if alive else "stopped"
    ev = evaluate(cfg, load_records(gate_dir))
    out["complete_looks"] = ev["complete_looks"]
    out["num_looks"] = cfg.num_looks
    out["summaries"] = ev["summaries"]
    return out
