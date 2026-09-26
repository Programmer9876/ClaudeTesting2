#!/usr/bin/env python3
"""Budgeted test queue: sequential verdicts, area-first priority with CPU-cost preemption, pinned snapshots,
fallbacks, confirmations and a ledger-driven report (docs/QUEUE.md; docs/PRIORITY_PLAN.md step 1).

    python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --explain --simulate   # approve
    nice -n 10 python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --max-hours 8  # run
    python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --status                 # where
    python3 scripts/run_queue.py --plan scripts/queue_plan.json --dir runs/queue1 --bump-code --areas trades

(Named ``run_queue.py`` rather than the design's ``queue.py``: a ``scripts/queue.py`` would shadow the standard
library's ``queue`` module, which ``concurrent.futures`` and ``multiprocessing`` import, for every script run as
``python3 scripts/<x>.py``.)

What it does, one slice at a time:
1. re-reads the plan (a parse error keeps the previous plan), applies intake (lint, power at the promised
   effect, headroom gates) and evaluates every look whose seed prefix is complete (scripts/seqtest.py);
   verdicts go to ``<dir>/ledger.jsonl`` and, as ``{kind: stop, source: queue}`` records, into the row's own
   JSONL, so campaign.py / ablate_catanatron.py never play a stopped candidate again;
2. picks the row to run: area first (harness, trades, diversification, ports, robber, counting, politics, other),
   headroom rows
   first within an area, then plan priority, then plan order.  A higher-ranked newcomer (a plan edit, a fallback,
   a confirmation) preempts the running row at the slice boundary only when
   ``w_q C_r > (1 + h) w_r (C_q + S)`` with remaining costs C in CPU-hours (getrusage of the chunks);
3. waits while the proof guard fires (a process whose command line matches ``--guard`` or a lock file), gates
   exclusive rows (AlphaBeta-class opponents) on the machine's load measured after its own work drained;
4. runs one time slice from the row's pinned code snapshot (``--max-minutes``: in-flight games finish) and
   books its CPU seconds; writes ``<dir>/QUEUE.md``.

Row kinds: ``catanatron`` (ablate_catanatron.py chunks aimed at the next look's prefix), ``selfplay`` (ablate.py in
240-game chunks, one value per call, explicit search base spec, seeds 5000+k), ``command`` (a zero-game shadow,
tune_joint, ...: ``verdict_from`` records PASS / FAIL), ``pool`` (a control-variate default pool,
``--default-only``), ``human`` (``route: human``: deferred to human testing, 0 games).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import resource
import shutil
import signal
import stat as statmod
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _load(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CAMP = _load("_campaign_for_queue", os.path.join(SCRIPTS, "campaign.py"))
AB = CAMP.AB
SEQ = _load("_seqtest_for_queue", os.path.join(SCRIPTS, "seqtest.py"))
MECH = _load("_mechanics_for_queue", os.path.join(SCRIPTS, "mechanics.py"))

AREAS = ("harness", "trades", "diversification", "ports", "robber", "counting", "politics", "other")
AREA_RANK = {a: i for i, a in enumerate(AREAS)}
POLARITIES = ("new", "knockout", "measure")
KINDS = ("catanatron", "selfplay", "command", "pool", "human")
FALLBACK_KINDS = ("simpler", "milder", "stronger")
NO_HOLM_TIERS = ("harness", "smoke")
DEFAULT_SEARCH_SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"
SEED_BLOCK = 10 ** 6                       # fallback children play on base + 10^6 x depth
POOL_BASE = 10 ** 7
SELFPLAY_CHUNK = 240
SELFPLAY_SEED = 5000
H_PREEMPT = 0.2
S_START_H = 0.7 / 3600.0                   # start-up cost of a slice, CPU-hours
SLICE_MINUTES = 15.0
LOAD_GATE_ALERT_S = 2 * 3600.0
GUARD_POLL_S = 60.0
MIN_SHADOW_SHARE = 0.02
COUNTING_HEADROOM_PP = 2.0
CRN_DROP = 0.25
# Politics scope (docs/ABLATIONS.md "Politics rule"): the design is forced to the fixed-N politics screen.
POLITICS_PREFIXES = ("politics.", "coalitions.")
POLITICS_NAMES = {"trading.feed_leader_guard", "opponent_model.stage_late_drop", "search.counters",
                  "search.counter_margin", "search.counter_aggr", "search.respond_lookahead",
                  "opponent_model.CALIB_RATE", "opponent_model.CALIB_PRIOR", "opponent_model.REJECT_STREAK",
                  "winpaths.SPOT_LEADER", "robber_eval.RETAL_W"}
# Terms that act only through player-to-player trades: against Catanatron they need --trades value/fair/native
# (3.3), else self-play (C++ evaluator only).
PLAYER_TRADE_PREFIXES = ("trading.", "politics.", "coalitions.", "opponent_model.")
PLAYER_TRADE_NAMES = {"search.trade_proposals", "search.acq_shapes", "search.acq_breadth", "search.acq_floor",
                      "acquisition.W_PREMIUM", "search.counters", "search.counter_margin", "search.counter_aggr",
                      "search.respond_lookahead", "acquisition.OFFER_COST", "acquisition.OFFER_LEAK",
                      "acquisition.OFFER_REPEAT"}
# Python-evaluator tunables (static_value / placement are mirrored in C++): self-play on them is refused.
PYEVAL_NAMES = {"heuristic.EXPOSURE_WEIGHT", "placement.RESOURCE_DEMAND", "placement.PLACEMENT_BLOCK_WEIGHT",
                "placement.PLACEMENT_ROBBER_Q"}
PYEVAL_PREFIXES = ("placement.PORT_", "heuristic.PORT_")
# D priors (discordant share) by class of change, measured (design F2)
D_PRIORS = {"full": 0.40, "trade": 0.25, "medium": 0.14, "low": 0.04}
# CPU seconds per game and arm (design F3 priors, until the row has measured its own)
CPU_PRIOR = {("value", False, "full"): 1.77, ("value", True, "full"): 5.5, ("vf", False, "full"): 1.15,
             ("vf", True, "full"): 3.3, ("alphabeta", False, "full"): 23.0, ("sameturn", False, "full"): 20.0,
             ("ab", False, "full"): 6.8, ("value", False, "counted"): 7.2, ("vf", False, "counted"): 3.9}
CPU_TRADES_VALUE = 3.0
CPU_SELFPLAY = 10.8
CPU_DEPTH2 = 2.4
CPU_HEURISTIC_BOT = 0.6


class PlanError(Exception):
    pass


def now() -> float:
    return time.time()


def value_text(v: Any, flag_off: bool = False) -> str:
    """The registry's text of a plan value (``Tunable.format`` without importing the registry)."""
    if flag_off or v is False:
        return "off"
    if v is True:
        return "on"
    if isinstance(v, (list, tuple)):
        return "/".join(f"{float(x):g}" for x in v)
    if isinstance(v, (int, float)):
        return f"{float(v):g}"
    return str(v)


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+@-]", "_", text)[:80]


def _jdump(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass
class Plan:
    path: str
    rows: List[Dict[str, Any]]
    settings: Dict[str, Any]
    interps: Dict[str, str]
    sha: str

    def row(self, name: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.rows if r["name"] == name), None)


def load_plan(path: str) -> Plan:
    """The queue plan: campaign.py's format plus the queue fields (all optional for campaign.py)."""
    try:
        with open(path) as fh:
            text = fh.read()
        raw = json.loads(text)
    except (OSError, ValueError) as ex:
        raise PlanError(f"cannot read plan {path}: {ex}")
    if not isinstance(raw, dict):
        raw = {"experiments": raw}
    interps = dict(CAMP.DEFAULT_INTERPRETERS)
    interps.update(raw.get("interpreters") or {})
    defaults = raw.get("defaults") or {}
    rows: List[Dict[str, Any]] = []
    names = set()
    for i, e in enumerate(raw.get("experiments") or []):
        e = dict(copy.deepcopy(defaults), **copy.deepcopy(e))
        bad = set(e) - CAMP.EXPERIMENT_KEYS
        if bad:
            raise PlanError(f"plan error: row {e.get('name', i)}: unknown field(s) {sorted(bad)}")
        if not e.get("name") or not CAMP.NAME_RE.match(str(e["name"])):
            raise PlanError(f"plan error: row {i}: missing or bad name {e.get('name')!r}")
        if e["name"] in names:
            raise PlanError(f"plan error: duplicate row name {e['name']!r}")
        names.add(e["name"])
        if e.get("route") == "human":
            e["kind"] = "human"
        e["kind"] = e.get("kind") or "catanatron"
        if e["kind"] not in KINDS:
            raise PlanError(f"plan error: {e['name']}: unknown kind {e['kind']!r} (choose from {KINDS})")
        e["enabled"] = e.get("enabled", True) is not False
        e["priority"] = e.get("priority", 100)
        seeds = e.get("seeds", {})
        if isinstance(seeds, int):
            seeds = {"count": seeds}
        e["seeds"] = {"count": int(seeds.get("count", e.get("games") or 100)), "base": int(seeds.get("base", 0))}
        e["_order"] = i
        rows.append(e)
    so = raw.get("screen_once")
    if so is not None and (not isinstance(so, list) or any(n not in names for n in so)):
        raise PlanError(f"plan error: screen_once must list plan rows (got {so!r})")
    rows = expand_bundles(rows)
    settings = {k: raw.get(k) for k in ("crn_pilot", "headroom", "removed", "notes", "screen_once")
                if raw.get(k) is not None}
    return Plan(path, rows, settings, interps, hashlib.sha1(text.encode()).hexdigest()[:12])


# ---------------------------------------------------------------------------
# Bundle-first rows (docs/ABLATIONS.md "Regrouping": test linked pieces together, then knock out)
# ---------------------------------------------------------------------------
KNOCKOUT_BLOCK = 5 * SEED_BLOCK            # bundle knockouts play on fresh seed blocks: base + 5e6 + i x 1e6


def _spec_with(base: str, keys: Dict[str, str]) -> str:
    name, _, rest = base.partition(":")
    kv: Dict[str, str] = {}
    for part in rest.split(","):
        if part.strip():
            k, _, v = part.partition("=")
            kv[k.strip()] = v.strip()
    kv.update(keys)
    return name + ":" + ",".join(f"{k}={v}" for k, v in kv.items())


def _spec_keys(spec: str) -> Dict[str, str]:
    _, _, rest = str(spec or "").partition(":")
    return {k.strip(): v.strip() for k, _, v in (p.partition("=") for p in rest.split(",") if p.strip())}


def _pieces(item: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A bundle member: ``KEY=VALUE`` (a bot-spec key), ``NAME=VALUE`` (a dotted registry tunable) or the name of a
    plan row that tests the piece alone (its tunable value, or the spec keys its candidate adds)."""
    text = str(item).strip()
    row = next((r for r in rows if r["name"] == text), None)
    if row is not None:
        if row.get("tunable") and len(row.get("values") or []) == 1:
            v = row["values"][0]
            return [{"kind": "tunable", "key": row["tunable"], "value": v, "label": row["tunable"].split(".")[-1],
                     "text": f"{row['tunable']}={value_text(v)}", "row": text}]
        add = {k: v for k, v in _spec_keys(row.get("cand_spec")).items()
               if _spec_keys(row.get("def_spec")).get(k) != v}
        if not add:
            raise PlanError(f"plan error: bundle member row {text!r} is neither one tunable value nor spec keys")
        return [{"kind": "spec", "key": k, "value": v, "label": k, "text": f"{k}={v}", "row": text}
                for k, v in add.items()]
    if "=" not in text:
        raise PlanError(f"plan error: bundle member {text!r} is not KEY=VALUE nor a row name")
    k, _, v = (x.strip() for x in text.partition("="))
    if "." in k:
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        return [{"kind": "tunable", "key": k, "value": val, "label": k.split(".")[-1], "text": text}]
    return [{"kind": "spec", "key": k, "value": v, "label": k, "text": text}]


def _tunable_default(name: str) -> Any:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from catanbot import tuning
    t = tuning.find(name)
    return t.default


def expand_bundles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A row with ``bundle: [pieces]`` plays every piece on against the default (its candidate spec / overrides are
    built from the pieces unless given).  With ``knockouts: true`` it gets one generated knockout row per piece,
    ``<bundle>-no-<piece>``: the full bundle is the default arm, the bundle minus that piece the candidate
    (knockout design: REMOVE = the piece hurts inside the bundle, KEEP(proven) = it carries gain), eligible only
    after the bundle ADOPTs, on a fresh seed block.  Member rows named in ``bundle`` wait for the bundle and are
    shelved with it (bundle first; no 2x2)."""
    out: List[Dict[str, Any]] = []
    for e in rows:
        out.append(e)
        if not e.get("bundle"):
            continue
        pieces = [p for item in e["bundle"] for p in _pieces(item, rows)]
        if len(pieces) < 2:
            raise PlanError(f"plan error: {e['name']}: a bundle needs at least two pieces")
        base = e.get("base_spec") or DEFAULT_SEARCH_SPEC
        spec_on = {p["key"]: p["value"] for p in pieces if p["kind"] == "spec"}
        tun_on = {p["key"]: p["value"] for p in pieces if p["kind"] == "tunable"}
        if not e.get("cand_spec") and not e.get("tunable"):
            e["cand_spec"] = _spec_with(base, spec_on)
            e["def_spec"] = base
            if tun_on:
                e["cand_set"] = dict(e.get("cand_set") or {}, **tun_on)
        e["_members"] = sorted({p["row"] for p in pieces if p.get("row")})
        e["_pieces"] = [p["text"] for p in pieces]
        if not e.get("knockouts"):
            continue
        for i, p in enumerate(pieces):
            k = {kk: copy.deepcopy(vv) for kk, vv in e.items()
                 if kk not in ("bundle", "knockouts", "promise_pp", "cand_set", "set", "bundle_of", "notes", "_members",
                               "_pieces", "mechanism", "d_prior")}
            k["name"] = f"{e['name']}-no-{p['label']}"
            k["polarity"] = "knockout"
            k["design"] = "knockout"
            keep = {kk: vv for kk, vv in spec_on.items() if not (p["kind"] == "spec" and kk == p["key"])}
            k["def_spec"] = _spec_with(base, spec_on)
            k["cand_spec"] = _spec_with(base, keep)
            common = dict(e.get("set") or {}, **tun_on)
            if common:
                k["set"] = common
            if p["kind"] == "tunable":
                k["cand_set"] = {p["key"]: _tunable_default(p["key"])}
            k["after"] = {"row": e["name"], "verdicts": ["ADOPT"]}
            k["seeds"] = {"count": e["seeds"]["count"], "base": e["seeds"]["base"] + KNOCKOUT_BLOCK + i * SEED_BLOCK}
            k["_order"] = e["_order"] + 0.001 * (i + 1)
            k["_knockout_of"] = e["name"]
            k["notes"] = (f"bundle knockout (generated): {e['name']} without {p['text']}; the default arm is the full "
                          f"bundle. REMOVE = the piece hurts inside the bundle; KEEP(proven) = it carries the gain. "
                          f"Eligible only after the bundle ADOPTs.")
            out.append(k)
    return out


def row_names(e: Dict[str, Any]) -> List[str]:
    """Every registry name a row changes (its tunable, cand_set / set keys, tune= entries of its specs)."""
    out = []
    if e.get("tunable"):
        out.append(e["tunable"])
    for key in ("cand_set", "set"):
        v = e.get(key)
        if isinstance(v, dict):
            out.extend(v)
    for key in ("cand_spec", "def_spec", "base_spec"):
        m = re.search(r"tune=([^,]*)", str(e.get(key) or ""))
        if m:
            out.extend(x.split(":")[0] for x in m.group(1).split(";") if x)
    return out


def is_politics(e: Dict[str, Any]) -> bool:
    return any(n in POLITICS_NAMES or n.startswith(POLITICS_PREFIXES) for n in row_names(e))


def is_player_trade(e: Dict[str, Any]) -> bool:
    return any(n in PLAYER_TRADE_NAMES or n.startswith(PLAYER_TRADE_PREFIXES) for n in row_names(e))


def needs_pyeval(e: Dict[str, Any]) -> bool:
    return any(n in PYEVAL_NAMES or n.startswith(PYEVAL_PREFIXES) for n in row_names(e))


def info_counted(e: Dict[str, Any], arm: str = "both") -> bool:
    both = CAMP.info_mode({k: e.get(k) for k in ("adapter_opts", "info", "extra_args")}) == "counted"
    if arm == "both":
        return both
    return both or CAMP.info_mode({"cand_adapter_opts": e.get("cand_adapter_opts")}) == "counted"


def is_aa(e: Dict[str, Any]) -> bool:
    return (e["kind"] == "catanatron" and not e.get("tunable") and e.get("cand_spec")
            and e.get("cand_spec") == e.get("def_spec") and not e.get("cand_set") and not e.get("cand_adapter_opts"))


def design_of(e: Dict[str, Any]) -> str:
    kind = e.get("kind") or ("human" if e.get("route") == "human" else "catanatron")
    if kind in ("command", "human"):
        return "gate" if kind == "command" else "human"
    if kind == "pool":
        return "pool"
    if e.get("design"):
        return e["design"]
    if e.get("confirms"):
        return "confirm"
    if is_politics(e):
        return "politics"
    return SEQ.POLARITY_DESIGN.get(e.get("polarity") or "new", "screen")


def d_prior(e: Dict[str, Any]) -> float:
    """Discordant-share prior of a row's class (design F2), or its own ``d_prior``."""
    if e.get("d_prior") is not None:
        return float(e["d_prior"])
    names = row_names(e)
    if not names and e.get("cand_spec"):
        return D_PRIORS["full"]
    if is_player_trade(e):
        return D_PRIORS["trade"]
    if any(n.startswith(("search.", "openings.", "placement.", "ports.", "heuristic.PORT", "winpaths."))
           for n in names):
        return D_PRIORS["full"]
    if any(n.startswith(("danger.", "devcards.", "robber", "search.robber", "search.kick")) for n in names):
        return D_PRIORS["low"]
    if any(n.startswith("heuristic.") for n in names):
        return D_PRIORS["medium"]
    return D_PRIORS["trade"]


def cand_values(e: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """``[(candidate key, plan value)]``: value texts for tunable rows, 'cand' for spec rows."""
    if e["kind"] in ("command", "human"):
        return [("row", None)]
    if e["kind"] == "pool":
        return [("pool", None)]
    if e.get("tunable"):
        if e.get("flag_off") and not e.get("values"):
            return [("off", False)]
        vals = e.get("values")
        if isinstance(vals, str):
            vals = [v for v in re.split(r"[;,]" if ";" not in vals else ";", vals) if v.strip()]
        out, seen = [], set()
        for v in vals or []:
            t = value_text(v)
            if t not in seen:
                seen.add(t)
                out.append((t, v))
        return out
    return [("cand", None)]


def fallback_depth(plan: Plan, e: Dict[str, Any]) -> int:
    d = 0
    cur = e
    seen = set()
    while cur.get("parent") and cur["name"] not in seen:
        seen.add(cur["name"])
        d += 1
        cur = plan.row(cur["parent"]) or {}
    return d


def effective_base(plan: Plan, e: Dict[str, Any]) -> int:
    base = int(e["seeds"]["base"])
    if e.get("parent") and e.get("fallback"):
        return base + SEED_BLOCK * fallback_depth(plan, e)
    return base


# ---------------------------------------------------------------------------
# Intake: lint, power at the promise, headroom
# ---------------------------------------------------------------------------
@dataclass
class Intake:
    status: str = "ok"                 # ok | refused | shelved | deferred | bundled
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    design: str = "screen"
    d: Optional[float] = None
    power: Optional[float] = None
    promise: Optional[float] = None
    reducer: Optional[str] = None


def lint(plan: Plan, e: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """``(errors, warnings)``: errors refuse the row before any game (routing and required fields)."""
    err: List[str] = []
    warn: List[str] = []
    name = e["name"]
    if e.get("area") not in AREAS:
        err.append(f"area must be one of {AREAS} (got {e.get('area')!r})")
    if e["kind"] not in ("command", "human", "pool") and e.get("polarity") not in POLARITIES:
        err.append(f"polarity must be one of {POLARITIES} (got {e.get('polarity')!r})")
    if not e.get("tier") and e["kind"] in ("catanatron", "selfplay"):
        warn.append(f"no tier: defaults to t{int(e['priority']) // 10} for the Holm family")
    if e.get("polarity") == "new" and e.get("promise_pp") is None and e.get("tier") != "smoke" \
            and not e.get("bundle_of") \
            and e["kind"] in ("catanatron", "selfplay") and design_of(e) not in ("politics", "confirm"):
        err.append("a new row needs promise_pp (the effect it is expected to deliver, in pp)")
    if e.get("design") and e["design"] not in SEQ.DESIGNS:
        err.append(f"unknown design {e['design']!r}")
    if is_politics(e) and e.get("design") not in (None, "politics") and e["kind"] != "human":
        warn.append("politics-scope term: design forced to the fixed-N politics screen")
    area = e.get("area")
    kind = e["kind"]
    trades = e.get("trades") or "off"
    cand_trades = (e.get("cand_adapter_opts") or {}).get("trades") if isinstance(e.get("cand_adapter_opts"), dict) \
        else None
    if kind == "catanatron" and is_player_trade(e) and trades == "off" and cand_trades in (None, "off"):
        err.append("acts only through player trades: needs trades value/fair/native (3.3) or kind selfplay "
                   "(hint: add \"trades\": \"value\")")
    if kind == "catanatron" and trades != "off" and e.get("interpreter") != "py330":
        err.append("domestic trading needs catanatron 3.3 (interpreter py330)")
    if area == "counting":
        if kind in ("selfplay",):
            err.append("counting rows cannot run in self-play (belief is None there; bots see the full state)")
        if kind == "catanatron" and not info_counted(e) and not e.get("headroom"):
            err.append("counting rows need --info counted on both arms (adapter_opts info=counted)")
    if kind == "selfplay":
        if needs_pyeval(e):
            err.append("self-play on the Python evaluator is refused (170 games/h: 2,400 games ~14 h); port the "
                       "weight to C++ with a parameter first")
        if not (is_politics(e) or is_player_trade(e) or area in ("robber", "trades") or e.get("noharm")):
            err.append("self-play is limited to politics screens, trade no-harm checks and mechanics inert against "
                       "Catanatron (player trades, robber on knight holders)")
        if not e.get("tunable"):
            err.append("a self-play row needs a tunable (one value per ablate.py call)")
    if kind == "catanatron" and e.get("tunable") and not e.get("values") and not e.get("flag_off"):
        err.append("give explicit values (the queue does not read the registry's candidates)")
    if kind == "command" and e["enabled"] and not e.get("cmd"):
        err.append("a command row needs cmd")
    if e.get("parent"):
        par = plan.row(e["parent"])
        if par is None:
            err.append(f"parent {e['parent']!r} is not in the plan")
        else:
            if par.get("polarity") == "knockout":
                err.append("a knockout row is never a parent (knockouts spawn no fallbacks)")
            if e.get("fallback") not in FALLBACK_KINDS:
                err.append(f"fallback must be one of {FALLBACK_KINDS}")
            if fallback_depth(plan, e) > 2:
                err.append("fallback ladders are at most 2 deep")
    if e.get("confirms") and plan.row(e["confirms"]) is None:
        err.append(f"confirms {e['confirms']!r}: no such row")
    for m in e.get("bundle_of") or []:
        if plan.row(m) is None:
            err.append(f"bundle_of: no row {m!r}")
    if e.get("estimator") == "cv" and d_prior(e) < 0.15:
        warn.append("estimator cv on a low-divergence row gains nothing (variance ratio >= 0.92)")
    if screen_once(plan, e) and design_of(e) != "politics" and (e.get("polarity") != "new"
                                                                  or design_of(e) != "screen"):
        warn.append("screen_once only changes a new row's screen (intake power); ignored here")
    return err, warn


def screen_once(plan: Plan, e: Dict[str, Any]) -> bool:
    """Plan decision 1 (the user, 2026-09-26): take this row's one screen despite a power below 0.5 at its promise.
    The row field ``"screen_once": true`` or the plan-level list ``"screen_once": [row names]`` (the same; the list
    keeps the file readable by a queue process started before the field existed, which refuses unknown row fields)."""
    return bool(e.get("screen_once")) or e.get("name") in (plan.settings.get("screen_once") or [])


def headroom_upper_pp(q: "Queue", area: str) -> Optional[Tuple[float, str]]:
    """Upper 95% bound (pp) of an area's headroom: a final headroom row's estimate, or a static plan entry."""
    for e in q.plan.rows:
        if e.get("headroom") and e.get("area") == area and e["enabled"]:
            v = q.ledger.verdict(e["name"], cand_values(e)[0][0])
            if v is None:
                return None
            # a headroom row's candidate REMOVES the capability (trade proposals 0): the gap is -delta
            delta, se = v.get("delta") or 0.0, v.get("se") or 0.0
            return 100.0 * (-delta + SEQ.Z95 * se), e["name"]
    st = (q.plan.settings.get("headroom") or {}).get(area)
    if isinstance(st, dict) and st.get("gap") is not None:
        return 100.0 * (float(st["gap"]) + SEQ.Z95 * float(st.get("se", 0.0))), st.get("source", "plan")
    return None


def headroom_pending(q: "Queue", area: str) -> Optional[str]:
    for e in q.plan.rows:
        if e.get("headroom") and e.get("area") == area and e["enabled"]:
            if q.ledger.verdict(e["name"], cand_values(e)[0][0]) is None:
                return e["name"]
    return None


def intake(q: "Queue", e: Dict[str, Any], power_check: bool = True) -> Intake:
    it = Intake(design=design_of(e))
    errs, warns = lint(q.plan, e)
    it.warnings = warns
    if errs:
        it.status = "refused"
        it.reasons = errs
        return it
    if e["kind"] == "human":
        it.status = "deferred"
        it.reasons = ["route human: deferred to human testing (0 games)"]
        return it
    if e.get("polarity") != "new" or it.design not in ("screen",) or e.get("tier") == "smoke" \
            or e["kind"] not in ("catanatron", "selfplay"):
        return it
    members = [q.plan.row(m) for m in e.get("bundle_of") or []]
    members = [m for m in members if m is not None]
    # a bundle row: promise = the sum of its members' promises, D = the largest member D (unless given)
    promise = float(e["promise_pp"]) if e.get("promise_pp") is not None else \
        sum(float(m.get("promise_pp") or 0.0) for m in members)
    area = e.get("area")
    if area == "counting" and e["kind"] == "catanatron":
        hr = headroom_upper_pp(q, "counting")
        if hr is not None and hr[0] < COUNTING_HEADROOM_PP:
            it.status = "shelved"
            it.reasons = [f"SHELVE(headroom): counted-vs-full gap upper bound {hr[0]:.1f} pp < "
                          f"{COUNTING_HEADROOM_PP:g} pp ({hr[1]})"]
            return it
    if is_player_trade(e):
        hr = headroom_upper_pp(q, "trades")
        if hr is not None:
            cap = max(1.0, 0.5 * hr[0])
            if promise > cap:
                it.warnings.append(f"promise capped at {cap:.1f} pp by the trades headroom ({hr[1]})")
                promise = cap
    it.promise = promise
    D = d_prior(e) if e["kind"] == "catanatron" else 1.0      # 2v2 self-play: almost every game is decided
    if members and e.get("d_prior") is None and e["kind"] == "catanatron":
        D = max([D] + [d_prior(m) for m in members])
    n_max = e["seeds"]["count"] if e["kind"] == "catanatron" else int(e.get("games") or e["seeds"]["count"])
    unit = 1.0 if e["kind"] == "catanatron" else 0.5
    it.d = D
    if not power_check:
        return it
    D_eff = D
    red = q.reducer_for(e)
    if red is not None:
        D_eff = D * red[1]
        it.reducer = red[0]
    it.power = SEQ.power_at(promise / unit if unit != 1.0 else promise, D_eff, n_max)
    if it.power < SEQ.UNPROVEN_POWER and screen_once(q.plan, e):
        # plan decision 1: one screen within the budget; lint, the cap, early stopping and the look-1
        # 'unprovable at promise' re-check still apply, an unclear screen ends SHELVE
        it.warnings.append(f"screen_once: power {it.power:.2f} < {SEQ.UNPROVEN_POWER:g} at +{promise:g} pp "
                           f"(D {D_eff:.2f}, {n_max} pairs); screened once anyway (plan decision 1)")
        return it
    if it.power < SEQ.UNPROVEN_POWER:
        bundle = next((b for b in q.plan.rows if b["enabled"] and e["name"] in (b.get("bundle_of") or [])), None)
        if bundle is not None:
            it.status = "bundled"
            it.reasons = [f"power {it.power:.2f} < {SEQ.UNPROVEN_POWER:g} at +{promise:g} pp (D {D_eff:.2f}): runs "
                          f"inside bundle {bundle['name']}"]
            return it
        sib = [s for s in q.plan.rows if s is not e and s["enabled"] and s.get("area") == area
               and s.get("polarity") == "new" and s.get("promise_pp") and s["kind"] == e["kind"]]
        prop = ""
        if sib:
            tot = promise + sum(float(s["promise_pp"]) for s in sib)
            dmax = max([D] + [d_prior(s) for s in sib])
            prop = (f"; a bundle with {', '.join(s['name'] for s in sib)} (promise {tot:g} pp, D {dmax:.2f}) would "
                    f"have power {SEQ.power_at(tot, max(dmax, tot / 100 + 1e-6), n_max):.2f} (add a row with "
                    f"bundle_of to run it)")
        it.status = "shelved"
        it.reasons = [f"SHELVE(intake: unprovable at budget): power {it.power:.2f} at +{promise:g} pp (D prior "
                      f"{D_eff:.2f}, {n_max} pairs){prop}"]
    return it


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
class Ledger:
    """``<dir>/ledger.jsonl``: epochs, pins, looks, verdicts (the first per candidate wins), alerts, chunks,
    gates and pools.  Rebuildable from the rows' ``{kind: stop, source: queue}`` records (verdicts only)."""

    def __init__(self, path: str):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self._offset = 0
        self.writer = AB.JsonlWriter(path)
        self.reload()

    def reload(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        end = data.rfind(b"\n")
        if end < 0:
            return
        for line in data[:end + 1].decode("utf-8", "replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                self.records.append(r)
        self._offset += end + 1

    def append(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(rec)
        rec.setdefault("v", 1)
        rec.setdefault("t", now())
        self.writer.append([rec])
        self.reload()
        return rec

    def of(self, kind: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("kind") == kind]

    def verdict(self, row: str, cand: str) -> Optional[Dict[str, Any]]:
        for r in self.records:
            if r.get("kind") == "verdict" and r.get("row") == row and r.get("cand") == cand:
                return r
        return None

    def verdicts(self, row: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            if r.get("kind") == "verdict" and r.get("row") == row:
                out.setdefault(r.get("cand"), r)
        return out

    def looks(self, row: str, cand: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("kind") == "look" and r.get("row") == row and r.get("cand") == cand]

    def pin(self, row: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.records if r.get("kind") == "pin" and r.get("row") == row), None)

    def chunks(self, row: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("kind") == "chunk" and r.get("row") == row]

    def gate(self, row: str) -> Optional[Dict[str, Any]]:
        return next((r for r in reversed(self.records) if r.get("kind") == "gate" and r.get("row") == row), None)

    def epochs(self) -> List[Dict[str, Any]]:
        return self.of("epoch")

    def area_epoch(self, area: str) -> Optional[Dict[str, Any]]:
        cur = None
        for r in self.epochs():
            areas = r.get("areas") or "all"
            if areas == "all" or area in areas:
                cur = r
        return cur

    def epoch(self, eid: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.epochs() if r.get("id") == eid), None)

    def has_alert(self, row: str, msg: str) -> bool:
        return any(r.get("kind") == "alert" and r.get("row") == row and r.get("msg") == msg for r in self.records)


# ---------------------------------------------------------------------------
# Snapshots (one code epoch per row, from an explicit allowlist)
# ---------------------------------------------------------------------------
SNAPSHOT_SCRIPTS = ("ablate_catanatron.py", "ablate.py", "bench_catanatron.py", "factorial_catanatron.py",
                    "tune_joint.py", "mechanics.py", "seqtest.py", "decision_shadow.py", "winpaths_shadow.py",
                    "port_gate.py", "dev_belief_calibrate.py", "belief_shadow.py")
CHECK_CODE = r"""
import importlib.util, json, os, sys
root = os.getcwd()
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("_ab_check", os.path.join(root, "scripts", "ablate_catanatron.py"))
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
out = {"fingerprint": ab.code_fingerprint(), "evaluator": ab.evaluator_mode(), "missing": []}
names = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
if names:
    from catanbot import tuning
    for n in names:
        try:
            tuning.find(n)
        except KeyError:
            out["missing"].append(n)
print(json.dumps(out))
"""


def snapshot_files(src: str) -> List[str]:
    """The allowlist, relative to ``src``: catanbot/**/*.py without vision/ and __pycache__, the C++ extension
    (not ``*.staged.so``) and the scripts the chunks run."""
    out = []
    base = os.path.join(src, "catanbot")
    for dirpath, dirnames, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, src)
        parts = rel.split(os.sep)
        if "vision" in parts[1:2] or "__pycache__" in parts:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "vision")]
        for fn in filenames:
            if fn.endswith(".py") or (fn.startswith("catanbot_core") and fn.endswith(".so") and "staged" not in fn):
                out.append(os.path.join(rel, fn))
    for fn in SNAPSHOT_SCRIPTS:
        if os.path.exists(os.path.join(src, "scripts", fn)):
            out.append(os.path.join("scripts", fn))
    return sorted(out)


def files_sha(root: str, files: Sequence[str]) -> str:
    h = hashlib.sha1()
    for f in files:
        h.update(f.encode())
        with open(os.path.join(root, f), "rb") as fh:
            h.update(hashlib.sha1(fh.read()).digest())
    return h.hexdigest()[:16]


def run_check(interp: str, root: str, names: Sequence[str] = (), timeout: float = 300.0) -> Dict[str, Any]:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=root)
    env.pop("CATANBOT_NO_ACCEL", None)
    proc = subprocess.run([interp, "-c", CHECK_CODE, json.dumps(list(names))], cwd=root, env=env,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"snapshot check failed ({interp} in {root}): {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def make_snapshot(src: str, dest: str, interps: Sequence[str], retries: int = 3) -> Dict[str, Any]:
    """Copy the allowlist from ``src`` to ``dest`` (read-only) and assert, per interpreter, that the snapshot's
    code fingerprint equals the source's at copy time and that the C++ evaluator loads there.  A copy that
    raced an edit (source fingerprint changed during the copy) is retried."""
    last = None
    for _ in range(retries):
        before = {i: run_check(i, src)["fingerprint"] for i in interps}
        if os.path.exists(dest):
            _make_writable(dest)
            shutil.rmtree(dest)
        files = snapshot_files(src)
        for f in files:
            d = os.path.join(dest, f)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(os.path.join(src, f), d)
        after = {i: run_check(i, src)["fingerprint"] for i in interps}
        snap = {i: run_check(i, dest) for i in interps}
        last = (before, after, snap)
        if before == after and all(snap[i]["fingerprint"] == after[i] for i in interps):
            bad = [i for i in interps if not str(snap[i]["evaluator"]).startswith("c++")]
            if bad:
                raise RuntimeError(f"snapshot {dest}: the C++ evaluator does not load for {bad}: "
                                   f"{[snap[i]['evaluator'] for i in bad]}")
            for dirpath, _, filenames in os.walk(dest):
                for fn in filenames:
                    p = os.path.join(dirpath, fn)
                    os.chmod(p, os.stat(p).st_mode & ~(statmod.S_IWUSR | statmod.S_IWGRP | statmod.S_IWOTH))
            return {"fingerprint": after, "files_sha": files_sha(dest, files), "files": len(files),
                    "evaluator": {i: snap[i]["evaluator"] for i in interps}}
    raise RuntimeError(f"snapshot {dest}: fingerprints kept changing during the copy: {last}")


def _make_writable(path: str) -> None:
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                os.chmod(p, os.stat(p).st_mode | statmod.S_IWUSR)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Machine: CPU accounting, proof guard, load gate
# ---------------------------------------------------------------------------
def children_cpu() -> float:
    """CPU seconds of every waited-for descendant (the chunk process and its pool workers)."""
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def list_processes() -> List[Tuple[int, str]]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if cmd:
            out.append((int(d), cmd))
    return out


def read_proc_stat() -> Tuple[List[int], int]:
    with open("/proc/stat") as fh:
        lines = fh.read().splitlines()
    cpu = [int(x) for x in lines[0].split()[1:]]
    n = sum(1 for ln in lines if re.match(r"cpu\d+ ", ln))
    return cpu, max(1, n)


def busy_cores(a: Tuple[List[int], int], b: Tuple[List[int], int]) -> float:
    """Busy cores between two /proc/stat samples (idle and iowait count as idle)."""
    da = [y - x for x, y in zip(a[0], b[0])]
    total = sum(da)
    if total <= 0:
        return 0.0
    idle = da[3] + (da[4] if len(da) > 4 else 0)
    return (total - idle) / total * b[1]


# ---------------------------------------------------------------------------
# Incremental JSONL indexes
# ---------------------------------------------------------------------------
class IndexCache:
    """ablate_catanatron ``Index`` per JSONL, extended by byte offset (a partial last line waits)."""

    def __init__(self):
        self.cache: Dict[str, Tuple[Any, int]] = {}

    def get(self, path: str):
        idx, off = self.cache.get(path, (None, 0))
        if idx is None:
            idx = AB.Index()
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size < off:            # truncated / rewritten: start over
                idx, off = AB.Index(), 0
            if size > off:
                with open(path, "rb") as fh:
                    fh.seek(off)
                    data = fh.read()
                end = data.rfind(b"\n")
                if end >= 0:
                    for line in data[:end + 1].decode("utf-8", "replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(r, dict):
                            idx.add(r)
                    off += end + 1
        self.cache[path] = (idx, off)
        return idx


# ---------------------------------------------------------------------------
# Row state
# ---------------------------------------------------------------------------
@dataclass
class Cand:
    key: str
    value: Any
    run: Optional[Dict[str, Any]] = None
    verdict: Optional[Dict[str, Any]] = None
    looks_done: int = 0
    last: Optional[Dict[str, Any]] = None       # last look record

    @property
    def open(self) -> bool:
        return self.verdict is None


@dataclass
class RowState:
    e: Dict[str, Any]
    intake: Intake
    status: str = "ELIGIBLE"          # ELIGIBLE | RUNNING | WAITING | BLOCKED | DISABLED | REFUSED | SHELVED |
    reason: str = ""                  # DEFERRED | BUNDLED | FINAL | NOT TRIGGERED
    cands: List[Cand] = field(default_factory=list)
    epoch: Optional[str] = None
    n_max: int = 0
    base: int = 0
    design: str = "screen"
    cost_h: float = 0.0               # remaining CPU-hours (preemption)
    cpu_s: float = 0.0                # spent CPU seconds
    games: int = 0

    @property
    def name(self) -> str:
        return self.e["name"]

    @property
    def area(self) -> str:
        return self.e.get("area") or "other"

    @property
    def open_cands(self) -> List[Cand]:
        return [c for c in self.cands if c.open]

    @property
    def final(self) -> bool:
        return self.status in ("FINAL", "NOT TRIGGERED", "SHELVED", "REFUSED", "DEFERRED", "BUNDLED")

    @property
    def runnable(self) -> bool:
        return self.status in ("ELIGIBLE", "RUNNING")

    def sort_key(self) -> Tuple:
        return (AREA_RANK.get(self.area, len(AREAS) - 1), 0 if self.e.get("headroom") else 1, self.e["priority"], self.e["_order"])


def looks_for(e: Dict[str, Any], design: str, n_max: int) -> List[int]:
    if e["kind"] == "selfplay":
        if design in ("screen", "knockout"):
            return SEQ.look_sizes(n_max, SEQ.K_LOOKS, multiple=2 * SELFPLAY_CHUNK)
        return [n_max]
    if design in ("screen", "knockout"):
        return SEQ.look_sizes(n_max)
    if design == "estimate" and is_aa(e):
        return [min(SEQ.AA_PAIRS, n_max)]
    if design == "politics" and n_max > SEQ.ALERT_PAIRS:
        return [SEQ.ALERT_PAIRS, n_max]
    return [n_max]


def needs_mode(e: Dict[str, Any]) -> Optional[str]:
    if is_player_trade(e):
        return "trades"
    if e.get("area") == "counting":
        return "counting"
    return None


def cpu_prior(e: Dict[str, Any]) -> float:
    """CPU seconds per game and arm before the row has measured its own (design F3)."""
    if e["kind"] == "selfplay":
        return CPU_SELFPLAY
    if e["kind"] == "command":
        return float(e.get("est_cpu_h") or 0.5) * 3600.0
    opp = e.get("opponent") or "value"
    py = needs_pyeval(e)
    info = "counted" if info_counted(e, "cand") else "full"
    base = CPU_PRIOR.get((opp, py, info)) or CPU_PRIOR.get((opp, False, info)) or CPU_PRIOR.get((opp, py, "full")) \
        or 1.77
    if (e.get("trades") or "off") != "off":
        base = max(base, CPU_TRADES_VALUE)
    if CAMP.search_depth(e) >= 2:
        base = base * (1 + CPU_DEPTH2) / 2.0          # one arm at depth 2
    if str(e.get("cand_spec") or "").startswith("heuristic"):
        base = base * (1 + CPU_HEURISTIC_BOT) / 2.0
    return base


def default_class(e: Dict[str, Any]) -> str:
    """Rows whose default arms are the same games (shared within an epoch)."""
    if e["kind"] not in ("catanatron", "pool"):
        return f"{e['kind']}:{e['name']}"
    def_spec = e.get("def_spec") or e.get("base_spec") or DEFAULT_SEARCH_SPEC
    return _jdump([e.get("interpreter"), e.get("opponent"), e.get("opponent_params"), e.get("trades") or "off",
                   def_spec, e.get("set"), e.get("adapter_opts"), needs_pyeval(e), CAMP.crn_mode(e),
                   e.get("vps_to_win"), e.get("discard_limit")])


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------
class Queue:
    def __init__(self, plan_path: str, d: str, home: Optional[str] = None, snapshot: bool = True,
                 slice_minutes: float = SLICE_MINUTES, max_workers: int = 3, guard: str = "proof_snapshot",
                 lock_files: Sequence[str] = (), no_intake: bool = False, hard: bool = False, order: str = "priority",
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 cpu: Callable[[], float] = children_cpu, proc_list: Callable[[], List[Tuple[int, str]]] = list_processes,
                 proc_stat: Callable[[], Tuple[List[int], int]] = read_proc_stat, runner=None, log=sys.stdout,
                 tunable_check: bool = True, src_root: str = ROOT, poll_s: float = 30.0):
        self.plan_path = plan_path
        self.dir = os.path.abspath(d)
        os.makedirs(os.path.join(self.dir, "logs"), exist_ok=True)
        inside = os.path.commonpath([self.dir, ROOT]) == ROOT
        self.home = os.path.abspath(home) if home else (
            os.path.join(os.path.dirname(ROOT), "queue_runs", os.path.basename(self.dir)) if inside else self.dir)
        if snapshot and os.path.commonpath([self.home, ROOT]) == ROOT:
            raise SystemExit(f"error: snapshots must live outside the repository ({self.home}); pass --home")
        self.snapshot = snapshot
        self.slice_minutes = slice_minutes
        self.max_workers = max_workers
        self.guard = guard
        self.lock_files = list(lock_files) or [os.path.join(os.path.dirname(ROOT), "PROOF_RUNNING")]
        self.no_intake = no_intake
        self.hard = hard
        self.order_mode = order
        self.clock, self.sleep, self.cpu = clock, sleep, cpu
        self.proc_list, self.proc_stat = proc_list, proc_stat
        self.runner = runner or self._run_process
        self.log = log
        self.tunable_check = tunable_check
        self.src_root = os.path.abspath(src_root)
        self.poll_s = poll_s
        self.ledger = Ledger(os.path.join(self.dir, "ledger.jsonl"))
        self.indexes = IndexCache()
        self.plan = load_plan(plan_path)
        self._plan_mtime = os.path.getmtime(plan_path)
        self.rows: Dict[str, RowState] = {}
        self._tcheck: Dict[Tuple[str, str], List[str]] = {}
        self.incumbent: Optional[str] = None
        self.exclusive_wait_since: Optional[float] = None

    # -- helpers ------------------------------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        print(msg, file=self.log, flush=True)

    def jsonl(self, e: Dict[str, Any]) -> str:
        return os.path.join(self.dir, f"{e['name']}.jsonl")

    def reducer_for(self, e: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """A variance reducer available for the row: ``(name, D multiplier)`` (CRN after a passed pilot, CV with a
        pool of M >= 4 N_max on its default arm), else None."""
        crn = self.crn_status()
        if crn and crn.get("pass") and CAMP.crn_mode(e) in ("dice", "auto"):
            return ("crn", crn["ratio"])
        if e.get("estimator") == "cv":
            pool = self.pool_for(e)
            if pool is not None and pool["m"] >= 4 * e["seeds"]["count"]:
                p = pool["mean"] or 0.5
                D = d_prior(e)
                return ("cv", SEQ.cv_variance_ratio(D, p, e["seeds"]["count"], pool["m"]))
        return None

    def crn_status(self) -> Optional[Dict[str, Any]]:
        cfg = self.plan.settings.get("crn_pilot")
        if not cfg:
            return None
        off, dice, aa = (self.plan.row(cfg.get(k, "")) for k in ("off", "dice", "aa"))
        if not off or not dice:
            return None
        vo = self.ledger.verdict(off["name"], cand_values(off)[0][0])
        vd = self.ledger.verdict(dice["name"], cand_values(dice)[0][0])
        if not vo or not vd:
            return {"pass": False, "pending": True}
        so, sd = vo.get("stats") or {}, vd.get("stats") or {}
        d_off = (so.get("plus", 0) + so.get("minus", 0)) / max(1, so.get("n", 1))
        d_dice = (sd.get("plus", 0) + sd.get("minus", 0)) / max(1, sd.get("n", 1))
        ratio = d_dice / d_off if d_off > 0 else 1.0
        aa_ok = True
        if aa is not None:
            va = self.ledger.verdict(aa["name"], cand_values(aa)[0][0])
            aa_ok = bool(va and va.get("verdict") == "PASS")
        drop = float(cfg.get("drop", CRN_DROP))
        return {"pass": ratio <= 1.0 - drop and aa_ok, "ratio": ratio, "d_off": d_off, "d_dice": d_dice,
                "aa_ok": aa_ok, "pending": False}

    def pool_for(self, e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cls = default_class(e)
        for p in reversed(self.ledger.of("pool")):
            prow = self.plan.row(p.get("row", ""))
            if prow is not None and default_class(prow) == cls:
                pin = self.ledger.pin(e["name"])
                if pin is None or pin.get("epoch") == p.get("epoch"):
                    return p
        return None

    # -- refresh --------------------------------------------------------------------------------------------------
    def reload_plan(self) -> None:
        try:
            mt = os.path.getmtime(self.plan_path)
        except OSError:
            return
        if mt == self._plan_mtime:
            return
        self._plan_mtime = mt
        try:
            new = load_plan(self.plan_path)
        except PlanError as ex:
            msg = f"plan re-read failed, keeping the previous plan: {ex}"
            self.say(msg)
            self.ledger.append({"kind": "alert", "row": "", "msg": msg})
            return
        if new.sha != self.plan.sha:
            self.say(f"plan re-read ({new.sha})")
        self.plan = new

    def refresh(self) -> None:
        self.ledger.reload()
        self.reload_plan()
        rows: Dict[str, RowState] = {}
        for e in self.plan.rows:
            it = intake(self, e, power_check=not self.no_intake) if e["enabled"] else Intake(design=design_of(e))
            if self.no_intake and it.status == "shelved":
                it.status = "ok"
            rs = RowState(e=e, intake=it, design=it.design)
            n_max = int(e.get("games") or e["seeds"]["count"]) if e["kind"] == "selfplay" else e["seeds"]["count"]
            pin = self.ledger.pin(e["name"])
            if pin is not None:
                rs.epoch = pin.get("epoch")
                if pin.get("n_max") and int(pin["n_max"]) != n_max:
                    self._alert_once(e["name"], f"cap change ignored: the row started with {pin['n_max']} "
                                                f"(caps are frozen once a row starts)")
                    n_max = int(pin["n_max"])
            rs.n_max = n_max
            rs.base = effective_base(self.plan, e) if e["kind"] != "selfplay" else (e["seeds"].get("base") or
                                                                                   SELFPLAY_SEED)
            if e["kind"] == "selfplay" and rs.base == 0:
                rs.base = SELFPLAY_SEED
            rs.cands = [Cand(k, v) for k, v in cand_values(e)]
            for c in rs.cands:
                c.verdict = self.ledger.verdict(e["name"], c.key)
                c.looks_done = len(self.ledger.looks(e["name"], c.key))
                lk = self.ledger.looks(e["name"], c.key)
                c.last = lk[-1] if lk else None
            chunks = self.ledger.chunks(e["name"])
            rs.cpu_s = sum(float(x.get("cpu_s") or 0.0) for x in chunks)
            rs.games = sum(int(x.get("games") or 0) for x in chunks)
            rows[e["name"]] = rs
        self.rows = rows
        for rs in self._ordered(rows.values()):
            self._status(rs)
            if rs.status in ("ELIGIBLE", "RUNNING") and rs.e["kind"] in ("catanatron", "selfplay", "pool", "command"):
                self.evaluate(rs)
                self._stalled(rs)
                self._status(rs)
        for rs in self._ordered(rows.values()):     # second pass: gates on rows that settled in the first
            if not rs.final:
                self._status(rs)
        for rs in rows.values():
            rs.cost_h = self.remaining_cost(rs) if rs.runnable else 0.0

    def _alert_once(self, row: str, msg: str) -> None:
        if not self.ledger.has_alert(row, msg):
            self.ledger.append({"kind": "alert", "row": row, "msg": msg})
            self.say(f"[{row}] {msg}")

    def _ordered(self, rows: Iterable[RowState]) -> List[RowState]:
        return sorted(rows, key=lambda r: r.sort_key())

    # -- status / eligibility -------------------------------------------------------------------------------------
    def _status(self, rs: RowState) -> None:
        e = rs.e
        it = rs.intake
        if not e["enabled"]:
            rs.status, rs.reason = "DISABLED", e.get("requires") or "enabled: false"
            return
        if it.status == "refused":
            rs.status, rs.reason = "REFUSED", "lint: " + "; ".join(it.reasons)
            return
        if it.status == "deferred":
            rs.status, rs.reason = "DEFERRED", it.reasons[0]
            return
        if all(not c.open for c in rs.cands) and rs.cands:
            rs.status, rs.reason = "FINAL", ""
            if any((c.verdict or {}).get("verdict") == "NOT TRIGGERED" for c in rs.cands):
                rs.status = "NOT TRIGGERED"
            return
        if it.status == "shelved":
            self._final_all(rs, "SHELVE", it.reasons[0], label="SHELVE(intake)", persist=False)
            rs.status, rs.reason = "SHELVED", it.reasons[0]
            return
        if it.status == "bundled":
            rs.status, rs.reason = "BUNDLED", it.reasons[0]
            return
        started = self.ledger.pin(e["name"]) is not None
        # identity of a started row
        if started:
            pin = self.ledger.pin(e["name"])
            if pin.get("identity") and pin["identity"] != self.identity(e):
                self._final_open(rs, "FAILED", "plan-changed", label="FAILED(plan-changed)")
                rs.status, rs.reason = "FINAL", "FAILED(plan-changed)"
                return
            pinned_vals = pin.get("values") or []
            for key in pinned_vals:
                if key not in {c.key for c in rs.cands} and self.ledger.verdict(e["name"], key) is None:
                    self.ledger.append({"kind": "verdict", "row": e["name"], "cand": key, "verdict": "WITHDRAWN",
                                        "label": "WITHDRAWN", "reason": "value removed from the plan",
                                        "run_key": self._run_key_of(e, key), "source": "queue"})
                    self._write_stop(e, key, self._run_key_of(e, key),
                                     {"verdict": "WITHDRAWN", "label": "WITHDRAWN", "reason": "value removed"})
        wait = self._gates(rs)
        if wait is not None:
            kind, why = wait
            rs.status, rs.reason = kind, why
            return
        rs.status = "RUNNING" if started else "ELIGIBLE"
        rs.reason = ""

    def identity(self, e: Dict[str, Any]) -> str:
        keys = ("kind", "interpreter", "opponent", "opponent_params", "tunable", "flag_off", "base_spec", "cand_spec",
                "def_spec", "cand_set", "set", "adapter_opts", "cand_adapter_opts", "trades", "players", "counters",
                "cmd", "estimator")
        doc = {k: e.get(k) for k in keys}
        doc["base"] = effective_base(self.plan, e)
        doc["design"] = design_of(e)
        doc["crn"] = CAMP.crn_mode(e)
        return hashlib.sha1(_jdump(doc).encode()).hexdigest()[:12]

    def _run_key_of(self, e: Dict[str, Any], key: str) -> Optional[str]:
        if e["kind"] != "catanatron":
            return None
        idx = self.indexes.get(self.jsonl(e))
        run = self._find_run(idx, e, key)
        return run["run_key"] if run else None

    def _gates(self, rs: RowState) -> Optional[Tuple[str, str]]:
        """``(status, reason)`` when the row may not run now (WAITING / BLOCKED / NOT TRIGGERED), else None."""
        e = rs.e
        # a member of an enabled bundle: bundle first; shelved with it, superseded by its knockouts after an ADOPT
        for b in self.plan.rows:
            if b["enabled"] and e["name"] in (b.get("_members") or []):
                bs = self.rows.get(b["name"])
                if bs is None or not bs.final:
                    return "WAITING", f"bundle first: {b['name']}"
                if any((c.verdict or {}).get("verdict") == "ADOPT" for c in bs.cands):
                    self._final_all(rs, "NOT TRIGGERED", f"superseded by the knockouts of bundle {b['name']}",
                                    persist=False)
                    return "NOT TRIGGERED", f"superseded by the knockouts of bundle {b['name']}"
                self._final_all(rs, "SHELVE", f"with bundle {b['name']}", label="SHELVE(with bundle)", persist=False)
                return "SHELVED", f"shelved with bundle {b['name']}"
        # after: other rows' verdicts
        afters = e.get("after") or []
        if isinstance(afters, dict):
            afters = [afters]
        for a in afters:
            tgt = self.rows.get(a.get("row", ""))
            if tgt is None:
                return "BLOCKED", f"after: no row {a.get('row')!r}"
            if not tgt.final:
                return "WAITING", f"after {tgt.name}"
            vs = [c.verdict for c in tgt.cands if c.verdict]
            want = a.get("verdicts")
            ok = True
            if want:
                ok = any(any(str(v.get("label", "")).startswith(w) or v.get("verdict") == w for w in want) for v in vs)
            if ok and a.get("cond"):
                ok = any(_cond(a["cond"], v) for v in vs)
            if not ok:
                self._final_all(rs, "NOT TRIGGERED", f"after {tgt.name}: condition not met", persist=False)
                return "NOT TRIGGERED", f"after {tgt.name}: condition not met"
        # fallback of a parent
        if e.get("parent"):
            par = self.rows.get(e["parent"])
            if par is None:
                return "BLOCKED", f"parent {e['parent']!r} missing"
            if not par.final:
                return "WAITING", f"parent {par.name} ({par.status})"
            trig, why = self.fallback_trigger(par, e.get("fallback"))
            if not trig:
                self._final_all(rs, "NOT TRIGGERED", f"fallback {e.get('fallback')} of {par.name}: {why}",
                                persist=False)
                return "NOT TRIGGERED", why
        # confirmation of a parent's ADOPT (Holm within the parent's tier)
        if e.get("confirms"):
            par = self.rows.get(e["confirms"])
            if par is None:
                return "BLOCKED", f"confirms {e['confirms']!r} missing"
            ok_vals, why = self.confirm_trigger(par)
            if ok_vals is None:
                return "WAITING", why
            if not ok_vals:
                self._final_all(rs, "NOT TRIGGERED", why, persist=False)
                return "NOT TRIGGERED", why
            if e.get("tunable"):
                for c in rs.cands:
                    if c.open and c.key not in ok_vals:
                        self._final_one(rs, c, "NOT TRIGGERED", f"{par.name}={c.key} was not confirmed-eligible",
                                        persist=False)
        # headroom first within the area
        if e.get("polarity") == "new" and not e.get("headroom"):
            pend = headroom_pending(self, rs.area)
            if pend is not None and (rs.area != "trades" or is_player_trade(e)):
                return "WAITING", f"headroom row {pend}"
        # zero-game shadow gate
        sg = e.get("shadow_gate")
        if sg:
            res = self.shadow_share(sg)
            if res is None:
                return "WAITING", f"shadow {sg.get('row')}"
            if isinstance(res, str):
                return "BLOCKED", res
            share, n = res
            if share < float(sg.get("min_share", MIN_SHADOW_SHARE)):
                why = (f"NOOP(shadow): {share:.1%} of {n} {'/'.join(sg.get('classes', []))} decisions changed "
                       f"(< {float(sg.get('min_share', MIN_SHADOW_SHARE)):.0%})")
                self._final_all(rs, "NOOP", why, label="NOOP(shadow)", persist=False)
                return "FINAL", why
        # exclusive / epoch / tunables
        if self.tunable_check and e["kind"] in ("catanatron", "selfplay"):
            missing = self.missing_tunables(rs)
            if missing:
                return "BLOCKED", f"needs --bump-code: {', '.join(missing)} not in epoch {self.row_epoch(rs)}"
        return None

    def shadow_share(self, sg: Dict[str, Any]):
        row = self.rows.get(sg.get("row", ""))
        if row is None:
            return f"shadow_gate: no row {sg.get('row')!r}"
        g = self.ledger.gate(row.name)
        if g is None or g.get("result") not in ("PASS", "FAIL"):
            return None
        if g.get("result") == "FAIL":
            return f"shadow {row.name} FAILED ({g.get('detail')})"
        path = g.get("path")
        try:
            with open(path) as fh:
                res = json.load(fh)
        except (OSError, TypeError, ValueError):
            return f"shadow {row.name}: result {path} unreadable"
        ds = _load("_decision_shadow_for_queue", os.path.join(SCRIPTS, "decision_shadow.py"))
        got = ds.gate_share(res, sg.get("key", ""), sg.get("classes") or ["robber", "knight"])
        if got is None:
            return f"shadow {row.name}: no candidate {sg.get('key')!r}"
        return got

    def fallback_trigger(self, par: RowState, kind: Optional[str]) -> Tuple[bool, str]:
        """Does the parent's outcome make this fallback kind eligible?"""
        vs = [c.verdict for c in par.cands if c.verdict]
        if not vs:
            return False, "parent has no verdict"
        if any(v.get("verdict") == "ADOPT" for v in vs):
            return False, "parent ADOPTed"
        if any(v.get("verdict") in ("FAILED", "BLOCKED", "WITHDRAWN") for v in vs):
            return False, "parent FAILED (fallbacks never follow a failure)"
        best = max(vs, key=lambda v: (v.get("delta") if v.get("delta") is not None else -9))
        verdict, reason = best.get("verdict"), str(best.get("reason") or "")
        flags = best.get("flags") or []
        if kind == "simpler":
            ok = verdict in ("SHELVE", "REJECT") or "promise not met" in flags
            return ok, f"parent {best.get('label')}"
        if kind == "milder":
            if not ((verdict == "REJECT" and reason.startswith("worse")) or
                    (verdict == "SHELVE" and reason.startswith("no gain"))):
                return False, f"parent {best.get('label')} (milder needs REJECT(worse) or SHELVE(no gain))"
            mech = par.e.get("mechanism") or {}
            summ = (best.get("mech") or {}).get(mech.get("metric", ""))
            if not mech or not summ:
                return False, "no mechanism readout for the overshoot rule"
            ok = MECH.overshoot(summ, mech.get("direction", "+"), best.get("delta"))
            return ok, (f"overshoot of {mech.get('metric')}" if ok else f"no overshoot of {mech.get('metric')}")
        if kind == "stronger":
            ok = verdict == "NOOP" or (verdict == "SHELVE" and reason.startswith("too small"))
            return ok, f"parent {best.get('label')}"
        return False, f"unknown fallback kind {kind!r}"

    def tier_family(self, tier: str) -> Tuple[Dict[str, float], bool]:
        """``({row|cand: p}, complete)`` over the candidates of a tier's rows that screen a hypothesis."""
        fam: Dict[str, float] = {}
        complete = True
        for rs in self.rows.values():
            e = rs.e
            if (e.get("tier") or f"t{int(e['priority']) // 10}") != tier or not e["enabled"]:
                continue
            if e["kind"] not in ("catanatron", "selfplay") or is_aa(e) or rs.status in ("REFUSED", "DEFERRED",
                                                                                         "BUNDLED", "DISABLED"):
                continue
            if e.get("confirms") or e.get("tier") in NO_HOLM_TIERS:
                continue
            for c in rs.cands:
                v = c.verdict
                if v is None:
                    if rs.status not in ("SHELVED", "NOT TRIGGERED"):
                        complete = False
                    continue
                if v.get("p") is not None and v.get("verdict") not in ("FAILED", "WITHDRAWN", "NOT TRIGGERED"):
                    fam[f"{e['name']}|{c.key}"] = float(v["p"])
        return fam, complete

    def holm(self, tier: str) -> Tuple[Dict[str, float], bool]:
        fam, complete = self.tier_family(tier)
        if not fam:
            return {}, complete
        res = SEQ.S().holm(fam, 0.05)
        return {k: v["p_adj"] for k, v in res.items()}, complete

    def confirm_trigger(self, par: RowState) -> Tuple[Optional[List[str]], str]:
        """Candidate keys of the parent that a confirmation row may test (ADOPT, or a significant estimate, with a
        Holm-adjusted stage-wise p < 0.05 within the parent's tier); None while the tier is incomplete."""
        tier = par.e.get("tier") or f"t{int(par.e['priority']) // 10}"
        adj, complete = self.holm(tier)
        if not par.final or not complete:
            return None, f"confirmation waits for tier {tier} to complete (Holm family)"
        ok = []
        for c in par.cands:
            v = c.verdict or {}
            p = adj.get(f"{par.name}|{c.key}")
            if p is None or p >= 0.05:
                continue
            if v.get("verdict") in ("ADOPT", "REMOVE") or (v.get("verdict") == "ESTIMATE" and v.get("delta")):
                ok.append(c.key)
        return ok, ("" if ok else f"{par.name}: no candidate ADOPTed with Holm p < 0.05 in tier {tier}")

    def _final_all(self, rs: RowState, verdict: str, reason: str, label: Optional[str] = None,
                   persist: bool = True) -> None:
        for c in rs.cands:
            if c.open:
                self._final_one(rs, c, verdict, reason, label, persist)

    def _final_open(self, rs: RowState, verdict: str, reason: str, label: Optional[str] = None) -> None:
        self._final_all(rs, verdict, reason, label)

    def _final_one(self, rs: RowState, c: Cand, verdict: str, reason: str, label: Optional[str] = None,
                   persist: bool = True) -> None:
        """Close a candidate.  ``persist=False`` (gate outcomes: intake SHELVE, NOT TRIGGERED, NOOP(shadow)) keeps
        the verdict in memory only: it is recomputed at every refresh from the plan and the ledger, so a plan edit
        can reopen the row; verdicts from games, FAILED and WITHDRAWN go to the ledger (the first one wins)."""
        if self.ledger.verdict(rs.name, c.key) is not None:
            c.verdict = self.ledger.verdict(rs.name, c.key)
            return
        if not persist:
            c.verdict = {"kind": "verdict", "row": rs.name, "cand": c.key, "verdict": verdict,
                         "label": label or verdict, "reason": reason, "design": rs.design, "pairs": 0,
                         "transient": True}
            return
        run_key = self._run_key_of(rs.e, c.key)
        rec = {"kind": "verdict", "row": rs.name, "cand": c.key, "run_key": run_key, "verdict": verdict,
               "label": label or verdict, "reason": reason, "design": rs.design, "source": "queue", "pairs": 0}
        c.verdict = self.ledger.append(rec)
        if run_key:
            self._write_stop(rs.e, c.key, run_key, rec)

    def _write_stop(self, e: Dict[str, Any], key: str, run_key: Optional[str], v: Dict[str, Any]) -> None:
        if not run_key or e["kind"] != "catanatron":
            return
        idx = self.indexes.get(self.jsonl(e))
        if run_key in idx.queue_stops:
            return
        rec = {"v": 1, "kind": "stop", "run_key": run_key, "pairs": v.get("pairs"), "delta": v.get("delta"),
               "se": v.get("se"), "verdict": v.get("verdict"), "label": v.get("label"), "reason": v.get("reason"),
               "look": v.get("look"), "design": v.get("design"), "p": v.get("p"), "flags": v.get("flags", []),
               "source": "queue", "row": e["name"], "value_text": key, "t": now()}
        AB.JsonlWriter(self.jsonl(e)).append([rec])

    # -- epochs / tunables ----------------------------------------------------------------------------------------
    def row_epoch(self, rs: RowState) -> str:
        if rs.epoch:
            return rs.epoch
        if not self.snapshot:
            return "live"
        ep = self.ledger.area_epoch(rs.area)
        return ep["id"] if ep else "A"

    def epoch_path(self, eid: str) -> str:
        if not self.snapshot or eid == "live":
            return self.src_root
        return os.path.join(self.home, "code", eid)

    def interp(self, e: Dict[str, Any]) -> str:
        return self.plan.interps.get(e.get("interpreter") or "py321", e.get("interpreter") or sys.executable)

    def missing_tunables(self, rs: RowState) -> List[str]:
        names = row_names(rs.e)
        if not names:
            return []
        eid = self.row_epoch(rs)
        root = self.epoch_path(eid)
        if self.snapshot and eid != "live" and not os.path.isdir(root):
            root = self.src_root      # the epoch is created at the first chunk: check the source it will copy
        key = (root, self.interp(rs.e))
        if key not in self._tcheck:
            wanted = sorted({n for r in self.plan.rows for n in row_names(r)})
            try:
                self._tcheck[key] = run_check(self.interp(rs.e), root, wanted)["missing"]
            except Exception as ex:  # noqa: BLE001
                self._tcheck[key] = []
                self.say(f"tunable check failed ({ex}); not blocking")
        return [n for n in names if n in self._tcheck[key]]

    def ensure_epoch(self, area: str) -> str:
        if not self.snapshot:
            return "live"
        ep = self.ledger.area_epoch(area)
        if ep is None:
            return self.bump(["all"], eid="A")["id"]
        return ep["id"]

    def bump(self, areas: Sequence[str], eid: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Create a code epoch (a snapshot) for the given areas ('all' = every area).  Rows already started stay
        pinned to their epoch; a bump while a started row of those areas is still open is refused unless
        ``force`` (default arms are shared within an epoch: bump at batch boundaries)."""
        areas = list(areas) or ["all"]
        if not force and self.ledger.epochs():
            self.refresh()
            open_rows = [rs.name for rs in self.rows.values()
                         if rs.status == "RUNNING" and ("all" in areas or rs.area in areas)]
            if open_rows:
                raise SystemExit(f"error: rows {open_rows} are open in the current epoch; finish them (a batch "
                                 f"boundary) or pass --force (they stay pinned to their epoch)")
        if eid is None:
            n = sum(1 for r in self.ledger.epochs() if r.get("id", "").startswith("B")) + 1
            eid = f"B{n}"
        if self.ledger.epoch(eid) is not None:
            raise SystemExit(f"error: epoch {eid} exists")
        dest = os.path.join(self.home, "code", eid)
        interps = sorted({self.interp(r) for r in self.plan.rows if r["kind"] in ("catanatron", "pool")} or
                         {sys.executable})
        interps = [i for i in interps if os.path.exists(i) or shutil.which(i)]
        self.say(f"snapshot {eid} ({', '.join(areas)}) -> {dest}")
        info = make_snapshot(self.src_root, dest, interps)
        rec = self.ledger.append({"kind": "epoch", "id": eid, "areas": "all" if "all" in areas else list(areas),
                                  "fingerprint": info["fingerprint"], "files_sha": info["files_sha"],
                                  "files": info["files"], "path": dest})
        return rec

    # -- looks ----------------------------------------------------------------------------------------------------
    def _find_run(self, idx, e: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
        runs = [r for r in idx.runs.values() if r.get("exp") == e["name"]]
        if e.get("tunable"):
            runs = [r for r in runs if str(r.get("value_text")) == key]
        if not runs:
            return None
        return max(runs, key=lambda r: r.get("t", 0))

    def evaluate(self, rs: RowState) -> None:
        e = rs.e
        if e["kind"] == "catanatron":
            self._evaluate_catanatron(rs)
        elif e["kind"] == "selfplay":
            self._evaluate_selfplay(rs)
        elif e["kind"] == "pool":
            self._evaluate_pool(rs)

    def _promise_for(self, rs: RowState) -> Optional[float]:
        return rs.intake.promise if rs.intake.promise is not None else rs.e.get("promise_pp")

    def _evaluate_catanatron(self, rs: RowState) -> None:
        e = rs.e
        idx = self.indexes.get(self.jsonl(e))
        looks = looks_for(e, rs.design, rs.n_max)
        for c in rs.cands:
            if not c.open:
                continue
            run = self._find_run(idx, e, c.key)
            c.run = run
            if run is None:
                continue
            qs = idx.queue_stops.get(run["run_key"])
            if qs is not None:        # a stop record the ledger lost (rebuild)
                self._verdict_from_stop(rs, c, qs)
                continue
            if rs.design == "legacy" and run["run_key"] in idx.stops:
                stop = idx.stops[run["run_key"]]
                st = SEQ.look_units(idx, run, rs.base, int(stop.get("pairs") or 0), AB)
                v = SEQ.decide(st, "legacy", 1, rs.n_max)
                v.reason = f"legacy --stop-at-se stop at {stop.get('pairs')} pairs"
                self._record_verdict(rs, c, v, run, idx, n=int(stop.get("pairs") or 0))
                continue
            j = c.looks_done
            while j < len(looks):
                n = looks[j]
                st = SEQ.look_units(idx, run, rs.base, n, AB)
                if not st.complete:
                    break
                v = self._decide(rs, st, j, looks, idx, run)
                look = {"kind": "look", "row": e["name"], "cand": c.key, "run_key": run["run_key"], "look": j + 1,
                        "n": n, "status": v.status, "z": v.z, "delta": v.delta, "se": v.se, "p": v.p,
                        "stats": st.summary()}
                c.last = self.ledger.append(look)
                j += 1
                c.looks_done = j
                for a in v.alerts:
                    self._alert_once(e["name"], a)
                if v.final:
                    self._record_verdict(rs, c, v, run, idx, n=n)
                    break

    def _decide(self, rs: RowState, st, j: int, looks: List[int], idx, run):
        e = rs.e
        design = rs.design
        if design == "politics" and len(looks) > 1 and j == 0:
            v = SEQ.Verdict(status="continue", look=1, pairs=st.ok, design=design)
            msg = SEQ.politics_alert(st)
            if msg:
                v.alerts.append(msg)
            return v
        if design == "legacy":
            stop = idx.stops.get(run["run_key"])
            v = SEQ.decide(st, "legacy", 1, rs.n_max)
            if stop:
                v.reason = "legacy --stop-at-se stop"
            return v
        if e.get("estimator") == "cv" and design in ("screen", "knockout"):
            fail = self._cv_check(rs, st, idx, run)
            if fail:
                return SEQ._final(SEQ.design_of(design), "FAILED", fail, j + 1, st, 0.0)
            pool = self.pool_for(e)
            if pool is not None:
                xs, ds = [], []
                for s in range(rs.base, rs.base + looks[j]):
                    c, d = idx.pair(run, s)
                    if c is not None and c.get("status") == "ok" and d.get("status") == "ok":
                        xs.append(float(int(bool(c.get("won"))) - int(bool(d.get("won")))))
                        ds.append(1.0 if d.get("won") else 0.0)
                if len(xs) >= 3:
                    return SEQ.decide_cv(st, xs, ds, float(pool["mean"]), int(pool["m"]), design, j + 1, rs.n_max,
                                         d_prior(e), aa=is_aa(e), needs=needs_mode(e),
                                         promise_pp=self._promise_for(rs) if e.get("polarity") == "new" else None)
        k = j + 1
        if design == "politics":
            k = 1
        return SEQ.decide(st, design, k, rs.n_max, aa=is_aa(e), needs=needs_mode(e),
                          promise_pp=self._promise_for(rs) if e.get("polarity") == "new" else None,
                          power_fn=(lambda pp, D, n: SEQ.power_at(pp, D, n)) if design == "screen" else None)

    def _cv_check(self, rs: RowState, st, idx, run) -> Optional[str]:
        pool = self.pool_for(rs.e)
        if pool is None:
            return None
        n = st.ok
        dbar = st.p_def
        if SEQ.cv_pool_mismatch(dbar, n, float(pool["mean"]), int(pool["m"])):
            return "pool-mismatch"
        return None

    def _stalled(self, rs: RowState, limit: int = 3) -> None:
        """Three slices in a row that played nothing (a failing command line, a harness error) end the row's open
        candidates as FAILED instead of looping forever."""
        chunks = self.ledger.chunks(rs.name)
        tail = 0
        for ch in reversed(chunks):
            if int(ch.get("games") or 0) > 0 or (rs.e["kind"] in ("selfplay", "command") and ch.get("exit") == 0):
                break
            tail += 1
        if tail >= limit and rs.open_cands:
            last = chunks[-1]
            why = f"stalled: {tail} slices without a game (last exit {last.get('exit')}; see logs/{rs.name}.log)"
            self._final_all(rs, "FAILED", why, label="FAILED(stalled)")

    def _record_verdict(self, rs: RowState, c: Cand, v, run: Optional[Dict[str, Any]], idx=None,
                        extra: Optional[Dict[str, Any]] = None, n: Optional[int] = None) -> None:
        e = rs.e
        rec = {"kind": "verdict", "row": e["name"], "cand": c.key, "run_key": run["run_key"] if run else None,
               "verdict": v.verdict, "label": v.label, "reason": v.reason, "look": v.look, "design": v.design,
               "pairs": v.pairs, "delta": v.delta, "se": v.se, "z": v.z, "p": v.p, "flags": list(v.flags),
               "stats": v.stats, "source": "queue", "epoch": rs.epoch}
        if idx is not None and run is not None:
            n = n if n is not None else v.pairs + int(v.stats.get("errors") or 0)
            pairs = [(cc, dd) for s in range(rs.base, rs.base + n)
                     for cc, dd in [idx.pair(run, s)] if cc is not None and cc.get("status") == "ok"
                     and dd.get("status") == "ok"]
            metrics = list(MECH.KEY_METRICS)
            mech = e.get("mechanism") or {}
            if mech.get("metric") and mech["metric"] not in metrics:
                metrics.append(mech["metric"])
            summ = {m: s for m, s in MECH.summarize_all(pairs, metrics).items() if s["n_pairs"]}
            if summ:
                rec["mech"] = summ
        if extra:
            rec.update(extra)
        c.verdict = self.ledger.append(rec)
        if run is not None:
            self._write_stop(e, c.key, run["run_key"], rec)
        self.say(f"[{e['name']}] {c.key}: {v.label} at look {v.look} ({v.pairs} pairs, delta "
                 f"{AB.fmt_pp(v.delta)} +- {AB.fmt_pp(v.se, sign=False)})")

    def _verdict_from_stop(self, rs: RowState, c: Cand, stop: Dict[str, Any]) -> None:
        rec = {"kind": "verdict", "row": rs.name, "cand": c.key, "run_key": stop.get("run_key"),
               "source": "rebuilt"}
        for k in ("verdict", "label", "reason", "look", "design", "pairs", "delta", "se", "z", "p", "flags"):
            rec[k] = stop.get(k)
        c.verdict = self.ledger.append(rec)

    # self-play ----------------------------------------------------------------------------------------------------
    def selfplay_dir(self, e: Dict[str, Any], key: str) -> str:
        return os.path.join(self.dir, "selfplay", e["name"], safe_key(key))

    def selfplay_chunks(self, e: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
        d = self.selfplay_dir(e, key)
        out = []
        k = 0
        while True:
            p = os.path.join(d, f"chunk_{k}.json")
            if not os.path.exists(p):
                break
            with open(p) as fh:
                out.append(json.load(fh))
            k += 1
        return out

    def _evaluate_selfplay(self, rs: RowState) -> None:
        e = rs.e
        looks = looks_for(e, rs.design, rs.n_max)
        for c in rs.cands:
            if not c.open:
                continue
            chunks = self.selfplay_chunks(e, c.key)
            if not chunks:
                continue
            keys = {(ch.get("base_spec"), ch.get("evaluator_mode"), ch.get("players"), ch.get("allow_counters"))
                    for ch in chunks}
            epochs = {x.get("epoch") for x in self.ledger.chunks(e["name"]) if x.get("cand") == c.key
                      and x.get("exit") == 0}
            if len(keys) > 1 or len(epochs) > 1:
                v = SEQ.Verdict(status="final", verdict="FAILED", reason="chunks-differ", label="FAILED(chunks-differ)",
                                design=rs.design, look=c.looks_done + 1)
                self._record_verdict(rs, c, v, None)
                continue
            records = [r for ch in chunks for r in ((ch.get("results") or [{}])[0].get("records") or [])]
            j = c.looks_done
            while j < len(looks):
                n = looks[j]
                if len(records) < n:
                    break
                st = SEQ.selfplay_units(records, n, one_candidate_only=bool(e.get("noharm")))
                st.n_target = st.ok
                design = rs.design
                if e.get("noharm"):
                    design = "confirm"
                v = SEQ.decide(st, design, j + 1 if design in ("screen", "knockout") else 1, rs.n_max,
                               promise_pp=self._promise_for(rs) if e.get("polarity") == "new" else None)
                if e.get("noharm") and v.final:
                    harm_p = SEQ.norm_cdf(v.z or 0.0)
                    v.verdict = "NO HARM" if harm_p >= 0.05 else "HARM"
                    v.label = v.verdict
                    v.p = harm_p
                c.last = self.ledger.append({"kind": "look", "row": e["name"], "cand": c.key, "look": j + 1, "n": n,
                                             "status": v.status, "z": v.z, "delta": v.delta, "se": v.se, "p": v.p,
                                             "stats": st.summary()})
                j += 1
                c.looks_done = j
                if v.final:
                    self._record_verdict(rs, c, v, None)
                    break

    def _evaluate_pool(self, rs: RowState) -> None:
        e = rs.e
        idx = self.indexes.get(self.jsonl(e))
        c = rs.cands[0]
        if not c.open:
            return
        run = self._find_run(idx, e, "cand")
        if run is None:
            return
        wins = []
        for s in range(rs.base, rs.base + rs.n_max):
            recs = [r for r in idx.get(run["def_key"], s) if r.get("status") == "ok"]
            if not recs:
                return
            wins.append(1.0 if recs[-1].get("won") else 0.0)
        mean = sum(wins) / len(wins)
        self.ledger.append({"kind": "pool", "row": e["name"], "def_key": run["def_key"], "epoch": rs.epoch,
                            "m": len(wins), "mean": mean})
        v = SEQ.Verdict(status="final", verdict="POOL", label="POOL", reason=f"M {len(wins)}, mean {mean:.4f}",
                        pairs=len(wins), design="estimate")
        self._record_verdict(rs, c, v, None)

    # -- cost / order / preemption ----------------------------------------------------------------------------------
    def cpu_per_game(self, rs: RowState) -> float:
        if rs.games > 0 and rs.cpu_s > 0:
            return rs.cpu_s / rs.games
        return cpu_prior(rs.e)

    def remaining_cost(self, rs: RowState) -> float:
        """Remaining CPU-hours: E_rem x (open candidate arms + (1 - rho) default arm) x CPU-s per game."""
        e = rs.e
        if e["kind"] == "command":
            g = self.ledger.gate(rs.name)
            return 0.0 if g and g.get("result") in ("PASS", "FAIL") else float(e.get("est_cpu_h") or 0.5)
        c_game = self.cpu_per_game(rs)
        rems = []
        for c in rs.open_cands:
            n_done = int((c.last or {}).get("n") or 0)
            z = (c.last or {}).get("z")
            sd = ((c.last or {}).get("stats") or {}).get("delta")
            rems.append(SEQ.expected_remaining(rs.design if e["kind"] != "pool" else "estimate", n_done, rs.n_max,
                                               z=z, look_done=c.looks_done if rs.design in ("screen", "knockout")
                                               else 0))
        if not rems:
            return 0.0
        if e["kind"] == "selfplay":
            return sum(rems) * c_game / 3600.0
        rho = self._rho(rs)
        return (sum(rems) + max(rems) * (1.0 - rho)) * c_game / 3600.0

    def _rho(self, rs: RowState) -> float:
        e = rs.e
        if e["kind"] != "catanatron":
            return 0.0
        idx = self.indexes.get(self.jsonl(e))
        run = next((c.run for c in rs.cands if c.run), None)
        if run is None:
            return 0.0
        nxt = [c for c in rs.open_cands]
        if not nxt:
            return 1.0
        looks = looks_for(e, rs.design, rs.n_max)
        j = min(c.looks_done for c in nxt)
        n = looks[min(j, len(looks) - 1)]
        have = sum(1 for s in range(rs.base, rs.base + n) if idx.get(run["def_key"], s))
        return have / max(1, n)

    def weight(self, rs: RowState) -> float:
        area_rows = [r for r in self.rows.values() if r.area == rs.area]
        p_min = min(float(r.e["priority"]) for r in area_rows) if area_rows else float(rs.e["priority"])
        w = 4.0 ** (-AREA_RANK.get(rs.area, len(AREAS) - 1)) * 2.0 ** (-(float(rs.e["priority"]) - p_min) / 10.0)
        return w * float(rs.e.get("weight") or 1.0)

    def order(self) -> List[RowState]:
        rows = [r for r in self.rows.values() if r.runnable]
        if self.order_mode == "index":
            def idx_key(r: RowState):
                p = 0.5
                if r.e.get("polarity") == "new" and r.intake.power is not None:
                    p = r.intake.power
                return (AREA_RANK.get(r.area, len(AREAS) - 1), -(self.weight(r) * p / max(r.cost_h, 1e-3)), r.e["_order"])
            return sorted(rows, key=idx_key)
        return self._ordered(rows)

    def should_preempt(self, q: RowState, r: RowState) -> bool:
        """``q`` (ranked above the incumbent ``r``) preempts at this slice boundary iff
        ``w_q C_r > (1 + h) w_r (C_q + S)``: a nearly finished incumbent finishes first."""
        return self.weight(q) * r.cost_h > (1.0 + H_PREEMPT) * self.weight(r) * (q.cost_h + S_START_H)

    def pick(self) -> Optional[RowState]:
        order = self.order()
        if not order:
            return None
        head = order[0]
        inc = self.rows.get(self.incumbent) if self.incumbent else None
        if inc is None or not inc.runnable or inc is head:
            return head
        if head.sort_key() < inc.sort_key() and not self.should_preempt(head, inc):
            return inc
        if head is not inc:
            self.say(f"preempt: {head.name} ({head.area}, {head.cost_h:.2f} CPU-h left) before {inc.name} "
                     f"({inc.cost_h:.2f} CPU-h left)")
        return head

    # -- guards ------------------------------------------------------------------------------------------------------
    def proof_guard(self) -> Optional[str]:
        me = os.getpid()
        if self.guard:
            for pid, cmd in self.proc_list():
                if pid == me or "run_queue.py" in cmd:
                    continue
                if self.guard in cmd:
                    return f"pid {pid}: {cmd[:160]}"
        for lf in self.lock_files:
            if lf and os.path.exists(lf):
                return f"lock file {lf}"
        return None

    def load_ok(self, rs: RowState, seconds: float = 10.0) -> Tuple[bool, float]:
        """Exclusive rows start only when externally busy cores <= cores - workers + 0.25 (sampled after the
        queue's own work drained)."""
        a = self.proc_stat()
        self.sleep(seconds)
        b = self.proc_stat()
        busy = busy_cores(a, b)
        workers = min(int(rs.e.get("workers") or 1), self.max_workers)
        return busy <= b[1] - workers + 0.25, busy

    @staticmethod
    def exclusive(e: Dict[str, Any]) -> bool:
        return bool(e.get("exclusive")) or e.get("opponent") in ("alphabeta", "sameturn", "ab")

    # -- slices -----------------------------------------------------------------------------------------------------
    def next_target(self, rs: RowState) -> int:
        looks = looks_for(rs.e, rs.design, rs.n_max)
        j = min((c.looks_done for c in rs.open_cands), default=len(looks))
        return looks[min(j, len(looks) - 1)]

    def chunk_command(self, rs: RowState, minutes: float) -> Tuple[List[str], str, Dict[str, str], Dict[str, Any]]:
        e = rs.e
        eid = rs.epoch or self.ensure_epoch(rs.area)
        root = self.epoch_path(eid)
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=root)
        env.pop("CATANBOT_NO_ACCEL", None)
        meta: Dict[str, Any] = {"epoch": eid}
        if e["kind"] in ("catanatron", "pool"):
            em = dict(e)
            em["seeds"] = {"count": self.next_target(rs), "base": rs.base} if e["kind"] == "catanatron" else \
                {"count": rs.n_max, "base": rs.base}
            em["workers"] = min(int(e.get("workers") or self.max_workers), self.max_workers)
            if e.get("tunable"):
                open_keys = {c.key for c in rs.open_cands}
                em["values"] = [v for k, v in cand_values(e) if k in open_keys]
                if e.get("flag_off") and not e.get("values"):
                    em["values"] = None
            if rs.design != "legacy":
                em.pop("stop_at_se", None)
                em.pop("stop_min_pairs", None)
            extra = [str(x) for x in e.get("extra_args") or []]
            if e.get("mech", True) and "--mech" not in extra:
                extra.append("--mech")
            crn = CAMP.crn_mode(e)
            if crn == "auto":
                st = self.crn_status()
                crn = "dice" if st and st.get("pass") else "off"
            if crn != "off" and "--crn" not in extra:
                extra += ["--crn", crn]
            if e["kind"] == "pool" and "--default-only" not in extra:
                extra.append("--default-only")
            if "--quiet" not in extra:
                extra.append("--quiet")
            em["extra_args"] = extra
            cmd = CAMP.command(em, self.dir, self.plan.interps, minutes)
            cmd[1] = os.path.join(root, "scripts", "ablate_catanatron.py")
            meta["n_target"] = em["seeds"]["count"]
            return cmd, root, env, meta
        if e["kind"] == "selfplay":
            c = next(iter(rs.open_cands))
            k = len(self.selfplay_chunks(e, c.key))
            chunk = int(e.get("chunk_games") or SELFPLAY_CHUNK)
            d = self.selfplay_dir(e, c.key)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f"chunk_{k}.json")
            cmd = [self.interp(e), os.path.join(root, "scripts", "ablate.py"), "--tunable", e["tunable"],
                   "--base-spec", e.get("base_spec") or DEFAULT_SEARCH_SPEC, "--games", str(chunk),
                   "--seed", str(rs.base + k), "--players", str(int(e.get("players") or 4)),
                   "--workers", str(min(int(e.get("workers") or self.max_workers), self.max_workers)),
                   "--json", out + ".tmp", "--quiet"]
            if c.value is False or (e.get("flag_off") and not e.get("values")):
                cmd += ["--flag-off"]
            else:
                cmd += ["--values", CAMP._values_text([c.value])]
            if e.get("counters"):
                cmd += ["--counters"]
            meta.update({"cand": c.key, "chunk": k, "out": out})
            return cmd, root, env, meta
        # command rows
        ctx = {"python": self.interp(e), "snapshot": root, "dir": self.dir, "row": e["name"],
               "minutes": f"{minutes:.2f}", "root": self.src_root}
        cmd = [str(x).format(**ctx) for x in e.get("cmd") or []]
        return cmd, root if e.get("cwd") != "repo" else self.src_root, env, meta

    def _run_process(self, cmd: List[str], cwd: str, env: Dict[str, str], log_path: str,
                     hard_check: Optional[Callable[[], bool]] = None) -> int:
        with open(log_path, "a") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        return proc.wait(timeout=self.poll_s)
                    except subprocess.TimeoutExpired:
                        if hard_check is not None and hard_check():
                            proc.send_signal(signal.SIGTERM)
                            try:
                                proc.wait(timeout=120)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            return -15
            except BaseException:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise

    def _played_games(self, e: Dict[str, Any]) -> int:
        if e["kind"] not in ("catanatron", "pool"):
            return 0
        idx = self.indexes.get(self.jsonl(e))
        return sum(1 for recs in idx.games.values() for r in recs if not r.get("reused_from")
                   and r.get("exp") == e["name"])

    def run_slice(self, rs: RowState, minutes: float) -> Dict[str, Any]:
        e = rs.e
        if self.ledger.pin(e["name"]) is None:
            eid = self.ensure_epoch(rs.area)
            self.ledger.append({"kind": "pin", "row": e["name"], "epoch": eid, "identity": self.identity(e),
                                "values": [k for k, _ in cand_values(e)], "n_max": rs.n_max, "base": rs.base})
            rs.epoch = eid
        cmd, cwd, env, meta = self.chunk_command(rs, minutes)
        games0 = self._played_games(e)
        cpu0 = self.cpu()
        t0 = self.clock()
        log_path = os.path.join(self.dir, "logs", f"{e['name']}.log")
        self.say(f"[{e['name']}] slice ({e['kind']}, epoch {meta.get('epoch')}, "
                 f"{'target ' + str(meta['n_target']) + ' seeds, ' if 'n_target' in meta else ''}"
                 f"{minutes:.0f} min)")
        hard = (lambda: self._hard_preempt(rs)) if self.hard else None
        code = self.runner(cmd, cwd, env, log_path, hard)
        wall = self.clock() - t0
        cpu_s = max(0.0, self.cpu() - cpu0)
        games = self._played_games(e) - games0
        if e["kind"] == "selfplay":
            tmp = meta["out"] + ".tmp"
            if code == 0 and os.path.exists(tmp):
                os.replace(tmp, meta["out"])
                with open(meta["out"]) as fh:
                    games = int(json.load(fh).get("games") or 0)
            elif os.path.exists(tmp):
                os.remove(tmp)            # a killed chunk is replayed whole
        rec = {"kind": "chunk", "row": e["name"], "cmd": cmd, "exit": code, "wall_s": round(wall, 3),
               "cpu_s": round(cpu_s, 3), "games": games, "epoch": meta.get("epoch")}
        if "cand" in meta:
            rec["cand"] = meta["cand"]
        self.ledger.append(rec)
        if e["kind"] == "command":
            self._command_result(rs, code)
        return rec

    def _hard_preempt(self, rs: RowState) -> bool:
        self.reload_plan()
        if self.plan.row(rs.name) is None:
            return True
        self.refresh()
        head = self.pick()
        return head is not None and head.name != rs.name

    def _command_result(self, rs: RowState, code: int) -> None:
        e = rs.e
        vf = e.get("verdict_from") or {}
        ctx = {"dir": self.dir, "row": e["name"], "snapshot": self.epoch_path(rs.epoch or "A"), "root": self.src_root}
        path = str(vf.get("json", "")).format(**ctx) if vf else None
        if code != 0:
            if "{minutes}" in " ".join(str(x) for x in e.get("cmd") or []):
                return            # a resumable command: the next slice continues it
            self.ledger.append({"kind": "gate", "row": e["name"], "result": "FAIL", "detail": f"exit {code}",
                                "path": path})
            self._final_all(rs, "FAILED", f"command exit {code}", label=f"FAILED(exit {code})")
            return
        if not path:
            self.ledger.append({"kind": "gate", "row": e["name"], "result": "PASS", "detail": "exit 0"})
            self._final_all(rs, "PASS", "exit 0")
            return
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            if "{minutes}" in " ".join(str(x) for x in e.get("cmd") or []):
                return
            self.ledger.append({"kind": "gate", "row": e["name"], "result": "FAIL", "detail": "no result", "path": path})
            self._final_all(rs, "FAILED", "no result file")
            return
        val = doc
        for part in str(vf.get("key", "")).split("."):
            if part:
                val = (val or {}).get(part) if isinstance(val, dict) else None
        if "pass_if" in vf:
            ok = val == vf["pass_if"]
        elif "min" in vf:
            ok = val is not None and float(val) >= float(vf["min"])
        elif "max" in vf:
            ok = val is not None and float(val) <= float(vf["max"])
        else:
            ok = bool(val)
        res = "PASS" if ok else "FAIL"
        self.ledger.append({"kind": "gate", "row": e["name"], "result": res, "detail": f"{vf.get('key')} = {val}",
                            "path": path})
        self._final_all(rs, res, f"{vf.get('key')} = {val}")

    # -- the loop ---------------------------------------------------------------------------------------------------
    def loop(self, max_hours: Optional[float] = None, once: bool = False, max_slices: Optional[int] = None) -> int:
        t_end = self.clock() + 3600.0 * max_hours if max_hours else None
        slices = 0
        rc = 0
        while True:
            self.refresh()
            self.write_report()
            if t_end is not None and self.clock() >= t_end - 60.0:
                self.say("queue: time budget used up; rerun to continue")
                break
            if max_slices is not None and slices >= max_slices:
                break
            rs = self.pick()
            if rs is None:
                self.say("queue: nothing eligible to run")
                break
            guard = self.proof_guard()
            if guard:
                self.say(f"proof guard: {guard}; no chunk starts")
                if once:
                    rc = 3
                    break
                self.sleep(GUARD_POLL_S)
                continue
            if self.exclusive(rs.e):
                ok, busy = self.load_ok(rs)
                if not ok:
                    self.say(f"load gate: {busy:.2f} cores busy; {rs.name} (exclusive) waits")
                    if self.exclusive_wait_since is None:
                        self.exclusive_wait_since = self.clock()
                    elif self.clock() - self.exclusive_wait_since > LOAD_GATE_ALERT_S:
                        self._alert_once(rs.name, "ALERT: exclusive row waited more than 2 h for the load gate")
                    alt = next((r for r in self.order() if not self.exclusive(r.e)), None)
                    if alt is None:
                        if once:
                            break
                        self.sleep(GUARD_POLL_S)
                        continue
                    rs = alt
                else:
                    self.exclusive_wait_since = None
            minutes = self.slice_minutes
            if t_end is not None:
                minutes = max(1.0, min(minutes, (t_end - self.clock()) / 60.0))
            self.run_slice(rs, minutes)
            self.incumbent = rs.name
            slices += 1
            if once:
                break
        self.refresh()
        self.write_report()
        return rc

    # -- reports ------------------------------------------------------------------------------------------------------
    def unit(self, rs: RowState) -> str:
        e = rs.e
        if e["kind"] == "selfplay":
            return "pp (per seat, 2v2)" if int(e.get("players") or 4) == 4 else "pp (3p mixed seats)"
        opp = e.get("opponent") or "?"
        suffix = ", vs value-rule responders" if (e.get("trades") or "off") in ("value", "fair") else ""
        return f"pp (1v3 win rate vs {opp}{suffix})"

    def next_action(self, rs: RowState, c: Cand, holm_p: Optional[float], complete: bool) -> str:
        v = c.verdict or {}
        verdict = v.get("verdict")
        area = rs.area
        kids = [r for r in self.rows.values() if r.e.get("parent") == rs.name]
        conf = [r for r in self.rows.values() if r.e.get("confirms") == rs.name]
        if rs.status == "DEFERRED":
            return "deferred to human testing (no games)"
        if rs.design == "politics" or (verdict == "SCREENED"):
            if verdict is None:
                return "politics: one fixed-N screen, " + ("running" if rs.status == "RUNNING" else
                                                           rs.reason or "queued")
            if holm_p is None:
                return "politics: label after the tier completes"
            lab = "SIGNIFICANT" if holm_p < 0.05 else "INCONCLUSIVE"
            tail = "" if complete else " (provisional: tier incomplete)"
            if lab == "INCONCLUSIVE":
                return f"INCONCLUSIVE{tail}: no more games, default unchanged, deferred to human testing"
            return f"SIGNIFICANT{tail}: may continue to factorial / joint tuning"
        kos = [r for r in self.rows.values() if r.e.get("_knockout_of") == rs.name]
        if rs.e.get("bundle") and verdict in ("SHELVE", "REJECT", "NOOP"):
            return f"on ice with its pieces ({', '.join(rs.e.get('_pieces') or [])}): bundle first, no 2x2"
        if verdict == "ADOPT" and kos:
            return "knockouts now eligible: " + ", ".join(k.name for k in kos) + "; then the league gate"
        if verdict == "ADOPT":
            gate = "league gate (--formats 4p2v2,3p1v2; power 0.38 at a 0.53 share, 0.86 at 0.55; ~4 h)"
            if area == "counting":
                return "fresh-seed confirmation vs Catanatron in counted mode, then " + gate
            steps = [gate]
            if conf:
                steps.insert(0, f"confirmation row {conf[0].name}")
            if area == "trades":
                steps.append("3p 1v2 no-harm check")
            return "; ".join(steps) + " (bundle the area's ADOPTs first)"
        if verdict == "REMOVE":
            return "removal candidate (the term hurts): confirm on fresh seeds before deleting code"
        if verdict in ("SHELVE", "REJECT", "NOOP", "KEEP"):
            elig = [k.name for k in kids if k.status in ("ELIGIBLE", "RUNNING")]
            if elig:
                return f"on ice; fallback now eligible: {', '.join(elig)}"
            if verdict == "NOOP":
                return "inert (never changes a decision here): drop, or re-route to a harness where it fires"
            if verdict == "KEEP":
                return "keep the term (default unchanged)"
            return "on ice (default unchanged)"
        if verdict == "FAILED":
            if "inert-harness" in str(v.get("label")):
                return "re-queue in a harness that exercises the mechanic"
            return "fix the cause and re-queue (FAILED never triggers fallbacks)"
        if verdict == "PASS":
            return "pipeline / pairing verified"
        if verdict == "ESTIMATE":
            return "measurement recorded" + (" (headroom for its area)" if rs.e.get("headroom") else "")
        if verdict in ("CONFIRMED", "NOT CONFIRMED"):
            return "confirmation done"
        if rs.status in ("WAITING", "BLOCKED"):
            return rs.reason
        return "running" if rs.status == "RUNNING" else ("queued" if rs.status == "ELIGIBLE" else rs.reason or "")

    def report_markdown(self, when: Optional[str] = None) -> str:
        when = when or time.strftime("%Y-%m-%d %H:%M:%S")
        L: List[str] = [f"# Test queue report", "",
                        f"Plan `{os.path.relpath(self.plan.path, ROOT) if self.plan.path.startswith(ROOT) else self.plan.path}`"
                        f" ({self.plan.sha}), results `{self.dir}`, written {when}.  One line per candidate.  Units: "
                        "win-rate percentage points of the candidate minus the default (paired); `p` = stage-wise "
                        "one-sided p (fixed two-sided p for politics / estimate rows), `Holm` = adjusted within the "
                        "row's tier (provisional `*` until the tier is complete).  Estimates of rows stopped early "
                        "are biased away from 0 (winner's curse): confirm on fresh seeds.", ""]
        tiers = sorted({(r.e.get("tier") or "") for r in self.rows.values()})
        holm = {t: self.holm(t) for t in tiers if t}
        L += ["| area | row | candidate | polarity | design | label | look | pairs | estimate (pp) +- se | unit "
              "| base rate / relative | p | Holm | dVP +- se | discordant / diverged | mechanism (cand - def) "
              "| CPU-h | next action |", "|" + "---|" * 18]
        for rs in self._ordered(self.rows.values()):
            if rs.status == "DISABLED":
                continue
            for c in rs.cands:
                v = c.verdict or {}
                last = c.last or {}
                src = v if v else last
                st = (src.get("stats") or {})
                tier = rs.e.get("tier") or ""
                adj, complete = holm.get(tier, ({}, True))
                hp = adj.get(f"{rs.name}|{c.key}")
                delta, se = src.get("delta"), src.get("se")
                est = f"{100 * delta:+.1f} +- {100 * se:.1f}" if delta is not None and se is not None else ""
                pdef = st.get("p_def")
                rel = f"{100 * pdef:.1f}% / {delta / pdef:+.0%}" if pdef and delta is not None else ""
                n = st.get("n") or 0
                disc = f"{(st.get('plus', 0) + st.get('minus', 0)) / n:.2f} / {st.get('diverged', 0) / n:.2f}" \
                    if n else ""
                vp = f"{st.get('vp_delta'):+.2f} +- {st.get('vp_se'):.2f}" \
                    if st.get("vp_delta") is not None and st.get("vp_se") is not None else ""
                mech = v.get("mech") or {}
                mtxt = "; ".join(f"{k} {m['diff']:+.2f}+-{(m.get('diff_se') or 0):.2f}" for k, m in mech.items()
                                 if m.get("diff") is not None)[:160]
                label = v.get("label") or (rs.status if rs.status not in ("ELIGIBLE", "RUNNING") else
                                           ("open" if c.looks_done else "queued"))
                if v.get("flags"):
                    label += " [" + ", ".join(v["flags"]) + "]"
                p = v.get("p") if v else last.get("p")
                L.append(f"| {rs.area} | {rs.name} | {c.key} | {rs.e.get('polarity') or ''} | {rs.design} | {label} "
                         f"| {v.get('look') or last.get('look') or ''} | {v.get('pairs') or last.get('n') or ''} | "
                         f"{est} | {self.unit(rs) if est else ''} | {rel} | {'' if p is None else f'{p:.3g}'} | "
                         f"{'' if hp is None else f'{hp:.3g}' + ('' if complete else '*')} | {vp} | {disc} | {mtxt} | "
                         f"{rs.cpu_s / 3600:.2f} | {self.next_action(rs, c, hp, complete)} |")
        # open rows in scheduler order
        L += ["", "## Open rows in scheduler order", "",
              "| # | area | row | status | remaining CPU-h | weight | why |", "|---|---|---|---|---|---|---|"]
        for i, rs in enumerate(self.order(), 1):
            L.append(f"| {i} | {rs.area} | {rs.name} | {rs.status} | {rs.cost_h:.2f} | {self.weight(rs):.4f} | "
                     f"{rs.reason} |")
        waiting = [rs for rs in self._ordered(self.rows.values()) if rs.status in ("WAITING", "BLOCKED")]
        if waiting:
            L += ["", "Waiting / blocked: " + "; ".join(f"{rs.name} ({rs.status}: {rs.reason})" for rs in waiting)]
        # roll-up per area
        L += ["", "## Per area", "", "| area | rows | final | open | ADOPT | on ice (SHELVE/REJECT) | NOOP | FAILED "
              "| deferred | CPU-h spent |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in AREAS:
            rr = [r for r in self.rows.values() if r.area == a and r.status != "DISABLED"]
            if not rr:
                continue
            vs = [c.verdict.get("verdict") for r in rr for c in r.cands if c.verdict]
            L.append(f"| {a} | {len(rr)} | {sum(1 for r in rr if r.final)} | "
                     f"{sum(1 for r in rr if r.runnable)} | {vs.count('ADOPT')} | "
                     f"{vs.count('SHELVE') + vs.count('REJECT')} | {vs.count('NOOP')} | {vs.count('FAILED')} | "
                     f"{sum(1 for r in rr if r.status == 'DEFERRED')} | {sum(r.cpu_s for r in rr) / 3600:.2f} |")
        # bundle proposals
        L += ["", "## Bundle proposals (a human commits them; then the league gate)", ""]
        any_b = False
        for a in AREAS:
            adopts = [(r, c) for r in self.rows.values() if r.area == a for c in r.cands
                      if c.verdict and c.verdict.get("verdict") == "ADOPT" and r.e.get("tier") not in NO_HOLM_TIERS]
            if not adopts:
                continue
            any_b = True
            tune = ";".join(f"{r.e['tunable']}:{c.key}" for r, c in adopts if r.e.get("tunable"))
            specs = [r.e.get("cand_spec") for r, c in adopts if not r.e.get("tunable")]
            spec = DEFAULT_SEARCH_SPEC + (f",tune={tune}" if tune else "")
            L.append(f"- **{a}**: {', '.join(f'{r.name}={c.key}' for r, c in adopts)}"
                     + (f"; spec changes {specs}" if specs else ""))
            L.append(f"  `python3 scripts/league.py gate --candidate-commit <sha> --candidate-spec \"{spec}\" "
                     f"--formats 4p2v2,3p1v2`")
        if not any_b:
            L.append("(no ADOPT yet)")
        # deferred to human testing
        L += ["", "## Deferred to human testing (paste into docs/ABLATIONS.md, Politics rule)", "",
              "| term | screen | result | status |", "|---|---|---|---|"]
        for rs in self._ordered(self.rows.values()):
            if rs.status == "DEFERRED":
                L.append(f"| {', '.join(row_names(rs.e)) or rs.name} | none (no bot harness reacts) | n/a | deferred |")
                continue
            for c in rs.cands:
                v = c.verdict or {}
                if v.get("verdict") != "SCREENED":
                    continue
                adj, complete = holm.get(rs.e.get("tier") or "", ({}, True))
                hp = adj.get(f"{rs.name}|{c.key}")
                if hp is not None and hp >= 0.05:
                    screen = f"{rs.name} ({v.get('pairs')} {'games' if rs.e['kind'] == 'selfplay' else 'pairs'})"
                    L.append(f"| {', '.join(row_names(rs.e))}={c.key} | {screen} | {100 * (v.get('delta') or 0):+.1f} "
                             f"+- {100 * (v.get('se') or 0):.1f} pp, Holm p {hp:.2g} | INCONCLUSIVE"
                             f"{'' if complete else ' (provisional)'} |")
        crn = self.crn_status()
        if crn and not crn.get("pending"):
            L += ["", f"CRN pilot: discordant share {crn['d_off']:.3f} (off) -> {crn['d_dice']:.3f} (dice), ratio "
                      f"{crn['ratio']:.2f}; A/A {'identical' if crn['aa_ok'] else 'NOT identical'} -> "
                      f"{'PASS: rows with crn auto use --crn dice' if crn['pass'] else 'FAIL: CRN stays off'}"]
        alerts = [r for r in self.ledger.of("alert")][-10:]
        if alerts:
            L += ["", "## Alerts", ""] + [f"- {r.get('row')}: {r.get('msg')}" for r in alerts]
        return "\n".join(L) + "\n"

    def write_report(self, when: Optional[str] = None) -> str:
        path = os.path.join(self.dir, "QUEUE.md")
        text = self.report_markdown(when)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
        return path

    def status_text(self) -> str:
        L = [f"queue {self.dir} (plan {self.plan.path}, {self.plan.sha}); epochs: "
             + (", ".join(f"{r['id']}({r.get('areas')})" for r in self.ledger.epochs()) or "none yet")]
        L.append(f"  {'#':>3} {'area':9} {'row':34} {'status':14} {'design':9} {'cands':>5} {'open':>4} "
                 f"{'CPU-h':>6} {'left':>6}  detail")
        for i, rs in enumerate(self._ordered(self.rows.values()), 1):
            if rs.status == "DISABLED":
                continue
            detail = rs.reason or ", ".join(f"{c.key}:{(c.verdict or {}).get('label') or ('look ' + str(c.looks_done))}"
                                            for c in rs.cands)
            L.append(f"  {i:>3} {rs.area:9} {rs.name[:34]:34} {rs.status:14} {rs.design:9} {len(rs.cands):>5} "
                     f"{len(rs.open_cands):>4} {rs.cpu_s / 3600:>6.2f} {rs.cost_h:>6.2f}  {detail[:110]}")
        dis = [rs.name for rs in self.rows.values() if rs.status == "DISABLED"]
        if dis:
            L.append(f"  disabled ({len(dis)}): {', '.join(dis)}")
        return "\n".join(L)

    # -- explain / simulate ---------------------------------------------------------------------------------------------
    def explain(self, simulate: bool = True) -> str:
        """The full order with CPU hours at the cap and at null, and the null verdict timeline per area."""
        shown = os.path.relpath(self.plan.path, ROOT) if os.path.abspath(self.plan.path).startswith(ROOT) \
            else self.plan.path
        L = [f"Queue plan {shown} ({self.plan.sha}): {len(self.plan.rows)} rows; order = area (harness, "
             "trades, diversification, ports, robber, counting, politics, other), headroom first, priority, plan order.",
             "CPU-s per game and arm from the design's priors (value 1.77, value+pyeval 5.5, trades 3.0, vf 1.15, "
             "vf+pyeval 3.3, alphabeta 23, counted 7.2 / 3.9, depth 2 x2.4, self-play 10.8 per game); default arms "
             "shared within a default class and epoch.  'null' = expected pairs when the candidate has no effect "
             "(simulated operating characteristics at the row's D prior)."]
        L.append("")
        L.append(f"{'#':>3} {'area':9} {'row':34} {'kind':10} {'design':9} {'D':>5} {'N':>5} {'cand':>4} {'s/g':>5} "
                 f"{'cap CPU-h':>9} {'null CPU-h':>10} {'cum null':>8}  status")
        paid: Dict[str, int] = {}
        cum = 0.0
        per_area: Dict[str, float] = {}
        cond: List[Tuple[RowState, float]] = []
        i = 0
        for rs in self._ordered(self.rows.values()):
            e = rs.e
            if rs.status in ("DISABLED",):
                continue
            i += 1
            cap_h, null_h = self._row_cost(rs, paid)
            gated = rs.status in ("WAITING", "NOT TRIGGERED") and (e.get("parent") or e.get("confirms") or
                                                                   e.get("after"))
            status = rs.status + (f": {rs.reason}" if rs.reason else "")
            D = d_prior(e) if e["kind"] == "catanatron" else (1.0 if e["kind"] == "selfplay" else float("nan"))
            s_g = cpu_prior(e) if e["kind"] in ("catanatron", "selfplay", "pool") else float("nan")
            counts = rs.status in ("ELIGIBLE", "RUNNING") or (rs.status == "WAITING" and not gated)
            if counts:
                cum += null_h
                per_area[rs.area] = per_area.get(rs.area, 0.0) + null_h
            elif gated:
                cond.append((rs, cap_h))
            n = rs.n_max if e["kind"] in ("catanatron", "selfplay", "pool") else 0
            L.append(f"{i:>3} {rs.area:9} {rs.name[:34]:34} {e['kind']:10} {rs.design:9} "
                     f"{'' if D != D else f'{D:.2f}':>5} {n:>5} {len(rs.cands):>4} "
                     f"{'' if s_g != s_g else f'{s_g:.2f}':>5} {cap_h:>9.2f} {null_h:>10.2f} "
                     f"{f'{cum:.2f}' if counts else '-':>8}  {status[:90]}")
        L.append("")
        L.append("Null verdict timeline per area (CPU-h if every enabled, unconditional row is null; wall-h at 3 "
                 "cores = CPU-h / 3):")
        t = 0.0
        for a in AREAS:
            if a in per_area:
                t += per_area[a]
                L.append(f"  {a:9} {per_area[a]:7.2f} CPU-h  -> cumulative {t:7.2f} CPU-h ({t / 3:.2f} h wall)")
        L.append(f"  total    {t:7.2f} CPU-h ({t / 3:.2f} h wall at 3 cores)")
        if cond:
            L.append(f"Conditional rows (run only if their gate triggers), at their caps: "
                     + ", ".join(f"{rs.name} {h:.2f}" for rs, h in cond)
                     + f" = {sum(h for _, h in cond):.2f} CPU-h")
        if simulate:
            L.append("")
            L.append("Null operating characteristics used (simulated, 4,000 rows per cell): "
                     + "; ".join(f"{d} D {D:.2f} N {n}: ADOPT {r['adopt']:.3f}, mean pairs {r['mean_pairs']:.0f}"
                                 for (d, D, n), r in sorted(self._oc_cache.items())))
        return "\n".join(L)

    _oc_cache: Dict[Tuple[str, float, int], Dict[str, float]] = {}

    def _null_pairs(self, design: str, D: float, n: int, e: Dict[str, Any]) -> float:
        if design in ("screen", "knockout"):
            key = (design, round(D, 3), n)
            if key not in self._oc_cache:
                unit = 0.5 if e["kind"] == "selfplay" else 1.0
                self._oc_cache[key] = SEQ.simulate(design, D, 0.0, n, rows=4000, seed=7, unit=unit)
            return self._oc_cache[key]["mean_pairs"]
        if design == "estimate" and is_aa(e):
            return float(min(SEQ.AA_PAIRS, n))
        return float(n)

    def _row_cost(self, rs: RowState, paid: Dict[str, int]) -> Tuple[float, float]:
        e = rs.e
        if e["kind"] in ("human",):
            return 0.0, 0.0
        if e["kind"] == "command":
            h = float(e.get("est_cpu_h") or 0.5)
            return h, h
        s_g = cpu_prior(e)
        ncand = len(rs.cands)
        n = rs.n_max
        if e["kind"] == "selfplay":
            null_n = self._null_pairs(rs.design, 1.0, n, e)
            return ncand * n * s_g / 3600.0, ncand * null_n * s_g / 3600.0
        if e["kind"] == "pool":
            return n * s_g / 3600.0, n * s_g / 3600.0
        null_n = self._null_pairs(rs.design, d_prior(e), n, e)
        dcls = default_class(e) + f"|{self.row_epoch(rs)}|{rs.base}"
        have = paid.get(dcls, 0)
        s_def = CPU_PRIOR.get((e.get("opponent"), needs_pyeval(e), "counted" if info_counted(e) else "full"), s_g)
        if (e.get("trades") or "off") != "off":
            s_def = max(s_def, CPU_TRADES_VALUE)
        def_cap = max(0, n - have) * s_def
        def_null = max(0.0, null_n - have) * s_def
        paid[dcls] = max(have, int(round(null_n)))
        cap = (ncand * n * s_g + def_cap) / 3600.0
        null = (ncand * null_n * s_g + def_null) / 3600.0
        return cap, null


def _cond(expr: str, v: Dict[str, Any]) -> bool:
    """``"delta >= 0.015 and z > 0"`` over a verdict record's numeric fields (no eval)."""
    for part in str(expr).split(" and "):
        m = re.match(r"\s*([A-Za-z_]+)\s*(>=|<=|>|<|==)\s*(-?[0-9.eE+-]+)\s*$", part)
        if not m:
            return False
        x = v.get(m.group(1))
        if x is None:
            return False
        y = float(m.group(3))
        x = float(x)
        if not {"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y, "==": x == y}[m.group(2)]:
            return False
    return True


def rebuild_ledger(q: Queue) -> int:
    """Append a verdict record for every queue stop record that the ledger lacks (the ledger is derived data)."""
    n = 0
    for e in q.plan.rows:
        if e["kind"] != "catanatron":
            continue
        idx = q.indexes.get(q.jsonl(e))
        for rk, stop in idx.queue_stops.items():
            key = stop.get("value_text") or "cand"
            if q.ledger.verdict(e["name"], key) is None:
                rec = {"kind": "verdict", "row": e["name"], "cand": key, "run_key": rk, "source": "rebuilt"}
                for k in ("verdict", "label", "reason", "look", "design", "pairs", "delta", "se", "z", "p", "flags"):
                    rec[k] = stop.get(k)
                q.ledger.append(rec)
                n += 1
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--dir", required=True, help="results: row JSONLs, ledger.jsonl, QUEUE.md, logs/")
    ap.add_argument("--home", help="where code snapshots live (default: <dir> when it is outside the repository, "
                                   "else ../queue_runs/<name>)")
    ap.add_argument("--max-hours", type=float, default=None, help="wall-clock budget of this invocation")
    ap.add_argument("--once", action="store_true", help="run one slice and exit")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--explain", action="store_true", help="print the order, CPU hours and null timeline; play nothing")
    ap.add_argument("--simulate", action="store_true", help="with --explain: show the operating characteristics used")
    ap.add_argument("--report-only", action="store_true", help="evaluate looks, rewrite QUEUE.md, play nothing")
    ap.add_argument("--bump-code", action="store_true", help="create a new code epoch (snapshot)")
    ap.add_argument("--areas", default="all", help="--bump-code: comma-separated areas (default all)")
    ap.add_argument("--epoch-id", help="--bump-code: the new epoch's id (default B<n>)")
    ap.add_argument("--force", action="store_true", help="--bump-code while rows of those areas are open")
    ap.add_argument("--no-snapshot", action="store_true", help="run chunks from the live repository")
    ap.add_argument("--no-intake", action="store_true", help="skip the power and headroom checks (lint still applies)")
    ap.add_argument("--hard", action="store_true", help="preempt mid-slice with SIGTERM (loses in-flight games)")
    ap.add_argument("--slice-minutes", type=float, default=SLICE_MINUTES)
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--guard", default="proof_snapshot", help="no chunk starts while a process matches this")
    ap.add_argument("--lock-file", action="append", default=[], help="no chunk starts while this file exists")
    ap.add_argument("--order", choices=("priority", "index"), default="priority")
    ap.add_argument("--rebuild-ledger", action="store_true", help="re-derive verdicts from the rows' stop records")
    args = ap.parse_args(argv)
    q = Queue(args.plan, args.dir, home=args.home, snapshot=not args.no_snapshot, slice_minutes=args.slice_minutes,
              max_workers=args.max_workers, guard=args.guard, lock_files=args.lock_file, no_intake=args.no_intake,
              hard=args.hard, order=args.order)
    if args.rebuild_ledger:
        print(f"{rebuild_ledger(q)} verdict(s) restored")
        return 0
    if args.bump_code:
        areas = [a.strip() for a in args.areas.split(",") if a.strip()]
        bad = [a for a in areas if a != "all" and a not in AREAS]
        if bad:
            raise SystemExit(f"error: unknown area(s) {bad}")
        rec = q.bump(areas, eid=args.epoch_id, force=args.force)
        print(f"epoch {rec['id']} for {rec['areas']}: {rec['path']} (fingerprint {rec['fingerprint']})")
        return 0
    q.refresh()
    for rs in q.rows.values():
        for w in rs.intake.warnings:
            if args.explain:
                print(f"note [{rs.name}]: {w}")
    if args.explain:
        print(q.explain(simulate=args.simulate))
        return 0
    if args.status:
        print(q.status_text())
        return 0
    if args.report_only:
        print(f"report -> {q.write_report()}")
        return 0
    return q.loop(max_hours=args.max_hours, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
