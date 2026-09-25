#!/usr/bin/env python3
"""Resumable campaign of paired ablations against Catanatron (runs scripts/ablate_catanatron.py per experiment).

    python3 scripts/campaign.py --plan plan.json --dir runs/campaign1                   # run / resume everything
    python3 scripts/campaign.py --plan plan.json --dir runs/campaign1 --max-minutes 15  # a bounded chunk
    python3 scripts/campaign.py --plan plan.json --dir runs/campaign1 --status          # progress and ETA
    python3 scripts/campaign.py --plan plan.json --dir runs/campaign1 --summary-only    # rewrite the summary
    python3 scripts/campaign.py --plan plan.json --dir runs/campaign1 --dry-run         # print the commands

Plan (JSON): either a list of experiments or ``{"interpreters": {...}, "defaults": {...}, "experiments": [...]}``.
Each experiment::

    {"name": "turns_half@value",          # unique; results go to <dir>/<name>.jsonl
     "interpreter": "py330",              # py321 (system python3, catanatron 3.2.1) | py330 (the 3.3 venv)
     "opponent": "value",                 # bench_catanatron preset (value/alphabeta/sameturn on 3.3; vf/ab/... on 3.2.1)
     "tunable": "danger.TURNS_HALF", "values": [2, 4.5],     # or "flag_off": true; optional "base_spec"
     # ... or instead of the tunable: "cand_spec": "...", "def_spec": "...", optional "cand_set": {"NAME": v}
     "seeds": {"count": 2000, "base": 0}, "workers": 2, "priority": 1,
     # optional: "set", "adapter_opts", "cand_adapter_opts", "trades", "opponent_params", "stop_at_se",
     #           "stop_min_pairs", "game_timeout", "vps_to_win", "discard_limit", "enabled", "notes", "extra_args"}

Experiments run in ascending ``priority`` (ties: plan order), one after another, each through the
right interpreter with ``PYTHONHASHSEED=0``; every experiment appends to its own JSONL and reuses the
default-arm games other experiments of the campaign already played (same arm key and code, ``--reuse``).
After every experiment the markdown summary (``--summary``, default ``<dir>/SUMMARY.md``) is rewritten:
one row per candidate.  The campaign is idempotent (a finished experiment is skipped; an unfinished one
resumes) and survives a crashing game (recorded by ablate_catanatron.py as an error record) and a
failing experiment (logged in ``<dir>/campaign_events.log``, the campaign moves on).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATE = os.path.join(ROOT, "scripts", "ablate_catanatron.py")
DEFAULT_INTERPRETERS = {
    "py321": "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else (shutil.which("python3") or "python3"),
    "py330": "/home/user/venv_cat33/bin/python",
}
EXPERIMENT_KEYS = {"name", "interpreter", "opponent", "tunable", "values", "flag_off", "base_spec", "cand_spec",
                   "def_spec", "cand_set", "set", "adapter_opts", "cand_adapter_opts", "trades", "opponent_params",
                   "seeds", "workers", "priority", "stop_at_se", "stop_min_pairs", "game_timeout", "vps_to_win",
                   "discard_limit", "enabled", "notes", "extra_args"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.@+=-]+$")
IDLE_GAP_S = 600.0     # gaps between records longer than this are not counted as playing time


def load_ablate():
    spec = importlib.util.spec_from_file_location("_ablate_catanatron", ABLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = load_ablate()


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def _kv_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, dict):
        return [f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in x.items()]
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _values_text(values: Any) -> Optional[str]:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    if all(isinstance(v, (list, tuple)) for v in values):
        return ";".join("/".join(f"{x:g}" for x in v) for v in values)
    return ",".join(("on" if v else "off") if isinstance(v, bool) else f"{v}" for v in values)


def load_plan(path: str, max_workers: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    with open(path) as fh:
        raw = json.load(fh)
    interps = dict(DEFAULT_INTERPRETERS)
    defaults: Dict[str, Any] = {}
    if isinstance(raw, dict):
        interps.update(raw.get("interpreters") or {})
        defaults = raw.get("defaults") or {}
        exps = raw.get("experiments") or []
    else:
        exps = raw
    out = []
    names = set()
    for i, e in enumerate(exps):
        e = dict(defaults, **e)
        bad = set(e) - EXPERIMENT_KEYS
        if bad:
            raise SystemExit(f"plan error: experiment {e.get('name', i)}: unknown field(s) {sorted(bad)}")
        for k in ("name", "interpreter", "opponent"):
            if not e.get(k):
                raise SystemExit(f"plan error: experiment {i}: '{k}' is required")
        if not NAME_RE.match(e["name"]):
            raise SystemExit(f"plan error: experiment name {e['name']!r} (letters, digits, _ . @ + = - only)")
        if e["name"] in names:
            raise SystemExit(f"plan error: duplicate experiment name {e['name']!r}")
        names.add(e["name"])
        if e["interpreter"] not in interps:
            raise SystemExit(f"plan error: {e['name']}: interpreter {e['interpreter']!r} not in {sorted(interps)}")
        if bool(e.get("tunable")) == bool(e.get("cand_spec") or e.get("def_spec")):
            raise SystemExit(f"plan error: {e['name']}: give either 'tunable' or 'cand_spec' + 'def_spec'")
        if not e.get("tunable") and not (e.get("cand_spec") and e.get("def_spec")):
            raise SystemExit(f"plan error: {e['name']}: 'cand_spec' and 'def_spec' are both required")
        seeds = e.get("seeds", 100)
        if isinstance(seeds, int):
            seeds = {"count": seeds, "base": 0}
        e["seeds"] = {"count": int(seeds.get("count", 100)), "base": int(seeds.get("base", 0))}
        e["workers"] = max(1, min(int(e.get("workers", max_workers)), max_workers))
        e["priority"] = e.get("priority", 100)
        e["_order"] = i
        out.append(e)
    out.sort(key=lambda x: (x["priority"], x["_order"]))
    return out, interps


def jsonl_path(d: str, e: Dict[str, Any]) -> str:
    return os.path.join(d, f"{e['name']}.jsonl")


def command(e: Dict[str, Any], d: str, interps: Dict[str, str], max_minutes: Optional[float]) -> List[str]:
    cmd = [interps[e["interpreter"]], ABLATE, "--out", jsonl_path(d, e), "--exp-name", e["name"],
           "--opponent", e["opponent"], "--seeds", str(e["seeds"]["count"]), "--seed-base", str(e["seeds"]["base"]),
           "--workers", str(e["workers"]), "--reuse", os.path.join(d, "*.jsonl")]
    if e.get("tunable"):
        cmd += ["--tunable", e["tunable"]]
        vt = _values_text(e.get("values"))
        if vt:
            cmd += ["--values", vt]
        if e.get("flag_off"):
            cmd += ["--flag-off"]
        if e.get("base_spec"):
            cmd += ["--base-spec", e["base_spec"]]
    else:
        cmd += ["--cand-spec", e["cand_spec"], "--def-spec", e["def_spec"]]
    for flag, key in (("--cand-set", "cand_set"), ("--set", "set"), ("--adapter-opt", "adapter_opts"),
                      ("--cand-adapter-opt", "cand_adapter_opts")):
        for item in _kv_list(e.get(key)):
            cmd += [flag, item]
    for flag, key in (("--trades", "trades"), ("--opponent-params", "opponent_params"), ("--stop-at-se", "stop_at_se"),
                      ("--stop-min-pairs", "stop_min_pairs"), ("--game-timeout", "game_timeout"),
                      ("--vps-to-win", "vps_to_win"), ("--discard-limit", "discard_limit")):
        if e.get(key) is not None:
            v = e[key]
            if isinstance(v, dict):
                v = ",".join(f"{k}={x}" for k, x in v.items())
            cmd += [flag, str(v)]
    if max_minutes is not None:
        cmd += ["--max-minutes", f"{max_minutes:.2f}"]
    cmd += [str(x) for x in e.get("extra_args") or []]
    return cmd


def expected_candidates(e: Dict[str, Any]) -> Optional[int]:
    """How many candidate runs the experiment declares (None = unknown until it has run)."""
    if not e.get("tunable"):
        return 1
    if e.get("flag_off"):
        return 1
    v = e.get("values")
    if isinstance(v, list):
        return len({json.dumps(x) for x in v})
    if isinstance(v, str) and v.strip():
        return len([x for x in re.split(r"[;,]" if ";" not in v else ";", v) if x.strip()])
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from catanbot import tuning
        return len(tuning.find(e["tunable"]).candidates)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def active_rate(times: List[float]) -> Optional[float]:
    """Games per hour from record end times, idle gaps (> IDLE_GAP_S) excluded."""
    if len(times) < 2:
        return None
    ts = sorted(times)
    active = sum(b - a for a, b in zip(ts, ts[1:]) if b - a <= IDLE_GAP_S)
    n = sum(1 for a, b in zip(ts, ts[1:]) if b - a <= IDLE_GAP_S)
    return 3600.0 * n / active if active > 0 and n else None


def progress(e: Dict[str, Any], d: str) -> Dict[str, Any]:
    path = jsonl_path(d, e)
    index, bad = AB.load_index(path)
    lo, n = e["seeds"]["base"], e["seeds"]["count"]
    rng = range(lo, lo + n)
    runs = [r for r in index.runs.values() if r.get("exp") == e["name"]]
    ncand = expected_candidates(e)
    if ncand is None:
        ncand = max(1, len(runs))
    per_run = []
    for r in runs:
        done = 0
        for s in rng:
            c, _ = index.pair(r, s)
            if c is not None:
                done += 1
        stopped = r["run_key"] in index.stops
        per_run.append({"run": r, "done": done, "stopped": stopped})
    keys = {k for r in runs for k in (r["cand_key"], r["def_key"])}
    games = [g for (k, s), recs in index.games.items() if k in keys and s in rng for g in recs]
    played = [g for g in games if not g.get("reused_from")]
    complete_runs = sum(1 for x in per_run if x["stopped"] or x["done"] >= n)
    complete = len(runs) >= ncand and complete_runs == len(runs) and len(runs) > 0
    planned_games = n * (ncand + 1)
    stopped_left = sum(n - x["done"] for x in per_run if x["stopped"])
    remaining = 0 if complete else max(0, planned_games - len(games) - stopped_left)
    rate = active_rate([g["t"] for g in played if g.get("t")])
    return {"name": e["name"], "path": path, "exists": os.path.exists(path), "bad_lines": bad, "runs": len(runs),
            "index": index, "last_exit": last_exit(d, e["name"]),
            "ncand": ncand, "planned_games": planned_games, "games": len(games), "played": len(played),
            "reused": len(games) - len(played), "errors": sum(1 for g in games if g.get("status") != "ok"),
            "remaining": remaining, "rate": rate, "complete": complete, "per_run": per_run,
            "mean_s": (sum(g.get("duration", 0.0) for g in played if g.get("status") == "ok")
                       / max(1, sum(1 for g in played if g.get("status") == "ok"))) if played else None}


def last_exit(d: str, name: str) -> Optional[int]:
    """Exit code of the experiment's latest finished attempt (``campaign_events.log``), None if never run."""
    code = None
    path = os.path.join(d, "campaign_events.log")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("name") == name and ev.get("event") == "finish":
                code = ev.get("code")
    return code


def cand_tag(run: Dict[str, Any]) -> str:
    return str(run.get("value_text")) if run.get("tunable") else "cand"


def fmt_h(hours: Optional[float]) -> str:
    if hours is None or math.isnan(hours):
        return "?"
    if hours < 1:
        return f"{60 * hours:.0f} min"
    return f"{hours:.1f} h"


def print_status(exps: List[Dict[str, Any]], d: str, plan: str) -> None:
    rows = [(e, progress(e, d)) for e in exps]
    rates: Dict[Tuple[str, str], float] = {}
    for e, p in rows:
        if p["rate"]:
            rates.setdefault((e["interpreter"], e["opponent"]), p["rate"])
    total_eta = 0.0
    unknown = False
    print(f"campaign {d} (plan {plan}): {len(exps)} experiment(s), "
          f"{sum(1 for _, p in rows if p['complete'])} complete")
    print(f"  {'prio':>4} {'name':34} {'interp':6} {'opponent':10} {'cands':>5} {'games done/planned':>19} "
          f"{'err':>4} {'reused':>6} {'rate g/h':>9} {'ETA':>8}  pairs per candidate")
    for e, p in rows:
        rate = p["rate"] or rates.get((e["interpreter"], e["opponent"]))
        eta = None
        if p["complete"]:
            eta = 0.0
        elif rate:
            eta = p["remaining"] / rate
        if eta is None:
            unknown = True
        else:
            total_eta += eta
        pr = ", ".join(f"{cand_tag(x['run'])}:{x['done']}" + ("(stopped)" if x["stopped"] else "")
                       for x in p["per_run"]) or "-"
        state = "done" if p["complete"] else fmt_h(eta)
        if not p["complete"] and p["last_exit"] not in (None, 0):
            state = f"FAILED({p['last_exit']})"
        rate_txt = f"{p['rate']:.0f}" if p["rate"] else (f"~{rate:.0f}" if rate else "?")
        print(f"  {e['priority']:>4} {e['name'][:34]:34} {e['interpreter']:6} {e['opponent'][:10]:10} {p['ncand']:>5} "
              f"{p['games']:>9}/{p['planned_games']:<9} {p['errors']:>4} {p['reused']:>6} {rate_txt:>9} {state:>8}  {pr}")
    print(f"  remaining (serial, at measured rates): {fmt_h(total_eta)}" + (" + experiments without a rate yet"
                                                                           if unknown else ""))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def summary_markdown(exps: List[Dict[str, Any]], d: str, plan: str) -> str:
    lines = [f"# Campaign summary: paired ablations against Catanatron",
             "",
             f"Plan `{plan}`, results `{d}`, written {time.strftime('%Y-%m-%d %H:%M:%S')}.  One row per candidate: "
             "each seed is played once per arm (candidate / default catanbot seat, same board, dice seed and seat "
             "`s % 4`, 3 copies of the opponent).  `delta` = candidate minus default win rate (paired over seeds, "
             "+- one standard error; 95% CI = +-1.96 s.e.), `dVP` = paired difference of catanbot's final VP.  "
             "`ms/dec` = mean catanbot decision time (candidate / default), `opp ms` = the opponents' mean decision "
             "time.  `ident` = pairs whose two games were identical (the change never altered a decision).  A verdict "
             "needs >= 30 pairs; `stopped` = sequential stop (--stop-at-se), whose estimate is biased away from 0.",
             "",
             "| prio | experiment | candidate | default | opponent | engine | pairs / planned | cand win % | def win % "
             "| delta (pp) +- s.e. | 95% CI (pp) | dVP +- s.e. | ms/dec c / d | opp ms c / d | ident | verdict | status |",
             "|" + "---|" * 17]
    for e in exps:
        p = progress(e, d)
        if not p["exists"] or not p["runs"]:
            state = "not started" if p["last_exit"] in (None, 0) else \
                f"FAILED (exit {p['last_exit']}, see logs/{e['name']}.log)"
            lines.append(f"| {e['priority']} | {e['name']} | {e.get('tunable') or e.get('cand_spec')} | | {e['opponent']} "
                         f"| {e['interpreter']} | 0 / {e['seeds']['count']} | | | | | | | | | | {state} |")
            continue
        index = p["index"]
        for x in p["per_run"]:
            run = x["run"]
            st = AB.run_stats(index, run)
            lo, hi = st["ci95"]
            ctx = run.get("ctx") or {}
            status = "stopped early" if x["stopped"] else ("done" if x["done"] >= e["seeds"]["count"] else
                                                          f"running ({x['done']}/{e['seeds']['count']})")
            if st["errors"]:
                status += f", {st['errors']} error pair(s)"
            if p["last_exit"] not in (None, 0):
                status += f", last attempt FAILED (exit {p['last_exit']})"
            cand = run["cand"]["label"]
            dflt = run["def"]["label"]
            vpd = "n/a" if math.isnan(st["vp_delta"]) else f"{st['vp_delta']:+.2f}"
            lines.append(
                f"| {e['priority']} | {e['name']} | `{cand}` | `{dflt}` | {ctx.get('opponent', e['opponent'])} | "
                f"{ctx.get('catanatron', '?')} | {st['pairs']} / {e['seeds']['count']} | {pct(st['win_rate_cand'])} | "
                f"{pct(st['win_rate_def'])} | {AB.fmt_pp(st['delta']).replace('pp', '')} +- "
                f"{pct(st['se'])} | [{pct(lo)}, {pct(hi)}] | "
                f"{vpd} +- {AB.fmt(st['vp_se'])} | {AB.fmt(st['ms_cand'], 1)} / {AB.fmt(st['ms_def'], 1)} | "
                f"{AB.fmt(st['opp_ms_cand'], 1)} / {AB.fmt(st['opp_ms_def'], 1)} | {st['identical']}/{st['pairs']} | "
                f"{st['verdict']} | {status} |")
    return "\n".join(lines) + "\n"


def write_summary(exps: List[Dict[str, Any]], d: str, plan: str, path: str) -> None:
    text = summary_markdown(exps, d, plan)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def log_event(d: str, **kw) -> None:
    kw["t"] = time.time()
    with open(os.path.join(d, "campaign_events.log"), "a") as fh:
        fh.write(json.dumps(kw, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
class Interrupted(Exception):
    pass


def run_campaign(args, exps: List[Dict[str, Any]], interps: Dict[str, str]) -> int:
    d = args.dir
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    summary = args.summary or os.path.join(d, "SUMMARY.md")
    deadline = time.time() + 60.0 * args.max_minutes if args.max_minutes else None
    child: List[Optional[subprocess.Popen]] = [None]

    def on_signal(signum, frame):
        raise Interrupted(f"signal {signum}")

    old_term = signal.signal(signal.SIGTERM, on_signal)
    old_int = signal.signal(signal.SIGINT, on_signal)
    rc = 0
    try:
        for e in exps:
            if e.get("enabled") is False:
                continue
            p = progress(e, d)
            if p["complete"]:
                print(f"[{e['name']}] complete ({p['games']} games); skipped", flush=True)
                continue
            left = None
            if deadline:
                left = (deadline - time.time()) / 60.0
                if left < 1.0:
                    print("campaign: time budget used up; rerun to continue", flush=True)
                    break
            cmd = command(e, d, interps, left)
            if args.dry_run:
                print(" ".join(cmd))
                continue
            env = dict(os.environ, PYTHONHASHSEED="0",
                       PYTHONPATH=ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
            log_path = os.path.join(d, "logs", f"{e['name']}.log")
            print(f"[{e['name']}] {e['interpreter']} vs {e['opponent']}: {p['games']}/{p['planned_games']} games "
                  f"so far; running (log {log_path})", flush=True)
            t0 = time.time()
            log_event(d, event="start", name=e["name"], cmd=cmd)
            with open(log_path, "a") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                log.flush()
                try:
                    child[0] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    code = child[0].wait()
                except Interrupted:
                    proc = child[0]
                    if proc is not None and proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    log_event(d, event="interrupted", name=e["name"], seconds=time.time() - t0)
                    raise
                finally:
                    child[0] = None
            p2 = progress(e, d)
            log_event(d, event="finish", name=e["name"], code=code, seconds=time.time() - t0,
                      games=p2["games"], complete=p2["complete"])
            tail = ""
            if code != 0:
                rc = 1
                try:
                    with open(log_path) as fh:
                        tail = fh.read().strip().splitlines()[-1][:200]
                except Exception:  # noqa: BLE001
                    pass
            print(f"[{e['name']}] exit {code} after {time.time() - t0:.0f} s: {p2['games']}/{p2['planned_games']} "
                  f"games, {'complete' if p2['complete'] else 'incomplete'}" + (f"; FAILED: {tail}" if code else ""),
                  flush=True)
            write_summary(exps, d, args.plan, summary)
    except Interrupted as ex:
        print(f"campaign interrupted ({ex}); rerun the same command to resume", flush=True)
        rc = 130
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if not args.dry_run:
            write_summary(exps, d, args.plan, summary)
            print(f"summary -> {summary}", flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="plan JSON")
    ap.add_argument("--dir", required=True, help="results directory (one JSONL per experiment, logs/, SUMMARY.md)")
    ap.add_argument("--summary", help="markdown summary path (default <dir>/SUMMARY.md)")
    ap.add_argument("--status", action="store_true", help="print progress and ETA, run nothing")
    ap.add_argument("--summary-only", action="store_true", help="rewrite the summary, run nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="start no new seed after this many minutes (in-flight games finish; rerun to continue)")
    ap.add_argument("--max-workers", type=int, default=2, help="cap on any experiment's workers (default 2)")
    args = ap.parse_args(argv)
    exps, interps = load_plan(args.plan, args.max_workers)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        missing = wanted - {e["name"] for e in exps}
        if missing:
            raise SystemExit(f"error: no experiment(s) {sorted(missing)} in the plan")
        exps = [e for e in exps if e["name"] in wanted]
    os.makedirs(args.dir, exist_ok=True)
    if args.status:
        print_status(exps, args.dir, args.plan)
        return 0
    if args.summary_only:
        path = args.summary or os.path.join(args.dir, "SUMMARY.md")
        write_summary(exps, args.dir, args.plan, path)
        print(f"summary -> {path}")
        return 0
    return run_campaign(args, exps, interps)


if __name__ == "__main__":
    raise SystemExit(main())
