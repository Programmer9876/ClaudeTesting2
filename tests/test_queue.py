"""Tests for scripts/run_queue.py, the budgeted test queue (docs/QUEUE.md).

Scheduler rules run on a synthetic clock with fake CPU counters, /proc/stat and process lists (no games).  The
end-to-end tests play synthetic games (``ablate_catanatron.py --fake-games``) through the real chunk commands,
JSONLs, stop records and ledger.  Snapshot tests copy the repository's allowlist to a temporary directory.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
PLAN = os.path.join(SCRIPTS, "queue_plan.json")
DATA = os.path.join(ROOT, "tests", "data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RQ = _load("run_queue_under_test", os.path.join(SCRIPTS, "run_queue.py"))
CAMP = RQ.CAMP
AB = RQ.AB
SEQ = RQ.SEQ
SPEC = "search:depth=1,beam=4,expand=8,evaluator=heuristic"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _plan(tmp_path, rows, **top):
    doc = {"interpreters": {"py321": sys.executable, "py330": sys.executable}, "defaults": {"workers": 1},
           "experiments": rows}
    doc.update(top)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(doc))
    return str(p)


def _q(tmp_path, rows, top=None, **kw):
    plan = _plan(tmp_path, rows, **(top or {}))
    kw.setdefault("snapshot", False)
    kw.setdefault("tunable_check", False)
    kw.setdefault("log", io.StringIO())
    kw.setdefault("guard", "NO_SUCH_GUARD_PATTERN_x")
    kw.setdefault("lock_files", [str(tmp_path / "no_lock")])
    kw.setdefault("poll_s", 0.2)
    return RQ.Queue(plan, str(tmp_path / "run"), **kw)


def _row(name, area="other", polarity="new", tier="t1", priority=1, **kw):
    r = {"name": name, "area": area, "polarity": polarity, "tier": tier, "priority": priority, "interpreter": "py321",
         "opponent": "vf", "tunable": "danger.TURNS_HALF", "values": [2], "seeds": {"count": 400, "base": 0}}
    if polarity == "new":
        r["promise_pp"] = 4
        r["d_prior"] = 0.04
    r.update(kw)
    return r


def _verdict(q, row, cand, verdict, label=None, reason="", p=None, delta=None, se=None, flags=(), mech=None,
             look=1, pairs=400, stats=None):
    q.ledger.append({"kind": "verdict", "row": row, "cand": cand, "verdict": verdict, "label": label or verdict,
                     "reason": reason, "p": p, "delta": delta, "se": se, "flags": list(flags), "mech": mech,
                     "look": look, "pairs": pairs, "stats": stats or {}, "source": "queue"})


def _fake_args(effect, disc=None):
    a = ["--fake-games", str(effect)]
    if disc is not None:
        a += ["--fake-discordance", str(disc)]
    return a


# ---------------------------------------------------------------------------
# the plan and intake
# ---------------------------------------------------------------------------
def test_every_current_plan_row_gets_area_polarity_tier(tmp_path):
    q = RQ.Queue(PLAN, str(tmp_path / "run"), snapshot=False, tunable_check=False, log=io.StringIO())
    q.refresh()
    from catanbot import tuning
    for rs in q.rows.values():
        e = rs.e
        assert e.get("area") in RQ.AREAS, e["name"]
        if not e["enabled"]:
            assert e.get("requires") or e.get("notes"), e["name"]
            continue
        if e["kind"] in ("catanatron", "selfplay"):
            assert e.get("polarity") in RQ.POLARITIES and e.get("tier"), e["name"]
            for n in RQ.row_names(e):
                tuning.find(n)                                  # epoch A runs on today's registry
        assert rs.status not in ("REFUSED", "SHELVED", "BLOCKED"), (e["name"], rs.status, rs.reason)
    names = {e["name"] for e in q.plan.rows}
    assert "t2_counted_info@value" not in names                 # removed as redundant (the proof measured it)
    assert any(r["name"] == "t2_counted_info@value" for r in q.plan.settings["removed"])
    for n in ("t2_danger_mult_off@value", "t2_block_need0@value", "t2_steal_factor_off@value",
              "t2_rob_break_off@value", "t2_turns_half2@value", "t2_knight03@value"):
        assert q.plan.row(n)["shadow_gate"]["row"] == "shadow_robber" and q.rows[n].status == "WAITING"
    for n in ("t2_max_slack0_vrule@value", "t2_coal_scale2_vrule@value", "t2_feed_leader_off_vrule@value",
              "t2_late_drop0_vrule@value", "counters_selfplay", "respond_lookahead_selfplay"):
        assert q.rows[n].design == "politics" and q.plan.row(n)["tier"] == "politics"
    assert RQ.looks_for(q.plan.row("counters_selfplay"), "politics", 1200) == [1200]
    op = q.plan.row("t1_openings@value")
    assert op["area"] == "ports" and op["values"] == ["pips_diversity", "standin_book", "setup_pick"]
    assert q.plan.row("demand_flat_pyeval@vf")["values"] == [[1, 1, 1, 1, 1]]
    child = q.plan.row("demand_mild_pyeval@vf")
    assert child["parent"] == "demand_flat_pyeval@vf" and child["fallback"] == "milder"
    assert q.plan.row("expansion_reach_credit@value")["enabled"] is False
    assert q.plan.row("expansion_reach_credit@value")["area"] == "ports"
    # area order: the ports rows run before every robber row, harness rows first
    order = [rs.name for rs in q.order()]
    assert order[0] == "t1_baseline_aa@value"
    assert order.index("t1_trades0_vrule@value") < order.index("t2_dump0@value") < order.index("t1_openings@value")
    assert order.index("demand_flat_pyeval@vf") < order.index("shadow_robber") < order.index("t1_depth2@value")


def test_campaign_py_loads_queue_plan_and_skips_queue_kinds(tmp_path):
    exps, _ = CAMP.load_plan(PLAN, 2)
    kinds = {e["name"]: e.get("_skip") for e in exps}
    assert kinds["shadow_robber"] and kinds["counters_selfplay"] and kinds["ports_spot_leader"]
    assert kinds["t1_openings@value"] is None
    out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "campaign.py"), "--plan", PLAN, "--dir",
                          str(tmp_path / "c"), "--dry-run"], capture_output=True, text=True, timeout=300,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr[-1500:]
    assert "[shadow_robber] kind command: run by scripts/run_queue.py, skipped" in out.stdout
    assert "--exp-name t1_openings@value" in out.stdout and "--exp-name shadow_robber" not in out.stdout


def test_existing_campaign_plan_json_loads_unchanged(tmp_path):
    exps, _ = CAMP.load_plan(os.path.join(SCRIPTS, "campaign_plan.json"), 2)
    assert len(exps) == 32 and not any(e.get("_skip") for e in exps)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"name": "x", "interpreter": "py330", "opponent": "value", "tunable": "t", "seed": 5}]))
    with pytest.raises(SystemExit, match="unknown field"):     # 'seed' stays a typo of 'seeds'
        CAMP.load_plan(str(bad), 2)


def test_lint_routing_rules(tmp_path):
    rows = [
        _row("count_sp", area="counting", kind="selfplay", tunable="devcards.KNIGHT_VALUE", games=1200),
        _row("count_full", area="counting"),
        _row("count_ok", area="counting", adapter_opts={"info": "counted"}),
        _row("trade_off", area="trades", tunable="politics.MAX_SLACK", values=[0.0], interpreter="py330",
             opponent="value"),
        _row("trade_ok", area="trades", tunable="search.trade_proposals", values=[0], interpreter="py330",
             opponent="value", trades="value"),
        _row("sp_pyeval", area="robber", kind="selfplay", tunable="heuristic.EXPOSURE_WEIGHT", values=[0.0],
             games=2400),
        _row("sp_other", area="other", kind="selfplay", tunable="search.beam", values=[2], games=2400),
        _row("no_area", area="nowhere"),
        _row("no_promise", promise_pp=None),
        _row("human", kind="human"),
    ]
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    st = {n: (rs.status, rs.reason) for n, rs in q.rows.items()}
    assert st["count_sp"][0] == "REFUSED" and "self-play" in st["count_sp"][1]
    assert st["count_full"][0] == "REFUSED" and "info counted" in st["count_full"][1]
    assert st["count_ok"][0] == "ELIGIBLE"
    assert st["trade_off"][0] == "REFUSED" and "trades value" in st["trade_off"][1]
    assert st["trade_ok"][0] == "ELIGIBLE"
    assert st["sp_pyeval"][0] == "REFUSED" and "port the weight to C++" in st["sp_pyeval"][1]
    assert st["sp_other"][0] == "REFUSED"
    assert st["no_area"][0] == "REFUSED" and st["no_promise"][0] == "REFUSED"
    assert st["human"][0] == "DEFERRED"
    assert all(rs.name not in ("count_sp", "human") for rs in q.order())


def test_politics_names_force_politics_design(tmp_path):
    q = _q(tmp_path, [_row("slack", area="politics", polarity="knockout", tunable="politics.MAX_SLACK",
                           values=[0.0], trades="value", interpreter="py330", opponent="value", design="screen",
                           tier="politics")])
    assert RQ.design_of({"kind": "catanatron", "tunable": "coalitions.SCALE"}) == "politics"
    assert RQ.design_of({"kind": "selfplay", "tunable": "search.respond_lookahead"}) == "politics"
    q.refresh()
    assert any("forced" in w for w in q.rows["slack"].intake.warnings)


def test_low_power_row_gets_reducer_then_bundle_then_intake_shelve(tmp_path):
    lo = _row("lo", promise_pp=2, d_prior=0.40, seeds={"count": 2000, "base": 0})
    sib = _row("sib", promise_pp=2, d_prior=0.40, seeds={"count": 2000, "base": 0}, tunable="danger.BLOCK_NEED",
               values=[0])
    q = _q(tmp_path, [lo, sib])
    q.refresh()
    rs = q.rows["lo"]
    assert rs.status == "SHELVED" and "unprovable at budget" in rs.reason and "a bundle with sib" in rs.reason
    assert abs(rs.intake.power - SEQ.power_at(2, 0.40, 2000)) < 1e-9 and rs.intake.power < 0.5
    # a declared bundle row takes over
    bundle = _row("bundle", promise_pp=None, d_prior=None, seeds={"count": 2000, "base": 0}, bundle_of=["lo", "sib"],
                  tunable="danger.BLOCK_FLOOR", values=[0.2])
    q = _q(tmp_path, [lo, sib, bundle])
    q.refresh()
    assert q.rows["lo"].status == "BUNDLED" and q.rows["bundle"].status == "ELIGIBLE"
    it = q.rows["bundle"].intake                    # promise = sum of the members', D = the largest member D
    assert it.promise == 4 and it.d == 0.40 and it.power >= 0.5
    # a reducer first: CV with a pool of M >= 4 N_max on the same default arm
    cv = _row("cv", promise_pp=3, d_prior=0.40, seeds={"count": 2000, "base": 0}, estimator="cv")
    pool = {"name": "pool", "kind": "pool", "area": "harness", "priority": 9, "interpreter": "py321", "opponent": "vf",
            "cand_spec": SPEC, "def_spec": SPEC, "seeds": {"count": 8000, "base": 10 ** 7}}
    q = _q(tmp_path, [cv, pool])
    q.refresh()
    assert q.rows["cv"].status == "SHELVED"                     # no pool games yet
    q.ledger.append({"kind": "pool", "row": "pool", "def_key": "x", "epoch": None, "m": 8000, "mean": 0.25})
    q.refresh()
    it = q.rows["cv"].intake
    assert q.rows["cv"].status == "ELIGIBLE" and it.reducer == "cv" and it.power >= 0.5


def test_crn_pilot_decides_the_crn_reducer(tmp_path):
    rows = [_row("off", area="harness", polarity="measure", tier="harness", design="estimate", tunable="search.expand",
                 values=[4]),
            _row("dice", area="harness", polarity="measure", tier="harness", design="estimate", tunable="search.expand",
                 values=[4], crn="dice"),
            _row("aa", area="harness", polarity="measure", tier="harness", tunable=None, values=None, cand_spec=SPEC,
                 def_spec=SPEC, crn="dice"),
            _row("row", promise_pp=3, d_prior=0.40, seeds={"count": 2000, "base": 0}, crn="auto")]
    q = _q(tmp_path, rows, top={"crn_pilot": {"off": "off", "dice": "dice", "aa": "aa", "drop": 0.25}})
    q.refresh()
    assert q.rows["row"].status == "SHELVED"
    _verdict(q, "off", "4", "ESTIMATE", stats={"n": 400, "plus": 70, "minus": 65})
    _verdict(q, "dice", "4", "ESTIMATE", stats={"n": 400, "plus": 45, "minus": 40})
    _verdict(q, "aa", "cand", "PASS")
    q.refresh()
    crn = q.crn_status()
    assert crn["pass"] and abs(crn["ratio"] - 85 / 135) < 1e-9
    assert q.rows["row"].intake.reducer == "crn" and q.rows["row"].status == "ELIGIBLE"
    cmd, _, _, _ = q.chunk_command(q.rows["row"], 5)
    assert cmd[cmd.index("--crn") + 1] == "dice"


def test_headroom_gate_blocks_counting_rows_when_gap_upper_below_2pp(tmp_path):
    row = _row("devread", area="counting", adapter_opts={"info": "counted"})
    q = _q(tmp_path, [row], top={"headroom": {"counting": {"gap": -0.03, "se": 0.005, "source": "test"}}})
    q.refresh()
    assert q.rows["devread"].status == "SHELVED" and "SHELVE(headroom)" in q.rows["devread"].reason
    q = _q(tmp_path, [row], top={"headroom": {"counting": {"gap": 0.02, "se": 0.01, "source": "test"}}})
    q.refresh()
    assert q.rows["devread"].status == "ELIGIBLE"


def test_trades_headroom_first_and_caps_player_trade_promises(tmp_path):
    head = _row("head", area="trades", polarity="measure", headroom=True, tunable="search.trade_proposals",
                values=[0], trades="value", interpreter="py330", opponent="value", priority=9)
    trade = _row("trade", area="trades", tunable="trading.accept_margin", values=[0.0], trades="value",
                 interpreter="py330", opponent="value", promise_pp=4, d_prior=0.10, priority=1,
                 seeds={"count": 2000, "base": 0})
    dump = _row("dump", area="trades", polarity="knockout", tunable="search.dump_candidates", values=[0], priority=2)
    q = _q(tmp_path, [trade, dump, head])
    q.refresh()
    assert [r.name for r in q.order()][0] == "head"           # headroom rows run first in their area
    assert q.rows["trade"].status == "WAITING" and "headroom" in q.rows["trade"].reason
    assert q.rows["dump"].status == "ELIGIBLE"                  # bank-side trades rows do not wait
    _verdict(q, "head", "0", "ESTIMATE", delta=-0.02, se=0.01)
    q.refresh()
    it = q.rows["trade"].intake
    assert q.rows["trade"].status == "ELIGIBLE" and abs(it.promise - 0.5 * (2.0 + 1.96)) < 0.01
    assert any("capped" in w for w in it.warnings)


def test_route_human_row_plays_nothing_and_is_deferred(tmp_path):
    q = _q(tmp_path, [{"name": "leader", "kind": "human", "area": "ports", "tier": "politics", "priority": 1,
                       "tunable": "winpaths.SPOT_LEADER"}])
    calls = []
    q.runner = lambda *a: calls.append(a) or 0
    assert q.loop(once=True) == 0 and not calls
    assert q.rows["leader"].status == "DEFERRED"
    rep = q.report_markdown(when="T")
    assert "| winpaths.SPOT_LEADER | none (no bot harness reacts) | n/a | deferred |" in rep


def test_explain_simulate_golden_current_plan(tmp_path):
    """Regenerate after a plan edit: python3 -c "import tests.test_queue as t; t.write_explain_golden()"."""
    q = RQ.Queue(PLAN, str(tmp_path / "run"), snapshot=False, tunable_check=False, log=io.StringIO())
    q.refresh()
    text = q.explain(simulate=True)
    with open(os.path.join(DATA, "queue_explain_golden.txt")) as fh:
        assert text == fh.read()
    assert "total" in text and "Conditional rows" in text


def write_explain_golden():
    import tempfile
    q = RQ.Queue(PLAN, tempfile.mkdtemp(), snapshot=False, tunable_check=False, log=io.StringIO())
    q.refresh()
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "queue_explain_golden.txt"), "w") as fh:
        fh.write(q.explain(simulate=True))


# ---------------------------------------------------------------------------
# scheduler (synthetic clock, fake counters)
# ---------------------------------------------------------------------------
def test_area_order_dominates_cost(tmp_path):
    rows = [_row("cheap_other", area="other", priority=1, seeds={"count": 100, "base": 0}),
            _row("dear_ports", area="ports", priority=99, seeds={"count": 2000, "base": 0}),
            _row("harness", area="harness", polarity="measure", tier="harness", priority=500)]
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    assert q.rows["dear_ports"].cost_h > 10 * q.rows["cheap_other"].cost_h
    assert [r.name for r in q.order()] == ["harness", "dear_ports", "cheap_other"]


def test_preempt_only_higher_area_and_exchange_rule(tmp_path):
    rows = [_row("r", area="robber", priority=36), _row("q", area="ports", priority=12),
            _row("low", area="other", priority=1)]
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    r, qq, low = q.rows["r"], q.rows["q"], q.rows["low"]
    assert abs(q.weight(qq) / q.weight(r) - 4.0) < 1e-12
    # the design's worked example: C_r 0.59 CPU-h left, C_q 1.5 -> 4 x 0.59 = 2.36 > 1.2 x 1.5 = 1.8: preempt
    r.cost_h, qq.cost_h = 0.59, 1.5
    assert q.should_preempt(qq, r)
    q.incumbent = "r"
    assert q.pick() is qq
    # at look 4, C_r = 0.25: 1.0 < 1.8, the incumbent finishes first (about 5 min wall)
    r.cost_h = 0.25
    assert not q.should_preempt(qq, r)
    assert q.pick() is r
    # a lower-area row never preempts: 'low' is behind r whatever its cost
    qq.status = "FINAL"
    low.cost_h = 0.0
    assert q.pick() is r


def test_cpu_seconds_from_getrusage_children_include_pool_workers(tmp_path):
    code = ("import multiprocessing as mp, time\n"
            "def burn(_):\n"
            "    t = time.process_time()\n"
            "    while time.process_time() - t < 0.4: pass\n"
            "if __name__ == '__main__':\n"
            "    with mp.get_context('fork').Pool(2) as p: p.map(burn, range(2))\n")
    script = tmp_path / "burn.py"
    script.write_text(code)
    c0 = RQ.children_cpu()
    subprocess.run([sys.executable, str(script)], check=True, timeout=60)
    assert RQ.children_cpu() - c0 >= 0.75                        # both pool workers' CPU is booked


def test_rate_prior_key_includes_info_depth_crn():
    base = {"interpreter": "py330", "opponent": "value", "tunable": "danger.TURNS_HALF", "values": [2]}
    k0 = CAMP.rate_class(base)
    assert CAMP.rate_class(dict(base, cand_adapter_opts={"info": "counted"})) != k0
    assert CAMP.rate_class(dict(base, tunable="search.depth", values=[2])) != k0
    assert CAMP.rate_class(dict(base, crn="dice")) != k0
    assert CAMP.info_mode(dict(base, extra_args=["--info", "counted"])) == "counted"
    e = dict(base, kind="catanatron")
    assert RQ.cpu_prior(e) == 1.77
    assert RQ.cpu_prior(dict(e, cand_adapter_opts={"info": "counted"})) == 7.2
    assert RQ.cpu_prior(dict(e, tunable="search.depth", values=[2])) == pytest.approx(1.77 * 3.4 / 2)
    assert RQ.cpu_prior(dict(e, tunable="heuristic.EXPOSURE_WEIGHT", values=[0.0])) == 5.5
    assert RQ.cpu_prior(dict(e, opponent="vf", tunable="placement.RESOURCE_DEMAND")) == 3.3
    assert RQ.cpu_prior(dict(e, trades="value")) == 3.0
    assert RQ.cpu_prior({"kind": "selfplay"}) == 10.8


def test_chunk_never_passes_stop_at_se_unless_legacy(tmp_path):
    rows = [_row("seq", values=[2, 4.5], stop_at_se=0.015, stop_min_pairs=100, seeds={"count": 2000, "base": 0}),
            _row("leg", design="legacy", stop_at_se=0.015, seeds={"count": 2000, "base": 0})]
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    _verdict(q, "seq", "4.5", "SHELVE")
    q.refresh()
    cmd, cwd, env, meta = q.chunk_command(q.rows["seq"], 15)
    assert "--stop-at-se" not in cmd and "--stop-min-pairs" not in cmd
    assert cmd[cmd.index("--seeds") + 1] == "400" and cmd[cmd.index("--values") + 1] == "2"   # open values only
    assert "--mech" in cmd and cmd[cmd.index("--max-minutes") + 1] == "15.00"
    assert cmd[1] == os.path.join(ROOT, "scripts", "ablate_catanatron.py") and env["PYTHONHASHSEED"] == "0"
    cmd2, _, _, _ = q.chunk_command(q.rows["leg"], 15)
    assert cmd2[cmd2.index("--stop-at-se") + 1] == "0.015"


def _play(q, name, seeds, base=0, extra=("--fake-games", "0.0")):
    e = q.plan.row(name)
    args = [sys.executable, os.path.join(SCRIPTS, "ablate_catanatron.py"), "--tunable", e["tunable"], "--values",
            ",".join(RQ.value_text(v) for v in e["values"]), "--opponent", "vf", "--seeds", str(seeds),
            "--seed-base", str(base), "--workers", "1", "--out", q.jsonl(e), "--exp-name", name, "--quiet"] + list(extra)
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-1500:]


def test_look_evaluated_only_on_complete_prefix(tmp_path):
    q = _q(tmp_path, [_row("r", seeds={"count": 2000, "base": 0})], no_intake=True)
    q.refresh()
    q.ledger.append({"kind": "pin", "row": "r", "epoch": "live", "identity": q.identity(q.plan.row("r")),
                     "values": ["2"], "n_max": 2000, "base": 0})
    _play(q, "r", 399, extra=RQ_FAKE_NULL)
    q.refresh()
    assert q.ledger.looks("r", "2") == []
    _play(q, "r", 400, extra=RQ_FAKE_NULL)
    q.refresh()
    looks = q.ledger.looks("r", "2")
    assert len(looks) == 1 and looks[0]["n"] == 400 and looks[0]["look"] == 1


RQ_FAKE_NULL = ["--fake-games", "0.0", "--fake-discordance", "0.4"]


def test_preempt_now_waits_for_slice_hard_loses_at_most_w_games(tmp_path):
    q = _q(tmp_path, [_row("r")], no_intake=True)
    log = str(tmp_path / "x.log")
    t0 = time.time()
    code = q._run_process([sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path), dict(os.environ),
                          log, hard_check=lambda: True)
    assert code == -15 and time.time() - t0 < 20
    code = q._run_process([sys.executable, "-c", "print('ok')"], str(tmp_path), dict(os.environ), log,
                          hard_check=None)
    assert code == 0
    # without --hard, preemption waits for the slice boundary: the runner gets no hard_check
    seen = []
    q.runner = lambda cmd, cwd, env, lp, hard: seen.append(hard) or 0
    q.loop(once=True)
    assert seen == [None]


def test_proof_guard_blocks_every_chunk(tmp_path):
    q = _q(tmp_path, [_row("r")], no_intake=True, guard="proof_snapshot",
           proc_list=lambda: [(os.getpid(), "python scripts/run_queue.py --guard proof_snapshot"),
                              (99999, "python /home/user/proof_snapshot2/scripts/bench_catanatron.py --info counted")])
    calls = []
    q.runner = lambda *a: calls.append(a) or 0
    assert q.loop(once=True) == 3 and not calls
    assert q.ledger.chunks("r") == []
    lock = tmp_path / "PROOF_RUNNING"
    lock.write_text("T12")
    q2 = _q(tmp_path, [_row("r")], no_intake=True, lock_files=[str(lock)], proc_list=lambda: [])
    q2.runner = lambda *a: calls.append(a) or 0
    assert q2.loop(once=True) == 3 and not calls
    q3 = _q(tmp_path, [_row("r")], no_intake=True, proc_list=lambda: [(os.getpid(), "run_queue.py proof_snapshot")],
            guard="proof_snapshot")
    assert q3.proof_guard() is None                               # the queue itself never trips its guard


def test_load_gate_samples_after_drain_fake_proc_stat(tmp_path):
    samples = []

    def stat_seq(busy_frac):
        state = {"n": 0}

        def f():
            state["n"] += 1
            tot = 1000 * state["n"]
            busy = int(tot * busy_frac)
            return [busy, 0, 0, tot - busy, 0, 0, 0, 0], 4
        return f

    slept = []
    rows = [_row("ab", opponent="alphabeta", interpreter="py330", priority=1, workers=1),
            _row("cheap", priority=2)]
    q = _q(tmp_path, rows, no_intake=True, proc_stat=stat_seq(3.5 / 4), sleep=lambda s: slept.append(s))
    ran = []
    q.runner = lambda cmd, cwd, env, lp, hard: ran.append(cmd[cmd.index("--exp-name") + 1]) or 0
    q.loop(once=True)
    assert slept and slept[0] == 10.0 and ran == ["cheap"]      # 3.5 busy cores > 4 - 1 + 0.25: the exclusive row waits
    q2 = _q(tmp_path, rows, no_intake=True, proc_stat=stat_seq(2.0 / 4), sleep=lambda s: None)
    ran.clear()
    q2.runner = lambda cmd, cwd, env, lp, hard: ran.append(cmd[cmd.index("--exp-name") + 1]) or 0
    q2.loop(once=True)
    assert ran == ["ab"]
    assert RQ.busy_cores(([0, 0, 0, 0], 4), ([50, 0, 50, 100, 0], 4)) == pytest.approx(2.0)


def test_fallback_kind_triggers_simpler_milder_stronger(tmp_path):
    rows = [_row("p"), _row("s", parent="p", fallback="simpler", priority=2),
            _row("m", parent="p", fallback="milder", priority=3, mechanism={"metric": "setup_distinct",
                                                                              "direction": "+"}),
            _row("g", parent="p", fallback="stronger", priority=4)]
    rows[0]["mechanism"] = {"metric": "setup_distinct", "direction": "+"}

    def status(verdict, reason="", flags=(), mech=None, delta=-0.01):
        q = _q(tmp_path, rows, no_intake=True)
        _verdict(q, "p", "2", verdict, label=f"{verdict}({reason})" if reason else verdict, reason=reason,
                 flags=flags, mech=mech, delta=delta)
        q.refresh()
        shutil.rmtree(tmp_path / "run")
        return {n: q.rows[n].status for n in ("s", "m", "g")}

    over = {"setup_distinct": {"diff": 0.5, "diff_se": 0.1}}
    assert status("SHELVE", "no gain; cap") == {"s": "ELIGIBLE", "m": "NOT TRIGGERED", "g": "NOT TRIGGERED"}
    assert status("SHELVE", "no gain; cap", mech=over) == {"s": "ELIGIBLE", "m": "ELIGIBLE", "g": "NOT TRIGGERED"}
    assert status("REJECT", "worse", mech=over) == {"s": "ELIGIBLE", "m": "ELIGIBLE", "g": "NOT TRIGGERED"}
    assert status("SHELVE", "too small to prove; cap", delta=0.01) == \
        {"s": "ELIGIBLE", "m": "NOT TRIGGERED", "g": "ELIGIBLE"}
    assert status("NOOP", "diverged share < 0.01") == {"s": "NOT TRIGGERED", "m": "NOT TRIGGERED", "g": "ELIGIBLE"}
    assert status("FAILED", "inert-harness") == {"s": "NOT TRIGGERED", "m": "NOT TRIGGERED", "g": "NOT TRIGGERED"}
    assert status("ADOPT") == {"s": "NOT TRIGGERED", "m": "NOT TRIGGERED", "g": "NOT TRIGGERED"}
    # 'promise not met' alone also unlocks a simpler idea
    assert status("REJECT", "futile", flags=["promise not met"])["s"] == "ELIGIBLE"
    # before the parent's verdict the children wait
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    assert q.rows["s"].status == "WAITING"


def test_knockout_never_parent_ladder_depth_2_fresh_seeds(tmp_path):
    rows = [_row("p", seeds={"count": 400, "base": 7}), _row("c1", parent="p", fallback="simpler"),
            _row("c2", parent="c1", fallback="simpler"), _row("c3", parent="c2", fallback="simpler"),
            _row("ko", polarity="knockout"), _row("kc", parent="ko", fallback="simpler")]
    q = _q(tmp_path, rows, no_intake=True)
    q.refresh()
    plan = q.plan
    assert RQ.effective_base(plan, plan.row("p")) == 7
    assert RQ.effective_base(plan, plan.row("c1")) == 10 ** 6
    assert RQ.effective_base(plan, plan.row("c2")) == 2 * 10 ** 6
    assert q.rows["c3"].status == "REFUSED" and "at most 2 deep" in q.rows["c3"].reason
    assert q.rows["kc"].status == "REFUSED" and "never a parent" in q.rows["kc"].reason


def test_confirmation_needs_adopt_and_holm_within_tier(tmp_path):
    rows = [_row("a"), _row("b", tunable="danger.BLOCK_NEED", values=[0]),
            _row("c", tunable="danger.BLOCK_FLOOR", values=[0.2]),
            _row("conf", tier="t3", confirms="a", opponent="alphabeta", interpreter="py330"),
            _row("other_tier", tier="t2", tunable="danger.BLOCK_NEED", values=[1])]
    q = _q(tmp_path, rows, no_intake=True)
    _verdict(q, "a", "2", "ADOPT", p=0.001, delta=0.05)
    _verdict(q, "b", "0", "SHELVE", p=0.3)
    q.refresh()
    assert q.rows["conf"].status == "WAITING" and "tier t1" in q.rows["conf"].reason
    assert q.rows["conf"].design == "confirm"
    _verdict(q, "c", "0.2", "SHELVE", p=0.5)
    q.refresh()
    adj, complete = q.holm("t1")
    assert complete and adj["a|2"] == pytest.approx(0.003)
    assert q.rows["conf"].status == "ELIGIBLE"
    # a weaker ADOPT that does not survive Holm in its tier: no confirmation
    shutil.rmtree(tmp_path / "run")
    q = _q(tmp_path, rows, no_intake=True)
    _verdict(q, "a", "2", "ADOPT", p=0.03, delta=0.05)
    _verdict(q, "b", "0", "SHELVE", p=0.3)
    _verdict(q, "c", "0.2", "SHELVE", p=0.5)
    q.refresh()
    assert q.rows["conf"].status == "NOT TRIGGERED"


def test_plan_reload_parse_error_keeps_previous_plan(tmp_path):
    q = _q(tmp_path, [_row("r")], no_intake=True)
    q.refresh()
    sha = q.plan.sha
    with open(q.plan_path, "w") as fh:
        fh.write("{ not json")
    os.utime(q.plan_path, (time.time() + 5, time.time() + 5))
    q.refresh()
    assert q.plan.sha == sha and "r" in q.rows
    assert any("keeping the previous plan" in r.get("msg", "") for r in q.ledger.of("alert"))


def test_removed_value_withdrawn_added_value_new_run(tmp_path):
    q = _q(tmp_path, [_row("r", values=[2, 4.5], seeds={"count": 400, "base": 0})], no_intake=True)
    q.refresh()
    q.ledger.append({"kind": "pin", "row": "r", "epoch": "live", "identity": q.identity(q.plan.row("r")),
                     "values": ["2", "4.5"], "n_max": 400, "base": 0})
    doc = json.loads(open(q.plan_path).read())
    doc["experiments"][0]["values"] = [2, 6]
    open(q.plan_path, "w").write(json.dumps(doc))
    os.utime(q.plan_path, (time.time() + 5, time.time() + 5))
    q.refresh()
    assert q.ledger.verdict("r", "4.5")["verdict"] == "WITHDRAWN"
    assert [c.key for c in q.rows["r"].open_cands] == ["2", "6"]
    # an identity field of a started row changes: FAILED(plan-changed); a cap change is ignored
    doc["experiments"][0]["opponent"] = "ab"
    doc["experiments"][0]["seeds"]["count"] = 2000
    open(q.plan_path, "w").write(json.dumps(doc))
    os.utime(q.plan_path, (time.time() + 10, time.time() + 10))
    q.refresh()
    assert q.ledger.verdict("r", "2")["label"] == "FAILED(plan-changed)"
    assert q.rows["r"].n_max == 400


def test_missing_tunable_in_snapshot_blocked_not_failed(tmp_path):
    q = _q(tmp_path, [_row("r", tunable="acquisition.NOT_BUILT_YET", values=[1])], no_intake=True,
           tunable_check=True)
    q.refresh()
    rs = q.rows["r"]
    assert rs.status == "BLOCKED" and "needs --bump-code" in rs.reason
    assert q.ledger.verdict("r", "1") is None


def test_stalled_row_fails_after_three_empty_slices(tmp_path):
    q = _q(tmp_path, [_row("r")], no_intake=True)
    q.runner = lambda *a: 2
    for _ in range(3):
        q.loop(once=True)
    q.refresh()
    assert q.ledger.verdict("r", "2")["label"] == "FAILED(stalled)"


# ---------------------------------------------------------------------------
# self-play rows
# ---------------------------------------------------------------------------
def _sp(name="sp", **kw):
    r = {"name": name, "kind": "selfplay", "area": "politics", "polarity": "new", "tier": "politics", "priority": 1,
         "tunable": "search.respond_lookahead", "values": [1], "games": 1200, "players": 4}
    r.update(kw)
    return r


def test_selfplay_chunk_command_seeds_spec_and_hashseed(tmp_path):
    q = _q(tmp_path, [_sp(), _sp("sp2", tunable="search.counters", counters=True,
                                 base_spec="search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=0")])
    q.refresh()
    cmd, cwd, env, meta = q.chunk_command(q.rows["sp"], 15)
    assert cmd[1].endswith("scripts/ablate.py")
    assert cmd[cmd.index("--base-spec") + 1] == SPEC                  # never ablate.py's cheap heuristic default
    assert cmd[cmd.index("--games") + 1] == "240" and cmd[cmd.index("--seed") + 1] == "5000"
    assert cmd[cmd.index("--values") + 1] == "1" and cmd.count("--values") == 1
    assert env["PYTHONHASHSEED"] == "0" and meta["chunk"] == 0 and cmd[cmd.index("--json") + 1].endswith(".tmp")
    cmd2, _, _, _ = q.chunk_command(q.rows["sp2"], 15)
    assert "--counters" in cmd2 and cmd2[cmd2.index("--base-spec") + 1].endswith("paths=0")
    assert RQ.looks_for(q.plan.row("sp"), "politics", 1200) == [1200]
    assert RQ.looks_for(dict(q.plan.row("sp"), polarity="new"), "screen", 2400) == [480, 960, 1440, 1920, 2400]


def _chunk_doc(records, base_spec=SPEC, mode="c++ (catanbot_core)"):
    return {"tunable": "search.respond_lookahead", "base_spec": base_spec, "players": 4, "games": len(records),
            "evaluator_mode": mode, "allow_counters": False, "results": [{"records": records}]}


def _sp_records(n, k, cand_share=0.3):
    out = []
    for g in range(n):
        pat = ["CCDD", "DDCC", "CDCD", "DCDC", "CDDC", "DCCD"][g % 6]
        w = (g * 7 + k) % 4
        side = pat[w]
        out.append({"pattern": pat, "winner": w, "diff": 0.5 if side == "C" else -0.5,
                    "winning_side": "cand" if side == "C" else "default", "vps": [10 if i == w else 5 for i in range(4)]})
    return out


def test_selfplay_rows_evaluate_fixed_1200_and_never_pool_differing_chunks(tmp_path):
    q = _q(tmp_path, [_sp()])
    q.refresh()
    runs = []

    def fake_runner(cmd, cwd, env, log, hard):
        out = cmd[cmd.index("--json") + 1]
        k = int(cmd[cmd.index("--seed") + 1]) - 5000
        with open(out, "w") as fh:
            json.dump(_chunk_doc(_sp_records(240, k)), fh)
        runs.append(k)
        return 0

    q.runner = fake_runner
    for _ in range(4):
        q.loop(once=True)
    assert runs == [0, 1, 2, 3] and q.ledger.verdict("sp", "1") is None      # 960 < 1200: no look yet
    q.loop(once=True)
    v = q.ledger.verdict("sp", "1")
    assert v["verdict"] == "SCREENED" and v["pairs"] == 1200 and v["design"] == "politics"
    # a chunk from another base spec is never pooled
    q2 = _q(tmp_path, [_sp("sp3")])
    q2.refresh()
    d = q2.selfplay_dir(q2.plan.row("sp3"), "1")
    os.makedirs(d, exist_ok=True)
    for k in range(5):
        with open(os.path.join(d, f"chunk_{k}.json"), "w") as fh:
            json.dump(_chunk_doc(_sp_records(240, k), base_spec=SPEC if k else "heuristic:temp=0.15"), fh)
    q2.refresh()
    assert q2.ledger.verdict("sp3", "1")["label"] == "FAILED(chunks-differ)"


def test_killed_chunk_replayed_whole(tmp_path):
    q = _q(tmp_path, [_sp()])
    q.refresh()
    seen = []

    def killed(cmd, cwd, env, log, hard):
        out = cmd[cmd.index("--json") + 1]
        open(out, "w").write('{"partial": ')                 # a kill mid-write
        seen.append(cmd[cmd.index("--seed") + 1])
        return -15

    q.runner = killed
    q.loop(once=True)
    q.loop(once=True)
    assert seen == ["5000", "5000"]                           # the same chunk again, from scratch
    d = q.selfplay_dir(q.plan.row("sp"), "1")
    assert not any(f.endswith(".json") for f in os.listdir(d))


def test_3p_noharm_uses_one_candidate_patterns_only():
    recs = []
    for g in range(60):
        pat = ["CCD", "DDC", "CDC", "DCD", "DCC", "CDD"][g % 6]
        recs.append({"pattern": pat, "winner": g % 3, "diff": 0.0, "vps": [8, 7, 6]})
    st = SEQ.selfplay_units(recs, one_candidate_only=True)
    assert st.ok == 30                                           # DDC, DCD, CDD only
    for r in recs:
        if r["pattern"].count("C") == 1:
            c = r["pattern"].index("C")
            assert (r["winner"] == c) == (r["winner"] in [i for i, x in enumerate(r["pattern"]) if x == "C"])
    x = [(1.0 if r["winner"] == r["pattern"].index("C") else -0.5) for r in recs if r["pattern"].count("C") == 1]
    assert abs(st.delta - sum(x) / len(x)) < 1e-12


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
def test_snapshot_allowlist_has_so_and_no_proof_or_run_dirs(tmp_path):
    src = tmp_path / "src"
    for rel in ("catanbot/__init__.py", "catanbot/x.py", "catanbot/league/y.py", "catanbot/vision/v.py",
                "catanbot/__pycache__/x.cpython-311.pyc", "catanbot/catanbot_core.cpython-311-x86_64-linux-gnu.so",
                "catanbot/catanbot_core.staged.so", "catanbot/data.json", "proof/T1/logs/a.jsonl.gz",
                "runs/campaign1/a.jsonl", "league/champions.json", "scripts/ablate_catanatron.py",
                "scripts/mechanics.py", "scripts/seqtest.py", "scripts/campaign.py", "scripts/run_queue.py",
                "docs/QUEUE.md"):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    files = RQ.snapshot_files(str(src))
    assert "catanbot/catanbot_core.cpython-311-x86_64-linux-gnu.so" in files
    assert {"catanbot/__init__.py", "catanbot/x.py", "catanbot/league/y.py", "scripts/ablate_catanatron.py",
            "scripts/mechanics.py", "scripts/seqtest.py"} <= set(files)
    for bad in ("catanbot/vision/v.py", "catanbot/catanbot_core.staged.so", "catanbot/data.json",
                "scripts/campaign.py", "scripts/run_queue.py"):
        assert bad not in files
    assert not any(f.startswith(("proof/", "runs/", "league/", "docs/")) or "__pycache__" in f for f in files)


def test_snapshot_fingerprint_equals_repo_and_cpp_evaluator(tmp_path):
    dest = tmp_path / "home" / "code" / "A"
    info = RQ.make_snapshot(ROOT, str(dest), [sys.executable])
    snap = RQ.run_check(sys.executable, str(dest))
    repo = RQ.run_check(sys.executable, ROOT)
    assert snap["fingerprint"] == repo["fingerprint"] == info["fingerprint"][sys.executable]
    assert snap["evaluator"].startswith("c++")
    assert os.path.exists(dest / "catanbot" / "catanbot_core.cpython-311-x86_64-linux-gnu.so")
    p = dest / "catanbot" / "tuning.py"
    assert not os.access(p, os.W_OK) or os.geteuid() == 0 or True     # read-only bits (root ignores them)
    assert not (os.stat(p).st_mode & 0o222)


def _mini_repo(tmp_path):
    """A copy of the repository's allowlist (a source tree another agent may edit)."""
    src = tmp_path / "src"
    for f in RQ.snapshot_files(ROOT):
        d = src / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), d)
    return src


def test_epochs_pin_rows_bump_refused_and_reuse_only_within_epoch(tmp_path):
    src = _mini_repo(tmp_path)
    home = tmp_path / "home"
    fake = RQ_FAKE_NULL
    rows = [_row("r1", extra_args=fake, seeds={"count": 2000, "base": 0}, d_prior=0.4),
            _row("r2", extra_args=fake, seeds={"count": 2000, "base": 0}, d_prior=0.4, tunable="danger.BLOCK_NEED",
                 values=[0], priority=2)]
    q = _q(tmp_path, rows, no_intake=True, snapshot=True, home=str(home), src_root=str(src), slice_minutes=5)
    q.loop(once=True)                                      # creates epoch A, pins r1, plays its first look
    assert q.ledger.pin("r1")["epoch"] == "A" and os.path.isdir(home / "code" / "A")
    code_a = {r["code"] for r in AB.read_jsonl(q.jsonl(q.plan.row("r1")))[0] if r.get("kind") == "game"}
    # another agent edits the source: a bump is refused while r1 is open, allowed with --force
    with open(src / "catanbot" / "__init__.py", "a") as fh:
        fh.write("\n# edited by another agent\n")
    with pytest.raises(SystemExit, match="open in the current epoch"):
        q.bump(["all"])
    rec = q.bump(["all"], force=True)
    assert rec["id"] == "B1" and os.path.isdir(home / "code" / "B1")
    q.loop(once=True)                                      # r1 continues (look 2) from epoch A
    cmd = q.ledger.chunks("r1")[-1]["cmd"]
    assert cmd[1] == str(home / "code" / "A" / "scripts" / "ablate_catanatron.py")
    codes = {r["code"] for r in AB.read_jsonl(q.jsonl(q.plan.row("r1")))[0] if r.get("kind") == "game"}
    assert codes == code_a                                  # the pinned row never mixes code versions
    q.ledger.append({"kind": "verdict", "row": "r1", "cand": "2", "verdict": "SHELVE", "label": "SHELVE",
                     "source": "test"})
    q.loop(once=True)                                      # r2 starts in the new epoch
    assert q.ledger.pin("r2")["epoch"] == "B1"
    g2 = [r for r in AB.read_jsonl(q.jsonl(q.plan.row("r2")))[0] if r.get("kind") == "game"]
    assert g2 and {r["code"] for r in g2}.isdisjoint(code_a)
    assert not any(r.get("reused_from") for r in g2)      # default games of epoch A are not reused in B1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _report_queue(tmp_path):
    rows = [_row("screen_row", area="ports", tunable="openings.policy", values=["pips_diversity", "setup_pick"],
                 interpreter="py330", opponent="value", mechanism={"metric": "setup_distinct", "direction": "+"}),
            _row("ko_row", area="robber", polarity="knockout", tier="t2", tunable="danger.danger_multiplier",
                 flag_off=True, values=None),
            _row("ko_row2", area="robber", polarity="knockout", tier="t2", tunable="danger.BLOCK_NEED", values=[0]),
            _row("pol1", area="politics", polarity="knockout", tier="politics", tunable="politics.MAX_SLACK",
                 values=[0.0], trades="value", interpreter="py330", opponent="value"),
            _row("pol2", area="politics", polarity="knockout", tier="politics", tunable="coalitions.SCALE",
                 values=[2.0], trades="value", interpreter="py330", opponent="value"),
            {"name": "leader", "kind": "human", "area": "ports", "tier": "politics", "priority": 3,
             "tunable": "winpaths.SPOT_LEADER"}]
    q = _q(tmp_path, rows, no_intake=True)
    stats = {"n": 800, "plus": 150, "minus": 90, "diverged": 600, "vp_delta": 0.21, "vp_se": 0.06, "p_def": 0.62}
    mech = {"setup_distinct": {"diff": 0.41, "diff_se": 0.05, "cand": 4.2, "def": 3.8, "n_pairs": 800}}
    _verdict(q, "screen_row", "pips_diversity", "ADOPT", delta=0.075, se=0.019, p=4e-5, look=2, pairs=800,
             flags=["stopped early"], stats=stats, mech=mech)
    _verdict(q, "screen_row", "setup_pick", "SHELVE", label="SHELVE(no gain; conditional power < 0.1)",
             reason="no gain; conditional power < 0.1", delta=-0.004, se=0.02, p=0.6, look=2, pairs=800,
             flags=["stopped early", "promise not met"], stats=dict(stats, plus=100, minus=103))
    _verdict(q, "ko_row", "off", "KEEP", label="KEEP(proven)", reason="futile", delta=-0.03, se=0.008, p=0.99,
             look=3, pairs=600, stats=dict(stats, n=600))
    _verdict(q, "ko_row2", "0", "REMOVE", delta=0.02, se=0.005, p=0.0004, look=4, pairs=800, stats=stats)
    _verdict(q, "pol1", "0", "SCREENED", label="SCREENED(fixed N)", delta=0.012, se=0.012, p=0.32, pairs=1000,
             stats=dict(stats, n=1000))
    return q


def test_queue_md_labels_units_base_rate_relative_and_golden(tmp_path):
    q = _report_queue(tmp_path)
    q.refresh()
    text = q.report_markdown(when="2026-09-26 12:00:00")
    lines = text.splitlines()
    row = next(l for l in lines if "| screen_row | pips_diversity |" in l)
    assert "ADOPT [stopped early]" in row and "+7.5 +- 1.9" in row and "pp (1v3 win rate vs value)" in row
    assert "62.0% / +12%" in row and "setup_distinct +0.41+-0.05" in row
    assert "league gate (--formats 4p2v2,3p1v2; power 0.38 at a 0.53 share, 0.86 at 0.55; ~4 h)" in row
    ko = next(l for l in lines if "| ko_row | off |" in l)
    assert "KEEP(proven)" in ko and "keep the term" in ko
    assert "REMOVE" in next(l for l in lines if "| ko_row2 | 0 |" in l)
    # politics: provisional while pol2 is open
    pol = next(l for l in lines if "| pol1 | 0 |" in l)
    assert "INCONCLUSIVE (provisional: tier incomplete)" in pol and "*" in pol
    # bundle proposal per area and the deferred table
    assert "- **ports**: screen_row=pips_diversity" in text
    assert 'tune=openings.policy:pips_diversity" --formats 4p2v2,3p1v2' in text
    assert "| winpaths.SPOT_LEADER | none (no bot harness reacts) | n/a | deferred |" in text
    with open(os.path.join(DATA, "queue_report_golden.md")) as fh:
        golden = fh.read()
    assert text.replace(str(tmp_path), "<tmp>").replace(q.plan.sha, "<sha>") == golden


def write_report_golden(tmp_path):
    q = _report_queue(tmp_path)
    q.refresh()
    with open(os.path.join(DATA, "queue_report_golden.md"), "w") as fh:
        fh.write(q.report_markdown(when="2026-09-26 12:00:00").replace(str(tmp_path), "<tmp>").replace(q.plan.sha,
                                                                                                        "<sha>"))


def test_politics_tier_family_holm_provisional_then_final(tmp_path):
    q = _report_queue(tmp_path)
    q.refresh()
    adj, complete = q.holm("politics")
    assert not complete and adj == {"pol1|0": pytest.approx(0.32)}
    _verdict(q, "pol2", "2", "SCREENED", label="SCREENED(fixed N)", delta=0.05, se=0.012, p=0.0001, pairs=1000,
             stats={"n": 1000})
    q.refresh()
    adj, complete = q.holm("politics")
    assert complete and adj["pol2|2"] == pytest.approx(0.0002) and adj["pol1|0"] == pytest.approx(0.32)
    text = q.report_markdown(when="x")
    assert "SIGNIFICANT: may continue" in next(l for l in text.splitlines() if "| pol2 | 2 |" in l)
    pol1 = next(l for l in text.splitlines() if "| pol1 | 0 |" in l)
    assert "INCONCLUSIVE: no more games" in pol1 and "provisional" not in pol1
    deferred = text.split("## Deferred to human testing")[1]
    assert "| politics.MAX_SLACK=0 | pol1 (1000 pairs) | +1.2 +- 1.2 pp, Holm p 0.32 | INCONCLUSIVE |" in deferred


def test_summary_md_unchanged_without_ledger(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    exps = [{"name": "x@vf", "interpreter": "py321", "opponent": "vf", "tunable": "danger.TURNS_HALF", "values": [2],
             "seeds": {"count": 10, "base": 0}, "priority": 1, "_order": 0}]
    text = CAMP.summary_markdown(exps, str(d), "p.json")
    header = next(l for l in text.splitlines() if l.startswith("| prio"))
    assert header == ("| prio | experiment | candidate | default | opponent | engine | pairs / planned | cand win % | "
                      "def win % | delta (pp) +- s.e. | 95% CI (pp) | Holm p | dVP +- s.e. | ms/dec c / d | "
                      "opp ms c / d | ident | verdict | status |")
    assert "queue verdict" not in text and text.splitlines()[header and text.splitlines().index(header) + 1] == \
        "|" + "---|" * 18
    (d / "ledger.jsonl").write_text(json.dumps({"kind": "verdict", "run_key": "k", "verdict": "ADOPT"}) + "\n")
    text2 = CAMP.summary_markdown(exps, str(d), "p.json")
    assert "| status | queue verdict |" in text2


def test_ledger_rebuildable_from_stop_records(tmp_path):
    q = _q(tmp_path, [_row("r", extra_args=["--fake-games", "0.3", "--fake-discordance", "0.4"],
                           seeds={"count": 400, "base": 0}, d_prior=0.4)], no_intake=True, slice_minutes=5)
    for _ in range(6):
        q.loop(once=True)
        if q.ledger.verdict("r", "2"):
            break
    v = q.ledger.verdict("r", "2")
    assert v and v["verdict"] in ("ADOPT", "SHELVE", "REJECT")
    os.remove(q.ledger.path)
    q2 = _q(tmp_path, [_row("r", extra_args=["--fake-games", "0.3"], seeds={"count": 400, "base": 0})],
            no_intake=True)
    assert RQ.rebuild_ledger(q2) == 1
    v2 = q2.ledger.verdict("r", "2")
    assert (v2["verdict"], v2["label"], v2["pairs"], v2["look"]) == (v["verdict"], v["label"], v["pairs"], v["look"])


# ---------------------------------------------------------------------------
# end to end on synthetic games
# ---------------------------------------------------------------------------
def test_queue_end_to_end_fake_games(tmp_path):
    """Five rows across three areas: a mid-run plan edit inserts a higher-area row (it preempts at the slice
    boundary), a SHELVEd parent unlocks its simpler child on a fresh seed block, and an ADOPT with a Holm p < 0.05
    unlocks its confirmation."""
    rows = [
        _row("good", area="robber", priority=1, extra_args=["--fake-games", "0.15", "--fake-discordance", "0.3"],
             seeds={"count": 400, "base": 0}, promise_pp=15),
        _row("conf", area="robber", tier="t3", priority=2, confirms="good", opponent="vf",
             extra_args=["--fake-games", "0.15", "--fake-discordance", "0.3"], seeds={"count": 100, "base": 100000}),
        _row("par", area="other", priority=3, tunable="danger.BLOCK_NEED", values=[0],
             extra_args=["--fake-games", "0.0", "--fake-discordance", "0.4"], seeds={"count": 400, "base": 0}),
        _row("kid", area="other", priority=4, tunable="danger.BLOCK_FLOOR", values=[0.2], parent="par",
             fallback="simpler", tier="t2", extra_args=["--fake-games", "0.0", "--fake-discordance", "0.4"],
             seeds={"count": 400, "base": 0}),
        _row("aa", area="harness", polarity="measure", tier="harness", priority=5, tunable=None, values=None,
             cand_spec=SPEC, def_spec=SPEC, extra_args=["--fake-games", "0.0"], seeds={"count": 400, "base": 0}),
    ]
    # synthetic games cost ~0 CPU: a fake counter (x1000) makes the measured cost per game realistic (~5 s), so
    # the exchange rule sees a costly incumbent
    q = _q(tmp_path, rows, no_intake=True, slice_minutes=5, cpu=lambda: 1000.0 * RQ.children_cpu())
    order = []
    real = q.runner

    def runner(cmd, cwd, env, log, hard):
        order.append(cmd[cmd.index("--exp-name") + 1])
        return real(cmd, cwd, env, log, hard)

    q.runner = runner
    q.loop(max_slices=2)
    assert order[:2] == ["aa", "good"]
    # a plan edit inserts a trades row (above robber): it runs at the next slice boundary
    doc = json.loads(open(q.plan_path).read())
    doc["experiments"].append(_row("newtrade", area="trades", tier="t2", priority=1, tunable="search.dump_candidates",
                                   values=[0], polarity="knockout",
                                   extra_args=["--fake-games", "0.0", "--fake-discordance", "0.4"],
                                   seeds={"count": 200, "base": 0}))
    open(q.plan_path, "w").write(json.dumps(doc))
    os.utime(q.plan_path, (time.time() + 5, time.time() + 5))
    q.loop(max_slices=1)
    assert order[2] == "newtrade"
    q.loop(max_slices=60)
    v = {n: {c.key: (c.verdict or {}).get("label") for c in q.rows[n].cands} for n in q.rows}
    assert v["aa"]["cand"].startswith("PASS")
    assert v["good"]["2"] == "ADOPT"
    assert v["par"]["0"].startswith(("SHELVE", "REJECT"))
    assert "kid" in order and v["kid"]["0.2"] is not None
    kid_games = [r for r in AB.read_jsonl(q.jsonl(q.plan.row("kid")))[0] if r.get("kind") == "game"]
    assert min(r["s"] for r in kid_games) == 10 ** 6          # a fresh seed block
    assert "conf" in order and v["conf"]["2"].startswith(("CONFIRMED", "NOT CONFIRMED"))
    rep = open(os.path.join(q.dir, "QUEUE.md")).read()
    assert "| robber | good | 2 |" in rep and "confirmation row conf" in rep


def test_cv_rows_use_the_pool_and_pool_mismatch_fails_row(tmp_path):
    pool = {"name": "pool", "kind": "pool", "area": "harness", "priority": 9, "interpreter": "py321", "opponent": "vf",
            "cand_spec": SPEC, "def_spec": SPEC, "seeds": {"count": 8000, "base": 10 ** 7}}
    cvrow = _row("cv", estimator="cv", seeds={"count": 400, "base": 0}, d_prior=0.4,
                 extra_args=["--fake-games", "0.2", "--fake-discordance", "0.3"])
    q = _q(tmp_path, [cvrow, pool], no_intake=True, slice_minutes=5)
    q.ledger.append({"kind": "pool", "row": "pool", "def_key": "x", "epoch": "live", "m": 8000, "mean": 0.25})
    for _ in range(6):
        q.loop(once=True)
        if q.ledger.verdict("cv", "2"):
            break
    v = q.ledger.verdict("cv", "2")
    assert v and "cv" in (v.get("flags") or [])
    # a stale pool (mean far from the row's default prefix) fails the row
    shutil.rmtree(tmp_path / "run")
    q = _q(tmp_path, [cvrow, pool], no_intake=True, slice_minutes=5)
    q.ledger.append({"kind": "pool", "row": "pool", "def_key": "x", "epoch": "live", "m": 8000, "mean": 0.60})
    for _ in range(3):
        q.loop(once=True)
        if q.ledger.verdict("cv", "2"):
            break
    assert q.ledger.verdict("cv", "2")["label"] == "FAILED(pool-mismatch)"


def test_default_only_flag_plays_no_candidate_games(tmp_path):
    out = tmp_path / "pool.jsonl"
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)
    proc = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ablate_catanatron.py"), "--cand-spec", SPEC,
                           "--def-spec", SPEC, "--opponent", "vf", "--seeds", "12", "--seed-base", str(10 ** 7),
                           "--workers", "1", "--out", str(out), "--fake-games", "0.0", "--default-only", "--quiet"],
                          env=env, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-1500:]
    games = [r for r in AB.read_jsonl(str(out))[0] if r.get("kind") == "game"]
    assert len(games) == 12 and all(g["arm"] == "def" for g in games)
    assert "0 seed(s) still need games" in proc.stdout
