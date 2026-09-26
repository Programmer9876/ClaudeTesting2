"""Tests for scripts/seqtest.py, the test queue's sequential verdict engine (docs/QUEUE.md).

Operating characteristics are checked by simulation against the numbers published in the design
(docs/designs/priority_areas_2026-09-26.json, infra F1/F2/F7); the per-look rules on hand-built look
statistics; the JSONL integration (stop records honoured by ablate_catanatron, the new synthetic-game hooks)
through the real script with synthetic games.
"""
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SEQ = _load("seqtest_under_test", os.path.join(SCRIPTS, "seqtest.py"))
AB = _load("ablate_catanatron_for_seqtest_tests", os.path.join(SCRIPTS, "ablate_catanatron.py"))
ABLATE = os.path.join(SCRIPTS, "ablate_catanatron.py")


def _env():
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONHASHSEED", None)
    env.pop("CATANBOT_NO_ACCEL", None)
    return env


def _run(args, check=True):
    proc = subprocess.run([sys.executable, ABLATE] + args, env=_env(), capture_output=True, text=True, timeout=600,
                          cwd=ROOT)
    if check:
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc


def _se_rate(p, n):
    return math.sqrt(max(p * (1 - p), 1e-6) / n)


def _look(n, plus=0, minus=0, diverged=None, inconsistent=0, codes=("c1",), errors=0, ident_diff=0, known=None,
          vp=0.0, adapter_c=None, adapter_d=None, target=None):
    """Hand-built look statistics: ``plus`` candidate-only wins, ``minus`` default-only wins, the rest concordant."""
    st = SEQ.LookStats(n_target=target if target is not None else n + errors)
    for _ in range(plus):
        st.add(1.0, vp)
    for _ in range(minus):
        st.add(-1.0, -vp, True)
    for _ in range(n - plus - minus):
        st.add(0.0, 0.0)
    st.errors = errors
    st.known = n if known is None else known
    st.diverged = (plus + minus) if diverged is None else diverged
    st.identical = st.known - st.diverged
    st.inconsistent = inconsistent
    st.ident_diff_winner = ident_diff
    st.cand_codes = tuple(codes)
    st.adapter_cand = dict(adapter_c or {})
    st.adapter_def = dict(adapter_d or {})
    return st


# ---------------------------------------------------------------------------
# boundary tables
# ---------------------------------------------------------------------------
def test_obf_and_hsd_tables_match_published_and_general_routine():
    ts = [0.2, 0.4, 0.6, 0.8, 1.0]
    a = SEQ.boundaries_general(ts, 0.025, "obf")
    r = SEQ.boundaries_general(ts, 0.05, "hsd", -2.0)
    assert SEQ.ADOPT_TABLE == (4.877, 3.357, 2.680, 2.290, 2.031)
    assert SEQ.REJECT_TABLE == (2.66, 2.46, 2.24, 2.02, 1.79)
    for got, pub in zip(a, SEQ.ADOPT_TABLE):
        assert abs(got - pub) < 0.01, (a, SEQ.ADOPT_TABLE)
    for got, pub in zip(r, SEQ.REJECT_TABLE):
        assert abs(got - pub) < 0.01, (r, SEQ.REJECT_TABLE)
    # the crossing probabilities spend exactly the O'Brien-Fleming alpha
    cross = SEQ.crossing_probs(a, ts)
    assert abs(sum(cross) - 0.025) < 2e-4
    # early-SHELVE thresholds: conditional power 0.10 under the current trend
    assert [round(x, 2) for x in SEQ.SHELVE_TABLE if x is not None] == [0.66, 0.95, 1.30]
    for k, thr in ((2, 0.66), (3, 0.95), (4, 1.30)):
        assert abs(SEQ.conditional_power(SEQ.shelve_thresholds()[k - 1], k / 5) - 0.10) < 1e-3
    # stage-wise p: fixed p at one look, and at look 3 it adds the null crossing probability of looks 1-2
    assert abs(SEQ.stagewise_p(1, 1.96) - 0.025) < 1e-3
    p3 = SEQ.stagewise_p(3, 2.0)
    assert sum(cross[:2]) < p3 < sum(cross[:2]) + SEQ.norm_sf(2.0)


def test_look_schedule_multiples_of_four():
    assert SEQ.look_sizes(2000) == [400, 800, 1200, 1600, 2000]
    assert SEQ.look_sizes(1000) == [200, 400, 600, 800, 1000]
    assert SEQ.look_sizes(400) == [80, 160, 240, 320, 400]
    assert all(n % 4 == 0 for n in SEQ.look_sizes(1234)[:-1]) and SEQ.look_sizes(1234)[-1] == 1234
    assert SEQ.look_sizes(2400, multiple=480) == [480, 960, 1440, 1920, 2400]


# ---------------------------------------------------------------------------
# operating characteristics (simulation, 4,000 rows per cell)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("D", [0.40, 0.25, 0.04])
def test_oc_null_false_adopt_below_alpha_screen_and_knockout(D):
    for design in ("screen", "knockout"):
        r = SEQ.simulate(design, D, 0.0, 2000, rows=4000, seed=11)
        assert r["adopt"] <= 0.025 + 3 * _se_rate(0.025, 4000), (design, D, r)


@pytest.mark.parametrize("D", [0.40, 0.25, 0.04])
def test_oc_expected_pairs_at_null_below_055_nmax(D):
    for design in ("screen", "knockout"):
        r = SEQ.simulate(design, D, 0.0, 2000, rows=4000, seed=12)
        assert r["mean_pairs"] <= 0.55 * 2000, (design, D, r)


@pytest.mark.parametrize("D", [0.40, 0.25])
def test_false_reject_at_delta_min_below_gamma(D):
    # at the smallest effect of interest (+1 pp) the screen may call REJECT at most gamma = 0.05 of the time
    r = SEQ.simulate("screen", D, 0.01, 2000, rows=4000, seed=13)
    assert r["reject"] <= 0.05 + 3 * _se_rate(0.05, 4000), r


def test_published_null_numbers_and_knockout_behaviour():
    r = SEQ.simulate("screen", 0.40, 0.0, 2000, rows=20000, seed=5)
    assert 0.018 <= r["adopt"] <= 0.024 and 0.024 <= r["reject"] <= 0.032 and 950 <= r["mean_pairs"] <= 990
    ko = SEQ.simulate("knockout", 0.40, 0.0, 2000, rows=20000, seed=5)
    assert 0.018 <= ko["adopt"] <= 0.024 and 0.007 <= ko["reject"] <= 0.013
    # a useful term (candidate -2 pp) is almost never removed, and stops early
    kd = SEQ.simulate("knockout", 0.40, -0.02, 2000, rows=4000, seed=6)
    assert kd["adopt"] <= 0.003 and kd["mean_pairs"] < 900
    k04 = SEQ.simulate("knockout", 0.04, -0.02, 2000, rows=4000, seed=6)
    assert 0.63 <= k04["reject"] <= 0.72 and k04["mean_pairs"] < 760


def test_power_grid_matches_simulation():
    for D, row in SEQ.POWER_TABLE.items():
        for pp, pub in row.items():
            got = SEQ.power_at(pp, D, 2000)
            assert abs(got - pub) <= 0.03, (D, pp, got, pub)
    assert SEQ.power_at(3, 0.40) < SEQ.UNPROVEN_POWER < SEQ.power_at(4, 0.40)


# ---------------------------------------------------------------------------
# per-look rules
# ---------------------------------------------------------------------------
def test_screen_adopt_reject_shelve_and_labels():
    v = SEQ.decide(_look(400, plus=90, minus=40), "screen", 1, 2000)        # z ~ 4.4 < 4.877
    assert v.status == "continue"
    v = SEQ.decide(_look(400, plus=110, minus=30), "screen", 1, 2000)       # z ~ 7
    assert v.final and v.verdict == "ADOPT" and v.label == "ADOPT" and "stopped early" in v.flags
    v = SEQ.decide(_look(800, plus=100, minus=160), "screen", 2, 2000)      # clearly worse
    assert v.final and v.label == "REJECT(worse)"
    v = SEQ.decide(_look(800, plus=5, minus=5, known=800), "screen", 2, 2000)    # D 0.0125: futile
    assert v.final and v.label == "REJECT(futile)"
    v = SEQ.decide(_look(800, plus=150, minus=150), "screen", 2, 2000)      # z 0 < 0.66 at look 2
    assert v.final and v.verdict == "SHELVE" and v.reason.startswith("no gain")
    v = SEQ.decide(_look(2000, plus=410, minus=380), "screen", 5, 2000)     # cap, small positive
    assert v.final and v.verdict == "SHELVE" and v.reason.startswith("too small to prove") \
        and "stopped early" not in v.flags


def test_knockout_labels_remove_keep_never_parent():
    v = SEQ.decide(_look(400, plus=110, minus=30), "knockout", 1, 2000)
    assert v.label == "REMOVE" and v.verdict == "REMOVE"
    v = SEQ.decide(_look(800, plus=100, minus=160), "knockout", 2, 2000)
    assert v.label == "KEEP(proven)"
    v = SEQ.decide(_look(2000, plus=400, minus=400), "knockout", 5, 2000)
    assert v.label == "KEEP(unproven)"
    RQ = _load("run_queue_for_seqtest_tests", os.path.join(SCRIPTS, "run_queue.py"))
    plan = RQ.Plan("x", [{"name": "ko", "kind": "catanatron", "polarity": "knockout", "enabled": True, "priority": 1,
                          "seeds": {"count": 1000, "base": 0}, "_order": 0},
                         {"name": "child", "kind": "catanatron", "parent": "ko", "fallback": "simpler", "enabled": True,
                          "polarity": "new", "priority": 2, "seeds": {"count": 1000, "base": 0}, "_order": 1}],
                   {}, {}, "sha")
    errs, _ = RQ.lint(plan, plan.rows[1])
    assert any("never a parent" in e for e in errs)


def test_estimate_design_fixed_n_and_aa_nondeterminism_failed():
    v = SEQ.decide(_look(2000, plus=300, minus=250), "estimate", 1, 2000)
    assert v.final and v.verdict == "ESTIMATE" and abs(v.delta - 0.025) < 1e-9 and v.p is not None
    v = SEQ.decide(_look(400, plus=0, minus=0, known=400), "estimate", 1, 400, aa=True)
    assert v.verdict == "PASS"
    v = SEQ.decide(_look(400, plus=0, minus=0, diverged=1, known=400), "estimate", 1, 400, aa=True)
    assert v.verdict == "FAILED" and v.label == "FAILED(nondeterminism)"


def test_politics_design_single_look_no_interim_stop_alert_on_zero_divergence():
    assert SEQ.DESIGNS["politics"].looks == 1
    RQ = _load("run_queue_for_seqtest_tests", os.path.join(SCRIPTS, "run_queue.py"))
    e = {"kind": "catanatron", "tunable": "politics.MAX_SLACK", "values": [0.0], "trades": "value"}
    assert RQ.design_of(e) == "politics"
    assert RQ.looks_for(e, "politics", 1000) == [400, 1000]      # the 400 look only raises an ALERT
    st = _look(400, known=400)
    assert SEQ.politics_alert(st) and "ALERT" in SEQ.politics_alert(st)
    assert SEQ.politics_alert(_look(400, plus=3, minus=2)) is None
    # a huge effect at the final look is still just SCREENED (label from Holm in the report), fixed two-sided p
    v = SEQ.decide(_look(1000, plus=200, minus=100), "politics", 1, 1000)
    assert v.verdict == "SCREENED" and abs(v.p - 2 * SEQ.norm_sf(abs(v.z))) < 1e-9


def test_noop_vs_failed_inert_harness_from_adapter_counters():
    st = _look(400, known=400)                     # 0 diverged of 400: CP upper 0.0092 < 0.01
    v = SEQ.decide(st, "screen", 1, 2000)
    assert v.verdict == "NOOP"
    v = SEQ.decide(_look(400, known=400, adapter_c={"offers": 0}, adapter_d={"offers_received": 0}),
                   "screen", 1, 2000, needs="trades")
    assert v.label == "FAILED(inert-harness)"
    v = SEQ.decide(_look(400, known=400, adapter_c={"offers": 12}), "screen", 1, 2000, needs="trades")
    assert v.verdict == "NOOP"
    v = SEQ.decide(_look(400, known=400, adapter_c={"info_samples": 0}), "screen", 1, 2000, needs="counting")
    assert v.label == "FAILED(inert-harness)"
    # 200 pairs are not enough to call NOOP (CP upper 0.018)
    assert SEQ.decide(_look(200, known=200), "screen", 1, 1000).status == "continue" or \
        SEQ.decide(_look(200, known=200), "screen", 1, 1000).verdict != "NOOP"
    assert abs(SEQ.clopper_pearson_upper(0, 400) - 0.0092) < 2e-4


def test_failed_inconsistent_needs_min_two_pairs():
    assert SEQ.decide(_look(400, plus=50, minus=50, inconsistent=2), "screen", 1, 2000).status == "continue"
    v = SEQ.decide(_look(400, plus=50, minus=50, inconsistent=3), "screen", 1, 2000)
    assert v.label == "FAILED(inconsistent)"
    # 2% of 300 diverged = 6 inconsistent allowed
    assert SEQ.decide(_look(400, plus=50, minus=50, diverged=300, inconsistent=6), "screen", 1, 2000).verdict \
        != "FAILED"


def test_failed_code_mixed_candidate_records():
    v = SEQ.decide(_look(400, plus=50, minus=50, codes=("aaa", "bbb")), "screen", 1, 2000)
    assert v.label == "FAILED(code-mixed)"


def test_identical_trace_different_winner_failed():
    v = SEQ.decide(_look(400, plus=50, minus=50, ident_diff=1), "screen", 1, 2000)
    assert v.label == "FAILED(identical-trace-different-winner)"


def test_errors_above_one_percent_failed():
    v = SEQ.decide(_look(400, plus=50, minus=50, errors=5, target=405), "screen", 1, 2000)
    assert v.label == "FAILED(errors)"
    assert SEQ.decide(_look(400, plus=50, minus=50, errors=4, target=404), "screen", 1, 2000).verdict != "FAILED"


def test_exact_mcnemar_crosscheck_on_few_discordant():
    # look 2 (800 pairs): 14 candidate-only vs 1 default-only gives z = 3.38 >= 3.357, but the exact binomial
    # one-sided p of 14/15 (4.9e-4) is above 1 - Phi(3.357) = 3.9e-4: no ADOPT
    st = _look(800, plus=14, minus=1)
    z = st.delta / st.se
    assert z >= SEQ.ADOPT_TABLE[1]
    assert SEQ.mcnemar_p(14, 1) > SEQ.norm_sf(SEQ.ADOPT_TABLE[1])
    assert SEQ.decide(st, "screen", 2, 2000).verdict != "ADOPT"
    # 15 vs 0 passes both
    assert SEQ.decide(_look(800, plus=15, minus=0), "screen", 2, 2000).verdict == "ADOPT"
    # with 60 discordant pairs the exact check no longer applies
    st2 = _look(400, plus=58, minus=2)
    assert SEQ.decide(st2, "screen", 1, 2000).verdict == "ADOPT"


def test_promise_not_met_and_small_flag():
    v = SEQ.decide(_look(2000, plus=400, minus=400), "screen", 5, 2000, promise_pp=4)
    assert "promise not met" in v.flags                          # upper bound ~ +2.8 pp < 4 pp
    v = SEQ.decide(_look(2000, plus=440, minus=400), "screen", 5, 2000, promise_pp=1)
    assert "promise not met" not in v.flags
    v = SEQ.decide(_look(400, plus=110, minus=30), "screen", 1, 2000, promise_pp=50)
    assert v.verdict == "ADOPT" and "small" in v.flags
    v = SEQ.decide(_look(400, plus=110, minus=30), "screen", 1, 2000, promise_pp=4)
    assert "small" not in v.flags


def test_look1_recheck_uses_observed_discordance():
    calls = []

    def power(pp, D, n):
        calls.append((pp, round(D, 3), n))
        return 0.2 if D > 0.3 else 0.9

    v = SEQ.decide(_look(400, plus=90, minus=90), "screen", 1, 2000, promise_pp=2, power_fn=power)
    assert calls == [(2, 0.45, 2000)] and v.verdict == "SHELVE" and "unprovable at promise" in v.reason
    v = SEQ.decide(_look(400, plus=20, minus=20), "screen", 1, 2000, promise_pp=2, power_fn=power)
    assert v.status == "continue"


def test_only_seed_prefix_and_ok_pairs_count(tmp_path):
    out = tmp_path / "p.jsonl"
    _run(["--tunable", "danger.TURNS_HALF", "--values", "2", "--opponent", "vf", "--seeds", "40", "--workers", "1",
          "--out", str(out), "--fake-games", "0.2", "--fake-crash-seeds", "3", "--quiet"])
    index, _ = AB.load_index(str(out))
    run = next(iter(index.runs.values()))
    st = SEQ.look_units(index, run, 0, 20, AB)
    assert st.complete and st.ok == 19 and st.errors == 1       # seed 3's candidate crashed: counted, excluded
    st = SEQ.look_units(index, run, 0, 60, AB)
    assert not st.complete and st.incomplete == 20              # only the first 40 seeds exist
    st = SEQ.look_units(index, run, 10, 10, AB)
    assert st.complete and st.ok == 10                           # a prefix of another base


def test_stop_record_written_once_and_progress_complete(tmp_path):
    camp = _load("campaign_for_seqtest_tests", os.path.join(SCRIPTS, "campaign.py"))
    out = tmp_path / "x@vf.jsonl"
    _run(["--tunable", "danger.TURNS_HALF", "--values", "2,4.5", "--opponent", "vf", "--seeds", "40", "--workers", "1",
          "--out", str(out), "--fake-games", "0.2", "--exp-name", "x@vf", "--quiet"])
    index, _ = AB.load_index(str(out))
    run = next(r for r in index.runs.values() if r["value_text"] == "2")
    v = SEQ.decide(_look(400, plus=110, minus=30), "screen", 1, 2000)
    rec = SEQ.stop_record(v, run["run_key"], "x@vf", "2")
    AB.JsonlWriter(str(out)).append([rec])
    AB.JsonlWriter(str(out)).append([dict(rec, verdict="SHELVE", label="SHELVE")])   # a second one never wins
    index, _ = AB.load_index(str(out))
    assert index.queue_stops[run["run_key"]]["verdict"] == "ADOPT"
    exps, _ = camp.load_plan(_write_plan(tmp_path, [{"name": "x@vf", "interpreter": "py321", "opponent": "vf",
                                                      "tunable": "danger.TURNS_HALF", "values": [2, 4.5],
                                                      "seeds": 400}]), 2)
    prog = camp.progress(exps[0], str(tmp_path))
    by = {x["run"]["value_text"]: x for x in prog["per_run"]}
    assert by["2"]["stopped"] and not by["4.5"]["stopped"]


def _write_plan(tmp_path, exps):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"experiments": exps}))
    return str(p)


def test_ablate_campaign_honours_queue_stop_records_only(tmp_path):
    out = tmp_path / "h.jsonl"
    args = ["--tunable", "danger.TURNS_HALF", "--values", "2,4.5", "--opponent", "vf", "--workers", "1",
            "--out", str(out), "--fake-games", "0.2", "--quiet"]
    _run(args + ["--seeds", "20"])
    index, _ = AB.load_index(str(out))
    r2 = next(r for r in index.runs.values() if r["value_text"] == "2")
    r45 = next(r for r in index.runs.values() if r["value_text"] == "4.5")
    # a legacy stop record (no source) is NOT honoured without --stop-at-se ...
    AB.JsonlWriter(str(out)).append([{"v": 1, "kind": "stop", "run_key": r45["run_key"], "pairs": 20, "delta": 0.1,
                                      "se": 0.05, "stop_se": 0.1, "t": 0}])
    # ... a queue stop record is
    AB.JsonlWriter(str(out)).append([{"v": 1, "kind": "stop", "run_key": r2["run_key"], "pairs": 20,
                                      "verdict": "SHELVE", "label": "SHELVE(no gain)", "source": "queue", "t": 0}])
    proc = _run(args + ["--seeds", "30"])
    assert "stopped by the test queue" in proc.stderr
    recs, _ = AB.read_jsonl(str(out))
    games = [r for r in recs if r.get("kind") == "game"]
    c2 = [g for g in games if g.get("arm_key") == r2["cand_key"]]
    c45 = [g for g in games if g.get("arm_key") == r45["cand_key"]]
    assert len(c2) == 20 and len(c45) == 30
    assert "STOPPED by the test queue" in _run(["--report", "--out", str(out)]).stdout


def test_legacy_design_reproduces_should_stop():
    st = {"pairs": 500, "se": 0.01, "delta": 0.05}
    assert AB.should_stop(st, 0.015, 100)
    v = SEQ.decide(_look(500, plus=100, minus=75), "legacy", 1, 500)
    assert v.final and v.verdict == "LEGACY" and v.design == "legacy"
    RQ = _load("run_queue_for_seqtest_tests", os.path.join(SCRIPTS, "run_queue.py"))
    # the queue passes --stop-at-se through only for the legacy design
    assert RQ.design_of({"kind": "catanatron", "design": "legacy"}) == "legacy"


def _orig_fake_body(s, effect, cand):
    """The synthetic game body of ablate_catanatron.py before the test-queue hooks (copied): must not change."""
    u = random.Random(f"fake-{s}").random()
    p = 0.25 + (effect if cand else 0.0)
    won = u < p
    seat = s % 4
    our_vp = 10 if won else 2 + int(u * 7)
    opp = [10 if (not won and i == 0) else 3 + (s + i) % 5 for i in range(3)]
    vps = opp[:seat] + [our_vp] + opp[seat:]
    differs = cand and (0.25 <= u < p)
    base = [f"a{s}-{i}" for i in range(60)]
    items = base[:40] + ([f"c{s}-{i}" for i in range(20)] if differs else base[40:])
    ours = [[10 * k + 3, AB.short_hash(f"{s}-{k}")] for k in range(6)]
    if differs:
        ours[4] = [43, AB.short_hash(f"{s}-cand")]
    return {"won": won, "our_vp": our_vp, "vps": vps, "trace": AB.make_trace(items), "ours": ours}


def test_new_fake_hooks_and_existing_fake_games_unchanged(tmp_path):
    ctx = {"fake": {"effect": 0.2, "crash": [], "die": [], "sleep": 0}}
    for s in range(12):
        for role in ("cand", "def"):
            body = AB._fake_game({"ctx": ctx}, {"role": role}, s)
            exp = _orig_fake_body(s, 0.2, role == "cand")
            assert {k: body[k] for k in exp} == exp and "mech" not in body
    # arm keys of a context without the new options are the old formula
    arm = {"role": "cand", "spec": "search:x", "overrides": {"a": 1}, "adapter": {"trades": "off"}}
    c = {"opponent": "vf", "opponent_params": {}, "python": "3.11", "catanatron": "3.2.1", "evaluator": "c++",
         "vps_to_win": 10, "discard_limit": 7, "hashseed": "0", "fake": None}
    old = hashlib.sha1(json.dumps({"role": "cand", "spec": "search:x", "overrides": {"a": 1},
                                   "adapter": {"trades": "off"},
                                   "ctx": {k: c[k] for k in ("opponent", "opponent_params", "python", "catanatron",
                                                             "evaluator", "vps_to_win", "discard_limit", "hashseed")}},
                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert AB.arm_key(arm, c) == old == AB.arm_key(arm, dict(c, mech=True))
    assert AB.arm_key(arm, dict(c, crn="dice")) != old
    # --fake-discordance: negative effects give differing traces; the discordant share is near D
    out = tmp_path / "n.jsonl"
    _run(["--tunable", "danger.TURNS_HALF", "--values", "2", "--opponent", "vf", "--seeds", "400", "--workers", "2",
          "--out", str(out), "--fake-games", "-0.05", "--fake-discordance", "0.3", "--quiet"])
    index, _ = AB.load_index(str(out))
    run = next(iter(index.runs.values()))
    st = SEQ.look_units(index, run, 0, 400, AB)
    assert st.ident_diff_winner == 0 and st.inconsistent == 0
    assert abs(st.discordant / st.ok - 0.3) < 0.07 and st.delta < 0
    v = SEQ.decide(st, "screen", 5, 400)
    assert v.verdict != "FAILED"


# ---------------------------------------------------------------------------
# control variate (F7, tooling only)
# ---------------------------------------------------------------------------
def test_cv_variance_ratio_matches_formula():
    assert abs(SEQ.cv_variance_ratio(0.40, 0.644, 2000, 8000) - 0.673) < 0.002
    assert abs(SEQ.cv_variance_ratio(0.40, 0.644, 2000, 4000) - 0.782) < 0.002
    assert abs(SEQ.cv_variance_ratio(0.25, 0.644, 2000, 8000) - 0.796) < 0.002
    assert abs(SEQ.cv_variance_ratio(0.40, 0.254, 2000, 8000) - 0.604) < 0.002
    # by simulation: the variance of theta over rows, relative to the paired estimate's
    import numpy as np
    rng = np.random.default_rng(3)
    rows, n, m, D, p = 1500, 2000, 8000, 0.40, 0.644
    a = D / (2 * (1 - p))
    b = D / (2 * p)
    th, xb = [], []
    for _ in range(rows):
        d = (rng.random(n) < p).astype(float)
        u = rng.random(n)
        c = np.where(d == 1, (u >= b), (u < a)).astype(float)
        x = c - d
        pool = rng.binomial(m, p) / m
        r = SEQ.cv_stat(list(x), list(d), pool, m)
        th.append(r["theta"])
        xb.append(x.mean())
    ratio = np.var(th) / np.var(xb)
    assert abs(ratio - SEQ.cv_variance_ratio(D, p, n, m)) < 0.08, ratio


def test_cv_sequential_type1_below_alpha_grid():
    for D, p, m in ((0.40, 0.644, 8000), (0.25, 0.25, 4000), (0.04, 0.644, 8000)):
        r = SEQ.simulate_cv(D, 0.0, p, m, 2000, rows=2000, seed=21)
        assert r["adopt"] <= 0.025 + 3 * _se_rate(0.025, 2000), (D, p, m, r)
    gain_cv = SEQ.simulate_cv(0.40, 0.03, 0.644, 8000, 2000, rows=2000, seed=22)["adopt"]
    gain_paired = SEQ.simulate_cv(0.40, 0.03, 0.644, 8000, 2000, rows=2000, seed=22, paired=True)["adopt"]
    assert gain_cv > gain_paired + 0.08


def test_cv_information_fractions_use_general_boundaries():
    ts, bounds = SEQ.cv_design_boundaries(0.40, 0.644, 2000, 8000)
    assert ts[-1] == 1.0 and all(a < b for a, b in zip(ts, ts[1:]))
    assert ts[0] > 0.2                                  # the pool term makes early looks carry more information
    assert bounds == pytest.approx(SEQ.boundaries_general(ts, 0.025, "obf"))
    assert bounds[0] < SEQ.ADOPT_TABLE[0]
    assert SEQ.cv_information_fractions([4.0, 2.0, 1.0], 1.0) == [0.25, 0.5, 1.0]


def test_pool_mismatch_detection():
    assert SEQ.cv_pool_mismatch(0.70, 1000, 0.64, 8000)
    assert not SEQ.cv_pool_mismatch(0.65, 1000, 0.64, 8000)


def test_expected_remaining_fixed_and_sequential():
    assert SEQ.expected_remaining("estimate", 100, 400) == 300
    assert SEQ.expected_remaining("screen", 0, 2000) == pytest.approx(980)
    far = SEQ.expected_remaining("screen", 1600, 2000, z=1.5, look_done=4)
    assert 0 < far <= 400
    assert SEQ.expected_remaining("screen", 400, 2000, z=4.0, look_done=1) < \
        SEQ.expected_remaining("screen", 400, 2000, z=0.8, look_done=1)


def test_decide_cv_uses_pool_and_predeclared_boundaries():
    import random as _r
    rng = _r.Random(4)
    n = 800
    d = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(n)]
    x = []
    for di in d:           # the candidate wins 12 pp more often when the default lost
        u = rng.random()
        x.append(0.0 if di else (1.0 if u < 0.40 else 0.0))
    st = _look(n, plus=int(sum(1 for v in x if v > 0)), minus=0, diverged=int(sum(1 for v in x if v > 0)))
    v = SEQ.decide_cv(st, x, d, 0.30, 8000, "screen", 2, 2000, 0.40)
    assert v.verdict == "ADOPT" and "cv" in v.flags and v.z > SEQ.cv_design_boundaries(0.40, 0.30, 2000, 8000)[1][1]
    # FAILED / NOOP pass through from the paired engine
    bad = _look(n, plus=10, minus=10, codes=("a", "b"))
    assert SEQ.decide_cv(bad, x, d, 0.30, 8000, "screen", 2, 2000, 0.40).label == "FAILED(code-mixed)"
    # no effect: continue at look 1
    x0 = [0.0] * 400
    d0 = [1.0 if i % 3 == 0 else 0.0 for i in range(400)]
    st0 = _look(400, known=400, diverged=40)
    assert SEQ.decide_cv(st0, x0, d0, 0.33, 8000, "screen", 1, 2000, 0.40).status == "continue"
