"""Tests of the strength-proof tooling: scripts/prove_strength.py (statistics, Holm, claim logic),
scripts/bench_catanatron.py (2v2 rotation, game-range chunks, crash re-runs, action logs) and
scripts/replay_catanatron.py (exact replays).  The statistics tests are pure Python (and compare
against scipy when it is installed); the bench / replay tests play a few short real games on
whichever catanatron the interpreter has (run the file with both interpreters)."""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PS = _load("prove_strength_under_test", "prove_strength.py")
F = Fraction


# ---------------------------------------------------------------------------
# Exact binomial statistics
# ---------------------------------------------------------------------------
def test_binomial_tails_known_values():
    assert PS.binom_sf_exact(8, 10, F(1, 2)) == F(56, 1024)          # C(10,8)+C(10,9)+C(10,10) = 56
    assert PS.binom_sf_exact(0, 10, F(1, 4)) == 1 and PS.binom_sf_exact(11, 10, F(1, 4)) == 0
    assert PS.binom_sf_exact(10, 10, F(1, 4)) == F(1, 4 ** 10)
    assert PS.binom_cdf_exact(2, 10, F(1, 2)) == F(1 + 10 + 45, 1024)
    for n, p in ((10, F(1, 4)), (37, F(1, 2)), (250, F(1, 4))):
        assert sum(PS.binom_pmf_exact(i, n, p) for i in range(n + 1)) == 1
        for k in (0, 1, n // 3, n // 2, n - 1, n):
            assert PS.binom_cdf_exact(k - 1, n, p) + PS.binom_sf_exact(k, n, p) == 1
    # float agreement with an independent direct sum
    direct = math.fsum(math.comb(1000, i) * 0.25 ** i * 0.75 ** (1000 - i) for i in range(350, 1001))
    assert abs(float(PS.binom_sf_exact(350, 1000, F(1, 4))) - direct) <= 1e-12 * direct
    # no underflow in the log: P(X >= 1000 | n = 1000, p = 1/4) = 4^-1000
    assert abs(PS.log10_fraction(PS.binom_sf_exact(1000, 1000, F(1, 4))) + 1000 * math.log10(4)) < 1e-9
    assert PS.fmt_p(PS.binom_sf_exact(1000, 1000, F(1, 4))).endswith("e-603")


def test_two_sided_exact_known_values():
    assert PS.binom_two_sided_exact(3, 10, F(1, 2)) == F(352, 1024)   # 2 * (1 + 10 + 45 + 120) / 1024
    assert PS.binom_two_sided_exact(5, 10, F(1, 2)) == 1
    assert PS.binom_two_sided_exact(250, 1000, F(1, 4)) == 1
    # k = 0 of 10 at p = 1/4: pmf(0) plus the upper outcomes no more likely than it (6..10)
    want = PS.binom_pmf_exact(0, 10, F(1, 4)) + PS.binom_sf_exact(6, 10, F(1, 4))
    assert PS.binom_two_sided_exact(0, 10, F(1, 4)) == want


def test_clopper_pearson_known_values():
    lo, hi = PS.clopper_pearson(0, 10, 0.95)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** 0.1)) < 1e-12
    lo, hi = PS.clopper_pearson(10, 10, 0.95)
    assert hi == 1.0 and abs(lo - 0.025 ** 0.1) < 1e-12
    lo, hi = PS.clopper_pearson(5, 10, 0.95)                        # textbook: [0.1871, 0.8129]
    assert abs(lo - 0.187086) < 1e-6 and abs(hi - 0.812914) < 1e-6
    # the bounds solve the defining tail equations exactly (checked in rational arithmetic)
    k, n = 37, 100
    for conf in (0.95, 0.99):
        lo, hi = PS.clopper_pearson(k, n, conf)
        half = (1 - conf) / 2
        assert abs(float(PS.binom_sf_exact(k, n, F(lo))) - half) < 1e-9 * half
        assert abs(float(PS.binom_cdf_exact(k, n, F(hi))) - half) < 1e-9 * half
    lo95, hi95 = PS.clopper_pearson(360, 1000, 0.95)
    lo99, hi99 = PS.clopper_pearson(360, 1000, 0.99)
    assert lo99 < lo95 < 0.36 < hi95 < hi99


GRID = [(0, 10, 0.25), (3, 10, 0.5), (360, 1000, 0.25), (90, 250, 0.25), (250, 1000, 0.25), (150, 1000, 0.25),
        (700, 1000, 0.5), (230, 400, 0.5), (1, 400, 0.25), (399, 400, 0.5), (132, 400, 0.25), (26, 100, 0.25)]


def test_statistics_agree_with_scipy():
    stats = pytest.importorskip("scipy.stats")
    worst = 0.0
    for k, n, p in GRID:
        pf = F(p).limit_denominator(100)
        mine = [float(PS.binom_sf_exact(k, n, pf)), float(PS.binom_two_sided_exact(k, n, pf))]
        mine += list(PS.clopper_pearson(k, n, 0.95)) + list(PS.clopper_pearson(k, n, 0.99))
        t = stats.binomtest(k, n, p, alternative="two-sided")
        ci95 = t.proportion_ci(0.95, "exact")
        ci99 = t.proportion_ci(0.99, "exact")
        ref = [stats.binomtest(k, n, p, alternative="greater").pvalue, t.pvalue, ci95.low, ci95.high,
               ci99.low, ci99.high]
        for a, b in zip(mine, ref):
            if a or b:
                worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
    assert worst < 1e-9, worst
    # the script's own cross-check helper sees the same numbers
    sv = PS.scipy_values(360, 1000, 0.25)
    assert abs(sv["p_greater"] - float(PS.binom_sf_exact(360, 1000, F(1, 4)))) < 1e-20


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------
def test_holm_step_down_and_adjusted_pvalues():
    p = {"a": F(1, 100), "b": F(4, 100), "c": F(3, 100), "d": F(5, 1000)}
    h = PS.holm(p, F(5, 100))
    assert h["order"] == ["d", "a", "c", "b"]
    assert h["rejected"] == {"d": True, "a": True, "c": False, "b": False} and not h["all_rejected"]
    assert h["threshold"]["d"] == F(5, 400) and h["threshold"]["c"] == F(5, 200) and h["threshold"]["b"] == F(5, 100)
    assert h["adjusted"] == {"d": F(2, 100), "a": F(3, 100), "c": F(6, 100), "b": F(6, 100)}
    # step-down stops at the first acceptance even if a later p is below its own threshold
    h = PS.holm({"x": F(1, 1000), "y": F(3, 100), "z": F(31, 1000)}, F(5, 100))
    assert h["rejected"] == {"x": True, "y": False, "z": False}
    h = PS.holm({"x": F(1, 1000), "y": F(2, 100), "z": F(4, 100)}, F(5, 100))
    assert h["all_rejected"] and max(h["adjusted"].values()) <= F(5, 100)
    # all rejected <=> max adjusted p <= alpha, on random inputs
    import random
    rng = random.Random(3)
    for _ in range(200):
        ps = {str(i): F(rng.randint(1, 10 ** 6), 10 ** 8) for i in range(6)}
        h = PS.holm(ps, F(1, 100))
        assert h["all_rejected"] == (max(h["adjusted"].values()) <= F(1, 100))


# ---------------------------------------------------------------------------
# Claim logic on synthetic protocol-shaped results
# ---------------------------------------------------------------------------
COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]
STAT0 = {"decisions": 100, "errors": 0, "observe_errors": 0, "fallback": 0, "unmapped_top": 0}
#: A counted-mode game's tracker counters (bench ``info_stats``) without tracker errors or resets.
INFO0 = {"info_samples": 160, "info_uncertain": 12, "info_errors": 0, "info_resets": 0, "info_max_hypotheses": 3,
         "hidden_steals": 2, "hidden_discards": 5, "hidden_dev_draws": 3}
COUNTED_META = {"mode": "counted", "samples": 4, "discards_public": False}


def _spread(i: int, rate: float) -> bool:
    """Exactly floor(n * rate) of the first n indices win, spread evenly."""
    return math.floor((i + 1) * rate + 1e-9) - math.floor(i * rate + 1e-9) == 1


def synthetic(tid: str, rate: float = None, seat_rates=None, games: int = None, seed: int = None,
              meta_overrides=None, record_hook=None):
    """A bench-like summary for protocol test ``tid`` (all fields as the protocol requires)."""
    t = PS.TESTS[tid]
    n = t.games if games is None else games
    base = t.seed if seed is None else seed
    recs = []
    per_group = Counter()
    for g in range(n):
        seats = PS.our_seats_for(g, t.fmt)
        j = per_group[seats]          # wins spread evenly within every seat (1v3) / arrangement (2v2)
        per_group[seats] += 1
        won = _spread(j, seat_rates[seats[0]] if seat_rates is not None else rate)
        others = [i for i in range(4) if i not in seats]
        ws = seats[0] if won else others[0]
        vps = [5, 5, 5, 5]
        vps[ws] = 10
        r = {"game": g, "seed": PS.game_seed(base, g), "our_seats": list(seats),
             "seat": seats[0] if t.fmt != "2v2" else None, "winner": COLORS[ws], "winner_seat": ws,
             "vps": vps, "turns": 80, "actions": 400, "won": won, "stats": dict(STAT0),
             "trades": {"opp_errors": 0}, "crashes": 0}
        if t.opponents:        # the mixed table (amendment 3): what bench_catanatron.py records per game
            lineup = list(PS.mixed_lineup(g, t.opponents))
            r.update(lineup=lineup, relative=[lineup[(seats[0] + k) % 4] for k in (1, 2, 3)],
                     winner_name=lineup[ws])
        if t.info == "counted":   # amendment 2: the counted mode's tracker counters
            r["info_stats"] = dict(INFO0)
        if record_hook:
            record_hook(r)
        recs.append(r)
    meta = {"spec": PS.PROTOCOL_SPEC, "opponent": t.opponent, "opponent_class": t.opponent_class,
            "opponent_params": {}, "trades": "off", "catanatron": t.engine,
            "hash_seed": "0", "seed": t.seed, "format": t.fmt, "our_seats": 2 if t.fmt == "2v2" else 1,
            "vps_to_win": 10, "discard_limit": 7}
    if tid in PS.AMENDED_TESTS:    # T1-T6 / R1-R2 keep the metadata of the original tooling
        meta["info"] = dict(COUNTED_META) if t.info == "counted" else {"mode": "full"}
        if t.opponents:
            meta["opponents"] = list(t.opponents)
    meta.update(meta_overrides or {})
    return dict(meta, results=recs, games=len(recs))


GOOD = {"T1": 0.5, "T2": 0.5, "T3": 0.5, "T4": 0.7, "T5": 0.7, "T6": 0.7, "R1": 0.3, "R2": 0.3,
        "T7": 0.5, "T8": 0.5, "T9": 0.5, "T10": 0.5, "T11": 0.5}
_BASELINE: dict = {}


def _analyze(tid, summ, source=None):
    return PS.analyze_test(tid, [(source or f"{tid}.json", {k: v for k, v in summ.items() if k != "results"},
                                  summ["results"])])


def verdict_for(overrides=None, drop=()):
    """Analysis and verdict of the passing baseline GOOD with some tests replaced / dropped (the
    baseline analyses are cached: analyze_test and evaluate never modify them)."""
    overrides = overrides or {}
    analysis = {}
    for tid, rate in GOOD.items():
        if tid in drop:
            continue
        if tid in overrides:
            analysis[tid] = _analyze(tid, overrides[tid])
        else:
            if tid not in _BASELINE:
                _BASELINE[tid] = _analyze(tid, synthetic(tid, rate))
            analysis[tid] = _BASELINE[tid]
    return analysis, PS.evaluate(analysis)


def failed(verdict, claim):
    return [c for c in verdict[claim]["conditions"] if not c["pass"]]


def test_claims_pass_on_strong_conforming_results():
    analysis, v = verdict_for()
    assert all(a["conforms"] for a in analysis.values())
    assert v["claim1"]["pass"] and v["claim2"]["pass"], [c["failures"] for c in failed(v, "claim2")]
    assert analysis["T1"]["seats"][0]["games"] == 250 and analysis["T4"]["arrangements"]["CCoo"]["games"] == 167
    txt = PS.report_text(analysis, v)
    assert "CLAIM 1 - better than Catanatron's strong bots: PASS" in txt and "CLAIM 2" in txt and ": PASS" in txt
    md = PS.report_markdown(analysis, v, ["x.json"])
    assert md.startswith("# Strength proof results") and "**PASS**" in md and "Holm-Bonferroni" in md


def test_claim1_fails_when_a_test_is_not_significant():
    analysis, v = verdict_for({"T2": synthetic("T2", 0.27)})
    assert not v["claim1"]["pass"] and not v["claim2"]["pass"]
    holm_fail = failed(v, "claim1")
    assert len(holm_fail) == 1 and "Holm" in holm_fail[0]["name"] and "T2" in holm_fail[0]["failures"][0]


def test_claim2_five_sigma_fails_while_claim1_passes():
    analysis, v = verdict_for({"T3": synthetic("T3", 0.33)})
    p = analysis["T3"]["p_one_sided"]
    assert F(57, 10 ** 8) < p < F(1, 600)
    assert v["claim1"]["pass"] and not v["claim2"]["pass"]
    names = [c["name"] for c in failed(v, "claim2")]
    assert any("5.7e-7" in n for n in names) and any("effect size" in n for n in names)


def test_effect_size_alone_fails_claim2():
    analysis, v = verdict_for({"T1": synthetic("T1", 0.36)})
    assert analysis["T1"]["cp99"][0] < 0.35 and analysis["T1"]["p_one_sided"] < F(1, 10 ** 10)
    assert v["claim1"]["pass"]
    bad = failed(v, "claim2")
    assert [c["name"][:11] for c in bad] == ["effect size"] and "T1" in bad[0]["failures"][0]


def test_effect_size_threshold_for_2v2_share():
    analysis, v = verdict_for({"T4": synthetic("T4", 0.59)})     # 586/1000: 5 sigma, but lower bound < 0.55
    assert analysis["T4"]["wins"] == 586 and analysis["T4"]["p_one_sided"] < F(57, 10 ** 8)
    assert 0.54 < analysis["T4"]["cp99"][0] < 0.55 and v["claim1"]["pass"]
    bad = failed(v, "claim2")
    assert len(bad) == 1 and "effect size" in bad[0]["name"] and bad[0]["failures"][0].startswith("T4")


def test_seat_robustness_alone_fails_claim2():
    analysis, v = verdict_for({"T2": synthetic("T2", seat_rates={0: 0.25, 1: 0.8, 2: 0.8, 3: 0.8})})
    assert analysis["T2"]["seats"][0]["p_one_sided"] > F(1, 2)
    assert v["claim1"]["pass"]
    bad = failed(v, "claim2")
    assert len(bad) == 1 and "seat robustness" in bad[0]["name"] and "T2 seat 0" in bad[0]["failures"][0]


@pytest.mark.parametrize("tid,field,value,text", [
    ("T4", "errors", 1, "adapter errors"),
    ("R2", "fallback", 2, "illegal-action fallbacks"),
    ("T1", "observe_errors", 1, "observe errors"),
])
def test_any_adapter_error_or_fallback_fails_claim2(tid, field, value, text):
    def hook(r):
        if r["game"] == 3:
            r["stats"][field] = value
    analysis, v = verdict_for({tid: synthetic(tid, GOOD[tid], record_hook=hook)})
    assert v["claim1"]["pass"]
    bad = failed(v, "claim2")
    assert len(bad) == 1 and "zero adapter errors" in bad[0]["name"] and text in bad[0]["failures"][0]


def test_crashes_fail_claim2_and_count_as_losses():
    def rerun(r):
        if r["game"] == 5:
            r["crashes"] = 1
    _, v = verdict_for({"T5": synthetic("T5", GOOD["T5"], record_hook=rerun)})
    bad = failed(v, "claim2")
    assert len(bad) == 1 and "crashed attempts" in bad[0]["failures"][0]

    def crashed(r):
        if r["game"] == 4:        # a win in the synthetic pattern (seat 0's second game)
            assert r["won"]
            r.update(crashed=True, crashes=2, winner=None, winner_seat=-1, won=False, vps=[0, 0, 0, 0])
    analysis, v = verdict_for({"T1": synthetic("T1", GOOD["T1"], record_hook=crashed)})
    a = analysis["T1"]
    assert a["crashed_games"] == 1 and a["turn_cap_games"] == 0 and a["wins"] == 499
    assert "games crashed twice" in failed(v, "claim2")[0]["failures"][0]


def test_turn_cap_games_are_losses_and_won_is_recomputed():
    def capped(r):
        if r["game"] in (0, 5):   # two games that reached the turn cap without a winner
            r.update(winner=None, winner_seat=-1, won=False)
    analysis, v = verdict_for({"T2": synthetic("T2", GOOD["T2"], record_hook=capped)})
    assert analysis["T2"]["turn_cap_games"] == 2 and analysis["T2"]["conforms"]

    def lying(r):
        if r["game"] == 1:
            r.update(winner=None, winner_seat=-1, won=True)
    analysis, v = verdict_for({"T2": synthetic("T2", GOOD["T2"], record_hook=lying)})
    assert not analysis["T2"]["conforms"] and any("disagrees" in p for p in analysis["T2"]["problems"])
    assert not v["claim1"]["pass"]


def test_r_tests_two_sided_condition():
    analysis, v = verdict_for({"R1": synthetic("R1", 0.15)})
    assert analysis["R1"]["p_two_sided_vs_025"] < F(1, 100)
    assert v["claim1"]["pass"]
    bad = failed(v, "claim2")
    assert len(bad) == 1 and "R1 / R2" in bad[0]["name"] and "significantly below" in bad[0]["failures"][0]
    # a rate far ABOVE 0.25 is significant two-sided too, but passes (only "below" fails)
    analysis, v = verdict_for({"R2": synthetic("R2", 0.6)})
    assert analysis["R2"]["p_two_sided_vs_025"] < F(1, 10 ** 6) and v["claim2"]["pass"]
    # slightly below 0.25 but not significantly: passes
    analysis, v = verdict_for({"R2": synthetic("R2", 0.22)})
    assert analysis["R2"]["p_two_sided_vs_025"] > F(1, 100) and v["claim2"]["pass"]


def test_protocol_conformance_failures():
    # missing games
    analysis, v = verdict_for({"T3": synthetic("T3", 0.5, games=390)})
    assert not v["claim1"]["pass"] and not v["claim2"]["pass"]
    assert any("390 of the 400 protocol games" in f for f in failed(v, "claim1")[0]["failures"])
    # wrong base seed (metadata and per-game seeds)
    analysis, v = verdict_for({"T6": synthetic("T6", 0.7, seed=5, meta_overrides={"seed": 5})})
    probs = analysis["T6"]["problems"]
    assert any("seed is 5" in p for p in probs) and any("per-game seed" in p for p in probs)
    assert not v["claim1"]["pass"]
    # R2 with another spec: claim 1 (T tests only) still passes, claim 2 does not
    analysis, v = verdict_for({"R2": synthetic("R2", 0.3, meta_overrides={"spec": "heuristic"})})
    assert v["claim1"]["pass"] and not v["claim2"]["pass"]
    assert [c["name"][:10] for c in failed(v, "claim2")] == ["results of"]
    # trades on / hash seed / engine / opponent params
    for over, text in (({"trades": "value"}, "trades"), ({"hash_seed": None}, "PYTHONHASHSEED"),
                       ({"catanatron": "3.2.1"}, "catanatron"), ({"opponent_params": {"depth": "3"}}, "params"),
                       ({"opponent": "vp", "opponent_class": "VictoryPointPlayer"}, "opponent")):
        analysis, _ = verdict_for({"T1": synthetic("T1", 0.5, meta_overrides=over)})
        assert any(text in p for p in analysis["T1"]["problems"]), (over, analysis["T1"]["problems"])
    # a whole test missing: Holm still runs over six tests (the missing one enters with p = 1)
    analysis, v = verdict_for(drop=("T5",))
    assert not v["claim1"]["pass"] and v["holm_claim1"]["adjusted"]["T5"] == 1
    assert any("T5: no results" in f for c in failed(v, "claim1") for f in c["failures"])


def test_conformance_needs_every_registered_field():
    # On 3.3 the stand-in presets have catanatron's class names: vf / ab data must not pass as T1 / T2.
    for tid, opp, cls in (("T1", "vf", "ValueFunctionPlayer"), ("T2", "ab", "AlphaBetaPlayer")):
        analysis, v = verdict_for({tid: synthetic(tid, 0.5, meta_overrides={"opponent": opp, "opponent_class": cls})})
        assert any("opponent" in p for p in analysis[tid]["problems"]) and not v["claim1"]["pass"]
    # a registered field that is missing is a deviation (it cannot be shown to be the registered one)
    for key in ("trades", "format", "vps_to_win", "discard_limit", "catanatron", "spec", "hash_seed", "seed"):
        s = synthetic("T3", 0.5)
        s.pop(key)
        analysis, v = verdict_for({"T3": s})
        assert analysis["T3"]["problems"] and not v["claim1"]["pass"], key
    # another catanatron release than the registered 3.3.0
    analysis, _ = verdict_for({"T4": synthetic("T4", 0.7, meta_overrides={"catanatron": "3.3.1"})})
    assert any("catanatron" in p for p in analysis["T4"]["problems"])
    # bare per-game records (no run metadata) cannot be verified
    recs = synthetic("T5", 0.7)["results"]
    an = PS.analyze_test("T5", [("bare.jsonl", {}, recs)])
    assert an["games"] == 400 and not an["conforms"] and "no run metadata" in an["problems"][0]


def test_chunks_merge_and_overlaps():
    full = synthetic("T2", 0.5)
    meta = {k: v for k, v in full.items() if k != "results"}
    a, b, c = full["results"][:150], full["results"][150:], full["results"][140:160]
    an = PS.analyze_test("T2", [("a", meta, a), ("b", meta, b)])
    assert an["games"] == 400 and an["conforms"]
    an = PS.analyze_test("T2", [("a", meta, a), ("b", meta, b), ("c", meta, c)])
    assert an["games"] == 400 and an["conforms"] and "20 game(s) found in two chunks" in an["notes"][0]
    changed = [dict(r) for r in c]
    changed[0]["vps"] = [1, 2, 3, 4]
    an = PS.analyze_test("T2", [("a", meta, a), ("b", meta, b), ("c", meta, changed)])
    assert not an["conforms"] and any("different outcomes" in p for p in an["problems"])


def test_cli_reads_files_directories_and_infers_ids(tmp_path, capsys):
    paths = {}
    for tid, rate in GOOD.items():
        s = synthetic(tid, rate)
        d = tmp_path / tid
        d.mkdir()
        half = len(s["results"]) // 2
        for i, part in enumerate((s["results"][:half], s["results"][half:])):
            with open(d / f"chunk{i}.json", "w") as fh:
                json.dump(dict(s, results=part, games=len(part)), fh)
        paths[tid] = str(d)
    md = tmp_path / "PROOF.md"
    out_json = tmp_path / "proof.json"
    rc = PS.main(sum((["--test", f"{t}={p}"] for t, p in paths.items()), []) + ["--markdown", str(md),
                                                                                  "--json", str(out_json)])
    out = capsys.readouterr().out
    assert rc == 0 and "CLAIM 1 - better than Catanatron's strong bots: PASS" in out
    assert "CLAIM 2 - ready for (supervised) human testing: PASS" in out
    assert md.read_text().startswith("# Strength proof results")
    data = json.loads(out_json.read_text())
    assert data["verdict"]["claim1"]["pass"] is True and data["analysis"]["T1"]["games"] == 1000
    # positional: test ids inferred from opponent / format / engine
    rc = PS.main([paths["T1"], paths["T4"], paths["R2"]])
    out = capsys.readouterr().out
    assert rc == 0 and "T1   value" in out and "T4   value" in out and "R2   ab" in out
    assert "T2: no results" in out
    # the amendments' tests: inferred by information mode and mixed format too
    rc = PS.main([paths["T1"], paths["T7"], paths["T10"], paths["T11"]])
    out = capsys.readouterr().out
    assert rc == 0 and "T7   value      1v3       counted" in out and "T1   value      1v3       full" in out
    assert "T10  mixed      1v3-mixed full" in out and "T11  mixed      1v3-mixed counted" in out
    assert "T8: no results" in out and "READINESS" in out and ": FAIL" in out


# ---------------------------------------------------------------------------
# Amendments 2 and 3: T7-T11, claims 3 and 4, readiness (synthetic results)
# ---------------------------------------------------------------------------
A57 = F(57, 10 ** 8)


def _set_outcome(r, won):
    """Make record ``r`` a catanbot win / loss (winner seat, colour, VPs, winner name all consistent)."""
    seats = r["our_seats"]
    ws = seats[0] if won else [i for i in range(4) if i not in seats][0]
    vps = [5, 5, 5, 5]
    vps[ws] = 10
    r.update(won=won, winner=COLORS[ws], winner_seat=ws, vps=vps)
    if "lineup" in r:
        r["winner_name"] = r["lineup"][ws]


def with_wins(tid, k, **kw):
    """Synthetic results of ``tid`` with exactly ``k`` catanbot wins (from the 0.5 baseline)."""
    s = synthetic(tid, 0.5, **kw)
    wins = [r for r in s["results"] if r["won"]]
    for r in reversed(wins[k:]):
        _set_outcome(r, False)
    assert sum(r["won"] for r in s["results"]) == k
    return s


def test_amended_tests_are_registered_as_in_the_protocol():
    want = {"T7": ("value", "ValueFunctionPlayer", "1v3", 1000, 900201, "counted"),
            "T8": ("alphabeta", "AlphaBetaPlayer", "1v3", 400, 900201, "counted"),
            "T9": ("sameturn", "SameTurnAlphaBetaPlayer", "1v3", 400, 900201, "counted"),
            "T10": ("value,alphabeta,sameturn", "ValueFunctionPlayer+AlphaBetaPlayer+SameTurnAlphaBetaPlayer",
                    "1v3-mixed", 400, 900301, "full"),
            "T11": ("value,alphabeta,sameturn", "ValueFunctionPlayer+AlphaBetaPlayer+SameTurnAlphaBetaPlayer",
                    "1v3-mixed", 400, 900401, "counted")}
    for tid, row in want.items():
        t = PS.TESTS[tid]
        assert (t.opponent, t.opponent_class, t.fmt, t.games, t.seed, t.info, t.engine) == row + ("3.3.0",), tid
    assert PS.TESTS["T10"].opponents == PS.TESTS["T11"].opponents == ("value", "alphabeta", "sameturn")
    assert PS.C3_TESTS == ("T7", "T8", "T9") and PS.C4_TESTS == ("T10", "T11")
    assert PS.CLAIM3_ALPHA == PS.CLAIM4_ALPHA == A57 and PS.PROTOCOL_INFO_SAMPLES == 4
    assert PS.NULL["1v3-mixed"] == F(1, 4) and PS.EFFECT_LOWER["1v3-mixed"] == PS.EFFECT_LOWER["1v3"] == 0.35
    # T1-T6 / R1-R2 are unchanged: full information, three copies of one opponent
    assert all(PS.TESTS[t].info == "full" and PS.TESTS[t].opponents == () for t in PS.T_TESTS + PS.R_TESTS)
    assert [PS.TESTS[t][:7] for t in ("T1", "T4", "R2")] == [
        ("T1", "value", "ValueFunctionPlayer", "1v3", 1000, 900001, "3.3.0"),
        ("T4", "value", "ValueFunctionPlayer", "2v2", 1000, 900001, "3.3.0"),
        ("R2", "ab", "AlphaBetaPlayer", "1v3", 400, 900101, "3.2.1")]
    # the protocol text registers the same tests
    doc = open(os.path.join(ROOT, "docs", "PROOF_PROTOCOL.md")).read()
    for line in ("| T7 | V (catanatron ValueFunctionPlayer) | 1v3 | 1000 |",
                 "| T8 | A (catanatron AlphaBetaPlayer) | 1v3 | 400 |",
                 "| T9 | S (catanatron SameTurnAlphaBetaPlayer) | 1v3 | 400 |",
                 "| T10 | full (as T1-T6) | 400 | 900301 |", "| T11 | counted (as T7-T9) | 400 | 900401 |",
                 "seed 900201", "default 4 samples per decision", "permutation `(g // 4) % 6`"):
        assert line in doc, line


def test_mixed_lineup_follows_the_protocol_rule():
    names = PS.MIXED_OPPONENTS
    orders = [tuple(names[i] for i in p) for p in PS.MIXED_PERMUTATIONS]
    assert orders == [("value", "alphabeta", "sameturn"), ("value", "sameturn", "alphabeta"),
                      ("alphabeta", "value", "sameturn"), ("alphabeta", "sameturn", "value"),
                      ("sameturn", "value", "alphabeta"), ("sameturn", "alphabeta", "value")]
    for g in range(1000):
        lu = PS.mixed_lineup(g)
        s = g % 4                                          # catanbot's seat rotates g % 4
        assert lu[s] == "catanbot" and sorted(lu) == sorted(("catanbot",) + names)
        assert tuple(lu[(s + k) % 4] for k in (1, 2, 3)) == orders[(g // 4) % 6]   # permutation (g // 4) % 6
        assert PS.our_seats_for(g, "1v3-mixed") == (s,)
    assert PS.mixed_lineup(0) == ("catanbot", "value", "alphabeta", "sameturn")
    assert PS.mixed_lineup(5) == ("alphabeta", "catanbot", "value", "sameturn")
    assert PS.mixed_lineup(23) == ("sameturn", "alphabeta", "value", "catanbot")
    assert PS.mixed_lineup(24) == PS.mixed_lineup(0)
    # "every opponent sits in every relative position equally often over each 24 consecutive games"
    # (and every player in every seat): all windows, not only the aligned blocks
    for start in range(0, 400 - 23):
        window = range(start, start + 24)
        rel = Counter((PS.mixed_lineup(g)[(g % 4 + k) % 4], k) for g in window for k in (1, 2, 3))
        assert len(rel) == 9 and set(rel.values()) == {8}
        seat = Counter((PS.mixed_lineup(g)[s], s) for g in window for s in range(4))
        assert len(seat) == 16 and set(seat.values()) == {6}
    # 400 games = 16 blocks of 24 + 16: every seat exactly 100 games
    assert Counter(PS.our_seats_for(g, "1v3-mixed")[0] for g in range(400)) == {0: 100, 1: 100, 2: 100, 3: 100}


def test_claims_3_and_4_pass_on_strong_conforming_results():
    analysis, v = verdict_for()
    assert all(a["conforms"] for a in analysis.values()), {t: a["problems"] for t, a in analysis.items()}
    for c in ("claim1", "claim2", "claim3", "claim4"):
        assert v[c]["pass"], (c, [x["failures"] for x in failed(v, c)])
    assert v["readiness"] == {"pass": True, "claims": {"claim2": True, "claim3": True, "claim4": True}, "failed": []}
    assert analysis["T7"]["seats"][3]["games"] == 250 and analysis["T10"]["seats"][0]["games"] == 100
    assert analysis["T11"]["info"] == "counted" and analysis["T10"]["info"] == "full"
    mw = analysis["T10"]["mixed_wins"]
    assert mw["catanbot"] == analysis["T10"]["wins"] == 200 and sum(mw.values()) == 400 and mw["none"] == 0
    # Holm families: claim 3 over three tests, claim 4 over two, at 5.7e-7
    assert sorted(v["holm_claim3"]["threshold"].values()) == [A57 / 3, A57 / 2, A57]
    assert sorted(v["holm_claim4"]["threshold"].values()) == [A57 / 2, A57]
    assert set(v["holm_claim3"]["order"]) == set(PS.C3_TESTS) and set(v["holm_claim4"]["order"]) == set(PS.C4_TESTS)
    txt = PS.report_text(analysis, v)
    for line in ("CLAIM 3 - strong under Colonist information (T7-T9): PASS",
                 "CLAIM 4 - beats a mixed Catanatron table (T10-T11): PASS",
                 "READINESS for (supervised) human testing = claims 2, 3 and 4 (amendments 2 and 3): PASS",
                 "Holm-Bonferroni over T7-T9 at family alpha 5.7e-7 (claim 3):",
                 "Holm-Bonferroni over T10-T11 at family alpha 5.7e-7 (claim 4):", "T10 wins by player"):
        assert line in txt, line
    md = PS.report_markdown(analysis, v, ["x.json"])
    for line in ("| 3 - strong under Colonist information (T7-T9) | **PASS** | - |",
                 "| 4 - beats a mixed Catanatron table (T10-T11) | **PASS** | - |",
                 "| readiness for supervised human testing (claims 2, 3 and 4) | **PASS** | - |",
                 "## Holm-Bonferroni over T7-T9 (claim 3)", "## Holm-Bonferroni over T10-T11 (claim 4)",
                 "**Claim 4: PASS**", "counted (K = 4, discards hidden)"):
        assert line in md, line


def test_claims_1_and_2_do_not_depend_on_the_amended_tests():
    base, vb = verdict_for(drop=PS.AMENDED_TESTS)
    assert vb["claim1"]["pass"] and vb["claim2"]["pass"]
    assert not vb["claim3"]["pass"] and not vb["claim4"]["pass"] and vb["readiness"]["failed"] == ["claim3", "claim4"]
    assert all("no results" in f for c in failed(vb, "claim3") for f in c["failures"])

    def broken(r):
        r["stats"]["errors"] = 1
        r["crashes"] = 1
    bad = {t: synthetic(t, 0.2, seed=5, meta_overrides={"seed": 5, "spec": "x"}, record_hook=broken)
           for t in PS.AMENDED_TESTS}
    analysis, v = verdict_for(bad)
    for key in ("claim1", "claim2", "holm_claim1", "holm_claim2"):
        assert v[key] == vb[key], key
    assert not v["claim3"]["pass"] and not v["claim4"]["pass"]
    assert len(failed(v, "claim3")) == 5 and len(failed(v, "claim4")) == 5


def test_readiness_needs_claims_2_3_and_4():
    _, v = verdict_for({"T1": synthetic("T1", 0.36)})          # claim 2 fails (effect size), 3 and 4 pass
    assert v["claim1"]["pass"] and not v["claim2"]["pass"] and v["claim3"]["pass"] and v["claim4"]["pass"]
    assert not v["readiness"]["pass"] and v["readiness"]["failed"] == ["claim2"]
    _, v = verdict_for({"T8": synthetic("T8", 0.27)})          # claim 3 fails, claims 1 and 2 are untouched
    assert v["claim1"]["pass"] and v["claim2"]["pass"] and not v["claim3"]["pass"] and v["claim4"]["pass"]
    assert v["readiness"]["failed"] == ["claim3"]
    assert "READINESS for (supervised) human testing = claims 2, 3 and 4 (amendments 2 and 3): FAIL (failed: claim 3)" \
        in PS.report_text(*verdict_for({"T8": synthetic("T8", 0.27)}))
    _, v = verdict_for(drop=("T11",))                          # a missing test fails its claim
    assert not v["claim4"]["pass"] and v["readiness"]["failed"] == ["claim4"]
    assert any("T11: no results (enters Holm with p = 1)" in f for c in failed(v, "claim4") for f in c["failures"])
    assert v["holm_claim4"]["adjusted"]["T11"] == 1


def test_claim3_and_claim4_holm_step_down():
    # T8 = T9 = 146/400 (p = 2.2e-7 in (alpha/3, alpha/2]) after a tiny T7 p: steps alpha/2 and alpha reject
    _, v = verdict_for({"T8": with_wins("T8", 146), "T9": with_wins("T9", 146)})
    holm3 = [c for c in v["claim3"]["conditions"] if "Holm" in c["name"]][0]
    assert holm3["pass"] and holm3["name"] == "T7-T9 reject their nulls at family-wise alpha = 5.7e-7 (Holm over T7-T9)"
    # 145/400 (p = 3.9e-7 > alpha/2): the second step accepts, so the claim fails although p < alpha
    analysis, v = verdict_for({"T8": with_wins("T8", 145), "T9": with_wins("T9", 145)})
    assert A57 / 2 < analysis["T8"]["p_one_sided"] <= A57
    holm3 = [c for c in v["claim3"]["conditions"] if "Holm" in c["name"]][0]
    assert not holm3["pass"] and any("T8" in f or "T9" in f for f in holm3["failures"])
    # claim 4: one test at 145 and the other strong passes (alpha/2 then alpha); both at 145 fail
    _, v = verdict_for({"T10": with_wins("T10", 145)})
    assert [c for c in v["claim4"]["conditions"] if "Holm" in c["name"]][0]["pass"]
    _, v = verdict_for({"T10": with_wins("T10", 145), "T11": with_wins("T11", 145)})
    holm4 = [c for c in v["claim4"]["conditions"] if "Holm" in c["name"]][0]
    assert not holm4["pass"] and "(Holm over T10-T11)" in holm4["name"]
    assert v["claim1"]["pass"] and v["claim2"]["pass"] and v["claim3"]["pass"]


def test_claim3_and_claim4_five_sigma_and_effect_size():
    # not significant at all
    analysis, v = verdict_for({"T9": synthetic("T9", 0.33)})       # 132/400: p = 2e-4
    assert F(57, 10 ** 8) < analysis["T9"]["p_one_sided"] < F(1, 1000) and not v["claim3"]["pass"]
    names = [c["name"] for c in failed(v, "claim3")]
    assert any("5.7e-7" in n for n in names) and any("effect size" in n for n in names)
    # effect size alone: 360/1000 in T7 and 160/400 in T10 are far beyond 5 sigma, lower 99 % bound < 0.35
    analysis, v = verdict_for({"T7": synthetic("T7", 0.36), "T10": synthetic("T10", 0.40)})
    assert analysis["T7"]["p_one_sided"] < F(1, 10 ** 14) and analysis["T10"]["p_one_sided"] < F(1, 10 ** 10)
    assert analysis["T7"]["cp99"][0] < 0.35 and 0.33 < analysis["T10"]["cp99"][0] < 0.35
    for claim, tid in (("claim3", "T7"), ("claim4", "T10")):
        bad = failed(v, claim)
        assert [c["name"][:11] for c in bad] == ["effect size"] and bad[0]["failures"][0].startswith(tid)
    assert v["claim1"]["pass"] and v["claim2"]["pass"]


@pytest.mark.parametrize("tid,claim", [("T9", "claim3"), ("T11", "claim4")])
def test_amended_seat_robustness(tid, claim):
    analysis, v = verdict_for({tid: synthetic(tid, seat_rates={0: 0.8, 1: 0.8, 2: 0.8, 3: 0.25})})
    assert analysis[tid]["seats"][3]["p_one_sided"] > F(1, 2) and analysis[tid]["p_one_sided"] < F(1, 10 ** 20)
    bad = failed(v, claim)
    assert len(bad) == 1 and "seat robustness" in bad[0]["name"] and f"{tid} seat 3" in bad[0]["failures"][0]
    assert v["claim1"]["pass"] and v["claim2"]["pass"] and not v["readiness"]["pass"]


@pytest.mark.parametrize("tid,where,field,text", [
    ("T7", "stats", "errors", "adapter errors"),
    ("T8", "stats", "fallback", "illegal-action fallbacks"),
    ("T9", "stats", "observe_errors", "observe errors"),
    ("T10", "stats", "errors", "adapter errors"),
    ("T7", "info_stats", "info_errors", "counted-mode tracker errors"),
    ("T9", "info_stats", "info_resets", "belief resets"),
    ("T11", "info_stats", "info_errors", "counted-mode tracker errors"),
    ("T11", "info_stats", "info_resets", "belief resets"),
    ("T11", "stats", "fallback", "illegal-action fallbacks"),
])
def test_any_error_fallback_or_tracker_error_fails_claims_3_4(tid, where, field, text):
    def hook(r):
        if r["game"] == 7:
            r[where][field] = 2
    analysis, v = verdict_for({tid: synthetic(tid, GOOD[tid], record_hook=hook)})
    claim, other = ("claim3", "claim4") if tid in PS.C3_TESTS else ("claim4", "claim3")
    bad = failed(v, claim)
    assert len(bad) == 1 and "zero adapter errors" in bad[0]["name"] and f"{tid}: 2 {text}" == bad[0]["failures"][0]
    assert v[other]["pass"] and v["claim1"]["pass"] and v["claim2"]["pass"]
    assert v["readiness"]["failed"] == [claim] and analysis[tid]["conforms"]
    assert "2/0/0" in PS.report_text(analysis, v) or "0/2/0" in PS.report_text(analysis, v)


def test_crashes_fail_claims_3_4_and_count_as_losses():
    def rerun(r):
        if r["game"] == 9:
            r["crashes"] = 1
    _, v = verdict_for({"T8": synthetic("T8", 0.5, record_hook=rerun)})
    bad = failed(v, "claim3")
    assert len(bad) == 1 and bad[0]["failures"] == ["T8: 1 crashed attempts"]

    def crashed(r):
        if r["game"] == 4:        # a win in the synthetic pattern
            assert r["won"]
            r.update(crashed=True, crashes=2, winner=None, winner_seat=-1, won=False, vps=[0, 0, 0, 0],
                     winner_name=None, info_stats={k: 0 for k in INFO0})
    analysis, v = verdict_for({"T11": synthetic("T11", 0.5, record_hook=crashed)})
    a = analysis["T11"]
    assert a["crashed_games"] == 1 and a["turn_cap_games"] == 0 and a["wins"] == 199 and a["conforms"]
    assert a["mixed_wins"]["none"] == 1 and a["mixed_wins"]["catanbot"] == 199
    assert failed(v, "claim4")[0]["failures"] == ["T11: 2 crashed attempts, 1 games crashed twice"]


def test_amended_conformance_information_mode():
    def problems(tid, summ):
        analysis, v = verdict_for({tid: summ})
        claim = "claim3" if tid in PS.C3_TESTS else "claim4"
        assert not analysis[tid]["conforms"] and not v[claim]["pass"] and not v["readiness"]["pass"]
        assert v["claim1"]["pass"] and v["claim2"]["pass"]
        conf = failed(v, claim)[0]
        assert conf["name"].startswith("results of") and all(f.startswith(tid) for f in conf["failures"])
        return analysis[tid]["problems"]

    # a counted-mode run labelled full: T7 (registered counted) with info mode "full"
    probs = problems("T7", synthetic("T7", 0.5, meta_overrides={"info": {"mode": "full"}}))
    assert any("information mode is 'full', protocol 'counted'" in p for p in probs), probs
    # ... and given as T10 (registered full): the metadata says full, the games carry the tracker counters
    def counted_records(r):
        r["info_stats"] = dict(INFO0)
    probs = problems("T10", synthetic("T10", 0.5, record_hook=counted_records))
    assert probs == ["400 game(s) carry counted-mode tracker counters (info_stats): played in counted mode, "
                     "protocol 'full'"], probs
    # a counted run honestly labelled, given as T10
    probs = problems("T10", synthetic("T10", 0.5, meta_overrides={"info": dict(COUNTED_META)},
                                      record_hook=counted_records))
    assert any("information mode is 'counted', protocol 'full'" in p for p in probs)
    # a full-information run labelled counted, given as T11: no tracker counters in the games
    def full_records(r):
        r.pop("info_stats")
    probs = problems("T11", synthetic("T11", 0.5, record_hook=full_records))
    assert probs == ["400 game(s) without the counted mode's tracker counters (info_stats): not played in counted "
                     "mode, or its tracker errors cannot be verified"], probs
    # other samples, public discards, missing fields
    for info, text in (({"mode": "counted", "samples": 8, "discards_public": False}, "info samples is 8"),
                       ({"mode": "counted", "samples": True, "discards_public": False}, "info samples is True"),
                       ({"mode": "counted", "discards_public": False}, "info samples is None"),
                       ({"mode": "counted", "samples": 4, "discards_public": True}, "discards_public is True"),
                       ({"mode": "counted", "samples": 4}, "discards_public is None"),
                       ({"samples": 4, "discards_public": False}, "information mode missing"),
                       ("counted", "information mode missing"), (None, "information mode missing")):
        for tid in ("T8", "T11"):
            probs = problems(tid, synthetic(tid, 0.5, meta_overrides={"info": info}))
            assert any(text in p for p in probs), (tid, info, probs)
    s = synthetic("T9", 0.5)
    s.pop("info")
    assert any("information mode missing" in p for p in problems("T9", s))


def test_amended_conformance_mixed_table():
    def problems(tid, summ):
        analysis, v = verdict_for({tid: summ})
        assert not analysis[tid]["conforms"] and not v["readiness"]["pass"] and v["claim2"]["pass"]
        return analysis[tid]["problems"]

    # the opponents in another order: another permutation rule, every game's lineup differs
    wrong = ("value", "sameturn", "alphabeta")

    def relineup(r):
        r["lineup"] = list(PS.mixed_lineup(r["game"], wrong))
    s = synthetic("T10", 0.5, record_hook=relineup,
                  meta_overrides={"opponent": ",".join(wrong), "opponents": list(wrong),
                                  "opponent_class": "ValueFunctionPlayer+SameTurnAlphaBetaPlayer+AlphaBetaPlayer"})
    probs = problems("T10", s)
    assert any("opponents ['value', 'sameturn', 'alphabeta'], protocol ['value', 'alphabeta', 'sameturn']" in p
               for p in probs), probs
    assert any(p.startswith("400 game(s) whose lineup is not the protocol's") for p in probs)
    assert any("opponent 'value,sameturn,alphabeta'" in p for p in probs)
    # a wrong opponent list in the metadata alone (lineups as registered)
    for opps in (["value", "alphabeta"], ["value", "alphabeta", "sameturn", "value"], ["value", "value", "value"],
                 ["alphabeta", "value", "sameturn"], "value,alphabeta,sameturn", None):
        probs = problems("T11", synthetic("T11", 0.5, meta_overrides={"opponents": opps}))
        assert len(probs) == 1 and "opponents" in probs[0], (opps, probs)
    # another table: three different presets
    probs = problems("T10", synthetic("T10", 0.5, meta_overrides={
        "opponent": "value,alphabeta,mcts", "opponent_class": "ValueFunctionPlayer+AlphaBetaPlayer+MCTSPlayer",
        "opponents": ["value", "alphabeta", "mcts"]}))
    assert any("opponent 'value,alphabeta,mcts'" in p for p in probs) and any("opponents" in p for p in probs)
    # a game without / with a shuffled lineup
    def drop_one(r):
        if r["game"] == 17:
            r.pop("lineup")
        if r["game"] == 30:
            r["lineup"] = r["lineup"][::-1]
    probs = problems("T11", synthetic("T11", 0.5, record_hook=drop_one))
    assert probs == ["2 game(s) whose lineup is not the protocol's (catanbot seat g % 4, opponents in permutation "
                     "(g // 4) % 6 of value, alphabeta, sameturn)"], probs
    # a mixed run given as a single-opponent test, and a single-opponent run given as a mixed test
    probs = problems("T7", synthetic("T11", 0.5, seed=900201, meta_overrides={"seed": 900201}))
    assert any("format is '1v3-mixed', protocol '1v3'" in p for p in probs)
    assert any("(a mixed table)" in p for p in probs) and any("no mixed lineup" in p for p in probs)
    probs = problems("T10", synthetic("T1", 0.5, seed=900301, meta_overrides={"seed": 900301, "info": {"mode": "full"}}))
    assert any("format is '1v3', protocol '1v3-mixed'" in p for p in probs)
    assert any("400 game(s) whose lineup" in p for p in probs) and any("opponents None" in p for p in probs)


@pytest.mark.parametrize("tid", ["T7", "T8", "T10", "T11"])
def test_amended_conformance_needs_every_registered_field(tid):
    keys = ["format", "info", "seed", "hash_seed", "trades", "spec", "catanatron", "vps_to_win", "discard_limit",
            "opponent", "opponent_class"] + (["opponents"] if PS.TESTS[tid].opponents else [])
    claim = "claim3" if tid in PS.C3_TESTS else "claim4"
    base = synthetic(tid, 0.5)
    for key in keys:
        s = dict(base)
        s.pop(key)
        analysis, v = verdict_for({tid: s})
        assert analysis[tid]["problems"] and not v[claim]["pass"] and v["claim2"]["pass"], key
    t = PS.TESTS[tid]
    for over, text in (({"catanatron": "3.3.1"}, "catanatron"), ({"catanatron": "3.2.1"}, "catanatron"),
                       ({"trades": "value"}, "trades"), ({"hash_seed": "1"}, "PYTHONHASHSEED"),
                       ({"spec": "search:depth=1,evaluator=heuristic"}, "spec"),
                       ({"opponent_params": {"depth": "3"}}, "params"), ({"vps_to_win": 12}, "vps_to_win"),
                       ({"discard_limit": 9}, "discard_limit"),
                       ({"opponent_class": "MCTSPlayer"}, "opponent")):
        analysis, v = verdict_for({tid: synthetic(tid, 0.5, meta_overrides=over)})
        assert any(text in p for p in analysis[tid]["problems"]), (over, analysis[tid]["problems"])
        assert not v[claim]["pass"]
    # another test's seed (e.g. T1's 900001 or T10's 900301), metadata and per-game seeds
    other = 900001 if tid != "T10" else 900401
    analysis, _ = verdict_for({tid: synthetic(tid, 0.5, seed=other, meta_overrides={"seed": other})})
    probs = analysis[tid]["problems"]
    assert any(f"seed is {other}, protocol {t.seed}" in p for p in probs) and any("per-game seed" in p for p in probs)
    # games missing / outside 0..N-1
    analysis, _ = verdict_for({tid: synthetic(tid, 0.5, games=t.games - 4)})
    assert any(f"{t.games - 4} of the {t.games} protocol games" in p for p in analysis[tid]["problems"])
    # bare per-game records cannot be verified
    an = PS.analyze_test(tid, [("bare.jsonl", {}, synthetic(tid, 0.5)["results"])])
    assert not an["conforms"] and "no run metadata" in an["problems"][0]


def test_infer_test_uses_information_mode_and_mixed_format():
    for tid in ("T1", "T2", "T4", "R1", "T7", "T8", "T9", "T10", "T11"):
        s = synthetic(tid, 0.5)
        assert PS.infer_test({k: v for k, v in s.items() if k != "results"}, s["results"]) == tid, tid
    # results without the info field (the original tooling) are full information
    s = synthetic("T7", 0.5)
    s.pop("info")
    assert PS.infer_test({k: v for k, v in s.items() if k != "results"}, s["results"]) == "T1"


# ---------------------------------------------------------------------------
# bench_catanatron.py: 2v2 rotation, game ranges, crash re-runs (catanatron needed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    pytest.importorskip("catanatron")
    return _load("bench_catanatron_under_test", "bench_catanatron.py")


@pytest.fixture(scope="module")
def replay_mod():
    pytest.importorskip("catanatron")
    return _load("replay_catanatron_under_test", "replay_catanatron.py")


SMALL = "search:depth=1,beam=2,expand=4,actions=3,evaluator=heuristic"


def test_2v2_rotation_covers_the_six_arrangements_equally(bench):
    arr = [bench.our_seats_for(g, 2) for g in range(600)]
    counts = Counter(arr)
    assert len(counts) == 6 and set(counts.values()) == {100}
    assert set(counts) == set(bench.ARRANGEMENTS_2V2)
    assert all(len(set(a)) == 2 and all(0 <= s < 4 for s in a) for a in counts)
    # every turn-order position is ours in exactly half of the arrangements
    assert Counter(s for a in bench.ARRANGEMENTS_2V2 for s in a) == {0: 3, 1: 3, 2: 3, 3: 3}
    assert [bench.seat_pattern(a) for a in bench.ARRANGEMENTS_2V2] == ["CCoo", "CoCo", "CooC", "oCCo", "oCoC", "ooCC"]
    # 1v3 unchanged: seat g % 4
    assert [bench.our_seats_for(g) for g in range(8)] == [(g % 4,) for g in range(8)]
    # the protocol's own sample sizes: 1000 = 166 * 6 + 4, 400 = 66 * 6 + 4
    assert sorted(Counter(bench.our_seats_for(g, 2) for g in range(1000)).values()) == [166, 166, 167, 167, 167, 167]
    # prove_strength uses the same rotation
    assert all(PS.our_seats_for(g, "2v2") == bench.our_seats_for(g, 2) for g in range(12))


def test_2v2_summary_won_and_arrangements(bench):
    recs = []
    for g in range(12):
        ours = bench.our_seats_for(g, 2)
        ws = [ours[0], ours[1], [s for s in range(4) if s not in ours][0], -1][g % 4]
        recs.append({"game": g, "seat": None, "our_seats": list(ours), "winner": None if ws < 0 else "X",
                     "winner_seat": ws, "won": ws in ours, "vps": [5, 6, 7, 8], "our_vp": 6.0,
                     "opp_vps": [7, 8], "turns": 80, "duration": 1.0, "stats": {}, "unmapped_kinds": {}})
    s = bench.summarize(recs, "heuristic", "random", 1, our_seats=2, game_range=(0, 12))
    assert s["format"] == "2v2" and s["wins"] == 6 and s["win_rate"] == 0.5 and s["null_win_rate"] == 0.5
    assert s["truncated"] == 3 and s["game_range"] == [0, 12]
    assert {k: v["games"] for k, v in s["by_arrangement"].items()} == {p: 2 for p in
                                                                        ["CCoo", "CoCo", "CooC", "oCCo", "oCoC", "ooCC"]}
    assert sum(v["wins"] for v in s["by_arrangement"].values()) == 6
    assert sum(v["wins"] for v in s["by_seat"].values()) == 6   # wins by the catanbot seat that won


def test_real_2v2_games_seat_two_independent_bots(bench, monkeypatch):
    from catanbot.bench import catanatron_adapter as AD
    made = []

    class Recording(AD.CatanbotPlayer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(bench, "CatanbotPlayer", Recording)
    for g in (0, 4):
        made.clear()
        r = bench.run_one((g, 31, SMALL, "random", 10, 7, "off", None, {"our_seats": 2, "log": True}))
        ours = bench.ARRANGEMENTS_2V2[g % 6]
        assert r["our_seats"] == list(ours) and r["pattern"] == bench.seat_pattern(ours) and r["seat"] is None
        assert len(made) == 2 and made[0] is not made[1] and made[0].bot is not made[1].bot
        assert [m.color for m in made] == [AD.COLORS[s] for s in ours]
        assert made[0].seed != made[1].seed and all(m.stats["decisions"] > 0 for m in made)
        assert r["won"] == (r["winner_seat"] in ours) and r["won"] == (r["winner"] is not None and r["winner"] in
                                                                         [AD.COLORS[s].value for s in ours])
        assert r["our_vps"] == [r["vps"][s] for s in ours] and len(r["opp_vps"]) == 2
        assert r["stats"]["decisions"] == sum(st["decisions"] for st in r["stats_by_seat"].values())
        kinds = [p["kind"] for p in r["_log"]["players"]]
        assert [i for i, k in enumerate(kinds) if k == "catanbot"] == list(ours)
        assert r["stats"]["errors"] == 0 and r["stats"]["fallback"] == 0


def test_default_1v3_job_is_unchanged(bench):
    r = bench.run_one((5, 3, "heuristic", "random", 10, 7, "off", None))
    assert r["seat"] == 1 and r["our_seats"] == [1] and r["won"] == (r["winner_seat"] == 1)
    assert "_log" not in r and "crashes" not in r and "pattern" not in r
    assert r["our_vp"] == r["vps"][1] and len(r["opp_vps"]) == 3


def test_game_range_chunks_reproduce_an_uninterrupted_run(bench, tmp_path, capsys):
    for n_ours in (1, 2):
        whole = bench.plan_games(900001, bench.game_indices(1000), n_ours)
        chunks = []
        for a in range(0, 1000, 37):
            chunks += bench.plan_games(900001, bench.game_indices(0, game_range=(a, min(1000, a + 37))), n_ours)
        assert chunks == whole
        assert whole[0][1] == bench.game_seed(900001, 0) == 900001 * 100003 + 1
    assert bench.game_indices(5, offset=10) == range(10, 15) and bench.parse_game_range("3:9") == (3, 9)
    for bad in ("5:5", "x", "-1:3"):
        with pytest.raises(Exception):
            bench.parse_game_range(bad)
    # real games: one run of 4 vs two chunks, per-game outcomes identical
    common = ["--opponent", "random", "--spec", "heuristic", "--seed", "12", "--our-seats", "2"]
    assert bench.main(common + ["--games", "4", "--json", str(tmp_path / "full.json")]) == 0
    assert bench.main(common + ["--game-range", "0:3", "--json", str(tmp_path / "c1.json")]) == 0
    assert bench.main(common + ["--games", "1", "--game-offset", "3", "--json", str(tmp_path / "c2.json")]) == 0
    capsys.readouterr()
    key = ("game", "seed", "our_seats", "winner", "vps", "turns", "actions", "won")
    full = json.loads((tmp_path / "full.json").read_text())
    c1 = json.loads((tmp_path / "c1.json").read_text())
    c2 = json.loads((tmp_path / "c2.json").read_text())
    assert c1["game_range"] == [0, 3] and c2["game_range"] == [3, 4]
    assert ([[r[k] for k in key] for r in full["results"]]
            == [[r[k] for k in key] for r in c1["results"] + c2["results"]])
    with pytest.raises(SystemExit):
        bench.main(common + ["--game-range", "0:3", "--game-offset", "1"])


def test_bench_mixed_lineup_is_the_protocol_rule(bench):
    names = ["value", "alphabeta", "sameturn"]
    assert bench.MIXED_PERMUTATIONS == PS.MIXED_PERMUTATIONS and bench.CATANBOT == PS.CATANBOT
    assert all(bench.mixed_lineup(g, names) == PS.mixed_lineup(g, tuple(names)) for g in range(2400))
    assert all(bench.our_seats_for(g) == PS.our_seats_for(g, "1v3-mixed") for g in range(48))
    # the bench's game-range chunks give the same lineups as one run (a function of the game index only)
    assert [bench.mixed_lineup(g, names) for g in range(40, 80)] == [PS.mixed_lineup(g) for g in range(40, 80)]


def test_bench_counted_defaults_are_the_registered_ones(bench):
    from catanbot.bench.catanatron_adapter import DEFAULT_INFO_SAMPLES
    assert DEFAULT_INFO_SAMPLES == PS.PROTOCOL_INFO_SAMPLES == 4
    # what `--info counted` (default --info-samples, no --discards-public) records, and plain full information
    opts = {"info": "counted", "info_samples": DEFAULT_INFO_SAMPLES, "discards_public": False}
    assert bench.info_meta(opts) == COUNTED_META
    assert bench.info_meta({"info": "full", "info_samples": 4, "discards_public": False}) == {"mode": "full"}
    assert bench.info_meta(None) == {"mode": "full"}


@pytest.mark.parametrize("tid", ["T7", "T10", "T11"])
def test_bench_summary_metadata_passes_the_conformance_check(bench, tid, monkeypatch):
    """The metadata bench_catanatron.py writes for a protocol chunk (no game played: summarize() on
    protocol-shaped records) is exactly what prove_strength.py checks - field names and values."""
    from catanbot.bench import catanatron_adapter as AD
    if not AD.API_33:
        pytest.skip("catanatron's value / alphabeta / sameturn players need the 3.3 engine")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    t = PS.TESTS[tid]
    recs = synthetic(tid, 0.5)["results"][:48]
    for r in recs:
        r.update(our_vp=r["vps"][r["seat"]], opp_vps=[v for i, v in enumerate(r["vps"]) if i != r["seat"]],
                 duration=1.0)
    mixed = list(t.opponents) or None
    info = bench.info_meta({"info": t.info, "info_samples": AD.DEFAULT_INFO_SAMPLES, "discards_public": False})
    opponent = ",".join(mixed) if mixed else t.opponent       # as main() names a mixed run
    s = bench.summarize(recs, PS.PROTOCOL_SPEC, opponent, t.seed, "off", {}, our_seats=1, game_range=(0, 48),
                        mixed=mixed, info=info)
    s.update(vps_to_win=10, discard_limit=7)                   # as main() adds them
    s = json.loads(json.dumps(bench._strip_raw(s), default=str))
    meta = {k: v for k, v in s.items() if k != "results"}
    an = PS.analyze_test(tid, [("chunk.json", meta, s["results"])])
    assert an["problems"] == [f"48 of the {t.games} protocol games (missing 48-{t.games - 1})"], an["problems"]
    assert PS.infer_test(meta, s["results"]) == tid
    # the same chunk under the other information mode is a deviation
    other = {"info": "full"} if t.info == "counted" else {"info": "counted", "info_samples": 4}
    meta2 = dict(meta, info=bench.info_meta(other))
    assert any("information mode" in p for p in PS.analyze_test(tid, [("c.json", meta2, s["results"])])["problems"])


def test_rerun_crashes(bench, monkeypatch):
    calls = []
    real = bench.play_game

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(*a, **kw)

    monkeypatch.setattr(bench, "play_game", flaky)
    job = (2, 5, "heuristic", "random", 10, 7, "off", None, {"rerun_crashes": True})
    r = bench.run_one(job)
    assert len(calls) == 2 and r["crashes"] == 1 and "boom" in r["crash_errors"][0] and not r.get("crashed")

    def always(*a, **kw):
        raise RuntimeError("always")

    monkeypatch.setattr(bench, "play_game", always)
    r = bench.run_one(job)
    assert r["crashed"] and r["crashes"] == 2 and r["won"] is False and r["seat"] == 2 and r["winner"] is None
    s = bench.summarize([r], "heuristic", "random", 5)
    assert s["crashed_games"] == 1 and s["crash_attempts"] == 2 and s["truncated"] == 0
    with pytest.raises(RuntimeError):
        bench.run_one(job[:8])      # without --rerun-crashes a crash still aborts, as before


# ---------------------------------------------------------------------------
# Action logs and exact replays (2 real short games on this interpreter's catanatron)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_log(tmp_path_factory, bench):
    from catanbot.bench import catanatron_adapter as AD
    d = tmp_path_factory.mktemp("actionlog")
    opponent = "value" if AD.API_33 else "vf"
    rc = bench.main(["--games", "2", "--opponent", opponent, "--spec", SMALL, "--seed", "4242", "--workers", "2",
                     "--log-actions", str(d), "--rerun-crashes", "--json", str(d / "r.json")])
    assert rc == 0
    files = sorted(p for p in os.listdir(d) if p.endswith(".jsonl.gz"))
    assert files == [f"{opponent}_1v3_seed4242_g00000-00002.jsonl.gz"]
    return str(d / files[0]), json.loads((d / "r.json").read_text()), opponent


def test_action_log_contents(real_log):
    from catanbot.bench import catanatron_adapter as AD
    path, summary, opponent = real_log
    with gzip.open(path, "rt") as fh:
        recs = [json.loads(line) for line in fh]
    assert sorted(r["game"] for r in recs) == [0, 1]
    assert summary["action_log"]["games"] == 2 and summary["action_log"]["bytes"] == os.path.getsize(path)
    by_game = {r["game"]: r for r in summary["results"]}
    for rec in recs:
        assert rec["format"] == AD.LOG_FORMAT and rec["catanatron"] == AD.CATANATRON_VERSION
        assert rec["seed"] == by_game[rec["game"]]["seed"] and rec["base_seed"] == 4242
        assert rec["hash_seed"] == os.environ.get("PYTHONHASHSEED")
        land = [t for t in rec["board"]["tiles"] if t["t"] == "land"]
        assert len(land) == 19 and sum(t["resource"] is None for t in land) == 1
        assert all((t["number"] is None) == (t["resource"] is None) for t in land)
        assert len([t for t in rec["board"]["tiles"] if t["t"] == "port"]) == 9 and rec["board"]["robber"]
        assert len(rec["dev_deck"]) == 25 and rec["colors"] == [c.value for c in AD.COLORS]
        players = rec["players"]
        assert [p["kind"] for p in players].count("catanbot") == 1
        me = next(p for p in players if p["kind"] == "catanbot")
        assert me["spec"] == SMALL and me["seat"] == rec["game"] % 4 == rec["our_seats"][0]
        opp = next(p for p in players if p["kind"] == "opponent")
        assert opp["preset"] == opponent and ":" in opp["class"] and opp["params"] == {}
        rolls = [a for a in rec["actions"] if a[1] == "ROLL"]
        assert rolls and all(len(a[3]) == 2 and 2 <= sum(a[3]) <= 12 for a in rolls)
        assert all(a[0] in rec["colors"] for a in rec["actions"])
        buys = [a for a in rec["actions"] if a[1] == "BUY_DEVELOPMENT_CARD"]
        assert all(a[3] in ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY") for a in buys)
        steals = [a for a in rec["actions"] if a[1] == "MOVE_ROBBER" and a[2][1] is not None]
        assert all(a[3] in (None, "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE") for a in steals)
        fin = rec["final"]
        assert fin["vps"] == by_game[rec["game"]]["vps"] and fin["turns"] == by_game[rec["game"]]["turns"]
        assert fin["num_actions"] == len(rec["actions"]) == by_game[rec["game"]]["actions"]


def test_replay_check_passes_on_real_games(real_log, replay_mod, capsys):
    path, summary, _ = real_log
    assert replay_mod.check_files([path], verbose=True) == 0
    out = capsys.readouterr().out
    assert "2 games, all OK" in out and "KiB/game" in out
    assert replay_mod.main([path, "--check"]) == 0
    recs, note = replay_mod.read_records(path)
    assert note is None
    for rec in recs:
        ok, msg = replay_mod.check_record(rec)
        assert ok, msg
        res = next(r for r in summary["results"] if r["game"] == rec["game"])
        assert f"VPs {res['vps']}" in msg


def test_replay_prints_positions(real_log, replay_mod, capsys):
    path, _, _ = real_log
    assert replay_mod.main([path, "--game", "1", "--turn", "30", "--last", "5"]) == 0
    out = capsys.readouterr().out
    for text in ("players in turn order:", "board (land tiles", "ports (nodes)", "state after", "hand {",
                 "dev cards held", "settlements", "roads", "VP ", "bank", "last 5 action(s)", "catanbot"):
        assert text in out, text
    assert replay_mod.main([path, "--game", "0", "--action", "25", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["actions_applied"] == 25 and doc["state"]["actions"] == 25 and len(doc["state"]["players"]) == 4
    assert replay_mod.main([path, "--list"]) == 0
    assert "game     0" in capsys.readouterr().out


def test_replay_detects_a_tampered_log(real_log, replay_mod):
    path, _, _ = real_log
    recs, _ = replay_mod.read_records(path)
    rec = next(r for r in recs if any(a[1] == "ROLL" and sum(a[3]) == 7 for a in r["actions"]))
    # a 7 turned into a 6 or 8: the logged robber / discard that follows is no longer playable
    bad = json.loads(json.dumps(rec))
    i = next(i for i, a in enumerate(bad["actions"]) if a[1] == "ROLL" and sum(a[3]) == 7)
    d1, d2 = bad["actions"][i][3]
    new = [d1, d2 + 1] if d2 < 6 else [d1, d2 - 1]
    if bad["actions"][i][2] is not None:
        bad["actions"][i][2] = list(new)
    bad["actions"][i][3] = list(new)
    ok, msg = replay_mod.check_record(bad)
    assert not ok and "replay failed" in msg, msg
    # a different chance result than the one logged for the action (engine record differs)
    bad = json.loads(json.dumps(rec))
    bad["actions"][i][3] = [d1, d2 + 1] if d2 < 6 else [d1, d2 - 1]
    ok, msg = replay_mod.check_record(bad)
    assert not ok, msg
    # a wrong robber start (board mismatch) and a wrong final state
    bad = json.loads(json.dumps(rec))
    other = next(t for t in bad["board"]["tiles"] if t["t"] == "land" and t["resource"] is not None)
    bad["board"]["robber"] = other["c"]
    assert not replay_mod.check_record(bad)[0]
    bad = json.loads(json.dumps(rec))
    bad["final"]["vps"] = [v + 1 for v in bad["final"]["vps"]]
    ok, msg = replay_mod.check_record(bad)
    assert not ok and "vps" in msg


def test_replay_needs_the_engine_that_played(real_log, replay_mod):
    from catanbot.bench import catanatron_adapter as AD
    path, _, _ = real_log
    rec = replay_mod.read_records(path)[0][0]
    rec["api"] = "3.2" if AD.API_33 else "3.3"
    with pytest.raises(ValueError, match="replay it with the interpreter that played it"):
        AD.rebuild_game(rec)


def test_replay_cli_subprocess_and_log_overwrite(real_log, bench, tmp_path, capsys):
    path, _, opponent = real_log
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_catanatron.py"), path, "--check"],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "all OK" in proc.stdout, proc.stdout + proc.stderr
    # the same run twice into one directory: the second log replaces the first (no duplicate games)
    d = tmp_path / "again"
    for _ in range(2):
        assert bench.main(["--games", "1", "--opponent", "random", "--spec", "heuristic", "--seed", "9",
                           "--log-actions", str(d)]) == 0
    capsys.readouterr()
    (f,) = os.listdir(d)
    with gzip.open(d / f, "rt") as fh:
        assert len(fh.read().splitlines()) == 1
