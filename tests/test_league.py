"""Champion league: bot-server protocol, match runner, statistics, sequential gate, promotion, resume.

Fast by design: bot servers run from a temporary *copy* of the current tree (no git worktrees are
created), games use quick heuristic bots, the sequential design is checked by simulation.
"""
from __future__ import annotations

import json
import math
import os
import random
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from catanbot import actions as A
from catanbot.league import cli as L
from catanbot.league import gate as G
from catanbot.league import sequential as Q
from catanbot.league import stats as S
from catanbot.league.match import BotServer, InProcessSeat, ServerSeat, VersionSkewError, play_match_game
from catanbot.league.registry import Champion, LeagueError, Registry, copy_tree, materialize, spec_with_model
from catanbot.selfplay import make_bot
from catanbot.state import new_game

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tree_copy(tmp_path_factory):
    """A temporary copy of the current source tree (stands in for a materialised champion)."""
    dst = tmp_path_factory.mktemp("league_tree") / "tree"
    copy_tree(REPO, dst)
    return dst


def _server(tree, spec, tmp_path, label="s"):
    return BotServer(tree, spec, label, log_path=tmp_path / f"{label}.log", timeout=120).start()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _exact_sf(k, n, p):
    p = Fraction(p)
    return sum(Fraction(math.comb(n, i)) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_binomial_tails_match_exact_rational():
    rng = random.Random(3)
    for _ in range(60):
        n = rng.randint(1, 120)
        k = rng.randint(0, n)
        for p in (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)):
            exact = _exact_sf(k, n, p)
            got = S.binom_sf(k, n, float(p))
            assert got == pytest.approx(float(exact), rel=1e-10, abs=1e-300)
            exact_cdf = 1 - _exact_sf(k + 1, n, p)
            assert S.binom_cdf(k, n, float(p)) == pytest.approx(float(exact_cdf), rel=1e-10, abs=1e-300)
    # tiny tails keep their relative precision
    assert S.binom_sf(900, 1000, 0.5) == pytest.approx(float(_exact_sf(900, 1000, Fraction(1, 2))), rel=1e-9)


def test_clopper_pearson_defining_equations_and_scipy():
    for k, n in [(0, 10), (3, 10), (10, 10), (120, 200), (637, 1188)]:
        lo, hi = S.clopper_pearson(k, n, 0.95)
        if k > 0:
            assert S.binom_sf(k, n, lo) == pytest.approx(0.025, rel=1e-8)
        else:
            assert lo == 0.0
        if k < n:
            assert S.binom_cdf(k, n, hi) == pytest.approx(0.025, rel=1e-8)
        else:
            assert hi == 1.0
    scipy_stats = pytest.importorskip("scipy.stats")
    for k, n, p in [(12, 30, 0.5), (60, 100, 1 / 3), (7, 40, 0.25)]:
        ci = scipy_stats.binomtest(k, n, p).proportion_ci(0.95, "exact")
        assert S.clopper_pearson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-10)
        assert S.binom_test_two_sided(k, n, p) == pytest.approx(scipy_stats.binomtest(k, n, p).pvalue, rel=1e-9)
        assert S.fisher_less(k, n, n - k, n) == pytest.approx(
            scipy_stats.fisher_exact([[k, n - k], [n - k, k]], alternative="less").pvalue, rel=1e-9)


def test_holm_step_down():
    h = S.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert h["a"]["reject"] and not h["c"]["reject"] and not h["b"]["reject"]
    assert h["a"]["p_adj"] == pytest.approx(0.03) and h["c"]["p_adj"] == pytest.approx(0.06)
    assert h["b"]["p_adj"] == pytest.approx(0.06)   # monotone: max(0.06, 0.04)
    h = S.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert all(v["reject"] for v in h.values())


# ---------------------------------------------------------------------------
# sequential boundary
# ---------------------------------------------------------------------------
NS = [198 * k for k in range(1, 7)]


@pytest.mark.parametrize("spending", ["obf", "pocock"])
def test_sequential_type1_error_by_simulation(spending):
    # exact: the null crossing probability summed over the looks never exceeds alpha
    b = Q.upper_boundary(NS, 1188, 0.01, 0.5, spending, [False] * 5 + [True])
    assert b.spent[-1] <= 0.01
    assert all(x <= y + 1e-15 for x, y in zip(b.spent, b.allowed))
    # simulation: 2000 gates under the null (equal strength)
    r = Q.simulate_type1(0.5, NS, 1188, 0.01, 0.5, spending, 2000, seed=11)
    assert r["rate"] <= 0.01 + 3 * math.sqrt(0.01 * 0.99 / 2000)
    # a naive design (one-sided p < 0.01 at every look) is anti-conservative: the spending rule matters
    rng = np.random.default_rng(5)
    wins = np.cumsum(np.stack([rng.binomial(198, 0.5, 20000) for _ in NS], 1), 1)
    naive = np.array([min(k for k in range(n + 1) if S.binom_sf(k, n, 0.5) < 0.01) for n in NS])
    assert (wins >= naive).any(1).mean() > 0.02
    # power / early stopping for a clearly better candidate
    good = Q.simulate_type1(0.6, NS, 1188, 0.01, 0.5, spending, 2000, seed=12)
    assert good["rate"] > 0.99 and good["mean_stop_look"] < 4


def test_sequential_type1_with_random_draws_and_lower_boundary():
    """Batch sizes that depend on how many games were draws, and the mirrored failure boundary (3p null 1/3)."""
    rng = np.random.default_rng(2)
    sims, rej_up, rej_low = 400, 0, 0
    for _ in range(sims):
        ns, cum_n, cum_w = [], 0, 0
        up_crossed = low_crossed = False
        for k in range(4):
            m = 150 - rng.binomial(150, 0.03)          # turn-cap draws reduce the decisive games
            cum_n += m
            cum_w += rng.binomial(m, 1 / 3)
            ns.append(cum_n)
            fin = [False] * k + [k == 3]
            up = Q.upper_boundary(ns, 600, 0.05, 1 / 3, "obf", fin)
            low = Q.lower_boundary(ns, 600, 0.05, 1 / 3, "obf", fin)
            up_crossed |= cum_w >= up.crit[-1]
            low_crossed |= (cum_n - cum_w) >= low.crit[-1]
            assert up.spent[-1] <= 0.05 + 1e-12 and low.spent[-1] <= 0.05 + 1e-12
        rej_up += up_crossed
        rej_low += low_crossed
    bound = 0.05 + 3 * math.sqrt(0.05 * 0.95 / sims)
    assert rej_up / sims <= bound and rej_low / sims <= bound


# ---------------------------------------------------------------------------
# seats, seeds, rotation
# ---------------------------------------------------------------------------
def _cfg(**kw):
    base = dict(gate_id="t", candidate={"name": "cand", "spec": "heuristic", "spec_resolved": "heuristic",
                                        "commit": "c" * 40, "dirty": False, "tree": "."},
                champions=[{"name": "champion-1", "commit": "b" * 40, "spec": "heuristic:temp=1",
                            "spec_resolved": "heuristic:temp=1", "tree": "."},
                           {"name": "champion-0", "commit": "a" * 40, "spec": "heuristic:temp=0.3",
                            "spec_resolved": "heuristic:temp=0.3", "tree": "."}],
                batch_games=12, max_games=24)
    base.update(kw)
    return G.GateConfig(**base)


def test_seat_rotation_is_balanced_and_seeds_are_fixed_per_champion():
    cfg = _cfg(formats=["4p2v2", "3p1v2"], batch_games=198, max_games=396)
    fmt = G.parse_format("4p2v2")
    assert fmt.arrangements == list(combinations(range(4), 2)) and fmt.p0 == 0.5
    plans = [G.game_plan(cfg, ("champion-0", "4p2v2"), i) for i in range(198)]
    assert len({p["pattern"] for p in plans}) == 6
    for seat in range(4):   # every seat is a candidate seat in exactly half of the batch
        assert sum(seat in p["cand_seats"] for p in plans) == 99
    for b in range(33):     # a block = the 6 arrangements on one board/dice seed, same bot seed per seat
        blk = plans[6 * b:6 * b + 6]
        assert len({p["seed"] for p in blk}) == 1 and len({tuple(p["bot_seeds"]) for p in blk}) == 1
        assert sorted(p["pattern"] for p in blk) == sorted({"CCHH", "CHCH", "CHHC", "HCCH", "HCHC", "HHCC"})
    assert len({p["seed"] for p in plans}) == 33
    p3 = [G.game_plan(cfg, ("champion-0", "3p1v2"), i) for i in range(9)]
    assert [p["pattern"] for p in p3[:3]] == ["CHH", "HCH", "HHC"] and G.parse_format("3p1v2").p0 == pytest.approx(1 / 3)
    # seeds depend on (champion, format, block) and the seed base only - not on the candidate
    other = _cfg(candidate={"name": "x", "spec": "random", "spec_resolved": "random", "commit": None, "tree": "."},
                 batch_games=198, max_games=396)
    assert [G.game_plan(other, ("champion-0", "4p2v2"), i)["seed"] for i in range(12)] == [p["seed"] for p in plans[:12]]
    assert G.game_plan(cfg, ("champion-1", "4p2v2"), 0)["seed"] != plans[0]["seed"]
    assert G.game_plan(cfg, ("champion-0", "3p1v2"), 0)["seed"] != plans[0]["seed"]
    with pytest.raises(ValueError):
        G.parse_format("4p3v2")


# ---------------------------------------------------------------------------
# bot servers
# ---------------------------------------------------------------------------
def test_serve_round_trip_real_game_matches_in_process(tree_copy, tmp_path):
    """Two server seats (a heuristic bot and a search bot with an observe hook) from a copy of the tree play
    exactly the game the same bots play in-process."""
    specs = ["heuristic:temp=0.3", "heuristic", "search:depth=1,beam=2,expand=3,trades=1,evaluator=heuristic",
             "heuristic:temp=0.5"]

    def run(server_seats):
        seats = []
        for i, sp in enumerate(specs):
            if i in server_seats:
                seats.append(ServerSeat(_server(tree_copy, sp, tmp_path, f"s{i}")))
            else:
                seats.append(InProcessSeat(lambda sp=sp: make_bot(sp), spec=sp))
        log = []
        try:
            r = play_match_game(seats, 4242, [1, 2, 3, 4], 4, max_turns=400, action_log=log)
            info = [s.info() for s in seats]
        finally:
            for s in seats:
                s.close()
        return r, log, info

    r1, log1, info = run({0, 2})
    r2, log2, _ = run(set())
    assert info[0]["catanbot_file"].startswith(os.path.realpath(tree_copy))
    assert log1 == log2 and len(log1) == r1["actions"]
    assert (r1["winner"], r1["vps"], r1["turns"]) == (r2["winner"], r2["vps"], r2["turns"])
    assert r1["seats"][2]["observes"] == r1["actions"]          # the search bot observed every action
    assert r1["seats"][0]["observes"] == 0                      # heuristic: no observe hook -> skipped
    assert all(s["illegal"] == s["errors"] == 0 for s in r1["seats"])
    assert r1["seats"][0]["overhead_ms"] > 0 and r1["seats"][1]["overhead_ms"] == 0


def test_version_skew_fails_loudly_with_the_field_and_shim_can_adapt(tree_copy, tmp_path):
    srv = _server(tree_copy, "heuristic", tmp_path, "skew")
    try:
        assert srv.request({"cmd": "new_game", "seat": 0, "seed": 1, "num_players": 4})["ok"]
        st = new_game(4, rng=random.Random(1))
        from catanbot import engine as E
        legal = [A.to_json(a) for a in E.legal_actions(st)]
        d = st.to_dict()
        d["vp_to_win"] = 12                      # a field this commit does not know
        with pytest.raises(VersionSkewError, match="vp_to_win"):
            srv.request({"cmd": "decide", "state_id": 0, "state": d, "legal": legal})
        d2 = st.to_dict()
        del d2["hexes"]                          # a field this commit needs
        with pytest.raises(VersionSkewError, match="hexes"):
            srv.request({"cmd": "decide", "state_id": 1, "state": d2, "legal": legal})
        ok = srv.request({"cmd": "decide", "state_id": 2, "state": st.to_dict(), "legal": legal})
        assert A.from_json(ok["action"]) in E.legal_actions(st)
    finally:
        srv.close()
    shim = tmp_path / "shim.py"
    shim.write_text("IGNORE_FIELDS = {'vp_to_win'}\n")
    srv = BotServer(tree_copy, "heuristic", "shimmed", shim=str(shim), log_path=tmp_path / "shim.log").start()
    try:
        srv.request({"cmd": "new_game", "seat": 0, "seed": 1, "num_players": 4})
        r = srv.request({"cmd": "decide", "state_id": 0, "state": d, "legal": legal})
        assert "action" in r
    finally:
        srv.close()


class _BadBot:
    """Returns an illegal action at every 3rd decision and raises at every 7th."""

    name = "bad"

    def __init__(self):
        self.k = 0
        self.inner = make_bot("heuristic")

    def reset(self):
        self.k = 0

    def decide(self, state, legal, rng):
        self.k += 1
        if self.k % 7 == 0:
            raise RuntimeError("boom")
        if self.k % 3 == 0:
            return ("build_city", 999)
        return self.inner.decide(state, legal, rng)

    def observe(self, state, action, player):
        pass


def test_illegal_actions_are_counted_and_fall_back_to_first_legal():
    seats = [InProcessSeat(_BadBot, spec="bad")] + [InProcessSeat(lambda: make_bot("heuristic"), spec="heuristic")
                                                     for _ in range(3)]
    log = []
    r = play_match_game(seats, 99, [5, 6, 7, 8], 4, max_turns=150, action_log=log)
    bad = r["seats"][0]
    assert bad["illegal"] > 0 and bad["errors"] > 0
    assert bad["decisions"] // 3 - bad["decisions"] // 21 == bad["illegal"]
    assert bad["errors"] == bad["decisions"] // 7
    assert all(s["illegal"] == s["errors"] == 0 for s in r["seats"][1:])
    assert any("illegal action ('build_city', 999)" in e for e in r["error_log"])
    assert any("boom" in e for e in r["error_log"])
    assert r["actions"] == len(log) and r["turns"] > 0


def test_check_action_accepts_wellformed_extra_proposals_only():
    from catanbot.league.match import check_action
    from catanbot import engine as E
    rng = random.Random(0)
    s = new_game(4, rng=rng)
    bots = [make_bot("heuristic") for _ in range(4)]
    for _ in range(3000):      # advance to a main phase where proposals are legal
        legal = E.legal_actions(s)
        if s.phase == "main" and any(a[0] == A.PROPOSE_TRADE for a in legal):
            break
        s = E.apply(s, bots[E.acting_player(s)].decide(s, legal, rng), rng)
    legal = E.legal_actions(s)
    assert any(a[0] == A.PROPOSE_TRADE for a in legal)
    res = s.players[s.current].resources
    r = next(i for i in range(5) if res[i] > 0)
    give = tuple(1 if i == r else 0 for i in range(5))
    get = tuple(1 if i == (r + 1) % 5 else 0 for i in range(5))
    assert check_action(s, (A.PROPOSE_TRADE, give, get), legal, set(legal)) is None
    assert check_action(s, (A.PROPOSE_TRADE, give, give), legal, set(legal)) is not None
    assert check_action(s, ("build_city", 999), legal, set(legal)) is not None


# ---------------------------------------------------------------------------
# gate: mirror identity, resume, promotion
# ---------------------------------------------------------------------------
def _inproc_factory(cfg):
    specs = {"candidate": cfg.candidate["spec_resolved"], **{c["name"]: c["spec_resolved"] for c in cfg.champions}}
    return lambda who, slot: InProcessSeat(lambda s=specs[who]: make_bot(s), label=f"{who}#{slot}", spec=specs[who])


def test_identical_candidate_wins_exactly_half_of_every_block(tmp_path):
    """Seat seeds do not depend on the side: a candidate identical to the champion wins exactly 3 of 6."""
    cfg = _cfg(candidate={"name": "cand", "spec": "heuristic:temp=0.3", "spec_resolved": "heuristic:temp=0.3",
                          "commit": "c" * 40, "tree": "."},
               champions=[{"name": "champion-0", "commit": "a" * 40, "spec": "heuristic:temp=0.3",
                           "spec_resolved": "heuristic:temp=0.3", "tree": "."}],
               batch_games=12, max_games=12, max_turns=200)
    v = G.run_gate(cfg, tmp_path / "g", seat_factory=_inproc_factory(cfg), log=lambda *a: None)
    recs = G.load_records(tmp_path / "g")[("champion-0", "4p2v2")]
    for b in range(2):
        blk = [recs[i] for i in range(6 * b, 6 * b + 6)]
        assert len({(r["winner"], tuple(r["vps"])) for r in blk}) == 1
        assert sum(bool(r["cand_win"]) for r in blk) == (3 if blk[0]["winner"] >= 0 else 0)
    s = v["comparisons"]["champion-0/4p2v2"]
    assert s["share"] == 0.5 and v["verdict"] == "FAIL"


def test_gate_resume_reproduces_uninterrupted_run(tmp_path):
    cfg = _cfg(batch_games=3, max_games=6, max_turns=200, rule=G.GateRule(futility_margin=None))
    fac = _inproc_factory(cfg)
    quiet = lambda *a: None  # noqa: E731
    full = G.run_gate(cfg, tmp_path / "full", seat_factory=fac, log=quiet)
    r = G.run_gate(cfg, tmp_path / "part", seat_factory=fac, log=quiet, stop_after=5)
    assert r == {"interrupted": True, "games_done": 5}
    assert json.loads((tmp_path / "part" / "progress.json").read_text())["state"] == "interrupted"
    st = G.gate_status(tmp_path / "part")
    assert st["state"] == "stopped" and st["complete_looks"] == 0
    # a line cut by a crash mid-write is ignored on resume
    with open(tmp_path / "part" / "games.jsonl", "a") as f:
        f.write('{"type": "game", "champion": "champion-1", "form')
    r = G.run_gate(cfg, tmp_path / "part", seat_factory=fac, log=quiet, stop_after=4)   # ends inside look 2
    assert r["interrupted"] and G.gate_status(tmp_path / "part")["complete_looks"] == 1
    resumed = G.run_gate(cfg, tmp_path / "part", seat_factory=fac, log=quiet)
    a, b = G.load_records(tmp_path / "full"), G.load_records(tmp_path / "part")
    assert a.keys() == b.keys()
    for c in a:
        assert sorted(a[c]) == sorted(b[c]) == list(range(6))
        for i in a[c]:
            assert (a[c][i]["winner"], a[c][i]["vps"], a[c][i]["seed"]) == (b[c][i]["winner"], b[c][i]["vps"], b[c][i]["seed"])
    stat = lambda v: {k: (s["wins"], s["decisive"], s["p_greater"], s["vp_diff"]) for k, s in v["comparisons"].items()}  # noqa: E731
    assert stat(full) == stat(resumed) and full["verdict"] == resumed["verdict"]
    looks = [json.loads(x) for x in (tmp_path / "part" / "looks.jsonl").read_text().splitlines()]
    assert [x["look"] for x in looks] == [1, 2]


def _synth(gate_dir: Path, cfg: G.GateConfig, shares, errors=0):
    """Fabricated game records: ``shares[(champion, format)] = list of candidate win shares per look``."""
    gate_dir.mkdir(parents=True, exist_ok=True)
    G.atomic_write_json(gate_dir / "gate.json", cfg.to_dict())
    lines = []
    for comp, per_look in shares.items():
        fmt = G.parse_format(comp[1])
        idx = 0
        for k, share in enumerate(per_look, 1):
            end = cfg.look_end(k)
            n = end - idx
            wins = round(share * n)
            for j in range(n):
                plan = G.game_plan(cfg, comp, idx)
                sides = [("C" if s in plan["cand_seats"] else "H") for s in range(fmt.n)]
                win = j < wins
                w = plan["cand_seats"][0] if win else next(s for s in range(fmt.n) if sides[s] == "H")
                st = [{"decisions": 10, "nontrivial": 8, "decide_ms_nontrivial": 1.0, "overhead_ms": 0.5,
                       "illegal": (errors if (idx == 0 and s == 0) else 0), "errors": 0, "observe_errors": 0}
                      for s in range(fmt.n)]
                lines.append({"type": "game", "champion": comp[0], "format": comp[1], "index": idx,
                              "pattern": plan["pattern"], "cand_seats": plan["cand_seats"], "sides": sides,
                              "winner": w, "cand_win": win, "vp_diff": 1.0 if win else -1.0, "turns": 80,
                              "seat_stats": st, "void": None, "seed": plan["seed"], "vps": [0] * fmt.n})
                idx += 1
    with open(gate_dir / "games.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")


def _registry(tmp_path, names=("champion-0", "champion-1")):
    reg = Registry(tmp_path / "league" / "champions.json",
                   [Champion(n, ("a", "b", "c")[i] * 40, "heuristic", created="2026-09-25") for i, n in enumerate(names)],
                   {"league": "test"})
    reg.save()
    return Registry.load(reg.path)


def test_promotion_logic_on_synthetic_results(tmp_path):
    quiet = lambda *a: None  # noqa: E731
    base = dict(batch_games=198, max_games=594)
    c1, c0 = ("champion-1", "4p2v2"), ("champion-0", "4p2v2")
    # PASS: clearly better than the current champion, level with the earlier one
    cfg = _cfg(**base)
    _synth(tmp_path / "pass", cfg, {c1: [0.6, 0.6], c0: [0.5, 0.52]})
    v = G.run_gate(cfg, tmp_path / "pass", seat_factory=lambda *a: None, log=quiet)
    assert v["verdict"] == "PASS" and v["stop"] == {"look": 2, "reason": "success", "crossed_fail": []}
    assert v["a"]["p_greater"] < 0.01 and v["a"]["ci"][0] > 0.5
    reg = _registry(tmp_path)
    champ = G.promote(tmp_path / "pass", reg, name="champion-2", notes="synthetic")
    reg2 = Registry.load(reg.path)
    assert [c.name for c in reg2.champions] == ["champion-0", "champion-1", "champion-2"]
    assert reg2.current.commit == "c" * 40 and champ.gate["verdict"] == "PASS"
    assert champ.gate["records_sha256"] and champ.gate["comparisons"]["champion-1/4p2v2"]["wins"] == 238
    with pytest.raises(LeagueError, match="ladder changed"):     # the ladder moved on: gate is stale
        G.promote(tmp_path / "pass", reg2, name="champion-3")
    # FAIL (b): better than the current champion but significantly worse than an earlier one
    _synth(tmp_path / "b", cfg, {c1: [0.6, 0.6], c0: [0.45, 0.42]})
    v = G.run_gate(cfg, tmp_path / "b", seat_factory=lambda *a: None, log=quiet)
    assert v["a"]["passed"] and not v["b"]["passed"] and v["verdict"] == "FAIL"
    with pytest.raises(LeagueError, match="FAIL"):
        G.promote(tmp_path / "b", _registry(tmp_path / "r2"))
    # interim failure: clearly worse than the current champion at the first look
    _synth(tmp_path / "f", cfg, {c1: [0.35], c0: [0.5]})
    v = G.run_gate(cfg, tmp_path / "f", seat_factory=lambda *a: None, log=quiet)
    assert v["stop"]["reason"] == "fail" and v["verdict"] == "FAIL" and "champion-1/4p2v2" in v["b"]["interim_fail"]
    # futility: level with the current champion after half of the games
    _synth(tmp_path / "fut", cfg, {c1: [0.5, 0.48], c0: [0.5, 0.5]})
    v = G.run_gate(cfg, tmp_path / "fut", seat_factory=lambda *a: None, log=quiet)
    assert v["stop"]["reason"] == "futility" and v["verdict"] == "FAIL"
    # invalid: an illegal action anywhere with max_errors = 0
    _synth(tmp_path / "inv", cfg, {c1: [0.6, 0.6], c0: [0.5, 0.52]}, errors=1)
    v = G.run_gate(cfg, tmp_path / "inv", seat_factory=lambda *a: None, log=quiet)
    assert v["verdict"] == "INVALID"


def test_holm_makes_criterion_b_family_wise(tmp_path):
    """Three earlier champions each at p(worse) ~0.03: raw tests would reject, Holm does not."""
    champs = [{"name": f"champion-{i}", "commit": str(i) * 40, "spec": "heuristic", "spec_resolved": "heuristic",
               "tree": "."} for i in (3, 2, 1, 0)]
    shares = {("champion-3", "4p2v2"): [0.6, 0.6]}
    for i in (2, 1, 0):
        shares[(f"champion-{i}", "4p2v2")] = [0.46, 0.445]
    quiet = lambda *a: None  # noqa: E731
    for holm_on, expect in ((True, "PASS"), (False, "FAIL")):
        cfg = _cfg(champions=champs, batch_games=198, max_games=594, rule=G.GateRule(holm=holm_on))
        d = tmp_path / f"holm{holm_on}"
        _synth(d, cfg, shares)
        v = G.run_gate(cfg, d, seat_factory=lambda *a: None, log=quiet)
        raw = [x["p"] for x in v["b"]["family"].values()]
        assert all(0.01 < p < 0.05 for p in raw)
        assert v["verdict"] == expect


def test_external_hook(tmp_path):
    out = tmp_path / "ext" / "r.json"
    tmpl = "{python} -c \"import json,sys; json.dump({{'wins': 31, 'games': 100}}, open(sys.argv[1], 'w'))\" {out}"
    res = G.run_external_command(tmpl, str(tmp_path), "heuristic", None, out, tmp_path / "ext.log")
    assert res["wins"] == 31 and res["games"] == 100
    assert G.compare_external(res, {"wins": 30, "games": 100})["status"] == "pass"
    assert G.compare_external(res, {"wins": 50, "games": 100})["status"] == "fail"
    assert G.compare_external(res, None)["status"] == "no baseline"
    quiet = lambda *a: None  # noqa: E731
    for champ_wins, expect in ((30, "PASS"), (60, "FAIL")):
        cfg = _cfg(batch_games=198, max_games=594,
                   external={"cmd": "unused", "key": "catanatron", "baseline": {"wins": champ_wins, "games": 100}})
        d = tmp_path / f"e{champ_wins}"
        _synth(d, cfg, {("champion-1", "4p2v2"): [0.6, 0.6], ("champion-0", "4p2v2"): [0.5, 0.5]})
        v = G.run_gate(cfg, d, seat_factory=lambda *a: None, log=quiet, external_runner=lambda c, g: res)
        assert v["verdict"] == expect and v["c"]["status"] == ("pass" if expect == "PASS" else "fail")


# ---------------------------------------------------------------------------
# CLI end to end with bot servers (materialised from a tree copy, not a worktree)
# ---------------------------------------------------------------------------
def test_cli_gate_with_servers_status_and_promote_refusal(tree_copy, tmp_path, capsys):
    reg = Registry(tmp_path / "league" / "champions.json",
                   [Champion("champion-0", "a" * 40, "heuristic", created="2026-09-25", notes="test seed")], {})
    reg.save()
    common = ["--registry", str(reg.path), "--gates-dir", str(tmp_path / "gates"), "--league-home",
              str(tmp_path / "home")]
    assert L.main(common + ["materialize", "champion-0", "--method", "copy", "--source", str(tree_copy),
                            "--no-build"]) == 0
    assert (tmp_path / "home" / "champion-0" / "materialized.json").exists()
    assert L.main(common + ["materialize", "champion-0", "--method", "copy", "--source", str(tree_copy),
                            "--no-build"]) == 0         # idempotent
    rc = L.main(common + ["gate", "--candidate-spec", "heuristic:temp=0.3", "--candidate-tree", str(tree_copy),
                          "--gate-id", "g1", "--batch-games", "2", "--max-games", "2", "--max-turns", "120",
                          "--allow-no-accel", "--method", "copy", "--source", str(tree_copy), "--no-build",
                          "--stop-after", "1"])
    assert rc == 3                                       # interrupted after one game
    assert L.main(common + ["gate", "--resume", "g1"]) == 1   # resumed; 2 games cannot pass
    gd = tmp_path / "gates" / "g1"
    recs = [json.loads(x) for x in (gd / "games.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and all(r["specs"].count("heuristic:temp=0.3") == 2 for r in recs)
    assert all(len(r["seat_stats"]) == 4 and r["seat_stats"][0]["decisions"] > 0 for r in recs)
    servers = json.loads((gd / "servers.json").read_text())
    assert servers["candidate"]["catanbot_file"].startswith(os.path.realpath(tree_copy))
    assert json.loads((gd / "verdict.json").read_text())["verdict"] == "FAIL"
    capsys.readouterr()
    assert L.main(common + ["status"]) == 0
    out = capsys.readouterr().out
    assert "champion-0" in out and "g1" in out and "FAIL" in out
    assert L.main(common + ["promote", "g1"]) == 2       # refused: not a PASS
    assert len(Registry.load(reg.path).champions) == 1


def test_materialize_weights_and_spec(tmp_path, tree_copy):
    w = tmp_path / "w.npz"
    w.write_bytes(b"weights")
    import hashlib
    sha = hashlib.sha256(b"weights").hexdigest()
    reg = Registry(tmp_path / "league" / "champions.json",
                   [Champion("c0", "a" * 40, "search:depth=1,model=models/value_net.npz",
                             weights={"path": str(w), "sha256": sha})], {})
    reg.save()
    m = materialize(reg.champions[0], tmp_path / "home", method="copy", source=tree_copy, build=False, registry=reg,
                    log=lambda *a: None)
    assert m.weights.read_bytes() == b"weights" and m.spec == spec_with_model(reg.champions[0].spec, str(m.weights))
    assert m.spec.endswith(f"model={m.weights}") and "models/value_net.npz" not in m.spec
    bad = Champion("c1", "b" * 40, "heuristic", weights={"path": str(w), "sha256": "0" * 64})
    with pytest.raises(LeagueError, match="sha256"):
        materialize(bad, tmp_path / "home", method="copy", source=tree_copy, build=False, log=lambda *a: None)


def test_seeded_registry_file():
    reg = Registry.load(REPO / "league" / "champions.json")
    c0 = reg.champions[0]
    assert c0.name == "champion-0" and c0.commit.startswith("9984181")
    assert c0.spec == "search:depth=1,beam=4,expand=8,evaluator=heuristic" and c0.weights is None
    assert c0.notes == "default bot at the strength proof"


def test_parallel_workers_give_the_same_records(tmp_path):
    """Game outcomes depend only on the seeds: two worker threads reproduce the single-worker run."""
    quiet = lambda *a: None  # noqa: E731
    out = {}
    for w in (1, 2):
        cfg = _cfg(batch_games=4, max_games=4, max_turns=150, workers=w)
        G.run_gate(cfg, tmp_path / f"w{w}", seat_factory=_inproc_factory(cfg), log=quiet)
        recs = G.load_records(tmp_path / f"w{w}")
        out[w] = {(c, i): (r["winner"], r["vps"], r["turns"]) for c, d in recs.items() for i, r in d.items()}
    assert len(out[1]) == 8 and out[1] == out[2]
