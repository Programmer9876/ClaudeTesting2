import random

import numpy as np
import pytest

from catanbot import board as B
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.state import PHASE_GAME_OVER, new_game
from catanbot.vision import synth
from catanbot.vision.colonist import (Calibration, calibrate_from_image, constrained_assignment, draw_debug,
                                      hungarian, parse_image)
from catanbot.vision.schema import state_to_parsed


def played_state(seed, players=4, turns=50):
    rng = random.Random(seed)
    s = new_game(players, rng=rng)
    bots = [HeuristicBot(temperature=0.3) for _ in range(players)]
    while s.phase != PHASE_GAME_OVER and s.turn < turns:
        i = E.acting_player(s)
        s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
    s.dice = rng.randint(2, 12)
    return s


def render(s, size, seed):
    geom = synth.board_geometry(size, seed, True)
    return synth.render_state(s, size=size, seed=seed, me=0, jitter=True, geometry=geom), geom


def test_hungarian_and_constrained_assignment():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    assign = hungarian(cost)
    assert sorted(assign) == [0, 1, 2]
    assert sum(cost[i, j] for i, j in enumerate(assign)) == 5
    # two rows want class 0 but only one slot: the cheaper one gets it
    cost2 = np.array([[0.0, 5.0], [1.0, 5.0], [9.0, 0.0]])
    a2 = constrained_assignment(cost2, [1, 2])
    assert a2 == [0, 1, 1]


@pytest.mark.parametrize("seed,players,size", [(0, 4, (1280, 800)), (1, 3, (1920, 1080)), (2, 4, (800, 500)),
                                               (3, 3, (1366, 768))])
def test_round_trip(seed, players, size):
    s = played_state(seed, players, 40 + 10 * seed)
    img, geom = render(s, size, seed)
    res = parse_image(img, me=s.players[0].color)
    g = res.debug["geometry"]
    hs = geom["hex_size"]
    assert abs(g["cx"] - geom["cx"]) < 0.15 * hs and abs(g["cy"] - geom["cy"]) < 0.15 * hs
    assert abs(g["hex_size"] / hs - 1.0) < 0.05
    truth = state_to_parsed(s, me=0)
    hex_ok = sum(a["resource"] == b["resource"] for a, b in zip(res.parsed["hexes"], truth["hexes"]))
    num_ok = sum(a["number"] == b["number"] for a, b in zip(res.parsed["hexes"], truth["hexes"]))
    assert hex_ok >= 18 and num_ok >= 18
    assert res.parsed["robber"] == truth["robber"]
    ps = {p["color"]: p for p in res.parsed["players"]}
    tp_all = 0
    fp = fn = 0
    for tp in truth["players"]:
        p = ps[tp["color"]]
        ts = set(tp["settlements"]) | {("c", v) for v in tp["cities"]}
        ds = set(p["settlements"]) | {("c", v) for v in p["cities"]}
        tp_all += len(ts & ds)
        fp += len(ds - ts)
        fn += len(ts - ds)
        assert set(p["roads"]) == set(tp["roads"])
        assert p["vp"] == tp["vp"] and p["cards"] == tp["cards"]
    f1 = 2 * tp_all / max(1, 2 * tp_all + fp + fn)
    assert f1 >= 0.9
    me = ps[s.players[0].color]
    assert me["resources"] == {B.RESOURCE_NAMES[r]: s.players[0].resources[r] for r in range(5)}
    assert res.parsed["dice"] == s.dice
    assert res.parsed["current_player"] == s.players[s.current].color
    # ports: those not hidden by the hand bar must be right
    tports = {p["edge"]: p["type"] for p in truth.get("ports") or []}
    dports = {p["edge"]: p["type"] for p in res.parsed.get("ports") or []}
    assert all(tports.get(e) == t for e, t in dports.items())
    assert len(dports) >= 6
    # the state built from the parse is usable by the engine
    assert res.state.num_players == players
    assert E.legal_actions(res.state)


def test_robber_on_numbered_hex_and_debug_overlay(tmp_path):
    s = played_state(5, 4, 45)
    s.robber = next(i for i, (r, n) in enumerate(s.hexes) if r != B.DESERT and n == 8)
    img, geom = render(s, (1280, 800), 5)
    res = parse_image(img, me=s.players[0].color)
    assert res.parsed["robber"] == s.robber
    assert res.parsed["hexes"][s.robber]["number"] == 8
    out = draw_debug(img, res)
    out.save(tmp_path / "debug.png")
    assert out.size == img.size


def test_parse_accepts_path_and_array(tmp_path):
    s = played_state(6, 3, 30)
    img, geom = render(s, (1000, 640), 6)
    path = tmp_path / "shot.png"
    img.save(path)
    a = parse_image(str(path))
    b = parse_image(np.asarray(img))
    assert a.parsed["hexes"] == b.parsed["hexes"]
    assert a.parsed["robber"] == b.parsed["robber"]
    assert "me" in a.parsed and a.warnings is not None


def test_calibrate_from_image():
    s = played_state(7, 4, 40)
    img, geom = render(s, (1280, 800), 7)
    cal = calibrate_from_image(img, s)
    assert isinstance(cal, Calibration)
    for r in range(6):
        ref = Calibration().tile[r]
        assert all(abs(cal.tile[r][k] - ref[k]) < 40 for k in range(3))
    res = parse_image(img, me=s.players[0].color, calibration=cal)
    truth = state_to_parsed(s, me=0)
    assert sum(a["resource"] == b["resource"] for a, b in zip(res.parsed["hexes"], truth["hexes"])) >= 18
