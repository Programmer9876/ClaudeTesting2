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


# ---------------------------------------------------------------------------
# Regression tests for the robustness review (vision-cv-1 .. vision-cv-15)
# ---------------------------------------------------------------------------
import tracemalloc
from dataclasses import replace

from PIL import Image, ImageFilter

from catanbot.vision import colonist as C


def _truth_pieces(s):
    truth = state_to_parsed(s, me=0)
    return {p["color"]: (set(p["settlements"]), set(p["cities"]), set(p["roads"])) for p in truth["players"]}


def _piece_f1(res, s):
    tp = fp = fn = 0
    found = {p["color"]: (set(p["settlements"]), set(p["cities"]), set(p["roads"])) for p in res.parsed["players"]}
    for c, (ts, tc, tr) in _truth_pieces(s).items():
        ds, dc, dr = found.get(c, (set(), set(), set()))
        t_all = {("s", v) for v in ts} | {("c", v) for v in tc} | {("r", e) for e in tr}
        d_all = {("s", v) for v in ds} | {("c", v) for v in dc} | {("r", e) for e in dr}
        tp += len(t_all & d_all)
        fp += len(d_all - t_all)
        fn += len(t_all - d_all)
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _board_ok(res, s):
    truth = state_to_parsed(s, me=0)
    hex_ok = sum(a["resource"] == b["resource"] for a, b in zip(res.parsed["hexes"], truth["hexes"]))
    num_ok = sum(a["number"] == b["number"] for a, b in zip(res.parsed["hexes"], truth["hexes"]))
    return hex_ok, num_ok


@pytest.mark.parametrize("seed", [52, 54])
def test_robber_found_on_crowded_hex_under_jpeg(seed):
    """vision-cv-1: the pawn's dark blob merges with the outlines of pieces on the hex corners."""
    s = played_state(seed, 4, 60 + (seed % 3) * 30)
    # put the robber on the numbered hex with the most buildings on its corners (>= 3)
    buildings = {v for p in s.players for v in p.settlements + p.cities}
    crowd = {h: sum(v in buildings for v in B.HEX_VERTICES[h]) for h, (r, n) in enumerate(s.hexes) if r != B.DESERT}
    s.robber = max(crowd, key=crowd.get)
    assert crowd[s.robber] >= 3
    geom = synth.board_geometry((1280, 800), seed, True)
    style = synth.ColonistStyle(jpeg_quality=45)
    img = synth.render_state(s, size=(1280, 800), seed=seed, me=0, geometry=geom, style=style, jpeg=True)
    res = parse_image(img, me=s.players[0].color)
    assert res.parsed["robber"] == s.robber
    assert res.confidence["robber"] > 0
    assert not any("robber pawn not found" in w for w in res.warnings)


def test_missing_robber_is_reported_with_zero_confidence():
    s = played_state(0, 4, 40)
    s.robber = -1   # nothing drawn
    desert = next(i for i, (r, n) in enumerate(s.hexes) if r == B.DESERT)
    img, geom = render(s, (1280, 800), 0)
    res = parse_image(img, me="red")
    assert res.parsed["robber"] == desert
    assert res.confidence["robber"] == 0.0
    assert any("robber pawn not found" in w and "--fix robber=" in w for w in res.warnings)


def test_board_partly_under_panel_registers_correctly():
    """vision-cv-2: the lattice must not lock onto a registration shifted by a row / column."""
    s = played_state(4, 4, 45)
    w, h = 1280, 800
    g0 = synth.board_geometry((w, h), 4, False)
    geom = dict(g0, cx=0.30 * w, cy=0.50 * h)   # left column under the player panel
    img = synth.render_state(s, size=(w, h), seed=4, me=0, geometry=geom, jitter=False)
    res = parse_image(img, me="red")
    g = res.debug["geometry"]
    hs = geom["hex_size"]
    assert abs(g["cx"] - geom["cx"]) < 0.1 * hs and abs(g["cy"] - geom["cy"]) < 0.1 * hs
    hex_ok, num_ok = _board_ok(res, s)
    assert hex_ok == 19 and num_ok >= 17
    assert res.confidence["geometry"] > 0.8
    # every player is still listed (green's panel row merges with its pieces on the board)
    assert {p["color"] for p in res.parsed["players"]} == {p.color for p in s.players}


def test_cropped_board_registers_and_warns():
    s = played_state(4, 4, 45)
    w, h = 1280, 800
    geom = synth.board_geometry((w, h), 4, False)
    img = synth.render_state(s, size=(w, h), seed=4, me=0, geometry=geom, jitter=False)
    x0 = int(geom["cx"] - 1.0 * geom["hex_size"])
    img = img.crop((x0, 0, w, h))
    res = parse_image(img, me="red")
    g = res.debug["geometry"]
    hs = geom["hex_size"]
    assert abs(g["cx"] - (geom["cx"] - x0)) < 0.1 * hs and abs(g["cy"] - geom["cy"]) < 0.1 * hs
    assert any("partially hidden" in w for w in res.warnings)
    truth = state_to_parsed(s, me=0)
    visible = [i for i, (cx, cy) in enumerate(C._lattice_pixels(g, "hex")) if cx > 0.9 * hs]
    assert all(res.parsed["hexes"][i]["resource"] == truth["hexes"][i]["resource"] for i in visible)


def test_missing_panel_row_does_not_delete_a_player():
    """vision-cv-3: pieces of a colour without a panel row are kept (with a warning)."""
    s = played_state(25, 4, 60)
    w, h = 1280, 800
    geom = synth.board_geometry((w, h), 25, False)
    img = synth.render_state(s, size=(w, h), seed=25, me=0, geometry=geom, jitter=False)
    row_h = min(0.10 * h, 0.7 * h / 4)
    y_cut = int(0.02 * h + 1.5 * (row_h + 0.06 * row_h))   # the first row (red) is cut away
    img = img.crop((0, y_cut, w, h))
    res = parse_image(img, me="red")
    colors = {p["color"] for p in res.parsed["players"]}
    assert colors == {p.color for p in s.players}
    assert res.parsed["me"] == "red"
    assert any(w.startswith("red: pieces found but no panel row") for w in res.warnings)
    assert _piece_f1(res, s) >= 0.95
    assert not any("--me red is not among" in w for w in res.warnings)


@pytest.mark.parametrize("factor", [0.7, 1.15])
def test_global_brightness_change(factor):
    """vision-cv-4: a darker / brighter screenshot is parsed after adapting the reference colours."""
    s = played_state(3, 4, 45)
    geom = synth.board_geometry((1280, 800), 3, False)
    img = synth.render_state(s, size=(1280, 800), seed=3, me=0, geometry=geom, jitter=False)
    arr = np.clip(np.asarray(img).astype(np.float32) * factor, 0, 255).astype(np.uint8)
    res = parse_image(arr, me="red")
    hex_ok, num_ok = _board_ok(res, s)
    assert hex_ok == 19 and num_ok == 19
    assert res.parsed["robber"] == s.robber
    assert _piece_f1(res, s) >= 0.95
    me = next(p for p in res.parsed["players"] if p["color"] == "red")
    assert me["resources"] == {B.RESOURCE_NAMES[r]: s.players[0].resources[r] for r in range(5)}
    assert res.parsed["dice"] == s.dice
    if factor < 0.9:
        assert "brightness_ratio" in res.debug and abs(res.debug["brightness_ratio"] - factor) < 0.1


def test_shifted_sea_and_tile_colours_give_no_false_pieces():
    """vision-cv-5 / vision-cv-13: a hue-shifted sea or shifted tiles must not read as player pieces."""
    import colorsys
    base = synth.ColonistStyle()
    hh, ss, vv = colorsys.rgb_to_hsv(*(c / 255.0 for c in base.sea))
    r, g, b = colorsys.hsv_to_rgb((hh + 20 / 360.0) % 1.0, ss, vv)
    s = played_state(3, 4, 45)
    geom = synth.board_geometry((1280, 800), 3, False)
    for style in (replace(base, sea=(int(r * 255), int(g * 255), int(b * 255))),
                  replace(base, tile={k: tuple(min(255, x + 25) for x in v) for k, v in base.tile.items()})):
        img = synth.render_state(s, size=(1280, 800), seed=3, me=0, geometry=geom, jitter=False, style=style)
        res = parse_image(img, me="red")
        assert _piece_f1(res, s) == 1.0, res.warnings
        assert _board_ok(res, s) == (19, 19)
        assert not any("classified as desert" in w for w in res.warnings)


@pytest.mark.parametrize("seed,size,turns", [(36, (1024, 640), 100), (37, (1440, 900), 140)])
def test_city_types_and_panel_numbers_late_game(seed, size, turns):
    """vision-cv-6 / vision-cv-7: no city/settlement confusion on inverted-Y vertices, no panel misreads."""
    s = played_state(seed, 4, turns)
    geom = synth.board_geometry(size, seed, True)
    img = synth.render_state(s, size=size, seed=seed, me=0, geometry=geom)
    res = parse_image(img, me=s.players[0].color)
    truth = state_to_parsed(s, me=0)
    n_cities = sum(len(p["cities"]) for p in truth["players"])
    assert n_cities >= 5
    for tp in truth["players"]:
        p = next(pp for pp in res.parsed["players"] if pp["color"] == tp["color"])
        assert set(p["cities"]) == set(tp["cities"]), (tp["color"], p["cities"], tp["cities"])
        assert set(p["settlements"]) == set(tp["settlements"])
        assert (p["vp"], p["cards"], p["dev_cards"], p["knights"]) == (
            tp["vp"], tp["cards"], tp["dev_cards"] if not isinstance(tp["dev_cards"], dict) else sum(tp["dev_cards"].values()),
            tp["knights"])
    assert res.parsed["robber"] == s.robber
    assert res.debug["city_hits"]


def test_hidden_number_tokens_are_reported_not_guessed():
    """vision-cv-10: tokens under the hand bar get number None and one warning with the leftovers."""
    s = played_state(4, 4, 45)
    w, h = 1280, 800
    g0 = synth.board_geometry((w, h), 4, False)
    geom = dict(g0, cx=0.55 * w, cy=0.60 * h)   # bottom row under the hand bar
    img = synth.render_state(s, size=(w, h), seed=4, me=0, geometry=geom, jitter=False)
    res = parse_image(img, me="red")
    truth = state_to_parsed(s, me=0)
    hidden = [i for i in range(B.NUM_HEXES) if res.parsed["hexes"][i]["number"] is None and truth["hexes"][i]["number"]]
    assert len(hidden) >= 2
    leftover = sorted(truth["hexes"][i]["number"] for i in hidden)
    assert any("hidden" in w and str(leftover) in w for w in res.warnings), res.warnings
    for i in range(B.NUM_HEXES):
        if i not in hidden:
            assert res.parsed["hexes"][i]["number"] == truth["hexes"][i]["number"]
    assert res.confidence["numbers_min"] == 0.0
    assert res.state.hexes[hidden[0]][1] == 0


def test_single_hidden_token_gets_the_leftover_number():
    s = played_state(6, 4, 40)
    img, geom = render(s, (1280, 800), 6)
    arr = np.asarray(img).copy()
    hidden = next(i for i, (r, n) in enumerate(s.hexes) if r != B.DESERT and i != s.robber)
    cx, cy = synth.pixel_of_hex(hidden, geom)
    hs = geom["hex_size"]
    ys, xs = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    disc = np.hypot(xs - cx, ys - cy) < 0.40 * hs
    arr[disc] = synth.ColonistStyle().tile[s.hexes[hidden][0]]   # paint the token away
    res = parse_image(arr, me="red")
    assert res.parsed["hexes"][hidden]["number"] == s.hexes[hidden][1]
    assert any(f"hex {hidden}: number token not clearly visible" in w for w in res.warnings)


def test_noisy_image_is_filtered_and_blurred_fallback_is_flagged():
    """vision-cv-14: Gaussian noise is median-filtered; an unreliable blob fallback is flagged."""
    s = played_state(40, 4, 80)
    geom = synth.board_geometry((1280, 800), 40, False)
    img = synth.render_state(s, size=(1280, 800), seed=40, me=0, geometry=geom, jitter=False)
    rng = np.random.default_rng(1)
    noisy = np.clip(np.asarray(img).astype(np.float32) + rng.normal(0, 20, np.asarray(img).shape), 0, 255).astype(np.uint8)
    res = parse_image(noisy, me="red")
    assert any("median filter" in w for w in res.warnings)
    assert _board_ok(res, s) == (19, 19)
    assert res.parsed["robber"] == s.robber
    assert res.confidence["geometry"] > 0.8
    blurred = img.filter(ImageFilter.GaussianBlur(2.5))
    res = parse_image(blurred, me="red")
    if res.debug["method"] == "blob":
        assert any("not located reliably" in w for w in res.warnings) or res.confidence["geometry"] > 0.5


def test_hand_bar_never_reads_board_tiles_as_cards():
    """vision-cv-12 / vision-cv-15: a board reaching into the bottom of the image is not a hand."""
    s = played_state(0, 4, 45)
    truth = {B.RESOURCE_NAMES[r]: s.players[0].resources[r] for r in range(5)}
    for size, seed in (((900, 1400), 0), ((1280, 800), 4)):
        geom = synth.board_geometry(size, seed, False)
        if size == (1280, 800):
            geom = dict(geom, hex_size=0.95 * size[1] / 8.0, cy=0.5 * size[1])
        img = synth.render_state(s, size=size, seed=seed, me=0, geometry=geom, jitter=False)
        res = parse_image(img, me="red")
        me = next(p for p in res.parsed["players"] if p["color"] == "red")
        assert me.get("resources") in (None, truth), (size, me.get("resources"))
        assert "hand" in res.confidence


def test_read_dice_rejects_unrelated_white_squares():
    arr = np.full((400, 640, 3), 66, np.uint8)
    arr[:, :, 1] = 150
    arr[:, :, 2] = 225
    arr[340:380, 460:500] = 255    # two white squares far apart, no pips
    arr[330:370, 600:640] = 255
    assert C.read_dice(arr) == (0, 0.0)
    s = played_state(2, 4, 40)
    s.dice = 9
    img, geom = render(s, (1024, 640), 2)
    assert C.read_dice(np.asarray(img))[0] == 9


def test_classify_pixels_matches_reference_formula():
    """vision-cv-8: the matmul distance computation must give the same labels as the naive one."""
    rng = np.random.default_rng(0)
    cal = Calibration()
    names, pal = C._player_palette(cal)
    bg = C._background_palette(cal)
    px = rng.integers(0, 256, size=(20000, 3)).astype(np.uint8)
    px[:2000] = np.asarray(pal[rng.integers(0, len(pal), 2000)] + rng.normal(0, 15, (2000, 3)), dtype=np.int64).clip(0, 255)
    got = C._classify_pixels(px, pal, bg, cal.player_max_dist)
    p = px.astype(np.float32)
    dp = np.linalg.norm(p[:, None, :] - pal[None, :, :], axis=2)
    db = np.linalg.norm(p[:, None, :] - bg[None, :, :], axis=2).min(axis=1)
    j = dp.argmin(axis=1)
    dmin = dp[np.arange(len(p)), j]
    want = np.where((dmin < cal.player_max_dist) & (dmin < db), j, -1)
    assert (got != want).mean() < 0.002    # ties / float rounding only
    assert C._classify_pixels(np.zeros((0, 3), np.uint8), pal, bg, 60.0).shape == (0,)


def test_large_screenshot_parses_within_memory_budget():
    """vision-cv-8: a 2560x1440 screenshot must not need hundreds of megabytes."""
    s = played_state(0, 4, 45)
    size = (2560, 1440)
    geom = synth.board_geometry(size, 0, True)
    img = synth.render_state(s, size=size, seed=0, me=0, geometry=geom)
    C._digit_classifier()
    tracemalloc.start()
    res = parse_image(img, me="red")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 450e6, peak
    assert _board_ok(res, s) == (19, 19)
    assert res.parsed["robber"] == s.robber
    assert _piece_f1(res, s) >= 0.95
    ps = {p["color"]: p for p in res.parsed["players"]}
    truth = state_to_parsed(s, me=0)
    for tp in truth["players"]:
        assert (ps[tp["color"]]["vp"], ps[tp["color"]]["cards"]) == (tp["vp"], tp["cards"])
    assert ps["red"]["resources"] == truth["players"][0]["resources"]


def test_parse_result_confidence_keys_and_phase():
    s = played_state(1, 3, 30)
    img, geom = render(s, (1000, 640), 1)
    res = parse_image(img, me=s.players[0].color)
    for key in ("geometry", "hexes", "numbers", "numbers_min", "robber", "ports", "pieces", "panel", "hand", "dice"):
        assert key in res.confidence and 0.0 <= res.confidence[key] <= 1.0
    assert res.state.dice == s.dice and res.state.phase == "main"
    assert 0.0 < res.confidence["geometry"] <= 1.0
