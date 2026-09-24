"""Tests for catanbot.vision.synth: the Colonist.io-style renderer."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
from PIL import Image

from catanbot import board as B
from catanbot.vision import schema as S
from catanbot.vision import synth

from tests.test_schema import make_midgame_state


@pytest.fixture(scope="module")
def state():
    st = make_midgame_state()
    assert S.validate(S.state_to_parsed(st, me=0)) == []
    return st


def _mean_rgb(img: Image.Image, x: float, y: float, r: int = 2) -> np.ndarray:
    arr = np.asarray(img)
    xi, yi = int(round(x)), int(round(y))
    patch = arr[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1].reshape(-1, 3)
    return patch.mean(axis=0)


@pytest.mark.parametrize("size", [(1280, 800), (800, 500)])
def test_render_sizes_and_mode(state, size):
    img = synth.render_state(state, size=size, seed=1, me=0)
    assert img.mode == "RGB"
    assert img.size == size


def test_geometry_without_jitter():
    g = synth.board_geometry((1280, 800), seed=0, jitter=False)
    assert math.isclose(g["hex_size"], 0.7 * 800 / 8)
    assert 0 < g["cx"] < 1280 and 0 < g["cy"] < 800
    # board fits inside the image
    for v in range(B.NUM_VERTICES):
        x, y = synth.pixel_of_vertex(v, g)
        assert 0 < x < 1280 and 0 < y < 800
    # vertex helpers agree with the hex corner geometry
    hx, hy = synth.pixel_of_hex(9, g)
    for k, v in enumerate(B.HEX_VERTICES[9]):
        ex, ey = B.hex_corner(hx, hy, k, g["hex_size"])
        vx, vy = synth.pixel_of_vertex(v, g)
        assert math.isclose(ex, vx, abs_tol=1e-6) and math.isclose(ey, vy, abs_tol=1e-6)
    ex, ey = synth.pixel_of_edge(B.HEX_EDGES[9][0], g)
    ax, ay = synth.pixel_of_vertex(B.HEX_VERTICES[9][0], g)
    bx, by = synth.pixel_of_vertex(B.HEX_VERTICES[9][1], g)
    assert math.isclose(ex, (ax + bx) / 2) and math.isclose(ey, (ay + by) / 2)


def test_geometry_jitter_is_bounded_and_seeded():
    base = synth.board_geometry((1280, 800), jitter=False)
    seen = set()
    for seed in range(20):
        g = synth.board_geometry((1280, 800), seed=seed, jitter=True)
        assert abs(g["cx"] - base["cx"]) <= 0.04 * 1280 + 1e-9
        assert abs(g["cy"] - base["cy"]) <= 0.04 * 800 + 1e-9
        assert abs(g["hex_size"] / base["hex_size"] - 1.0) <= 0.08 + 1e-9
        seen.add((round(g["cx"], 3), round(g["cy"], 3), round(g["hex_size"], 3)))
        assert synth.board_geometry((1280, 800), seed=seed, jitter=True) == g
    assert len(seen) > 10


@pytest.mark.parametrize("size", [(1280, 800), (800, 500)])
def test_hex_and_sea_colours(state, size):
    seed = 4
    img = synth.render_state(state, size=size, seed=seed, me=0, jpeg=False)
    g = synth.board_geometry(size, seed=seed, jitter=True)
    wood_hex = next(i for i, (r, _) in enumerate(state.hexes) if r == B.WOOD)
    hx, hy = synth.pixel_of_hex(wood_hex, g)
    # sample just right of the number token, still well inside the tile
    rgb = _mean_rgb(img, hx + 0.55 * g["hex_size"], hy)
    assert rgb[1] > rgb[0] and rgb[1] > rgb[2], rgb
    # the token itself is cream and its centre is drawn
    tok = _mean_rgb(img, hx, hy - 0.28 * g["hex_size"])
    assert tok[0] > 150 and tok[1] > 130
    # sea far outside the board (right of the middle row)
    ex, ey = synth.pixel_of_hex(11, g)
    sea = _mean_rgb(img, ex + 2.6 * g["hex_size"], ey)
    assert sea[2] > sea[0] and sea[2] > sea[1], sea
    assert abs(sea[2] - 225) < 30
    # sea left of the board too
    wx, wy = synth.pixel_of_hex(7, g)
    sea2 = _mean_rgb(img, wx - 1.6 * g["hex_size"], wy)
    assert sea2[2] > sea2[0] and sea2[2] > sea2[1]


def test_pieces_are_drawn_in_player_colour(state):
    img = synth.render_state(state, size=(1280, 800), seed=2, jitter=False, jpeg=False)
    g = synth.board_geometry((1280, 800), jitter=False)
    for p in state.players:
        target = np.array(synth.PLAYER_RGB[p.color], dtype=float)
        for v in p.settlements + p.cities:
            x, y = synth.pixel_of_vertex(v, g)
            rgb = _mean_rgb(img, x, y + 0.09 * g["hex_size"], r=1)  # house body, below the roof
            assert np.abs(rgb - target).max() < 40, (p.color, v, rgb)
        for e in p.roads:
            x, y = synth.pixel_of_edge(e, g)
            rgb = _mean_rgb(img, x, y, r=1)
            assert np.abs(rgb - target).max() < 40, (p.color, e, rgb)


def test_deterministic_for_same_seed(state):
    a = synth.render_state(state, size=(640, 400), seed=7)
    b = synth.render_state(state, size=(640, 400), seed=7)
    assert a.tobytes() == b.tobytes()
    c = synth.render_state(state, size=(640, 400), seed=8)
    assert c.tobytes() != a.tobytes()


def test_jpeg_and_me_by_colour(state):
    img = synth.render_state(state, size=(640, 400), seed=3, me="blue", jpeg=True)
    assert img.mode == "RGB" and img.size == (640, 400)
    plain = synth.render_state(state, size=(640, 400), seed=3, me="blue", jpeg=False)
    assert img.tobytes() != plain.tobytes()


def test_render_to_file(state, tmp_path):
    path = tmp_path / "shot.png"
    geom = synth.render_to_file(state, str(path), size=(800, 500), seed=5)
    assert os.path.exists(path)
    with Image.open(path) as im:
        assert im.size == (800, 500)
    assert geom == synth.board_geometry((800, 500), seed=5, jitter=True)


def test_render_states_with_edge_cases():
    # fresh game with no pieces and a random board, 3 players, nothing rolled
    import random
    from catanbot.state import new_game
    st = new_game(3, rng=random.Random(1))
    img = synth.render_state(st, size=(1024, 640), seed=0)
    assert img.size == (1024, 640)
    # unknown player colour does not crash
    st.players[0].color = "teal"
    synth.render_state(st, size=(400, 250), seed=0, jitter=False)
