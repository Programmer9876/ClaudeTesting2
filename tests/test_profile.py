"""Tests for catanbot.vision.profile (UI regions measured on the user's screen)."""
from __future__ import annotations

import json

import pytest

from catanbot.vision.profile import ProfileError, UiProfile, parse_box


def test_parse_box_fractions_and_pixels():
    assert parse_box("0.8,0.15,0.99,0.78") == (0.8, 0.15, 0.99, 0.78)
    assert parse_box("1000,100,200,400", size=(1280, 800)) == pytest.approx((1000 / 1280, 0.125, 1200 / 1280, 0.625))
    # a pixel box running off the screen is clipped
    assert parse_box("1200,700,200,200", size=(1280, 800))[2:] == (1.0, 1.0)
    with pytest.raises(ProfileError):
        parse_box("1000,100,200,400")          # pixels without a screen size
    with pytest.raises(ProfileError):
        parse_box("1,2,3")
    with pytest.raises(ProfileError):
        parse_box("a,b,c,d")
    with pytest.raises(ProfileError):
        parse_box("0.5,0.5,0.4,0.9")           # x1 < x0
    with pytest.raises(ProfileError):
        parse_box("10,10,0,5", size=(100, 100))


def test_pixel_box_and_unknown_region():
    p = UiProfile()
    p.set_region("log", (0.8, 0.15, 0.99, 0.78))
    assert p.pixel_box("log", (1280, 800)) == (1024, 120, 1267, 624)
    assert p.pixel_box("log", (2560, 1600)) == (2048, 240, 2534, 1248)
    assert p.pixel_box("bank", (1280, 800)) is None
    with pytest.raises(ProfileError):
        p.set_region("chat", (0, 0, 1, 1))


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "screen.json")
    p = UiProfile.open(path)
    assert p.regions == {} and p.path == path
    p.set_region("log", (0.8, 0.15, 0.99, 0.78))
    p.set_region("player_panel", (0.0, 0.0, 0.3, 0.45))
    p.screen = (1920, 1080)
    p.ocr = {"text": [40, 44, 54], "exemplars": "screen.ocr.npz"}
    p.save()
    q = UiProfile.open(path)
    assert q.regions == p.regions and q.screen == (1920, 1080) and q.ocr == p.ocr
    assert q.sidecar_path("screen.ocr.npz") == str(tmp_path / "screen.ocr.npz")
    assert not (tmp_path / "screen.json.tmp").exists()


def test_refuses_to_overwrite_other_files(tmp_path):
    other = tmp_path / "session.json"
    other.write_text(json.dumps({"format": "catanbot.colonist_log session"}))
    with pytest.raises(ProfileError):
        UiProfile.open(str(other))
    with pytest.raises(ProfileError):
        UiProfile(path=str(other)).save()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ProfileError):
        UiProfile.load(str(bad))
    with pytest.raises(ProfileError):
        UiProfile().save()                     # no path
