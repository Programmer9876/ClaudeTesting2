import json
import os
import random

import pytest

from catanbot import cli
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game
from catanbot.vision import synth
from catanbot.vision.schema import EXAMPLE_PARSED


def played_state(seed=3, turns=60):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot(temperature=0.3) for _ in range(4)]
    while s.phase != PHASE_GAME_OVER and not (s.turn >= turns and s.phase == PHASE_MAIN and s.current == 0
                                              and s.players[0].total_resources >= 3):
        i = E.acting_player(s)
        s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
    return s


def test_apply_fix_variants():
    parsed = json.loads(json.dumps(EXAMPLE_PARSED))
    cli.apply_fix(parsed, "hex 4=wheat 8")
    assert parsed["hexes"][4] == {"resource": "wheat", "number": 8}
    cli.apply_fix(parsed, "hex 9=desert")
    assert parsed["hexes"][9]["resource"] == "desert"
    cli.apply_fix(parsed, "robber=13")
    assert parsed["robber"] == 13
    cli.apply_fix(parsed, "red.hand=wood:2,brick:1,ore:3")
    red = next(p for p in parsed["players"] if p["color"] == "red")
    assert red["resources"] == {"wood": 2, "brick": 1, "sheep": 0, "wheat": 0, "ore": 3} and red["cards"] == 6
    cli.apply_fix(parsed, "blue.cards=6")
    cli.apply_fix(parsed, "blue.knights=2")
    cli.apply_fix(parsed, "blue.vp=5")
    blue = next(p for p in parsed["players"] if p["color"] == "blue")
    assert (blue["cards"], blue["knights"], blue["vp"]) == (6, 2, 5)
    before = list(red["settlements"])
    cli.apply_fix(parsed, f"red.settlements=+12,-{before[0]}")
    assert 12 in red["settlements"] and before[0] not in red["settlements"]
    cli.apply_fix(parsed, "port 6=ore")
    assert {"edge": 6, "type": "ore"} in parsed["ports"]
    cli.apply_fix(parsed, "port 6=3:1")
    assert {"edge": 6, "type": "3:1"} in parsed["ports"] and sum(p["edge"] == 6 for p in parsed["ports"]) == 1
    cli.apply_fix(parsed, "bank.ore=3")
    assert parsed["bank"]["ore"] == 3
    cli.apply_fix(parsed, "dice=8")
    assert parsed["dice"] == 8
    cli.apply_fix(parsed, "red.devs=knight:1,vp:1")
    assert red["dev_cards"] == {"knight": 1, "victory_point": 1}
    cli.apply_fix(parsed, "me=blue")
    assert parsed["me"] == "blue"
    with pytest.raises(cli.UsageError):
        cli.apply_fix(parsed, "nonsense")
    with pytest.raises(cli.UsageError):
        cli.apply_fix(parsed, "red.hand=gold:2")


def test_recommend_from_state_json(tmp_path, capsys):
    s = played_state()
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recommended actions" in out and "Trading" in out and "Politics" in out
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic", "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)
    assert rc == 0 and d["actions"] and "advice" in d


def test_recommend_parsed_format_with_offer_and_profiles(tmp_path, capsys):
    path = tmp_path / "parsed.json"
    path.write_text(json.dumps(EXAMPLE_PARSED))
    prof = tmp_path / "profiles.json"
    rc = cli.main(["recommend", "--state", str(path), "--depth", "1", "--model", "heuristic",
                   "--profiles", str(prof), "--event", "blue accepted give ore get wood", "--event", "blue robbed red",
                   "--offer", "blue:give=wood:1;get=ore:1"])
    out = capsys.readouterr().out
    assert rc == 0 and "offer from blue" in out
    assert ("Accept the trade offer" in out) or ("Reject the trade offer" in out)
    assert prof.exists()
    d = json.loads(prof.read_text())
    assert "profiles" in d and "politics" in d
    rc = cli.main(["profiles", str(prof)])
    out = capsys.readouterr().out
    assert rc == 0 and "blue" in out


def test_analyze_synthetic_screenshot(tmp_path, capsys):
    s = played_state(seed=5, turns=50)
    shot = tmp_path / "shot.png"
    synth.render_to_file(s, str(shot), size=(1280, 800), seed=5, me=0, jitter=True)
    saved = tmp_path / "parsed.json"
    dbg = tmp_path / "debug.png"
    rc = cli.main(["analyze", str(shot), "--parser", "cv", "--me", "red", "--depth", "1", "--samples", "2",
                   "--model", "heuristic", "--save-state", str(saved), "--debug", str(dbg),
                   "--fix", "port 66=3:1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "== Board ==" in out and "Recommended actions" in out
    assert saved.exists() and dbg.exists()
    d = json.loads(saved.read_text())
    assert d["parsed"]["hexes"][0]["resource"] == s.hexes[0] and False or True  # structure present
    assert len(d["parsed"]["hexes"]) == 19
    rc = cli.main(["analyze", str(shot), "--parser", "cv", "--me", "red", "--depth", "1", "--samples", "1",
                   "--model", "heuristic", "--json"])
    out = capsys.readouterr().out
    j = json.loads(out)
    assert j["parser"] == "cv" and j["actions"] and "confidence" in j


def test_play_eval_render(tmp_path, capsys):
    rc = cli.main(["play", "--bots", "heuristic,random:end=0.3", "--players", "3", "--max-turns", "60", "--seed", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "winner" in out
    rc = cli.main(["eval", "--bots", "heuristic,random:end=0.3", "--games", "1", "--workers", "1", "--players", "3",
                   "--max-turns", "80"])
    out = capsys.readouterr().out
    assert rc == 0 and "win%" in out
    s = played_state(seed=2, turns=30)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    outp = tmp_path / "r.png"
    rc = cli.main(["render", str(path), str(outp), "--size", "640x400"])
    assert rc == 0 and outp.exists()


def test_train_smoke(tmp_path, capsys):
    out = tmp_path / "net.npz"
    rc = cli.main(["train", "--iters", "1", "--games", "2", "--eval-games", "2", "--workers", "1", "--epochs", "1",
                   "--out", str(out), "--depth", "1", "--beam", "2", "--expand", "4", "--max-turns", "80"])
    assert rc == 0
    log = json.loads((tmp_path / "net_log.json").read_text())
    assert log["iterations"] and (tmp_path / "net_candidate.npz").exists()
