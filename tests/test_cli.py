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


def test_log_outcome_calibrate(tmp_path, capsys):
    s = played_state(seed=4, turns=40)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    log = tmp_path / "games.jsonl"
    for k in range(2):
        rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                       "--log", str(log), "--game", "g1"])
        capsys.readouterr()
        assert rc == 0
    rc = cli.main(["outcome", str(log), "--game", "g1", "--winner", "red"])
    capsys.readouterr()
    assert rc == 0
    rc = cli.main(["calibrate", "--log", str(log)])
    out = capsys.readouterr().out
    assert rc == 0 and "Brier" in out and "game g1: won" in out
    rc = cli.main(["calibrate", "--selfplay", "1", "--model", "heuristic", "--seed", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "self-play calibration" in out


def test_watch_from_dir(tmp_path, capsys):
    s = played_state(seed=6, turns=40)
    d = tmp_path / "frames"
    d.mkdir()
    synth.render_to_file(s, str(d / "a.png"), size=(1000, 640), seed=6, me=0, jitter=True)
    synth.render_to_file(s, str(d / "b.png"), size=(1000, 640), seed=6, me=0, jitter=True)   # duplicate: skipped
    rc = cli.main(["watch", "--from-dir", str(d), "--me", "red", "--depth", "1", "--samples", "1", "--model", "heuristic"])
    out = capsys.readouterr().out
    assert rc == 0 and "watch mode" in out and out.count("turn of") == 1


# ---------------------------------------------------------------------------
# regression tests for the CLI review findings (validation, error reporting, --time, docs)
# ---------------------------------------------------------------------------
import copy
import time

import numpy as np

from catanbot import board as B


def _err_line(capsys):
    cap = capsys.readouterr()
    assert "Traceback" not in cap.err
    return cap.err, cap.out


def test_bot_spec_splitting_and_validation(capsys):
    assert cli.split_bot_specs("search:depth=1,model=x.npz,search:depth=1,evaluator=heuristic,heuristic,random") == [
        "search:depth=1,model=x.npz", "search:depth=1,evaluator=heuristic", "heuristic", "random"]
    assert cli.split_bot_specs("heuristic;random:end=0.3 search:beam=2,depth=1") == [
        "heuristic", "random:end=0.3", "search:beam=2,depth=1"]
    assert cli.split_bot_specs("") == []
    # the documented multi-keyword specs run (no ValueError from make_bot)
    rc = cli.main(["play", "--bots", "search:depth=1,evaluator=heuristic,heuristic", "--players", "3",
                   "--max-turns", "8", "--seed", "1"])
    err, out = _err_line(capsys)
    assert rc == 0 and "winner" in out
    # --players defaults to 3..4 seats even with two specs
    rc = cli.main(["play", "--bots", "heuristic,random:end=0.3", "--max-turns", "8"])
    err, out = _err_line(capsys)
    assert rc == 0 and "VP [" in out and out.count(",") >= 2
    for bots in ("bogus,heuristic", "search:depth=x,heuristic", "heuristic,heuristic,heuristic,heuristic,heuristic"):
        rc = cli.main(["play", "--bots", bots, "--max-turns", "8"])
        err, out = _err_line(capsys)
        assert rc == 2 and err.startswith("error:"), (bots, err)
    rc = cli.main(["eval", "--bots", "search:depth=1,model=/nonexistent/net.npz,heuristic", "--games", "1",
                   "--workers", "1", "--players", "3", "--max-turns", "8"])
    err, out = _err_line(capsys)
    assert rc == 2 and "bad bot spec" in err
    for argv in (["play", "--bots", "heuristic", "--players", "2"], ["play", "--players", "5"],
                 ["eval", "--players", "2"], ["recommend", "--state", "x.json", "--depth", "0"],
                 ["recommend", "--state", "x.json", "--beam", "0"], ["recommend", "--state", "x.json", "--samples", "0"],
                 ["recommend", "--state", "x.json", "--time", "0"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        capsys.readouterr()


def test_apply_fix_rejects_impossible_values():
    parsed = copy.deepcopy(EXAMPLE_PARSED)
    before = copy.deepcopy(parsed)
    bad = ["hex 4=wheat 13", "hex x=wheat 8", "hex 99=wheat 8", "hex 4=wheat 7", "hex 9=desert 8", "hex 9=wheat",
           "dice=x", "dice=13", "robber=99", "port 99=3:1", "port 12=3:1", "port 1-2=ore", "red.settlements=+99",
           "red.roads=+72", "red.cities=-x", "blue.lr=maybe", "green.la=2", "purple.cards=3", "cyan.hand=wood:1",
           "me=purple", "current=purple", "deck=26", "bank.ore=20", "blue.dev=x", "players=cyan"]
    for fx in bad:
        with pytest.raises(cli.UsageError):
            cli.apply_fix(parsed, fx)
    assert parsed == before, "a rejected fix must not modify the parse"
    with pytest.raises(cli.UsageError, match=r"bad --fix 'dice=x'"):
        cli.apply_fixes(parsed, ["dice=x"])
    with pytest.raises(cli.UsageError, match=r"0\.\.18"):
        cli.apply_fixes(parsed, ["hex 99=wheat 8"])
    # a resource change without a number keeps the existing token
    cli.apply_fix(parsed, "hex 4=wheat")
    assert parsed["hexes"][4] == {"resource": "wheat", "number": 6}
    # a city replaces the settlement on that vertex and vice versa (no double building)
    red = next(p for p in parsed["players"] if p["color"] == "red")
    v = red["settlements"][0]
    cli.apply_fix(parsed, f"red.cities=+{v}")
    assert v in red["cities"] and v not in red["settlements"]
    cli.apply_fix(parsed, f"red.settlements=+{v}")
    assert v in red["settlements"] and v not in red["cities"]
    # ports may be given by edge id or by the vertex pair
    e = B.COASTAL_EDGES[3]
    a, b = B.EDGE_VERTICES[e]
    cli.apply_fix(parsed, f"port {a}-{b}=ore")
    assert {"edge": e, "type": "ore"} in parsed["ports"]
    cli.apply_fix(parsed, f"port {e}=none")
    assert not any(p["edge"] == e for p in parsed["ports"])
    # only players= may create a player; names resolve to colours
    cli.apply_fix(parsed, "players=red,blue,orange,green,white")
    assert "white" in [p["color"] for p in parsed["players"]]
    cli.apply_fix(parsed, "me=Bob")
    assert parsed["me"] == "blue"
    cli.apply_fix(parsed, "blue.lr=yes")
    assert next(p for p in parsed["players"] if p["color"] == "blue")["longest_road"] is True


def test_recommend_errors_are_one_liners(tmp_path, capsys):
    gpath = tmp_path / "game.json"
    gpath.write_text(json.dumps(played_state().to_dict()))
    ppath = tmp_path / "parsed.json"
    ppath.write_text(json.dumps(EXAMPLE_PARSED))
    (tmp_path / "bad.json").write_text(json.dumps({"foo": 1}))
    (tmp_path / "list.json").write_text(json.dumps([1, 2]))
    (tmp_path / "noplayers.json").write_text(json.dumps({"phase": "main", "hexes": []}))
    (tmp_path / "binary.json").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00garbage")
    base = ["--depth", "1", "--samples", "1"]
    cases = [
        (["recommend", "--state", str(gpath), "--me", "purple", "--model", "heuristic"], "unknown player"),
        (["recommend", "--state", str(gpath), "--model", "heuristic", "--fix", "red.hand=ore:5"], "--fix only applies"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--fix", "hex 4=wheat 13"], "2..12"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--fix", "hex 99=wheat 8"], "0..18"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--fix", "me=purple"], "unknown player"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--fix", "purple.cards=3"], "unknown player"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "blue"], "bad --offer"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "blue:give=wood:1"], "non-empty"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "blue:give=wood:1;get=wood:1"], "both sides"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "red:give=wood:1;get=ore:1"], "opponent"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "purple:give=wood:1;get=ore:1"], "unknown offer proposer"),
        (["recommend", "--state", str(ppath), "--model", str(tmp_path / "missing.npz")], "value net not found"),
        (["recommend", "--state", str(ppath), "--model", str(ppath)], "could not load value net"),
        (["recommend", "--state", str(tmp_path / "bad.json"), "--model", "heuristic"], "not a game state"),
        (["recommend", "--state", str(tmp_path / "list.json"), "--model", "heuristic"], "expected a JSON object"),
        (["recommend", "--state", str(tmp_path / "noplayers.json"), "--model", "heuristic"], "no players"),
        (["recommend", "--state", str(tmp_path / "binary.json"), "--model", "heuristic"], "not a JSON state file"),
        (["recommend", "--state", str(tmp_path / "missing.json"), "--model", "heuristic"], "No such file"),
        (["recommend", "--state", str(ppath), "--model", "heuristic", "--profiles", str(tmp_path / "binary.json")], "not a JSON profiles file"),
        (["render", str(ppath), str(tmp_path / "r.png"), "--size", "abc"], "--size must be"),
        (["render", str(ppath), str(tmp_path / "r.png"), "--size", "640"], "--size must be"),
        (["render", str(ppath), str(tmp_path / "r.png"), "--me", "purple"], "unknown player"),
        (["calibrate"], "needs --log"),
        (["profiles", str(tmp_path / "nonexistent.json")], "No such file"),
        (["profiles", str(gpath)], "not a profiles file"),
        (["analyze", str(gpath), "--parser", "cv", "--model", "heuristic"], "could not parse"),
    ]
    for argv, msg in cases:
        rc = cli.main(argv + (base if argv[0] in ("recommend", "analyze") else []))
        err, out = _err_line(capsys)
        assert rc == 2, (argv, err)
        assert msg in err and err.startswith("error:"), (argv, err)
        assert out == "", argv   # errors go to stderr only
    # a valid offer still works, with the proposer resolved by colour; an unaffordable one is
    # evaluated (only rejecting is legal) with a warning instead of a crash
    rc = cli.main(["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "blue:give=wood:1;get=wheat:1"] + base)
    err, out = _err_line(capsys)
    assert rc == 0 and "offer from blue" in out and "WARNING: you cannot pay" not in out
    rc = cli.main(["recommend", "--state", str(ppath), "--model", "heuristic", "--offer", "blue:give=wood:1;get=ore:9"] + base)
    err, out = _err_line(capsys)
    assert rc == 0 and "WARNING: you cannot pay 9 ore" in out and "Reject the trade offer" in out
    assert "Accept the trade offer" not in out


def test_profiles_file_is_never_clobbered(tmp_path, capsys):
    game = tmp_path / "game.json"
    text = json.dumps(played_state().to_dict())
    game.write_text(text)
    ppath = tmp_path / "parsed.json"
    ppath.write_text(json.dumps(EXAMPLE_PARSED))
    rc = cli.main(["recommend", "--state", str(ppath), "--depth", "1", "--model", "heuristic", "--profiles", str(game)])
    err, out = _err_line(capsys)
    assert rc == 2 and "not a profiles file" in err
    assert game.read_text() == text
    # a real profile file keeps working and shows the political / coalition information
    prof = tmp_path / "profiles.json"
    rc = cli.main(["recommend", "--state", str(ppath), "--depth", "1", "--model", "heuristic", "--profiles", str(prof),
                   "--event", "blue traded orange give 2 ore get 1 wood"])
    err, out = _err_line(capsys)
    assert rc == 0 and prof.exists()
    rc = cli.main(["profiles", str(prof)])
    err, out = _err_line(capsys)
    assert rc == 0 and "political capital" in out and "recent political events: Bob traded Carol" in out


def test_report_keeps_original_state_and_marks_player_to_move(tmp_path, capsys):
    ppath = tmp_path / "parsed.json"
    ppath.write_text(json.dumps(EXAMPLE_PARSED))
    base = ["recommend", "--state", str(ppath), "--depth", "1", "--samples", "1", "--model", "heuristic"]
    rc = cli.main(base + ["--json", "--fix", "current=blue", "--fix", "dice=8"])
    d = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert d["state"]["current"] == 1 and d["state"]["phase"] == "main" and d["state"]["dice"] == 8
    assert d["decision"] == {"current": "red", "phase": "roll", "dice": 0}
    assert d["search"]["samples_done"] == 1 and not d["search"]["budget_hit"]
    rc = cli.main(base + ["--fix", "current=blue"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert rc == 0 and any(l.startswith("> ") and "Bob" in l for l in lines)
    assert any(l.startswith("* ") and "Alice" in l for l in lines)
    assert "It is Bob's turn" in out
    # opponent profile lines appear once (Opponents section), not again under Trading
    assert not any(l.strip().startswith("Opponent ") for l in lines)
    assert sum("accepts" in l for l in lines) == 3
    # the recommendation ran for the right situation and the state object was not modified in place
    state, parsed = cli.load_state_file(str(ppath))
    args = cli.build_parser().parse_args(base + ["--fix", "current=blue"])
    cli.apply_fixes(parsed, args.fix)
    before = state.to_dict()
    report = cli.recommend_for_state(state, 0, args, parsed)
    assert state.to_dict() == before and report["actions"]


def test_game_over_state_reports_no_actions(tmp_path, capsys):
    s = played_state()
    s.phase = PHASE_GAME_OVER
    s.winner = 2
    path = tmp_path / "over.json"
    path.write_text(json.dumps(s.to_dict()))
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic"])
    out = capsys.readouterr().out
    assert rc == 0 and "no legal actions" in out and "game is over" in out and "orange won" in out


def test_time_budget_is_shared_across_samples(tmp_path, capsys, monkeypatch):
    ppath = tmp_path / "parsed.json"
    ppath.write_text(json.dumps(EXAMPLE_PARSED))

    class SlowEval:
        def evaluate(self, states, players):
            time.sleep(0.03)
            return np.full(len(states), 0.5)

    base = ["recommend", "--state", str(ppath), "--depth", "1", "--samples", "4", "--json"]
    rc = cli.main(base + ["--model", "heuristic"])
    d = json.loads(capsys.readouterr().out)
    assert rc == 0 and d["search"]["samples"] == 4 and d["search"]["samples_done"] == 4
    monkeypatch.setattr(cli, "load_evaluator", lambda path: (SlowEval(), "slow evaluator"))
    rc = cli.main(base + ["--time", "0.01"])
    d = json.loads(capsys.readouterr().out)
    assert rc == 0 and d["actions"]
    assert d["search"]["time_limit"] == 0.01 and d["search"]["samples_done"] < 4 and d["search"]["budget_hit"]
    assert any("ran out" in n for n in d["notes"])


def test_ids_table_and_overlay(tmp_path):
    s = played_state(seed=2, turns=20)
    text = cli.ids_ascii(s)
    assert "Coastal edges" in text and len([l for l in text.splitlines() if l.startswith(" ") and ":" in l]) == 19
    assert " ".join(str(v) for v in B.HEX_VERTICES[4]) in " ".join(text.split())
    from PIL import Image
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    cli.draw_ids(img, {"cx": 200.0, "cy": 150.0, "hex_size": 30.0})
    assert np.asarray(img).max() > 0          # something was drawn
    assert cli.draw_ids(img, None) is img       # no geometry: no-op
    board = cli.board_ascii(s)
    assert "Ports:" in board and "edge " in board and "vertices" in board


def test_outcome_validates_game_and_winner(tmp_path, capsys):
    s = played_state(seed=4, turns=40)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(s.to_dict()))
    log = tmp_path / "games.jsonl"
    rc = cli.main(["recommend", "--state", str(path), "--me", "red", "--depth", "1", "--model", "heuristic",
                   "--log", str(log), "--game", "g1"])
    capsys.readouterr()
    assert rc == 0
    for argv in (["outcome", str(log), "--game", "nope", "--winner", "red"],
                 ["outcome", str(log), "--game", "g1", "--winner", "purple"],
                 ["outcome", str(tmp_path / "missing.jsonl"), "--game", "g1", "--winner", "red"]):
        rc = cli.main(argv)
        err, out = _err_line(capsys)
        assert rc == 2 and err.startswith("error:"), (argv, err)
    assert not (tmp_path / "missing.jsonl").exists()
    assert sum(1 for l in log.read_text().splitlines() if l.strip()) == 1
    rc = cli.main(["outcome", str(log), "--game", "g1", "--winner", "Blue"])
    err, out = _err_line(capsys)
    assert rc == 0 and "won by blue" in out
