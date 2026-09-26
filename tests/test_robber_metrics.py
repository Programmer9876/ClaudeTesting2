"""Tests for catanbot/robber_metrics.py: the observation-only robber counters of self-play games (the seat observer
"robber" of tuning.play_paired_game; docs/PRIORITY_PLAN.md step 5, robber.shadow_metrics)."""
import hashlib
import random

from catanbot import actions as A
from catanbot import board as B
from catanbot import robber_eval as R
from catanbot import tuning
from catanbot.robber_metrics import RobberCounters
from catanbot.selfplay import make_bot, play_game
from catanbot.state import PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL
from tests.test_robber_eval import blocked_state, owners, roll_of

K = B.DEV_KNIGHT


def _game(spec, seed, observer=None, turns=60):
    h = hashlib.sha256()

    def on_action(st, a, p):
        if observer is not None:
            observer.on_action(st, a, p)
        h.update(repr((p, a)).encode())

    res = play_game([make_bot(spec) for _ in range(4)], rng=random.Random(seed), seed=seed, max_turns=turns,
                    on_action=on_action)
    h.update(repr((res.winner, list(res.vps), res.turns, res.actions)).encode())
    return h.hexdigest()


def test_robber_counters_do_not_change_games():
    """6 games (heuristic bots, full length) plus 1 search-bot game: identical with and without the counters; the
    paired-game summary is identical too, with the counters under seat_events."""
    for seed in range(1, 7):
        obs = RobberCounters(4)
        assert _game("heuristic:temp=0.15", seed, obs, turns=400) == _game("heuristic:temp=0.15", seed, None, turns=400)
        r = obs.result()
        assert sum(r["turn_starts"]) > 0 and set(r) == set(RobberCounters.KEYS)
    obs = RobberCounters(4)
    assert _game(tuning.DEFAULT_SEARCH_SPEC, 3, obs, turns=40) == _game(tuning.DEFAULT_SEARCH_SPEC, 3, None, turns=40)
    plain = tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=200)
    seen = tuning.play_paired_game("heuristic:temp=0.15", {}, "CDDC", 5, max_turns=200, observers=["robber"])
    ev = seen.pop("seat_events")["robber"]
    for k in ("winner", "vps", "turns", "actions", "seat_decisions"):
        assert seen[k] == plain[k]
    assert set(ev) == set(RobberCounters.KEYS) and all(len(v) == 4 for v in ev.values())
    assert sum(ev["rob_moves"]) > 0 and sum(ev["cards_lost_block"]) > 0
    st = tuning.paired_stats([dict(seen, seat_events={"robber": ev})])
    assert st["records"][0]["seat_events"]["robber"] == ev


def test_robber_counters_match_replay_on_fixture():
    """A hand-built sequence: a blocked turn start with a knight, a roll of the robber hex's number (a settlement
    and a city on it), an END_TURN holding the knight as the strict leader, a 7 whose robber move hits the next
    roller's knight-holding block and the leader, and a kick that takes Largest Army."""
    s, h, j = blocked_state()
    v = next(v for v in B.HEX_VERTICES[h] if not any(v in p.settlements or v in p.cities for p in s.players))
    other = (j + 2) % 4
    s.players[other].cities.append(v)            # a second victim on h (a city: 2 cards per roll)
    s.hexes = list(s.hexes)
    num = s.hexes[h][1]
    obs = RobberCounters(4)
    # 1. j's turn start while blocked, holding a knight; it rolls the robber hex's number
    roll_of(s, j, 0)
    s.players[j].dev_cards[K] = 1
    obs.on_action(s, (A.ROLL,), j)
    s.dice, s.phase = num, PHASE_MAIN
    # 2. the roll's production: j lost its pips' cards there, `other` 2 per city
    n_j = sum(1 for v in B.HEX_VERTICES[h] if v in s.players[j].settlements) + \
        2 * sum(1 for v in B.HEX_VERTICES[h] if v in s.players[j].cities)
    lead = max(s.public_vp(i) for i in range(4) if i != j)
    while s.public_vp(j) <= lead:                # make j the strict public-VP leader (cities away from h)
        s.players[j].cities.append(next(x for x in range(B.NUM_VERTICES) if h not in B.VERTEX_HEXES[x] and not any(
            x in p.settlements or x in p.cities for p in s.players)))
    obs.on_action(s, (A.END_TURN,), j)
    r = obs.result()
    assert r["turn_starts"][j] == 1 and r["blocked_turns"][j] == 1 and r["blocked_turns_knight"][j] == 1
    assert r["cards_lost_block"][j] == n_j and r["cards_lost_block"][other] == 2
    assert r["knights_held_end"][j] == 1 and r["knights_held_exposed"][j] == 1
    # 3. the seat before j rolls a 7 and moves the robber onto h (j rolls next and holds a knight)
    mover = (j - 1) % 4
    s.robber = next(x for x in range(B.NUM_HEXES) if x != h and s.hexes[x][0] == B.DESERT)
    s.current, s.phase, s.dice = mover, PHASE_ROBBER, 7
    s.players[j].resources = [1, 0, 0, 0, 0]
    obs.on_action(s, (A.MOVE_ROBBER, h, j), mover)
    r = obs.result()
    assert r["rob_moves"][mover] == 1 and r["rob_on_knight_holder"][mover] == 1
    assert r["rob_on_next_kicker"][mover] == 1 and r["rob_on_leader"][mover] == 1
    assert r["cards_lost_steal"][j] == 1
    # 4. j kicks (blocked) with its third knight: Largest Army at the end, a kick
    s.robber = h
    roll_of(s, j, 0)
    s.players[j].played_knights = 2
    s.largest_army_owner = -1
    away = next(x for x in range(B.NUM_HEXES) if x != h and s.hexes[x][0] == B.DESERT)
    obs.on_action(s, (A.PLAY_KNIGHT, away, -1), j)
    r = obs.result()
    assert r["knights_played"][j] == 1 and r["kicks"][j] == 1 and r["army_end"] == [int(i == j) for i in range(4)]
    assert r["rob_moves"][j] == 1 and r["rob_on_knight_holder"][j] == 0
    # a 7 is never a block loss
    obs2 = RobberCounters(4)
    obs2.on_action(s, (A.ROLL, 7), j)
    assert sum(obs2.result()["cards_lost_block"]) == 0


def test_classify_move_matches_robber_eval():
    s, h, j = blocked_state()
    s.players[j].dev_cards[K] = 1
    mover = (j + 2) % 4
    s.robber = next(x for x in range(B.NUM_HEXES) if x != h and s.hexes[x][0] == B.DESERT)
    s.current, s.phase, s.dice = mover, PHASE_ROBBER, 7
    info = RobberCounters.classify_move(s, (A.MOVE_ROBBER, h, j), mover)
    t = [(k - mover - 1) % 4 for k in range(4)]
    assert info["kick"] == [(t[j], j, R.KICK_LAMBDA)] and s.robber != h     # the state is not changed
    s.phase, s.dice = PHASE_ROLL, 0
    s.players[mover].dev_cards[K] = 1
    info = RobberCounters.classify_move(s, (A.PLAY_KNIGHT, h, j), mover)    # a pre-roll knight: mover rolls first
    assert info["kick"] == [((j - mover) % 4, j, R.KICK_LAMBDA)]
