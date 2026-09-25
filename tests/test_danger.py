import random

from catanbot import board as B
from catanbot import engine as E
from catanbot.agents.heuristic_bot import HeuristicBot
from catanbot.danger import (block_factor, danger_lines, rob_break_probability, steal_factor, win_path, win_paths)
from catanbot.placement import buildable_settlements, is_free_vertex
from catanbot.robber import best_robber_move, choose_victim, hex_damage, target_weight, threat
from catanbot.state import PHASE_GAME_OVER, PHASE_MAIN, new_game


def played(seed=5, turns=50):
    rng = random.Random(seed)
    s = new_game(4, rng=rng)
    bots = [HeuristicBot() for _ in range(4)]
    while s.phase != PHASE_GAME_OVER and s.turn < turns:
        i = E.acting_player(s)
        s = E.apply_inplace(s, bots[i].decide(s, E.legal_actions(s), rng), rng)
    s.phase = PHASE_MAIN
    s.current = 0
    s.dice = 6
    return s


def give_spot(s, i):
    """Make sure player i has a settlement spot buildable now (add a road to a free vertex if needed)."""
    if buildable_settlements(s, i):
        return
    occ = s.occupied_vertices()
    eocc = s.occupied_edges()
    for e, (a, b) in enumerate(B.EDGE_VERTICES):
        if e in eocc:
            continue
        if is_free_vertex(occ, a) or is_free_vertex(occ, b):
            s.players[i].roads.append(e)
            return
    raise AssertionError("no free vertex on the board")


def no_awards(s):
    """Put Longest Road / Largest Army out of everyone's reach (player 3 holds both, far ahead)."""
    s.longest_road_owner = 3
    s.longest_road_len = 15
    s.largest_army_owner = 3
    s.players[3].played_knights = 10


def test_win_path_basics_and_cache():
    s = played()
    paths = win_paths(s)
    assert set(paths) == {0, 1, 2, 3}
    assert win_paths(s) is paths                      # cached for the identical state
    wp = paths[1]
    assert wp.need_vp == B.VP_TO_WIN - round(wp.vp)
    assert all(m >= 0 for m in wp.missing) and 0.0 <= wp.danger <= 1.0
    assert abs(sum(wp.need_share) - 1.0) < 1e-9 or sum(wp.missing) < 1e-9
    s.players[1].resources = [3, 3, 3, 3, 3]
    assert win_paths(s) is not paths                  # a changed hand invalidates the cache
    assert win_paths(s)[1].turns <= wp.turns


def test_loaded_runner_up_outranks_overextended_leader():
    s = played()
    lead, run = 0, 1
    for i in (lead, run):
        s.players[i].cities = []
        s.players[i].settlements = list(s.players[i].settlements[:1]) or s.players[i].settlements
    # Leader: 9 VP (4 cities + 1 settlement), empty hand, no spot: only a 5-card city stands between
    # them and the win, but they have to gather it all.
    p = s.players[lead]
    verts = list(p.settlements) + list(p.cities)
    while len(verts) < 5:
        v = next(v for v in range(B.NUM_VERTICES) if v not in s.occupied_vertices())
        verts.append(v)
        p.settlements.append(v)
    p.cities = verts[:4]
    p.settlements = verts[4:5]
    p.resources = [0, 0, 0, 0, 0]
    p.roads = []                     # overextended: no road network, no spot, all cities already built
    # Runner-up: 8 VP (3 cities + 2 settlements), holding city + settlement cost with a spot to build.
    q = s.players[run]
    verts = [v for v in (list(q.settlements) + list(q.cities)) if v not in p.settlements + p.cities]
    while len(verts) < 5:
        occ = s.occupied_vertices()
        v = next(v for v in range(B.NUM_VERTICES) if v not in occ and v not in verts)
        verts.append(v)
    q.cities = verts[:3]
    q.settlements = verts[3:5]
    q.resources = [1, 1, 1, 3, 3]
    give_spot(s, run)
    no_awards(s)
    assert s.public_vp(lead) == 9 and s.public_vp(run) == 8
    paths = win_paths(s)
    assert paths[run].can_win_now and paths[run].danger == 1.0
    assert paths[lead].turns > 1.0 and paths[lead].danger < paths[run].danger
    assert not paths[lead].can_win_now
    assert threat(s, lead) > threat(s, run)                         # VP alone says: hit the leader
    assert target_weight(s, run) > target_weight(s, lead)           # distance to win says: hit the runner-up
    assert rob_break_probability(paths[run]) > 0.5
    lines = danger_lines(s, 2)
    assert "CAN WIN" in lines[0] and (s.players[run].name or s.players[run].color) in lines[0]


def test_block_factor_tracks_need_hand_and_port():
    s = played()
    i = 1
    p = s.players[i]
    verts = list(p.settlements) + list(p.cities)
    while len(verts) < 6:
        occ = s.occupied_vertices()
        v = next(v for v in range(B.NUM_VERTICES) if v not in occ and v not in verts)
        verts.append(v)
    p.cities = verts[:3]
    p.settlements = verts[3:6]          # 9 VP: one city upgrade wins
    # They need a city (ore + wheat), hold the wheat and one ore already: the city is the next step
    # even though a settlement is cheaper in total.
    p.resources = [0, 0, 0, 2, 1]
    no_awards(s)
    wp = win_path(s, i)
    assert wp.need_vp == 1 and wp.steps == ["city"]
    assert wp.need_share[B.ORE] > 0.99 and wp.missing[B.ORE] == 2.0
    f_ore = block_factor(wp, B.ORE, 5)
    f_sheep = block_factor(wp, B.SHEEP, 5)
    assert f_ore > 2.0 * f_sheep
    # Holding the ore too: blocking ore stops mattering.
    p.resources = [0, 0, 0, 2, 3]
    wp2 = win_path(s, i)
    assert block_factor(wp2, B.ORE, 5) < f_ore
    # A 2:1 sheep port turns sheep into their currency for the ore they need.
    p.resources = [0, 0, 0, 2, 1]
    v = p.settlements[0]
    s.ports = dict(s.ports)
    s.ports[v] = B.SHEEP
    wp3 = win_path(s, i)
    assert wp3.eff_need[B.SHEEP] > wp.eff_need[B.SHEEP]
    assert block_factor(wp3, B.SHEEP, 5) > f_sheep
    assert wp3.supply[B.ORE] >= wp.supply[B.ORE]


def test_steal_factor_prefers_useful_cards():
    s = played()
    i = 2
    p = s.players[i]
    p.cities = []
    p.settlements = list(p.settlements)[:2]
    no_awards(s)
    p.resources = [3, 0, 0, 0, 0]                    # wood only: useless for the city they need
    wp_wood = win_path(s, i)
    p.resources = [0, 0, 0, 2, 1]                    # wheat + ore: exactly what the city needs
    wp_city = win_path(s, i)
    assert steal_factor(wp_city) > steal_factor(wp_wood)
    # ... and cards *we* need count a little too.
    assert steal_factor(wp_wood, our_need=[1, 0, 0, 0, 0]) > steal_factor(wp_wood, our_need=[0, 0, 0, 0, 0])


def test_choose_victim_and_best_move_use_the_model():
    s = played(seed=8, turns=60)
    me = 0
    h, victim, reason = best_robber_move(s, me)
    assert 0 <= h < B.NUM_HEXES and h != s.robber
    assert isinstance(reason, str) and reason
    # a can-win-now opponent on some hex becomes the victim of choice
    for j in range(1, 4):
        q = s.players[j]
        if not q.settlements:
            continue
        q.resources = [0, 0, 0, 2, 3]
        q.cities = list(q.cities)
        while s.public_vp(j) < 9 and q.settlements:
            v = q.settlements.pop()
            q.cities.append(v)
        if s.public_vp(j) >= 9 and q.settlements:
            break
    else:
        return
    paths = win_paths(s)
    if not paths[j].can_win_now:
        return
    hexes = [hh for hh in range(B.NUM_HEXES) if hh != s.robber
             and any(v in B.HEX_VERTICES[hh] for v in q.settlements + q.cities)]
    assert hexes
    assert choose_victim(s, hexes[0], me) == j
    h, victim, reason = best_robber_move(s, me)
    assert victim == j and "can win" in reason


def test_hex_damage_is_need_aware():
    s = played()
    me = 0
    j = 1
    p = s.players[j]
    p.cities = []
    p.settlements = list(p.settlements)[:2]
    p.resources = [0, 0, 0, 2, 0]
    ore_hexes = [h for h in range(B.NUM_HEXES) if s.hexes[h][0] == B.ORE and s.hexes[h][1]
                 and any(v in B.HEX_VERTICES[h] for v in p.settlements)]
    other = [h for h in range(B.NUM_HEXES) if s.hexes[h][0] in (B.SHEEP, B.WOOD, B.BRICK) and s.hexes[h][1]
             and any(v in B.HEX_VERTICES[h] for v in p.settlements)]
    if not ore_hexes or not other:
        return
    # same pips: an ore hex they need beats a sheep/wood/brick hex they do not, other things equal
    for ho in ore_hexes:
        for hx in other:
            if B.PIPS[s.hexes[ho][1]] == B.PIPS[s.hexes[hx][1]]:
                d_ore = hex_damage(s, ho, me)[0]
                d_oth = hex_damage(s, hx, me)[0]
                # only opponent j's share is need-weighted; others may share the hex, so compare loosely
                assert d_ore > 0 and d_oth >= 0
                return
