#!/usr/bin/env python3
"""Replay logged catanatron 3.3 games and measure the finite-supply rules (docs/SCRUTINY.md, Q21).

    /home/user/venv_cat33/bin/python scripts/audit_limits.py proof/T1 ... proof/T11 bench_1v1/H1 bench_1v1/H2

Reports: pieces over the 15 roads / 5 settlements / 4 cities limit at any point (must be 0); player-games that
used up every piece of a kind; games where the bank hit 0 of a resource; rolls where a bank shortage cancelled
production (catanatron pays nobody that resource), and how many of those had a single owed player, whom the
official rule would pay what is left; Road Building plays by the number of free roads built.
"""
import sys, os, glob, gzip, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catanbot.bench import catanatron_adapter as AD
from catanatron.apply_action import yield_resources
from catanatron.state_functions import player_key
RES = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
tests = sys.argv[1:]
g = 0; rolls = 0; pgames = 0
short_rolls = 0; short_games = set(); short_cards = 0; short_by_res = collections.Counter()
single_owed_short = 0
bank_zero_games = collections.Counter(); bank_min = [19]*5
out_of = collections.Counter(); over = 0
rb = collections.Counter()
for t in tests:
    for f in sorted(glob.glob(f'{t}/logs/*.jsonl.gz')):
        for line in gzip.open(f):
            rec = json.loads(line); g += 1; pgames += len(rec['colors'])
            game = AD.rebuild_game(rec)
            acts = rec['actions']
            zero_seen = set(); ran_out = set()
            i = 0
            while i < len(acts):
                c, k, v, r = acts[i]
                st = game.state
                if k == 'ROLL':
                    rolls += 1
                    num = sum(r or v)
                    if num != 7:
                        payout, depleted = yield_resources(st.board, st.resource_freqdeck, num)
                        # who was owed what (intended), to see if a depleted resource hit anyone
                        owed = collections.defaultdict(lambda: collections.Counter())
                        for coord, tile in st.board.map.land_tiles.items():
                            if tile.number != num or st.board.robber_coordinate == coord: continue
                            for nid in tile.nodes.values():
                                b = st.board.buildings.get(nid)
                                if b: owed[tile.resource][b[0]] += 2 if b[1] == 'CITY' else 1
                        lost = [res for res in depleted if sum(owed[res].values()) > 0]
                        if lost:
                            short_rolls += 1; short_games.add((t, rec['game']))
                            for res in lost:
                                short_cards += sum(owed[res].values()); short_by_res[res] += 1
                                if len(owed[res]) == 1: single_owed_short += 1
                if k == 'PLAY_ROAD_BUILDING':
                    free = 0; j = i + 1
                    while j < len(acts) and acts[j][1] == 'BUILD_ROAD' and acts[j][0] == c: free += 1; j += 1
                    rb[min(free, 2)] += 1
                AD.replay_log_action(game, acts[i], check=False)
                st = game.state
                for ri, n in enumerate(st.resource_freqdeck):
                    bank_min[ri] = min(bank_min[ri], n)
                    if n == 0: zero_seen.add(RES[ri])
                for col in st.colors:
                    key = player_key(st, col)
                    for piece in ('ROADS', 'SETTLEMENTS', 'CITIES'):
                        a = st.player_state[f'{key}_{piece}_AVAILABLE']
                        if a < 0: over += 1
                        if a == 0: ran_out.add((col, piece))
                i += 1
            for res in zero_seen: bank_zero_games[res] += 1
            for col, piece in ran_out: out_of[piece] += 1
print(f'games {g}, rolls {rolls}')
print(f'pieces over the limit at any point: {over}')
print('player-games that used up every piece of a kind:', {p: out_of[p] for p in ('ROADS','SETTLEMENTS','CITIES')}, f'(of {pgames} player-games)')
print('games where the bank hit 0 of a resource at some point:', dict(bank_zero_games))
print(f'rolls where a shortage cancelled production: {short_rolls} ({100*short_rolls/rolls:.2f}% of rolls), in {len(short_games)} games; cards not paid: {short_cards}; by resource {dict(short_by_res)}; cases with a single owed player (official rule would pay them what is left): {single_owed_short}')
print('Road Building plays by free roads built:', dict(rb))
