# Benchmarks against catanatron

[catanatron](https://pypi.org/project/catanatron/) is an independent Python
Catan engine with its own rules implementation and a few built-in players.
`catanbot/bench/catanatron_adapter.py` lets a catanbot bot sit at a
catanatron table: it converts catanatron's state into a `GameState`, runs the
catanbot search on it and hands catanatron back one of its own
`playable_actions`.  `scripts/bench_catanatron.py` plays batches of games
against catanatron's players and prints win rates.  This gives an *external*
yardstick that does not share any code with our engine, heuristics or value
net.

Two catanatron generations are supported and tested, selected by whichever
interpreter runs the script: the **3.2.1 PyPI wheel** (system `python3`
here: only the weak stock bots) and the **3.3.0 engine of the GitHub
checkout**, installed editable into a virtualenv (here
`/home/user/venv_cat33/bin/python`, `pip install -e
/home/user/bcollazo/catanatron`: ships the strong `ValueFunctionPlayer`,
`AlphaBetaPlayer`, `SameTurnAlphaBetaPlayer`, `GreedyPlayoutsPlayer` and
`MCTSPlayer`).  The adapter detects the API by feature
(`catanatron_adapter.API_33`): 3.3 keeps the log as
`State.action_records` of `ActionRecord(action, result)`, the playable
actions on the `Game`, `apply_action` in its own module (and it must be
given the record to replay a logged roll / robbery exactly), discards one
`DISCARD_RESOURCE` per prompt with `State.discard_counts`, uses 2-tuple
robber values, follows the official "dev cards are not playable the turn
they are bought" rule and adds domestic-trade prompts.  See
"Running the full ladder" below for the commands on each version.

```bash
pip install catanatron                     # 3.2.1 wheel; only needed for this benchmark (pulls networkx)
python3 scripts/bench_catanatron.py --games 20 --opponent vp \
    --spec "search:depth=1,evaluator=heuristic" [--workers 2] [--seed 0] [--json out.json] [--verbose] \
    [--vps-to-win 10] [--discard-limit 7] [--trades off|native|value|fair] \
    [--opponent-params KEY=VAL,...] [--hash-seed 0] [--info full|counted [--info-samples 4] [--discards-public]]
python3 scripts/bench_catanatron.py --games 24 --mixed-opponents value,alphabeta,sameturn   # 1 catanbot + 3 different presets
python3 scripts/bench_catanatron.py --list-opponents       # which presets resolve on the running catanatron
python3 scripts/bench_catanatron.py --probe-trades 200 --opponent value,alphabeta,random   # 3.3: how opponents answer offers
python3 -m pytest tests/test_catanatron_adapter.py tests/test_bench_script.py -q   # adapter tests (skipped without catanatron)
```

* `--trades` (3.3 only; default `off`): whether catanbot offers domestic
  trades and how the opponents answer them - see "Domestic trading against
  catanatron 3.3" below.  `--opponent-params` passes constructor parameters
  to every opponent (a 3.3 player's `Params` fields such as `depth=3` for
  `alphabeta`, `num_simulations=50` for `mcts`, `num_playouts=10` for
  `playouts`, `value_fn=contender` for `value`; or the keyword arguments of
  an older-style class such as `ab`'s `budget=4000`), coerced to the
  declared types; a class that takes none of the keys is a one-line error
  (exit code 2).  Every run reports the compute per decision of both sides
  ("Compute per decision" below).
* Run as a script, the bench re-executes itself with
  `PYTHONHASHSEED=--hash-seed` (default 0; `-1` keeps random hashing), so the
  same `--seed` gives the same games in every process (verified: identical
  winners, turns, VP and action counts in two processes; unpinned runs
  differ).  Two runs that differ only in a catanbot-side option are therefore
  *paired* game by game until the option first changes a decision.

* `--opponent vp` = `VictoryPointPlayer` (greedy one-ply VP maximiser, the
  strongest player in the core `catanatron` 3.2.1 package), `weighted` =
  `WeightedRandomPlayer` (random, biased towards cities > settlements > dev
  cards), `random` = `RandomPlayer` (uniform); `vf` / `ab` are our own
  stand-in players built inside the engine (`docs/BENCHMARKS_OPPONENTS.md`,
  both versions); `value`, `alphabeta`, `sameturn`, `playouts`, `mcts` are
  catanatron's own strong players (3.3 only).  A preset the running
  catanatron does not ship fails with a one-line
  `... is not available on catanatron 3.2.1 ...` message and exit code 2
  (inside a comma list or a `--ladder` it is skipped with the same line).
* `--spec` is any `catanbot.selfplay.make_bot` spec (`search:...`,
  `heuristic`, `random`, `search:model=models/value_net.npz,...`).
* Every game seats one catanbot player against three copies of the opponent;
  game `g` puts catanbot in seat `g % 4` (colours RED, BLUE, ORANGE, WHITE by
  seat), so seats and first-move advantage rotate exactly.  Boards, ports,
  dev-deck order and dice are catanatron's, seeded per game.

## Results (2026-09-25, catanatron 3.2.1 stock bots, 2 worker processes on 4 shared cores)

| catanbot spec | opponent (x3) | games | wins | win rate | avg VP catanbot | avg VP opponents | avg turns | s / game |
|---|---|---|---|---|---|---|---|---|
| `search:depth=1,evaluator=heuristic` | VictoryPointPlayer | 100 | 99 | 99% | 10.03 | 2.74 | 81.8 | 0.53 |
| `search:depth=1,evaluator=heuristic` | WeightedRandomPlayer | 100 | 100 | 100% | 10.08 | 2.85 | 81.5 | 0.40 |
| `search:depth=1,evaluator=heuristic` | RandomPlayer | 100 | 100 | 100% | 10.09 | 2.40 | 78.4 | 0.41 |
| `search:depth=2,evaluator=heuristic` | VictoryPointPlayer | 20 | 20 | 100% | 10.05 | 2.77 | 82.9 | 2.14 |
| `heuristic` (rule-based bot, no search) | VictoryPointPlayer | 100 | 97 | 97% | 10.04 | 2.94 | 86.4 | 0.16 |
| `random` (control: uniform random catanbot bot) | VictoryPointPlayer | 100 | 13 | 13% | 4.48 | 6.01 | 214.3 | 0.45 |

The first two rows are the batches requested for this benchmark (20-game
versions of them gave 20/20 and 20/20; the 100-game batches above took 27 s
and 21 s of wall time).  A random seat wins 25 % of the time.  "avg turns" is
catanatron's `num_turns` (setup turns included); no game hit catanatron's
1000-turn cap.  Adapter statistics over all 520 games: 0 errors, 0 fallbacks,
0 observe errors; 53 of ~22 000 bot decisions (0.4 % of the search rows'
decisions) had a top-ranked action without catanatron equivalent (every one
of them `PLAY_ROAD_BUILDING`, see limitation 3) and used the next-ranked
action instead.

Reading the numbers:

* catanatron's built-in players are weak.  `VictoryPointPlayer` only looks
  one action ahead and picks uniformly among all actions that do not gain a
  point immediately (bank trades, `END_TURN`, ...), so it rarely accumulates
  resources: after ~80 turns the three opponents average 2.7 VP, barely above
  the 2 VP everybody starts with.  The `random` control row shows the
  catanatron players are nevertheless better than uniform random play (13 %
  for catanbot's random bot, below the 25 % seat baseline).
* The ranking is what it should be: random (13 %) << catanatron players <
  catanbot heuristic bot (97 %) < depth-1 search (99 %) <= depth-2 search
  (100 %).  Against these opponents the benchmark is saturated, so it can
  confirm the adapter and gross bot strength but cannot separate the search
  depths or a value net from the heuristic evaluator; use self-play
  tournaments (`python -m catanbot eval`, `docs/USAGE.md`) or a stronger
  external opponent (`catanatron_experimental`'s `AlphaBetaPlayer`) for that.
* Cost: a depth-1 game is ~0.4-0.5 s (0.35 s of it in the search, ~40
  searched decisions per game; the other ~35 decisions are trivial: only one
  playable action, or every legal catanbot action maps to the same catanatron
  action).  Depth 2 is ~5x slower.  Opponents cost almost nothing.

## Results (2026-09-25, full ladders: stand-ins on catanatron 3.2.1, strong players on 3.3.0)

Two ladders were run on the same day with the same bot,
`search:depth=1,evaluator=heuristic` (the bench default), the C++
acceleration on (`catanbot.accel.AVAILABLE` is `True` under both
interpreters), one catanbot player against three copies of the opponent,
seat `g % 4` (verified in the per-game records: 50/50/50/50 games per seat
in every 200-game batch, 40 per seat in the 160-game batch), `--seed 0`
and `--workers 2` on the 4 shared cores / 15 GB machine.  catanatron 3.2.1
is the PyPI wheel under the system `python3`; catanatron 3.3.0 is the
GitHub checkout installed editable into a virtualenv (here
`/home/user/venv_cat33/bin/python`).  The stand-ins `vf` / `ab` are our own
players built inside the engine, described and measured against each other
in [`docs/BENCHMARKS_OPPONENTS.md`](BENCHMARKS_OPPONENTS.md).  Every batch
was a single command under a 25-minute `timeout`, and the batches ran one
after another (never two benchmarks at once).  `s / game` is catanatron's
per-game wall time inside a worker; it was measured while another workflow
kept two to three more CPU-bound processes running (load average 5-13 on
the 4 cores), so it is 2-3x the idle cost (0.4-0.5 s for the stock bots,
6.5 s for `ab` in an uncontended 8-game probe, 0.73 s for `vp` on 3.3 in a
4-game timing run); win rates and VP are unaffected by load.  Win-rate
intervals are 95 % Wilson intervals.

### Stand-in ladder, catanatron 3.2.1 (system `python3`)

```bash
cd /home/user/ClaudeTesting2
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 python3 scripts/bench_catanatron.py \
    --opponent random,weighted,vp,vf --games 200 --workers 2 --seed 0 \
    --spec "search:depth=1,evaluator=heuristic" --verbose --json results_321_fast.json   # 7.6 min wall
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 python3 scripts/bench_catanatron.py \
    --opponent ab --games 160 --workers 2 --seed 0 \
    --spec "search:depth=1,evaluator=heuristic" --verbose --json results_321_ab.json     # 19.0 min wall under load (~9 min idle)
```

| opponent (x3) | games | wins | win rate (95 % CI) | avg VP catanbot / opponents | avg turns | s / game | wins by seat 0 / 1 / 2 / 3 | adapter: errors / fallbacks / unmapped top |
|---|---|---|---|---|---|---|---|---|
| `random` = RandomPlayer (stock control) | 200 | 200 | 100 % (98-100) | 10.08 / 2.38 | 75.6 | 1.16 | 50 / 50 / 50 / 50 of 50 | 0 / 0 / 37 |
| `weighted` = WeightedRandomPlayer (stock control) | 200 | 200 | 100 % (98-100) | 10.10 / 2.74 | 77.3 | 0.90 | 50 / 50 / 50 / 50 of 50 | 0 / 0 / 51 |
| `vp` = VictoryPointPlayer (stock control) | 200 | 199 | 99.5 % (97-100) | 10.06 / 2.74 | 76.7 | 0.97 | 50 / 49 / 50 / 50 of 50 | 0 / 0 / 32 |
| `vf` = our ValueFunctionPlayer (1-ply value function) | 200 | 49 | 24.5 % (19-31) | 7.33 / 7.39 | 90.0 | 1.52 | 13 / 18 / 9 / 9 of 50 | 0 / 0 / 50 |
| `ab` = our AlphaBetaPlayer (depth-2 expectimax) | 160 | 39 | 24.4 % (18-32) | 7.16 / 7.27 | 91.3 | 14.1 (3-136; 6.5 idle) | 14 / 11 / 5 / 9 of 40 | 0 / 0 / 51 |

Over the 960 games: 0 adapter errors, 0 fallbacks, 0 observe errors, no game
near the 1000-turn cap (longest 204 turns); the 221 "unmapped top" decisions
are all `PLAY_ROAD_BUILDING` ranked first while catanatron 3.2.1 did not
offer it (limitation 3: it needs wood + brick there), and the next-ranked
action was played each time.  The search itself costs ~0.8-1.1 s per game
(37-38 searched decisions).

### Strong ladder, catanatron 3.3.0 (virtualenv with the GitHub checkout)

```bash
cd /home/user/ClaudeTesting2
PY=/home/user/venv_cat33/bin/python      # a virtualenv with the GitHub checkout installed editable
PYTHONPATH=/home/user/ClaudeTesting2 $PY scripts/bench_catanatron.py --list-opponents                 # all ten presets "available"
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent vp        --games 100 --workers 2 --seed 0 --verbose --json run_vp.json         # 0.9 min wall
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent value     --games 200 --workers 2 --seed 0 --verbose --json run_value.json      # 3.1 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent alphabeta --games 30  --workers 2 --seed 0 --verbose --json run_alphabeta.json  # 8.7 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent sameturn  --games 60  --workers 2 --seed 0 --verbose --json run_sameturn.json   # 11.8 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent mcts      --games 40  --workers 2 --seed 0 --verbose --json run_mcts.json       # 12.0 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent playouts  --games 2   --workers 2 --seed 0 --verbose --json run_playouts.json   # 18.1 min
```

(`--spec` left at its default, which is the same `search:depth=1,evaluator=heuristic`.)

| opponent (x3) | games | wins | win rate (95 % CI) | avg VP catanbot / opponents | avg turns | s / game | wins by seat 0 / 1 / 2 / 3 | adapter: errors / fallbacks / unmapped top |
|---|---|---|---|---|---|---|---|---|
| `vp` = VictoryPointPlayer (stock control) | 100 | 100 | 100 % (96-100) | 10.11 / 2.72 | 77.3 | 1.03 | 25 / 25 / 25 / 25 of 25 | 0 / 0 / 0 |
| `value` = ValueFunctionPlayer (catanatron's, default weights) | 200 | 127 | 63.5 % (57-70) | 8.90 / 5.86 | 83.4 | 1.87 | 34 / 32 / 32 / 29 of 50 | 0 / 0 / 0 |
| `alphabeta` = AlphaBetaPlayer (default depth 2) | 30 | 19 | 63.3 % (46-78) | 8.77 / 6.26 | 83.4 | 34.5 (longest game 70 s) | 4/8, 4/8, 5/7, 6/7 | 0 / 0 / 0 |
| `sameturn` = SameTurnAlphaBetaPlayer | 60 | 45 | 75.0 % (63-84) | 9.08 / 5.86 | 80.9 | 23.3 | 11 / 12 / 11 / 11 of 15 | 0 / 0 / 0 |
| `mcts` = MCTSPlayer (default 10 simulations) | 40 | 40 | 100 % (91-100) | 10.03 / 2.87 | 82.1 | 35.4 (14-119) | 10 / 10 / 10 / 10 of 10 | 0 / 0 / 0 |
| `playouts` = GreedyPlayoutsPlayer (default 25 playouts per action) | 2 | 2 | 100 % (34-100) | 10.50 / 3.83 | 108.5 | 950 (813 and 1088) | 1/1, 1/1, -, - | 0 / 0 / 0 |

Over the 432 games: 0 adapter errors, 0 fallbacks, 0 unmapped top actions
(3.3 generates the two free Road Building roads without checking the road
cost, so limitation 3 does not arise there), 0 observe errors, 0
domestic-trade prompts (no stock 3.3 player offers trades), no game near the
turn cap (longest 189 turns, in the `mcts` batch); 2479 discard cards were
planned once and handed over card by card.  The search costs 0.8-1.0 s per
game; everything else in `s / game` is the opponents' thinking time.

### Controls, assessment and what the losses look like

* **Stock controls (both versions).**  The sanity rule for the adapter
  holds: >= 99 % against `random`, `weighted` and `vp` on 3.2.1 (200 games
  each; the earlier section had 100 / 100 / 99 on 100 games) and 100/100
  against `vp` on 3.3.0, with the opponents stuck at 2.4-2.9 VP.  These rows
  are saturated and only show that the adapter and the bot work; they
  cannot rank bots.
* **Our stand-ins `vf` / `ab` (3.2.1).**  The default depth-1 heuristic bot
  sits exactly on the 25 % seat par against both (24.5 % and 24.4 %, avg
  VP 7.3 vs 7.4 and 7.2 vs 7.3): it is as strong as one `vf` / `ab`
  player, not stronger, and `ab` is not measurably harder for it than `vf`
  (the two stand-ins also tie each other at par, see
  `docs/BENCHMARKS_OPPONENTS.md`).  There is a pronounced seat effect: 31 of
  100 wins from seats 0-1 but 18 of 100 from seats 2-3 against `vf`, 25 of
  80 vs 14 of 80 against `ab`, because the stand-ins' opening book takes
  the best production spots first (the stand-ins themselves win far more
  often from seats 0-1: winner seats 59 / 59 / 36 / 46 in the `vf` games).
  62 of the 151 losses to `vf` and 53 of the 121 losses to `ab` were
  second places; 71 and 52 of them were lost by 1-3 VP.
* **catanatron's own strong players (3.3.0).**  Clearly above par but no
  longer dominant: 63.5 % against `value` (the only tightly measured row,
  CI 57-70 %, confirmed by 66 % in 116 further games), 63 % against
  `alphabeta` and 75 % against `sameturn` (30 and 60 games; the three
  intervals overlap, so these opponents cannot be ordered against each
  other from this run, and the `sameturn` figure came out at 50 % in 22
  verification games, see below: read it as "about 60-66 %, like the other
  two").  The games are competitive
  rather than blowouts (8.8-9.1 VP for catanbot against 5.9-6.3 for the
  opponents; 29 of the 73 losses to `value` ended with catanbot at 8-9 VP,
  41 of them in second place).  Note the asymmetry with the stand-ins: our
  `vf` beats catanatron's `value` player 50 % vs 3 of them
  (`docs/BENCHMARKS_OPPONENTS.md`), and the search bot lands on par against
  `vf` but at 63.5 % against `value`, i.e. the pictures are consistent.
* **`mcts` and `playouts` at their defaults (3.3.0).**  Not informative:
  `MCTSPlayer` with catanatron's default of 10 simulations loses like the
  stock bots (40/40, opponents at 2.9 VP) and `GreedyPlayoutsPlayer` (25
  random playouts per playable action, ~1 s per opponent decision) costs
  813 and 1088 s per game here, so only 2 games fit the 25-minute limit and
  2/2 (CI 34-100 %) is not a sample.  Stronger settings would need a way to
  pass opponent parameters (no `--opponent-params` flag yet).  `playouts`
  also prints one `Greedy took ... secs` line per opponent decision, which
  the script does not silence.
* **Why it loses** (per-decision traces of 8 losses to `vf`, 5 to `ab`, 3
  to `value`, 1 each to `alphabeta` / `sameturn`, replayed with
  `PYTHONHASHSEED=0` on fresh seeds because the ladder games themselves are
  not replayable, limitation 8): nothing mechanical.  0 errors, 0
  fallbacks, 0 timeouts (`SearchConfig.time_limit` is `None`; the slowest
  decision took 0.16 s, at most 4.3 s of search per game), no stalls
  (`END_TURN` was never chosen while a settlement or city was affordable),
  normal 78-130-turn games.  The losses are economic and start in the
  opening / mid game: the winner had 3-4 cities in every inspected 3.3
  loss while catanbot ended with 0-2, having put its resources into roads
  (11-15 in several losses, once all 15 with Longest Road and one city),
  settlements and development cards (12 dev cards / 8 knights in one `vf`
  loss); it ends turns holding 4-5 of one resource with a 4:1 trade legal
  and then loses them to 7s and the robber (the 3.3 players rob the
  leader); against the stand-ins it picks 3rd / 4th and starts from the
  weakest production at the table.  A one-ply search with the heuristic
  evaluator does not see the multi-turn payoff of a city or of a 4:1 trade,
  which is exactly what the value-function / alpha-beta opponents optimise.

### Verification of these numbers

* Every table entry above was recomputed from the per-game records in the
  JSON files (`wins`, win rate, wins by seat, average VP / turns / duration,
  the summed adapter statistics, `won` == `winner_seat == seat`, every
  winner at >= 10 VP): all agree, seats rotate `g % 4`, and in every batch
  `sum(duration) / 2` equals the wall time (both workers busy throughout),
  so the `s / game` figures are consistent with the wall times.
* Independent re-runs on the current tree with different games (`--seed
  11`, `12` and `13`, `--workers 2`, same spec, 0 adapter errors /
  fallbacks / observe errors in all of them; p = two-sided binomial
  probability of the re-run given the table's rate).  3.2.1: `vp` 8/8;
  `vf` 5/16 and 10/48, pooled 15/64 = 23 % (p = 1.0 against 24.5 %); `ab`
  4/8 and 4/32, pooled 8/40 = 20 % (p = 0.59 against 24.4 %): the
  stand-in table reproduces, on the newer `search.py` as well.  3.3.0:
  `value` 8/16 and 68/100, pooled 76/116 = 66 % (p = 0.41 for the 100-game
  run against 63.5 %); `alphabeta` 2/6 (p = 0.20 against 63 %);
  `sameturn` 2/6 and 9/16, pooled 11/22 = 50 % (p = 0.012 against 75 %).
  The `sameturn` row is therefore **not** reproduced at the 95 % level: its
  60-game result looks like a high draw.  An earlier 64-game batch in the
  scratch directory (seeds 0-3, 16 games each, partly on the older
  `search.py`) gave `sameturn` 40/64 = 62.5 %, `alphabeta` 39/64 = 61 %
  and `value` 263/400 = 66 %; pooling everything, the best estimates are
  `value` 466/716 = 65 % (62-68), `sameturn` 96/146 = 66 % (58-73) and
  `alphabeta` 60/100 = 60 % (50-69): the three strong players are
  statistically indistinguishable for this bot, at roughly 60-66 %.
* Caveats.  (1) `scripts/bench_catanatron.py` fixes the per-game seeds but
  does not pin `PYTHONHASHSEED`, so the same seeds give different games in
  different processes (limitation 8; `scripts/catanatron_ladder.py` pins
  it): the tables are statistically, not trajectory-wise, reproducible.
  (The bench pins `PYTHONHASHSEED=0` by default since the trading work of
  the same day, `--hash-seed`; runs from then on are trajectory-reproducible.)
  (2) `catanbot/search.py` and `catanbot/heuristic.py` were being edited by
  another workflow during the day: the 3.2.1 ladder ran on the 04:35
  `search.py` / 05:26 `heuristic.py`, the 3.3.0 ladder and the re-runs
  above on the 06:46 `search.py`; each table is internally consistent but
  may not reproduce exactly on a later tree.  (3) All timings are inflated
  by the concurrent load; use them as upper bounds when budgeting a run.
  (4) The JSON is written only when a batch completes, so a batch that hits
  its `timeout` loses everything but the `--verbose` log; keep the per-run
  game counts of the commands above.

## Domestic trading against catanatron 3.3

catanatron 3.3.0 (the GitHub checkout) has player-to-player trading; 3.2.1
has none.  The adapter used to decline every trade prompt and never offer.
It now plays catanbot's trade decisions through catanatron's protocol, and
`--trades` decides how the opponents answer (2026-09-25).

### The protocol (from `catanatron/game.py`, `models/actions.py`, `apply_action.py`)

1. **Offer.**  `OFFER_TRADE`, value `(5 offered counts, 5 asked counts)` in
   WOOD, BRICK, SHEEP, WHEAT, ORE order (catanbot's order).  It is never in
   `playable_actions`: `Game.execute` accepts one (`is_valid_action`) from
   the current player at a `PLAY_TURN` prompt after the roll when both halves
   are non-empty and share no resource.  There is **no per-turn limit** and
   **no check that the offerer holds the offered cards** (`CONFIRM_TRADE`
   would drive its hand negative).  `State.current_trade` becomes
   `offer + (offerer seat,)` and the prompt `DECIDE_TRADE`.
2. **Answers.**  The seats are asked one at a time in seat order, starting
   at the first seat that is not the offerer.  Playable: `REJECT_TRADE` and,
   if the seat holds the asked cards, `ACCEPT_TRADE` (both carry
   `current_trade`).  Accepting only sets `State.acceptees[seat]`; no card
   moves yet.  **Engine quirk:** the next seat asked is "the next higher seat
   that is not the one answering", so the offerer is asked about its *own*
   offer whenever it is not seat 0 (offerer in seat 2: seats 0, 1, 2, 3 are
   asked).  `CatanbotPlayer` answers that prompt `REJECT_TRADE` without a
   search (`stats["self_offer_prompts"]`), `BenchOpponent` likewise.
3. **Resolution.**  After the last seat: nobody accepted -> back to
   `PLAY_TURN` (the offerer may offer again); otherwise `DECIDE_ACCEPTEES`
   for the offerer with `CANCEL_TRADE` and one `CONFIRM_TRADE` per accepter
   (value: the 10 counts + the accepter's `Color`).  `CONFIRM_TRADE` swaps
   the cards; either way the offerer is back at `PLAY_TURN`.

### How catanatron's players answer an offer (measured)

`--probe-trades 200 --seed 1` on 3.3.0: 100 mid-game positions (post-roll,
turns 20-90 of games between four catanatron `ValueFunctionPlayer`s); at
each the turn player offers the first seat catanatron asks a random trade
that seat can pay, in one of three categories - **1:1** (the responder
receives one card and pays one), **2:1** (receives two, pays one:
favourable to it) and **1:2** (receives one, pays two: unfavourable).  200
offers per category, the same 600 offers for every player.  "Native" is the
player's own `decide` (errors: the call raised); the two rule columns are
our response rules evaluated with that player's value function (below).
The standard error of a 200-offer rate is at most 3.5 points.

| player (preset) | native 1:1 / 2:1 / 1:2 | native errors | `value` rule 1:1 / 2:1 / 1:2 | `fair` rule 1:1 / 2:1 / 1:2 | ms / answer |
|---|---|---|---|---|---|
| `ValueFunctionPlayer` (`value`) | 0 / 0 / 0 % | 0 | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.9 |
| `AlphaBetaPlayer` (`alphabeta`) | - | 600 / 600 `RuntimeError` | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.1 |
| `SameTurnAlphaBetaPlayer` (`sameturn`) | - | 600 / 600 `RuntimeError` | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.1 |
| `MCTSPlayer` (`mcts`) | - | 600 / 600 `RuntimeError` | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.1 |
| `GreedyPlayoutsPlayer` (`playouts`, 25 playouts) | 35.5 / 39.5 / 46.5 % | 0 | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 1215 |
| `RandomPlayer` (`random`) | 54.0 / 49.5 / 49.0 % | 0 | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.0 |
| `WeightedRandomPlayer` (`weighted`) | 45.5 / 50.5 / 49.0 % | 0 | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.0 |
| `VictoryPointPlayer` (`vp`) | 44.0 / 55.0 / 52.5 % | 0 | 29.5 / 65.0 / 23.5 % | 28.5 / 60.0 / 0 % | 0.3 |
| our stand-in `vf` | 0 / 0 / 0 % | 0 | 32.0 / 67.5 / 27.5 % | 30.5 / 63.0 / 0 % | 0.4 |
| our stand-in `ab` | 0 / 0 / 0 % | 0 | 32.0 / 67.5 / 27.5 % | 30.5 / 63.0 / 0 % | 0.4 |

Why:

* **`ValueFunctionPlayer` always rejects.**  It executes each playable
  action on a copy and evaluates its value function; `ACCEPT_TRADE` moves no
  card (the swap happens at `CONFIRM_TRADE`), so both answers score the same
  and its strict `>` keeps the first listed one, `REJECT_TRADE`.  The terms
  of the offer never enter the decision.  Our stand-ins `vf` / `ab` make the
  same 1-ply comparison and also always reject.
* **`AlphaBetaPlayer`, `SameTurnAlphaBetaPlayer` and `MCTSPlayer` crash.**
  Their outcome expansion (`tree_search_utils.execute_spectrum`) has no case
  for trade actions and raises `RuntimeError: Unknown ActionType
  ActionType.REJECT_TRADE` on every offer they could accept (a seat that
  cannot pay has a single playable action, returned before any search).
  Unwrapped, the exception leaves `Game.play()` and ends the game.
* **`RandomPlayer`, `WeightedRandomPlayer` and `VictoryPointPlayer` flip a
  coin** (the VP player sees a VP tie and picks at random): about 50 %
  whatever the terms.
* **`GreedyPlayoutsPlayer` answers by noise.**  It plays 25 random
  playouts after each answer (in which the offerer confirms or cancels at
  random) and keeps the answer with more wins; the terms barely matter
  (35.5 / 39.5 / 46.5 % - it accepts the unfavourable 1:2 offers *most*
  often) and each answer costs 1.2 s.
* **No catanatron player ever offers a trade** (none of them generates
  `OFFER_TRADE`), so in the bench only catanbot proposes.

"Native" trading is therefore degenerate against every catanatron player:
against `value` every offer is refused (trading can only cost catanbot
compute), against `alphabeta` / `sameturn` / `mcts` the game would crash
(the bench turns the exception into a rejection and counts it), and against
the random players and `vp` half of all offers succeed regardless of their
terms, which a proposer can exploit.  Hence the switch.

### `--trades {off,native,value,fair}`

| mode | catanbot | the opponents answer an offer |
|---|---|---|
| `off` (default) | never offers, declines any prompt | - (nobody offers) |
| `native` | offers | with their own `decide` (degenerate, above); an exception is counted (`opp_errors`) and answered `REJECT_TRADE` |
| `value` | offers | by **our** rule: accept iff the player's value function is strictly higher after the trade (both hands changed as `CONFIRM_TRADE` would) than before |
| `fair` | offers | `value`, and never give more cards than received, and never trade with a proposer holding `vps_to_win - 2` (8) or more public VP |

`value` and `fair` are **our model of a sensible opponent, not catanatron's
behaviour**.  `catanatron_adapter.BenchOpponent` wraps every opponent (it
also times them, next section); only the answer to an offer is replaced,
every other decision is the catanatron player's.  The value function is
the player's own where it has one (catanatron's `value` / `alphabeta` /
`sameturn`: their `value_fn` and weights, `base_fn` by default; our `vf` /
`ab`: their `_value`), otherwise catanatron's `base_fn` with its default
weights (`random`, `weighted`, `vp`, `mcts`, `playouts`).  Caveats:
`base_fn` sees a trade only through the responder's own hand - the
`hand_synergy` term (distance to a city / settlement, weight 100), the card
count (weight 1) and the >7-card penalty - so the `value` rule ignores what
the proposer gains and accepts some trades that cost it a card (23.5 % of
the 1:2 offers above); `fair` removes those and refuses near-winners.  Our
stand-ins' value function subtracts the best opponent's score, so for `vf` /
`ab` the rule also weighs the proposer's gain.

The catanbot side (`CatanbotPlayer(suppress_trades=False)`, any `--trades`
but `off`): the search's `PROPOSE_TRADE` choices are played as
`OFFER_TRADE` (only cards we hold, at most catanbot's 4 offers per turn, the
search's own cap of 4 early / 2 late applies first), `DECIDE_ACCEPTEES`
is decided in `PHASE_TRADE_SELECT` (`EXECUTE_TRADE` -> `CONFIRM_TRADE` with
that partner; the search's "never hand a build to a player about to win"
filter applies) and an incoming `DECIDE_TRADE` in `PHASE_TRADE_RESPONSE`.
Every logged trade action is observed (`OFFER_TRADE` -> `PROPOSE_TRADE`,
answers -> `ACCEPT_TRADE` / `REJECT_TRADE`, `CONFIRM_TRADE` ->
`EXECUTE_TRADE`, `CANCEL_TRADE`), so the opponent model learns each
opponent's acceptance rate and resource valuations exactly as in self-play;
the offerer's answer to its own offer and the forced rejection of a seat
that cannot pay (which catanbot's engine never asks) are not observed.

### Smoke: 40 games against `value`, trades off vs on (same seeds)

```bash
PY=/home/user/venv_cat33/bin/python
for t in off native value fair; do
  PYTHONPATH=/home/user/ClaudeTesting2 timeout 1150 $PY scripts/bench_catanatron.py --opponent value \
      --games 40 --workers 2 --seed 7 --trades $t --verbose --json smoke_$t.json
done
```

Depth-1 heuristic bot (default spec), `PYTHONHASHSEED=0`, so the four runs
play the same 40 boards / dice sequences and coincide until catanbot's
first offer (every game had one).

| `--trades` | catanbot wins | avg VP (catanbot / opp.) | turns | offers / game | accepted by >= 1 seat | executed | opponents accept (answerable offers) | could not pay | adapter errors / fallbacks | s / game |
|---|---|---|---|---|---|---|---|---|---|---|
| `off` | 28/40 = 70.0 % | 9.07 / 5.97 | 84.6 | 0 | 0 | 0 | - | - | 0 / 0 | 1.63 |
| `native` | 27/40 = 67.5 % | 9.03 / 5.99 | 84.8 | 59.0 | 0 | 0 | 0 / 5 087 (0 %) | 1 999 | 0 / 0 | 5.38 |
| `value` | 34/40 = 85.0 % | 9.78 / 5.13 | 68.7 | 41.0 | 10.1 | 8.6 | 515 / 3 504 (14.7 %) | 1 416 | 0 / 0 | 4.21 |
| `fair` | 34/40 = 85.0 % | 9.78 / 5.18 | 69.6 | 41.5 | 9.8 | 8.4 | 502 / 3 544 (14.2 %) | 1 430 | 0 / 0 | 5.07 |

No adapter errors, fallbacks, unmapped actions or observe errors in any
run, and no opponent error (`value` never raises).  Paired against `off`
(the same 40 games): `value` **+15.0 points** (9 games won only with
trading, 3 only without; paired s.e. 8.4), `fair` +15.0 (the same 9 / 3:
catanbot proposes 1-for-1 and 2-for-1 deals in the responder's favour, so
`fair`'s card-count condition never binds and its near-winner guard hardly
fires), `native` -2.5 (4 / 5; s.e. 7.6: against a player that refuses
everything the offers only cost time).  The direction agrees with
self-play's +17.5 points for search trade proposals (`docs/ABLATIONS.md`),
but 40 games resolve nothing below ~15 points.  With pinned hashing the
on / off runs are paired; the discordant-game rate here (12/40 = 30 %)
gives a paired standard error of about `sqrt(0.30 / n)`: 1.2 points at
2 000 games.

Two catanbot behaviours show up.  (1) It offers a lot: 41 offers per game
under `value` (2.4 per own turn, 16-62 per game), of which 25 % find an
accepter and 21 % are executed (catanbot cancels 1.4 accepted offers per
game in `PHASE_TRADE_SELECT`).  (2) It does not stop offering to players who
never accept: 59 offers per game and not one accepted under `native`.  The
opponent model's acceptance rate for such a player bottoms out near 0.075
(decayed average with its prior; `OpponentModel.predict_accept` then still
gives roughly 10-20 % per seat) and a refused offer costs nothing in the
search, so offering keeps a positive expected value.  The result does not
suffer (-2.5 +- 7.6) but our compute per game triples.

Cost: catanbot's time per game rises from 0.93 s to 3.5-4.4 s (151-187
decisions per game instead of 84, proposals in every main-phase search),
so at `--workers 2` against `value` the bench plays about 4 300 games per
hour with trades off and 1 300-1 700 with trades on (wall 33 s / 85-110 s
for 40 games, measured while a trade probe loaded a third core).  Games
with trading are 16 turns shorter (68.7 vs 84.6 catanatron turns).

## Compute per decision

`BenchOpponent` times every `decide` of the three opponents and
`CatanbotPlayer` every one of its own (including the adapter's state
conversion and the replay of the log, i.e. everything the seat costs); each
game record carries `timing` (mean / p50 / p95 / max ms, count, total
seconds per side, both over all decisions and over "choices" - decisions
with more than one playable action - plus our search time alone) and the
summary pools them.  `--trades off`, default spec (depth-1 heuristic
search), `--workers 2`, 3.3.0 unless noted:

| opponent (`--opponent-params`) | trades | games | catanbot ms / decision: mean / p95 (choices: mean / p95) | catanbot s / game | opponent ms / decision: mean / p95 / max (choices: mean / p95) | opponent s / game / seat | catanbot / opponent compute | wall s / game at 2 workers (games / hour) | catanbot wins |
|---|---|---|---|---|---|---|---|---|---|
| `value` | off | 40 | 11.2 / 43 (17.0 / 75) | 0.93 | 3.0 / 13 / 102 (4.8 / 16) | 0.23 | 4.14 | 0.8 (4307) | 28/40 |
| `alphabeta` | off | 8 | 8.5 / 30 (13.0 / 54) | 0.72 | 105.4 / 232 / 6057 (189.7 / 1432) | 8.10 | 0.09 | 13.2 (273) | 7/8 |
| `alphabeta` (depth=1) | off | 4 | 12.1 / 76 (18.2 / 89) | 0.96 | 4.5 / 22 / 133 (7.8 / 35) | 0.30 | 3.25 | 1.0 (3497) | 4/4 |
| `sameturn` | off | 8 | 9.8 / 38 (15.2 / 67) | 0.77 | 103.0 / 249 / 4607 (190.8 / 1547) | 7.17 | 0.11 | 11.9 (303) | 7/8 |
| `mcts` | off | 4 | 11.2 / 39 (18.3 / 60) | 0.63 | 123.7 / 375 / 601 (236.7 / 431) | 5.90 | 0.11 | 10.0 (361) | 4/4 |
| `playouts` (num_playouts=5) | off | 2 | 8.9 / 39 (15.5 / 50) | 0.60 | 734.5 / 3556 / 10951 (1507.9 / 7303) | 42.72 | 0.01 | 78.9 (46) | 2/2 |
| `vf` | off | 8 | 7.3 / 26 (10.7 / 38) | 0.59 | 1.2 / 5 / 19 (2.0 / 7) | 0.09 | 6.25 | 0.5 (7198) | 4/8 |
| `ab` | off | 8 | 7.9 / 36 (12.1 / 51) | 0.60 | 30.7 / 172 / 711 (49.3 / 225) | 2.22 | 0.27 | 3.9 (932) | 1/8 |
| `value` | value | 40 | 23.3 / 98 (37.6 / 132) | 3.51 | 2.2 / 12 / 113 (3.5 / 15) | 0.22 | 15.82 | 2.1 (1693) | 34/40 |

(3.3.0, `--workers 2 --seed 7 --verbose --json ...` with `--opponent
alphabeta --games 8`, `--opponent alphabeta --opponent-params depth=1
--games 4`, `--opponent sameturn --games 8`, `--opponent mcts --games 4`,
`--opponent playouts --opponent-params num_playouts=5 --games 2`,
`--opponent vf,ab --games 8`; `PYTHONHASHSEED=0`, no other benchmark
running; the `value` rows are the smoke runs above, during which a
single-process trade probe also ran.  "Choices" = decisions with
more than one playable action; the opponents' numbers pool all three
seats.  Win counts of 2-8 games are only a sanity check.)

* **Compute is not equal, and the direction depends on the opponent.**
  catanbot's depth-1 heuristic search costs 0.6-1.0 s per game (8-12 ms
  per decision, p95 30-75 ms).  Against catanatron's `value` player (0.23 s
  per seat per game) and our `vf` stand-in it spends 4-6x more; against
  `alphabeta` (8.1 s per seat per game, 105 ms per decision, p95 232 ms,
  slowest 6.1 s), `sameturn` (7.2 s) and `mcts` (5.9 s) it spends about
  **10x less**, against `ab` about 4x less and against `playouts` (even at
  `num_playouts=5`: 43 s per seat per game) 70x less.  So catanbot's 60-66 %
  against `alphabeta` / `sameturn` (results section) is not bought with more
  compute, while its ~65 % against `value` uses about 4x the opponent's.
* **Trading changes our side only**: with `--trades value` catanbot spends
  3.5 s per game (23 ms per decision, p95 98 ms) - 16x a `value` seat.
  Compute-matched comparisons with trading on need that factor in mind (or
  a cheaper spec).
* catanatron's `AlphaBetaPlayer` stops a search after 20 s of wall time;
  its slowest decision here was 6.1 s (`sameturn` 4.6 s), so no search was
  cut short, but a machine loaded far beyond 2 workers could reach the
  limit and change its play.
* **Throughput for the campaign** (2 workers): `value` about 4 300 games /
  hour (1 700 with `--trades value`), `vf` 7 200, `ab` 930, `alphabeta`
  270, `sameturn` 300, `mcts` 360, `playouts` (5 playouts) 46.  A 2 000-game
  run against `alphabeta` or `sameturn` therefore takes about 7 hours of
  2-worker time, i.e. about 25-35 commands of 60-80 games under a 20-minute
  limit (distinct `--seed`s, merged afterwards; the JSON is written only
  when a command completes); against `value` it is
  about 30 minutes with trades off and 70 minutes with trades on.
  `--opponent-params depth=1` makes `alphabeta` 25x cheaper (0.30 s per
  seat per game) but it is then a different, weaker opponent (4/4 wins).

## How the mapping works

### Board ids

| | catanatron | catanbot |
|---|---|---|
| tiles | cube coordinate `(x, y, z)`; `LandTile.id` 0..18 in template order (centre, ring 1, ring 2) | hex index 0..18, rows 3-4-5-4-3 top to bottom, left to right; axial `(q, r)` pointy-top (`board.HEX_COORDS`) |
| vertices | node id 0..53 (the water ring adds ids >= 54 that never carry buildings) | vertex id 0..53 sorted by `(y, x)` |
| edges | `(node_a, node_b)` tuples (both orientations stored in `board.roads`, `playable_actions` use `a < b`) | edge id 0..71 |
| corners | `NodeRef` NORTH, NORTHEAST, SOUTHEAST, SOUTH, SOUTHWEST, NORTHWEST | corner 0 (top) .. 5, clockwise (`board.HEX_VERTICES`) |

`derive_mapping(catan_map)` derives the bijections structurally instead of
hard-coding them:

1. **Tiles**: the 12 symmetries of the hexagonal board (6 rotations x
   mirror, `board_symmetries()`, identity first) are applied to the cube
   coordinates and projected to axial `(q, r) = (x, z)`; a symmetry is kept
   only if all 19 tiles land on 19 distinct catanbot hexes.
2. **Nodes** (`_match_nodes`): every node's *incidence set* (the tiles it
   belongs to, mapped to hexes) must equal the incidence set
   `board.VERTEX_HEXES[v]` of its vertex.  This pins down the 24 interior
   nodes (3 tiles) and the 12 coastal nodes shared by two tiles (the other
   vertex of that shared edge touches three hexes).  The 18 single-tile
   corners come in ambiguous pairs on the six corner hexes; they are resolved
   by propagation along catanatron's edges: a node adjacent to an already
   mapped node must map to a neighbour (`board.VERTEX_NEIGHBORS`) of that
   node's vertex.
3. **Edges**: `board.edge_between(node_to_vertex[a], node_to_vertex[b])`
   for every tile edge.
4. `verify_mapping` re-checks everything (tile/node/edge bijections, every
   tile's six nodes are exactly the hex's six corners, every tile edge is one
   of the hex's six sides, both orientations of every edge agree) and the
   first symmetry that passes wins.

Because the coordinate set of a hexagonal board is invariant under all 12
symmetries and the node incidence structure is too, every symmetry yields
*a* consistent isomorphism; trying the identity first keeps the natural
orientation whenever catanatron's convention matches ours - which it does in
3.2.1: `NORTH` is corner 0 and the refs run clockwise (asserted by
`test_natural_orientation_is_preferred`; `test_non_identity_symmetry_search`
covers the fallbacks).  Tile and node ids are identical for every `BASE_MAP`
game (only resources, numbers and ports are shuffled), so the mapping is
cached by structure (`mapping_for`).  The derived table:

```
              h0=t15 (0,2,-2)     h1=t16 (1,1,-2)     h2=t17 (2,0,-2)
      h3=t14 (-1,2,-1)    h4=t5 (0,1,-1)      h5=t6 (1,0,-1)      h6=t18 (2,-1,-1)
h7=t13 (-2,2,0)   h8=t4 (-1,1,0)    h9=t0 (0,0,0)     h10=t1 (1,-1,0)   h11=t7 (2,-2,0)
      h12=t12 (-2,1,1)    h13=t3 (-1,0,1)     h14=t2 (0,-1,1)     h15=t8 (1,-2,1)
              h16=t11 (-2,0,2)    h17=t10 (-1,-1,2)   h18=t9 (0,-2,2)
```

(`hN` = catanbot hex, `tM` = catanatron tile id, then the cube coordinate.)

### State (`to_catanbot_state(game, me_color)` / `state_to_catanbot(state)`)

| catanatron | catanbot |
|---|---|
| `LandTile.resource / number` (`None` = desert) | `hexes[h] = (resource, number)` (desert = `(DESERT, 0)`) |
| `board.robber_coordinate` | `robber` |
| `map.port_nodes` (`None` = 3:1) | `ports = {vertex: type}` (18 vertices) |
| `state.colors` (seating order) | `players[i]`, colour names `red/blue/orange/white` |
| `P{i}_{RES}_IN_HAND` | `resources` |
| `P{i}_{DEV}_IN_HAND` | 3.2.1: `dev_cards` (all playable, see limitation 4), `dev_cards_new = 0`.  3.3: a type with `P{i}_{DEV}_OWNED_AT_START` set (playable) goes to `dev_cards`, a type bought this turn to `dev_cards_new`; VP cards always to `dev_cards` |
| `P{i}_PLAYED_KNIGHT` | `played_knights` |
| `buildings_by_color[c][SETTLEMENT / CITY / ROAD]` | `settlements / cities / roads` |
| `P{i}_HAS_ROAD` + `LONGEST_ROAD_LENGTH`, `P{i}_HAS_ARMY` | `longest_road_owner / len`, `largest_army_owner` |
| `resource_freqdeck` | `bank` |
| `development_listdeck` | `dev_deck` counts by type |
| `current_turn_index`, `num_turns` | `current`, `turn` (`max_turns` = catanatron's 1000-turn cap) |
| `P{cur}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN` | `dev_played_this_turn` |
| `is_road_building` / `free_roads_available` | `free_roads` |
| prompt `BUILD_INITIAL_SETTLEMENT` / `BUILD_INITIAL_ROAD` | `PHASE_SETUP_SETTLEMENT` / `PHASE_SETUP_ROAD` (+ `setup_round`, `setup_last_settlement`) |
| prompt `PLAY_TURN`, not rolled / rolled | `PHASE_ROLL` / `PHASE_MAIN` (`dice` from the turn's logged `ROLL`) |
| prompt `DISCARD` | `PHASE_DISCARD`, `discard_queue` = current discarder + later seats holding > 7 cards (3.2.1, catanatron's hard-coded rule for the later discarders, limitation 12) / + later seats with `discard_counts > 0` (3.3, which also prompts each seat once per card: mid-way the hand is already reduced) |
| prompt `MOVE_ROBBER` | `PHASE_ROBBER` |
| prompt `DECIDE_TRADE` (3.3 domestic trades) | `PHASE_TRADE_RESPONSE`: `pending_trade` = `current_trade` (proposer = the turn player), `trade_responder` = the asked seat; seats asked before it (catanatron asks in seat order) answered accepted iff their `acceptees` flag is set; later seats that cannot pay are marked rejected, as catanbot's engine does |
| prompt `DECIDE_ACCEPTEES` | `PHASE_TRADE_SELECT` for the offerer, `responses` = the `acceptees` flags |
| `ACTUAL_VICTORY_POINTS >= vps_to_win` | `winner`, `PHASE_GAME_OVER` |

`total_vp(i)` of the converted state equals catanatron's
`ACTUAL_VICTORY_POINTS` and `public_vp(i)` its `VICTORY_POINTS` at every
tick of a game (tested).  The Longest Road holder and length are *copied*
from catanatron, not recomputed, because the two engines count roads
differently (limitation 11).  **All hands are exact** in the default
information mode (`--info full`): catanatron hands every `Player` the complete
`game.state`, so every seat is converted with `hand_known=True` /
`dev_known=True` - our "me" and the three opponents alike (catanbot's own
self-play engine is perfect-information too, so the search takes the same code
path).  `--info counted` redacts the opponents' cards to what a Colonist.io
player knows and decides on determinizations of a card-counting posterior
instead ("Information modes" below).  With trades suppressed (3.2.1, and `--trades off` on
3.3) `trades_this_turn` is set to the per-turn maximum so
`engine.legal_actions` never proposes player trades and the search runs with
`SearchConfig.trade_proposals = 0`; otherwise it is the number of
`OFFER_TRADE`s logged this turn.

### Actions (`catanbot_action_to_key` <-> `playable_key`)

| catanbot | catanatron `Action.value` |
|---|---|
| `(SETUP_SETTLEMENT / BUILD_SETTLEMENT, v)` | `BUILD_SETTLEMENT`, node id |
| `(SETUP_ROAD / BUILD_ROAD, e)` | `BUILD_ROAD`, `(node_a, node_b)` with `a < b` |
| `(BUILD_CITY, v)` | `BUILD_CITY`, node id |
| `(ROLL,)` | `ROLL`, `None` (the log stores the two dice) |
| `(MOVE_ROBBER, hex, victim)` | `MOVE_ROBBER`, `(cube_coordinate, Color or None, None)` on 3.2.1 (the log fills the stolen card), `(cube_coordinate, Color or None)` on 3.3 (the stolen card is the `ActionRecord.result`) |
| `(PLAY_KNIGHT, hex, victim)` | `PLAY_KNIGHT_CARD`, `None`; the `(hex, victim)` is remembered and answered at the following `MOVE_ROBBER` prompt |
| `(BUY_DEV,)` | `BUY_DEVELOPMENT_CARD`, `None` (log: the card) |
| `(PLAY_ROAD_BUILDING,)` | `PLAY_ROAD_BUILDING`, `None` |
| `(PLAY_YEAR_OF_PLENTY, r1, r2)` | `PLAY_YEAR_OF_PLENTY`, `("WOOD", "ORE")`-style pair (single-card picks exist only in catanatron) |
| `(PLAY_MONOPOLY, r)` | `PLAY_MONOPOLY`, resource string |
| `(BANK_TRADE, give, get)` | `MARITIME_TRADE`, 5-tuple: `ratio` copies of the given resource, `None` padding to four, then the asked resource, e.g. `("WHEAT","WHEAT","WHEAT",None,"BRICK")` |
| `(DISCARD, counts)` | 3.2.1: `DISCARD`, `None` (catanatron discards randomly; log: list of cards).  3.3: `DISCARD_RESOURCE`, the plan's first card; `CatanbotPlayer` queues the other cards for the engine's following one-card prompts, and merges a logged run of one player's `DISCARD_RESOURCE`s back into one `(DISCARD, counts)` observation |
| `(END_TURN,)` | `END_TURN`, `None` |
| `(PROPOSE_TRADE, give, get)` | 3.3: `OFFER_TRADE`, 10-tuple `give + get` (same resource order); never in `playable_actions`, built by `CatanbotPlayer` |
| `(ACCEPT_TRADE,)` / `(REJECT_TRADE,)` / `(CANCEL_TRADE,)` | 3.3: the same-named types (`ACCEPT` / `REJECT` carry `current_trade`, identified by type alone) |
| `(EXECUTE_TRADE, j)` | 3.3: `CONFIRM_TRADE`, `give + get + (Color of seat j,)` |
| forced `(ROLL, v)`; the player-trade actions on 3.2.1 | no equivalent |

`test_action_round_trips_for_every_action_type` checks, over whole random
games, that every catanatron playable action converts to a catanbot action
that converts back to the same key, that catanbot's `legal_actions` on the
converted state contains it (except the discards and, on 3.3, the non-knight
dev cards offered before the roll, limitation 14), and that every mappable
catanbot action catanatron does *not* offer is one of the two documented
rule differences below.  The 3.3 domestic-trade actions never appear in a
game between stock players (none of them offers);
`test_trade_actions_convert_both_ways` round-trips them through a hand-built
offer, including the two logged answers without a catanbot equivalent (the
offerer's answer to its own offer and the forced rejection of a seat that
cannot pay).

### `CatanbotPlayer.decide`

0. 3.3 only: a `DISCARD` prompt runs the bot once, on the full hand, over
   catanbot's `(DISCARD, counts)` options whose size is the engine's
   `discard_counts`; the first card is played and the rest are queued for the
   following one-card prompts (`stats["pending_discard"]`; if the queue no
   longer matches the hand the bot plans again).  A `DECIDE_TRADE` /
   `DECIDE_ACCEPTEES` prompt is answered with `REJECT_TRADE` /
   `CANCEL_TRADE` when trades are suppressed (the default); otherwise the
   bot decides it in `PHASE_TRADE_RESPONSE` / `PHASE_TRADE_SELECT` like any
   other decision, except the engine's question about our *own* offer,
   which is rejected without a search (`stats["trade_prompts"]`,
   `"offers_received"`, `"self_offer_prompts"`, ...).  With trades on, at
   our post-roll `PLAY_TURN` the catanbot
   `PROPOSE_TRADE` candidates are added as `OFFER_TRADE` actions (cards we
   hold, fewer than 4 offers so far this turn), also when the only playable
   catanatron action is `END_TURN`; a proposal the search ranks from outside
   the engine's candidate list (intermediary / political deals) is played
   too if well formed.
1. One playable action (roll-only turns, the 3.2.1 `DISCARD None` prompt): return it.
2. `MOVE_ROBBER` prompt right after our `PLAY_KNIGHT_CARD`: execute the
   remembered `(hex, victim)`; if catanatron no longer offers it, fall through.
3. Convert the state, take `engine.legal_actions`, keep the actions that map
   onto a playable catanatron action.  If they all map to the same catanatron
   action (e.g. only `ROLL`, or `MOVE_ROBBER` with one option) there is nothing
   to search.
4. Otherwise `bot.decide` (the `SearchBot` from the spec) runs on the converted
   state; its ranked `last_results` are walked and the best action with a
   catanatron equivalent is played.  Should nothing rank, the bot's own
   decision, then the highest `heuristic.action_priors` mappable action, then
   `fallback_action` (city > settlement > dev card > road > roll > robber >
   end turn) are used.  With `strict=False` (the default) any adapter
   exception also ends in `fallback_action`, so a benchmark never crashes;
   the tests run with `strict=True`.
5. Between decisions every action catanatron logged (`game.state.actions`
   on 3.2.1, `game.state.action_records` on 3.3; fully specified: dice,
   stolen card, drawn dev card, discarded cards) is replayed on a private
   *shadow* copy of catanatron's state (on 3.3 together with its
   `ActionRecord`, because `apply_action` would otherwise roll fresh dice /
   steal a fresh card from the `random.Random` the state shares with its
   copies), converting the state before each action and calling
   `bot.observe(state, action, seat)` - the same hook the self-play runner
   uses - so the opponent model and the political tracker see the whole
   game.  `PLAY_KNIGHT_CARD` + `MOVE_ROBBER` are merged into one
   `PLAY_KNIGHT` observation, delivered with the state the *card* was played
   in (`PHASE_ROLL` / `PHASE_MAIN`, knight still in hand) rather than the
   `PHASE_ROBBER` state after it, so that `SearchBot.observe` predicts the
   decision from the same legal-action set the observed
   `(PLAY_KNIGHT, hex, victim)` came from (tested).  Likewise a 3.3 run of
   one player's `DISCARD_RESOURCE` actions is one `(DISCARD, counts)`
   observation with the state before the first card, where it is a legal
   half-hand discard (tested on both versions).

## What Catanatron's bots are, and what they are not (read before quoting any result)

Catanatron is the standard open-source Catan benchmark; HexMachina (Belle et al.) also measured against its
AlphaBeta player, which is why we use it.  But its bots are far from complete players.  Every gap below was
checked in its source (GitHub checkout `ecf9311`, catanatron 3.3.0) and, where possible, measured in our logs.

| gap | where in Catanatron | what we see in our games |
|---|---|---|
| **Development cards are almost worthless to it.**  A card in hand scores 10 and a Knight played 10.1, against 100,000,000 for production. | `players/value.py`: `DEFAULT_WEIGHTS` (`hand_devs`, `army_size`) and `base_fn` | each AlphaBeta buys about 0.15 cards a game (our bot 6.2); when it does buy, it mostly holds them (T2 game 62: five of six cards never played) |
| **Victory Point cards are invisible to it.**  Its score counts only public points; a VP card raises only the hidden total, even for its own cards, and a won game gets no extra credit in its search. | `players/value.py` (`VICTORY_POINTS`); `state_functions.py:267`; `players/minimax.py`, `alphabeta` | it never suspects hidden points: in 168 of our 218 T2 wins, hidden VP cards were part of the winning total |
| **It models one enemy.**  Its production term counts only the next player in turn order; the other two opponents' production is ignored. | `players/value.py`, `base_fn` (the `P1` features; `features.py`, `iter_players`); the `ValueFunctionPlayer` docstring: "only considers 1 enemy player" | its robber and blocking choices focus on one neighbour, not on the leader |
| **It looks two moves ahead.** | `players/minimax.py:13`, `ALPHABETA_DEFAULT_DEPTH = 2` (20 s deadline, never reached in our runs) | no plans that pay off later: Largest Army, Monopoly, holding for a city |
| **No trading between players.**  Its bots never offer.  `ValueFunctionPlayer` rejects every offer; AlphaBeta, SameTurnAlphaBeta and MCTS crash on one. | `tree_search_utils.execute_spectrum` has no trade case (section "How catanatron's players answer an offer" above) | the strength proof and the 1v1 benchmark were played with trading off |
| **No table politics.**  No leader targeting and no coalitions; in its search every other player plays against it. | `players/minimax.py` (every other seat minimises) | it neither gangs up on a runaway leader nor spares a weak player |
| **Hand-set weights with no documented rationale.**  The alternative "contender" weights, which look machine-tuned, barely change the development-card weights (10.7 and 12.9). | `players/value.py`, `CONTENDER_WEIGHTS` | - |
| **It sees everything,** in every test we ran: all hands, all development cards. | Catanatron exposes the full state to its players | an advantage *to them*; in our Colonist-information tests our bot sees only public information and still wins |

**What this means for our numbers**
- **The results show that our bot beats an open-source baseline with these gaps.**  They are not evidence of
  strength against good human players.
- **Part of the edge exploits the gaps**, above all development cards, hidden Victory Point cards and an
  uncontested Largest Army.  Two proposed checks would measure how much (docs/SCRUTINY.md Q22): our bot with
  development-card purchases off, and an AlphaBeta patched to count its own hidden points.  They are not run
  yet.
- **Human testing through the advisor is the real test.**  The strength proof only claims readiness for it
  (docs/PROOF.md, claim 2).

## Limitations and rule differences

1. **Player trading is off by default and one-sided when on.**  catanatron
   3.2.1 has no domestic trades.  On 3.3 the default `--trades off` keeps
   `PROPOSE_TRADE` out of the search (`trades_this_turn` = max,
   `trade_proposals = 0`; the trade-related parts of the opponent model keep
   their priors, and feature `trades_this_turn` reads 4/4 for a value net).
   With `--trades native|value|fair` catanbot offers, but catanatron's
   players never offer, so only catanbot's proposal side and its choice of
   partner are exercised in the bench (its response side is covered by
   `test_incoming_offers_are_decided_by_the_bot`, two catanbot seats), and
   the answers come from catanatron's degenerate responders or from OUR
   response rule - see "Domestic trading against catanatron 3.3".
2. **Discards are random on 3.2.1.**  Its only `DISCARD` action has value
   `None`; the engine samples the cards.  Our 7-protection (dumping surplus
   before the roll) still applies, our discard *choice* does not.  On 3.3
   the bot chooses its discard (one search on the full hand, handed over
   card by card).
3. **Road Building needs wood + brick in catanatron** (`road_building_possibilities`
   checks the road cost even for the free roads), and during the free roads
   only `BUILD_ROAD` is offered (no `END_TURN`).  catanbot may rank
   `PLAY_ROAD_BUILDING` first without the resources; it is then skipped for
   the next-ranked action (about 0.4 % of searched decisions; counted as
   `unmapped_top`, reported per kind by the script).
4. **Dev cards are playable the turn they are bought** in catanatron 3.2.1
   (official rules and catanbot: next turn).  All held cards go into
   `dev_cards`; the search still assumes a card bought *during* the search is
   playable next turn only, which only makes it slightly pessimistic about
   `BUY_DEV`.  3.3 follows the official rule (`{DEV}_OWNED_AT_START`) and the
   adapter converts it exactly (`dev_cards` / `dev_cards_new`).
5. **Knight before rolling** is two catanatron decisions; the robber target is
   carried over.  If the search is re-run at the `MOVE_ROBBER` prompt before
   the roll, the converted state is `PHASE_ROBBER` with `dice = 0` and the
   simulated continuation skips the roll.
6. **Perfect information.**  Opponents' hands are exact (they are in
   catanatron), so robber-victim and monopoly decisions are easier than in
   real play where catanbot works from hand sizes and card counting.
7. **Weak stock opponents** (see above): win rates saturate near 100 % against
   `random` / `weighted` / `vp` (and against 3.3's `mcts` / `playouts` at their
   default parameters).  The stand-ins `vf` / `ab` and catanatron 3.3's
   `value` / `alphabeta` / `sameturn` are real opponents (ladder results above).
8. **Reproducibility.**  Seeds fix boards, decks, dice and our bot's RNG, but
   catanatron builds some action lists from `set`s of `Color` enums whose
   order depends on Python's per-process hash seed, so game trajectories are
   only reproducible within one process or with `PYTHONHASHSEED` fixed -
   which `scripts/bench_catanatron.py` now does by default (`--hash-seed 0`,
   it re-executes itself; results before that change were not pinned).
   catanatron's `AlphaBetaPlayer` also stops a search after 20 s of wall
   time (`MAX_SEARCH_TIME_SECS`), so a heavily loaded machine could in
   principle change its moves.
   The tests therefore loop over seeds rather than pin one trajectory
   (confirmed: the same 30 `WeightedRandomPlayer` seeds gave 18 701, 17 042
   and 18 646 ticks in three separate processes).
9. Actions logged before our first decision (setup placements of earlier
   seats) are not replayed through `observe`; single-card Year of Plenty logs
   are skipped (no catanbot equivalent).
10. `CatanbotPlayer.reset_state()` is not called by catanatron's `Game`
    itself; `play_game` calls it, and a new `game.id` also resets the bot.
11. **Longest Road: a road ending at an opponent's building does not count
    in catanatron.**  Its `Board.longest_acyclic_path` refuses to step onto
    any enemy-owned node, so the last road of a path that *ends* at an
    opponent's settlement / city is never counted (and `Board.build_road`
    never adds that node to the road's component, although
    `buildable_edges` does allow building the road into it).  catanbot's
    `engine.longest_road_length`, an independent brute-force longest-trail
    search and the official rules count that road: the opponent's building
    only stops the path from continuing *through* the vertex.  Over 30
    `WeightedRandomPlayer` games catanatron's `P{i}_LONGEST_ROAD_LENGTH` was
    exactly one below catanbot's on 363 of 10 192 ticks on which the award
    was held (every mismatch was catanatron = catanbot - 1; the brute force
    agreed with catanbot on every tick), and on 37 ticks in 1 of the 30 games
    the unique holder under catanbot's counting was a different seat from
    catanatron's `HAS_ROAD` holder.  This is an engine rule difference, not an
    adapter bug: the adapter copies catanatron's holder and length, so
    `total_vp` / `public_vp` always equal `ACTUAL_VICTORY_POINTS` /
    `VICTORY_POINTS`.  The consequence for play is that the search, whose
    simulated road building uses catanbot's counting, can expect a Longest
    Road award (or a takeover) that catanatron will not grant when the
    decisive road ends at an opponent's building.
    `test_road_ending_at_enemy_settlement_counts_in_catanbot_but_not_catanatron`
    reproduces this on a hand-built board (catanbot's engine books +2 VP for
    the fifth road, catanatron scores the path 4 and awards nothing) and
    `test_longest_road_lengths_never_differ_by_more_than_one_through_games`
    checks the `cat in (cb, cb - 1)` invariant through whole games.
12. **`discard_limit` only governs the first discarder (3.2.1).**  On a 7
    catanatron 3.2.1 picks the first player to discard with
    `state.discard_limit` (default 7) but advances to the *later* discarders
    with a hard-coded `> 7` (`apply_action`'s `DISCARD` branch).
    `state_to_catanbot` builds the `discard_queue` with the same `> 7` rule
    for the later seats so it lists exactly the seats catanatron goes on to
    prompt; with the default limit the two rules coincide (every benchmark
    game), with a non-default `--discard-limit` they do not (e.g. limit 9,
    hands `[10, 8, 6, 9]`: catanatron prompts seats 0, 1 and 3 although only
    seats 0 and 3 exceed 9; limit 5, hands `[6, 6, 8, 3]`: only seats 0 and
    2 are prompted).  `test_discard_queue_mirrors_catanatron_hard_coded_limit`
    replays these.  3.3 applies the limit to everyone and fixes the counts at
    the roll (`State.discard_counts`), which the queue follows
    (`test_discard_queue_follows_discard_counts`).
13. **3.3 domestic trades are wired, but off by default.**  With `--trades
    off` (and `CatanbotPlayer(suppress_trades=True)`) our player never
    offers, answers a `DECIDE_TRADE` prompt with `REJECT_TRADE` and a
    `DECIDE_ACCEPTEES` prompt with `CANCEL_TRADE`
    (`test_trade_prompts_are_declined`).  With trades on it plays catanbot's
    trade decisions through catanatron's protocol ("Domestic trading
    against catanatron 3.3").  catanatron itself has no per-turn offer limit
    and does not check that the offerer holds the offered cards; the adapter
    enforces catanbot's rules (at most 4 offers per turn, only held cards).
14. **3.3 offers Year of Plenty, Monopoly and Road Building before the
    roll** (3.2.1 and catanbot: only the knight).  Those pre-roll options
    have no legal catanbot counterpart in `PHASE_ROLL`, so they are never
    chosen; the same card is played after the roll instead.  An opponent's
    logged pre-roll play is observed with the `PHASE_ROLL` state (not a
    legal action there; `SearchBot.observe` makes no prediction for those
    kinds, so nothing breaks).  When Road Building was played before the
    roll, 3.3 prompts the two free roads first: `state_to_catanbot` converts
    that (`is_road_building` while not rolled) to a `PHASE_MAIN` state with
    `free_roads` set and `dice = 0`, where the roads are legal.
15. **`GreedyPlayoutsPlayer` runs its playouts in-process.**  catanatron
    3.3's `playouts` preset opens a `multiprocessing.Pool(cpu_count())` per
    decision; the bench script sets its `USE_MULTIPROCESSING` flag off when
    it resolves the preset (the seeded playouts give the same result), so
    `--workers` stays the only parallelism and the preset also works inside
    the worker processes (which may not spawn children).

## Running the full ladder (stand-ins on 3.2.1, strong Catanatron players on 3.3.0)

The PyPI release of Catanatron (3.2.1) ships only the weak stock bots.  The
strong players live in the GitHub checkout (3.3.0 engine); install it
editable into a virtualenv of its own, so the wheel stays available as well
(the paths below are the ones used on this machine; any virtualenv with the
GitHub checkout works, the script only needs to be run by that
interpreter):

```bash
git clone https://github.com/bcollazo/catanatron.git        # here: /home/user/bcollazo/catanatron
python3 -m venv /home/user/venv_cat33 && /home/user/venv_cat33/bin/python -m pip install -e catanatron numpy pillow pytest
cd /home/user/ClaudeTesting2
PYTHONPATH=/home/user/ClaudeTesting2 /home/user/venv_cat33/bin/python scripts/bench_catanatron.py --list-opponents   # all ten presets "available"
PYTHONPATH=/home/user/ClaudeTesting2 python3 scripts/bench_catanatron.py --list-opponents                          # 3.2.1: value/alphabeta/... "not available"
```

The same script and adapter run on both versions (`--list-opponents` says
which presets resolve on the interpreter in use; a preset that is not
available is a one-line error with exit code 2 when asked for alone, and
skipped inside a list or ladder).  Rules that keep a run safe on this
machine (4 shared cores, 15 GB): always run from the repository root with
`PYTHONPATH` set, keep to `--workers 2`, run one benchmark at a time, wrap
every command in `timeout 1500` (25 minutes) and pass `--verbose` (the JSON
is written only when a batch completes, so the per-game log is the fallback
if a batch is killed).  `--spec` defaults to
`search:depth=1,evaluator=heuristic`.  The 2026-09-25 results section above
was produced with exactly the commands below.

Stand-in ladder on catanatron 3.2.1 (system `python3`; our `vf` / `ab`
players from `docs/BENCHMARKS_OPPONENTS.md` plus the stock controls):

```bash
cd /home/user/ClaudeTesting2
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 python3 scripts/bench_catanatron.py \
    --opponent random,weighted,vp,vf --games 200 --workers 2 --seed 0 --verbose --json ladder_321_fast.json   # 7.6 min under load
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 python3 scripts/bench_catanatron.py \
    --opponent ab --games 160 --workers 2 --seed 0 --verbose --json ladder_321_ab.json                        # 19 min under load, ~9 min idle
# shorter presets: --ladder controls (random, weighted, vp) and --ladder standins (vf, ab) with one --games for all,
# e.g. --ladder standins --games 100 (~1.5 min for vf, ~6-12 min for ab)
```

Strong ladder on catanatron 3.3.0 (the virtualenv; catanatron's own
players), one command per opponent because their costs differ by three
orders of magnitude:

```bash
cd /home/user/ClaudeTesting2
PY=/home/user/venv_cat33/bin/python      # a virtualenv with the GitHub checkout installed editable
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent vp        --games 100 --workers 2 --seed 0 --verbose --json ladder_330_vp.json         # ~1 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent value     --games 200 --workers 2 --seed 0 --verbose --json ladder_330_value.json      # ~3 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent alphabeta --games 30  --workers 2 --seed 0 --verbose --json ladder_330_alphabeta.json  # ~9 min (40 on an idle machine)
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent sameturn  --games 60  --workers 2 --seed 0 --verbose --json ladder_330_sameturn.json   # ~12 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent mcts      --games 40  --workers 2 --seed 0 --verbose --json ladder_330_mcts.json       # ~12 min
PYTHONPATH=/home/user/ClaudeTesting2 timeout 1500 $PY scripts/bench_catanatron.py --opponent playouts  --games 2   --workers 2 --seed 0 --verbose --json ladder_330_playouts.json   # ~18 min for 2 games
# --ladder strong runs value, alphabeta, sameturn, playouts, mcts with one --games for all, so with playouts in it
# only --games 2 fits the time limit; prefer the per-opponent commands (or a comma list without playouts).
```

Per-game cost of the opponents with the depth-1 heuristic bot on this
machine, `--workers 2` while another workflow loaded the other cores (the
idle one-worker cost in brackets): 3.2.1 stock controls 0.9-1.2 s (0.4-0.5
s), `vf` 1.5 s (1.3 s), `ab` 14 s (6.5-7 s); 3.3.0 `vp` 1.0 s (0.7 s),
`value` 1.9 s (1.1 s), `alphabeta` 35 s, longest game 70 s (19 s),
`sameturn` 23 s, 7-39 s per game (17-22 s), `mcts` 35 s, 14-119 s per game
(20-48 s), `playouts` 813-1088 s
= 14-18 minutes per game (696 s idle; 25 random playouts per playable
action, about 1 s per opponent decision).  The game counts in the commands
above are the ones that fit 25 minutes with margin on the loaded machine.
`playouts` is the one preset that is too slow for a real ladder with its
default `num_playouts`, and `mcts` with catanatron's default of 10
simulations is too weak to be informative; `--opponent-params` now passes
their parameters (e.g. `--opponent mcts --opponent-params
num_simulations=50`, `--opponent playouts --opponent-params num_playouts=5`;
per-decision costs in "Compute per decision").  The playouts module's
per-decision `print` is silenced when the preset is resolved.

Presets: `random`, `weighted`, `vp` (stock controls), `vf`, `ab` (our
stand-in value-function / alpha-beta players built inside the Catanatron
engine, `docs/BENCHMARKS_OPPONENTS.md`), `value`, `alphabeta`, `sameturn`,
`playouts`, `mcts` (Catanatron's own strong players).  Any other opponent
can be given as an import path, e.g. `--opponent mypkg.bots:MyPlayer`, and
several as a comma list.  In 4-player games the seat baseline is 25 %; use
at least 100 games per opponent for a win rate of 40 %+ to be statistically
clear (a 95 % interval spans about +-7 points at 200 games, +-12 at 60 and
+-17 at 30, see the results section).

Smoke results of the dual-version work (2026-09-25, `--games 2 --workers 1
--seed 1`, depth-1 heuristic bot; only a check that both paths run cleanly):
3.2.1 vs `random` 2/2 wins, 0.40 s/game, 0 adapter errors / fallbacks /
observe errors; 3.3.0 vs `value` 0/2 wins (4.5 vs 6.8 VP), 1.1 s/game, 0
errors, 3 planned discard cards handed over card by card.  The full ladders
on both versions, run the same day, are in the results section above.

## Information modes

`--info full` (the default, `CatanbotPlayer(info="full")`) hands the bot
catanatron's true state: every hand and every development card is known, as
described in "State" above.  This is *more* than a human sees at a Colonist
table, so a win rate measured this way is an upper bound for play there.
The default is unchanged by the information-mode work: the same seeded games
replay action for action (identical logs and final-state fingerprints) with
the adapter from before the change, and `tests/test_public_info.py` pins the
decision procedure on 10 positions.

`--info counted` (`CatanbotPlayer(info="counted", info_samples=4,
discards_public=False)`) gives the bot exactly what a Colonist.io player
knows (`catanbot/bench/public_info.py`):

* **always known**: board, robber, buildings, roads, the bank per resource,
  the deck size, every hand *size* and development-card *count*, every
  played development card, awards, public VP; our own cards exactly;
* **public events with their content**: dice and each player's production,
  build costs, bank / port trades, domestic trades, Monopoly (the amount each
  victim lost is its hand-size drop), Year of Plenty, Road Building, card
  purchases (count and cost);
* **hidden**: the card of a steal between two opponents, the cards of a
  discard on a 7 (the count is public; `--discards-public` shows the cards),
  the type of every development card an opponent drew and still holds (only
  the public pool is known: 25 cards minus every played card minus ours).
  The discards of one 7 are *simultaneous* as on Colonist: the bank shows
  only their total once the last discarder is done, so a lone hidden
  discarder is still pinned down by the bank and several keep only their
  split open.

A `PublicInfoTracker` follows catanatron's action log on its own shadow game
(from the initial position, so a player that joins late still counts only
public events) and feeds `catanbot.counting.CardCounter`: an exact weighted
mixture of joint hand hypotheses (one hypothesis while nothing hidden
happened; a hidden steal branches over the victim's cards with their
probabilities; spends, monopolies and the bank prune what became
impossible).  A chance result (stolen card, drawn card, discarded cards) is
read only when the model allows it.  The bot then sees a public view (every
opponent `hand_known=False` / `dev_known=False` with exact sizes and counts,
cards zeroed; observations and explanations use it too, and opponents'
hidden discards are not observed) and every searched decision averages its
ranking over `K` determinizations sampled from the tracker (joint hands from
the mixture, development cards from the public pool, re-dealt while an
opponent would already hold enough VP cards to have won).  Our legal actions
depend only on public facts (own cards, board, bank, opponents' hand sizes
for robber victims) and are identical in every sample.

Measured (2026-09-25; replays of 20 games of four `WeightedRandomPlayer`s per
engine, following seat `seed % 4`, discards hidden): the true hands were in
the support of the counter at 100 % of 11,103 (3.2.1) / 12,428 (3.3) log
steps, with mean probability 0.80 / 0.76; each opponent's hand was known
exactly at 73 % of steps (all three at 63 % / 61 %); at most 44 / 23
hypotheses, never a pruned hypothesis or a reset, with 11-14 hidden steals,
29-33 hidden discard cards and 12-14 hidden card draws per game.  With every
hidden event revealed (a test switch) the counter equals the true hands after
every entry of 5 games per engine.  Changing the opponents' true cards in a
way the model cannot see (a card swapped between two opponents, a held card
exchanged with the deck) never changes a counted-mode decision or its
ranking on 20 positions per engine, while a full-information search's
values move at 18 (3.2.1) and 20 (3.3) of them.

Cost: `K` searches per decision.  On the same positions with the default
spec (`search:depth=1,evaluator=heuristic`), K = 4: 3.3 vs `value` 94 ms vs
22 ms per searched decision in full mode (4.2x), 3.2.1 vs `vf` 37 ms vs 11 ms
(3.4x); the two modes chose the same action at 35/35 and 39/40 of those
positions.

Approximations: an unseen discard is modelled as a uniformly random subset of
the hand (exact on 3.2.1, whose engine discards at random; on 3.3 players
choose); actions condition the belief through feasibility only (a city
proves the cards were held; not building proves nothing); hidden
development cards are uniform over the public pool apart from the "has not
won yet" condition; catanatron's discards are sequential while the counted
mode treats one 7's discards as simultaneous; the opponent model's
predictions of opponents' moves (used for trade acceptance only) are made
without their hands.  3.2.1 logs every chance outcome in the action values,
3.3 in `ActionRecord.result`: nothing the model allows is missing on either.
`scripts/ablate_catanatron.py --info counted [--info-samples K]
[--discards-public]` (or `--cand-adapter-opt info=counted` for one arm)
passes the mode to the arms.

## Mixed opponents

`--mixed-opponents value,alphabeta,sameturn` (any three presets) seats
catanbot in seat `g % 4` exactly as the 1v3 rotation and one copy of each
preset in the other seats, ordered by permutation `(g // 4) % 6` of the list
over the relative positions 1-3 after catanbot in turn order.  `g mod 24`
fixes both, so any 24 consecutive games put every preset in every relative
position 8 times (and every player in every seat 6 times).  Each game records
its `lineup`, `relative` order and `winner_name` (`catanbot` or the preset);
the summary (`format: "1v3-mixed"`, `opponents`, `mixed`: wins, win rate and
average VP per player, games per preset and relative position) and the
compute table report every preset separately.  It composes with `--info`,
`--game-range` chunks (the lineup depends on the game index only),
`--hash-seed` and `--log-actions` (records carry the lineup; `--check`
replays them); `--our-seats 2` is not supported.  Smoke (2026-09-25, 3.3.0,
default spec, games 0-2, one worker): 0 adapter errors; catanbot 1/3,
`value` 1/3, `alphabeta` 1/3, `sameturn` 0/3; average VP 8.0 / 7.0 / 7.7 / 6.3, 20 s per game.
