# catanbot

*Last updated 2026-09-26 (UTC).  Every result below names its date, our code commit and the Catanatron
version; the History section at the end keeps the trail.*

An ML-trained Settlers of Catan bot for 3-4 player games that

* **plays**: expectimax search over dice / draws / steals / trade acceptance
  with a self-play-trained value network at the leaves, max^n opponent
  simulation and hidden-hand determinization;
* **thinks like a good player**: explicit logic for bank vs port vs player
  trading, 7-protection, knight timing, dev-card buying, card counting,
  placement, and
* **plays politics**: exploitative opponent profiles (implied valuations,
  acceptance habits, robber targets) with decay, political capital between
  players, target pressure on the visible leader, game-stage-aware trading and
  kingmaker-lite moves - the objective is win probability, not VP;
* **reads Colonist.io screenshots**: a computer-vision parser (and an optional
  Claude-vision parser) turns a screenshot into a game state and the bot
  recommends the move with explanations.

```bash
pip install -e . opencv-python-headless
python -m catanbot analyze screenshot.png --me red          # what should I do?
python -m catanbot play --bots search,heuristic,heuristic,heuristic --verbose
python -m catanbot train --iters 6 --games 150 --workers 4  # self-play training
python -m pytest -q                                          # tests
```

See `docs/USAGE.md` for every command and the `--fix` / `--event` syntax,
`docs/STRATEGY.md` for how the bot decides, `docs/DESIGN.md` for the module
contracts.

## Layout

| module | what |
| --- | --- |
| `catanbot/board.py`, `state.py`, `actions.py`, `engine.py` | complete base-game rules (setup, production, robber, dev cards, awards, bank/port/player trades) |
| `heuristic.py`, `placement.py`, `trading.py`, `discard.py`, `robber.py`, `danger.py`, `devcards.py`, `counting.py`, `inference.py` | strategy modules (incl. distance-to-win targeting and steal exposure) and card counting / determinization |
| `opponent_model.py`, `politics.py` | exploitative opponent profiles, political capital, coalitions |
| `features.py`, `model.py`, `selfplay.py`, `train.py` | 302-dim features, numpy value net, self-play data + training loop |
| `search.py`, `agents/` | expectimax + beam search, the bots |
| `vision/` | Colonist.io parsers (`colonist.py` CV, `llm.py` Claude), synthetic renderer, digit classifier, state schema |
| `cli.py` | `analyze`, `recommend`, `play`, `eval`, `train`, `render`, `profiles` |

## Screenshot parsing

`python -m catanbot analyze shot.png --me red --debug overlay.png`

![debug overlay](docs/example_parse_debug.png)

On synthetic Colonist-style renders (four resolutions, 3-4 players, jitter
and JPEG compression) the parser reads the board geometry, all 19 tiles,
all 18 numbers, the robber, every road and building, the panel numbers,
the hand and the dice correctly; ports hidden under the hand bar are
reported as missing and can be added with `--fix "port 66=3:1"`.  Real
screenshots use the same pipeline (the token discs anchor the lattice, the
standard tile / number multisets constrain the classification); anything
uncertain is flagged and correctable with `--fix`.

## Strength against Catanatron, and what it does not show

*As of 2026-09-26.  Our code: commits `9984181` and `9599eed` (strength proof), `35ef224` (1v1).  Opponents:
Catanatron 3.3.0 at GitHub commit `ecf9311` (committed 2026-09-08; still the latest upstream on 2026-09-26)
and 3.2.1 from PyPI.  A newer bot or a newer Catanatron can change these numbers: check the History section.*

Pre-registered results, with every game archived and replayable:
- 4-player, our bot against three Catanatron bots (docs/PROOF.md): 62.8 % against ValueFunction and 54.5 %
  against AlphaBeta, where chance is 25 %; 58.5 % against AlphaBeta when our bot sees only what a Colonist
  player sees.
- 1v1 against AlphaBeta: 75.5 % (docs/BENCH_1V1.md).

**Read this first.**  Catanatron is the standard open-source benchmark, but its bots are far from complete
players:
- development cards are nearly worthless to them, and Victory Point cards invisible;
- they model one enemy and look two moves ahead;
- they never trade with other players and play no politics.
Part of our edge exploits exactly these gaps, so these results say nothing yet about strength against good
humans.  Human testing with the advisor is the real test.  Details, with the lines in Catanatron's source:
docs/BENCHMARKS.md, "What Catanatron's bots are, and what they are not".

## Training and strength

`python -m catanbot train` runs the self-play loop (see `docs/USAGE.md`).
Results of the training run shipped in `models/value_net.npz` are in
`docs/RESULTS.md`.
Champion league and promotion gate for every new training run / strategy (exact old bots, sequential tests): `docs/LEAGUE.md`.

## History

A dated trail of results and of what they were measured against.  When a result is re-measured (a newer
version of our bot, a newer Catanatron), add a row; never overwrite an old one.

| date (UTC) | what | our code | opponents | where |
|---|---|---|---|---|
| 2026-09-25 | Catanatron ladders: stand-ins and strong players | see the doc | Catanatron 3.2.1 (PyPI), 3.3.0 (`ecf9311`) | docs/BENCHMARKS.md |
| 2026-09-26 | Strength proof complete, claims 1-4 PASS: 62.8 % vs 3x ValueFunction, 54.5 % vs 3x AlphaBeta (chance 25 %), 58.5 % vs 3x AlphaBeta with Colonist information | `9984181`, `9599eed` | Catanatron 3.3.0 (`ecf9311`), 3.2.1 | docs/PROOF.md, proof/ |
| 2026-09-26 | 1v1 benchmark: 75.5 % vs AlphaBeta, 73.2 % vs ValueFunction | `35ef224` | Catanatron 3.3.0 (`ecf9311`) | docs/BENCH_1V1.md, bench_1v1/ |
| 2026-09-26 | Catanatron's gaps documented (development cards, hidden VP, one enemy, depth 2, no trading, no politics) | - | checked against Catanatron `ecf9311` | docs/BENCHMARKS.md |
| 2026-09-26 | Test queue started; first verdicts (player trading worth about 24 points against value-rule responders; wider offers shelved; acceptance calibration failed its behaviour check) | epochs A `3c7089c`, B1 `aefdd4a` | Catanatron 3.3.0 (`ecf9311`), self-play | docs/ABLATIONS.md, docs/queue/ |

