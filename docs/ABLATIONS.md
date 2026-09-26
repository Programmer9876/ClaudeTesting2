# Ablations: are the "alpha strategies" worth their weight (and their compute)?

Every strategy module in catanbot (`danger`, `robber`, `coalitions`, `politics`,
`opponent_model`, `trading`, `placement`, `devcards`, the static evaluator and the search
knobs) is a plausible idea whose *constants were set by hand*.  A strategy can be right in
principle and still lose games because its weight is off, or win games at a compute cost that
would buy more elsewhere.  The ablation harness answers both questions per constant with
paired self-play games:

* `catanbot/tuning.py` - the registry of tunables (`TUNABLES`), `apply` / `restore`, the
  paired-game runner and the statistics;
* `catanbot/agents/param_bot.py` - `ParamBot(inner, overrides)`: any bot with its constants
  overridden for the duration of its own decisions;
* `scripts/ablate.py` - the command line;
* `tests/test_ablate.py` - the tests.

## How to run

```bash
export PYTHONPATH=/path/to/ClaudeTesting2
python3 scripts/ablate.py --list                                   # the registry
python3 scripts/ablate.py --tunable danger.TURNS_HALF --values 2,4.5 \
    --games 40 --workers 2 --seed 1 --players 4 --json out.json    # one weight, two candidates
python3 scripts/ablate.py --tunable danger.danger_multiplier --flag-off --games 40 --players 3
python3 scripts/ablate.py --tunable search.beam --values 2,6 --games 20 --workers 2   # search knob
python3 scripts/ablate.py --tunable search.opp_roll_samples --values 1,6 --games 20      # opponent knob: depth-2 spec
python3 scripts/ablate.py --tunable heuristic.EXPOSURE_WEIGHT --values 0,0.5 --games 20 --workers 2
python3 scripts/ablate.py --tunable TURNS_HALF --plan              # plan only: spec, mode, seats
python3 scripts/ablate.py --sweep-all --games 4 --workers 2 --out docs/ABLATIONS.md   # every tunable
python3 scripts/ablate.py --sweep-all --kinds weight,flag --only danger.TURNS_HALF,coalitions.SCALE ...
```

* `--base-spec` is the bot both sides play.  Default `heuristic:temp=0.15` (a game takes about
  0.3 s) for cheap runs; tunables that only the search bot reads (the evaluator weight and the
  search knobs) default to `search:depth=1,beam=4,expand=8,evaluator=heuristic` (about 10 s a
  game with the C++ evaluator, 35 s with the Python one).  The four opponent knobs
  (`search.opp_roll_samples`, `opponent_actions`, `opponent_expand`, `opponent_proposals`) are only
  read by `Searcher._future_values`, which runs at `depth >= 2`: at depth 1 both sides would do
  identical work, so those default to `search:depth=2,beam=4,expand=8,evaluator=heuristic`
  (`depth` column of `--list`) and a shallower `--base-spec` is refused.  A tunable can be named by
  its registry name (`danger.TURNS_HALF`) or its bare attribute when unique (`TURNS_HALF`).
* `--values` lists the candidates (registry candidates when omitted); flags take `on`/`off` or
  `--flag-off`; `placement.RESOURCE_DEMAND` takes vectors `w/b/s/wh/o` separated by `;`.
* `--json` stores the full report: per candidate the summary row plus `records`, one entry per
  game (`seed`, `pattern`, `winner`, `winning_side`, the paired difference `diff` that game
  contributes, `vps`, `turns`, `evaluator_mode` and `pid` of the process that played it, and per
  seat the decision count, mean / p95 decision time and the override overhead), so every statistic
  can be recomputed by hand.  `--sweep-all` runs every tunable (or `--kinds` / `--only` subsets,
  `--max-candidates N` for the first N candidates) as subprocesses and writes one markdown table
  into the results block of `--out` (between the `ablate:results` markers).
* Evaluator mode.  `heuristic.static_value` has a bit-exact C++ port (`cpp/heuristic.cpp`) that
  reads **no Python constants**, so a tunable read by the evaluator (`heuristic.EXPOSURE_WEIGHT`,
  `placement.RESOURCE_DEMAND`, and - flagged since 2026-09-25, see "Adversarial verification" below -
  `placement.PLACEMENT_BLOCK_WEIGHT` / `PLACEMENT_ROBBER_Q`, which `static_value` reads through
  `score_settlement_spot`; `needs_python_evaluator` in the registry) only takes effect on the
  Python evaluator.  For such tunables the script re-executes itself with `CATANBOT_NO_ACCEL=1`
  set *before* `catanbot` is imported (`catanbot/accel.py` reads it at import time) so **both**
  sides run the Python evaluator, and prints `evaluator mode: python (CATANBOT_NO_ACCEL=1)`.
  Other tunables run `evaluator mode: c++ (catanbot_core)` when the extension is built.  Every
  game record also carries the mode of the (forked) process that played it, the script prints
  `evaluator mode in the game processes: ...` and warns if it differs from its own, and both are
  stored in the JSON (`evaluator_mode`, `worker_evaluator_modes`).  Compare decision times only
  within one mode.

## The paired design

Each game seats the *same* bot spec on every seat; the only difference is the tunable: seats
marked `C` play the candidate value, seats marked `D` the default.

* 4 players: 2 vs 2, seat patterns `CCDD DDCC CDCD DCDC CDDC DCCD` rotated per game;
  3 players: a rotating 2 vs 1, `CCD DDC CDC DCD DCC CDD`.  Consecutive patterns are complements,
  so any even number of games gives both sides exactly the same seats.
* Game `g` uses seed `seed * 100003 + g` for every candidate value, so the board (and the seat
  pattern) is identical across candidates.  `play_game` hands one `random.Random` to the engine
  (dice, steals) and to the bots (temperature sampling), so the dice and steals coincide across
  candidates only until the first decision that differs; from there on the games diverge, as in
  any common-random-numbers design over a sequential game.  Within a game the two sides always
  share everything, which is what the paired statistics rest on.
* The override lives only inside the candidate bot's hooks.  `ParamBot.decide` (and `reset`,
  `observe`, `explain`) applies the overrides, calls the inner bot and restores everything in a
  `finally` block, so the default seats always see the defaults even though the constants are
  module globals and both sides run in one process; an exception inside the inner bot cannot
  leak an override.  `danger`'s win-path cache is cleared before a candidate hook when the
  tunable feeds it and after every candidate hook, so a value computed with one side's constants
  is never reused by the other; since `observe` is wrapped too the cache is in effect emptied
  after every action of a paired game, for both sides alike (it still serves repeated lookups
  within one decision).  Both sides are wrapped (the default side with an empty override) so
  decision times are measured identically.
* Decision time is the wall time of `inner.decide` alone.  The apply / restore around it (up to
  ~0.1 ms for a tunable that rewrites function defaults, ~0.005 ms for the empty override) is
  accounted separately (`stats["overhead"]`, `overhead_ms_cand` / `overhead_ms_def` in the JSON
  and printed) and never enters the decision times or `extra_ms`.
* Search knobs (`kind = search`) are `SearchConfig` fields set on the bot's own config object,
  which is what the bot spec (`search:beam=2`) would have done; no global is touched.

### Statistics

Per game `g` the candidate side's *seat win rate* is `c_g = 1/|C|` if a candidate seat won, else 0,
and `d_g` likewise for the default side.  The report gives

* wins per side and per-seat win rates (`cand_wins / cand_seats`, `def_wins / def_seats`);
* `delta` = mean over games of `c_g - d_g`, the paired win-rate difference.  It is zero in
  expectation when both sides are equally strong, for 2-vs-2 and for the rotating 2-vs-1 design
  alike, and every game contributes one paired observation (`diff` in the game records).  In
  2-vs-2 it equals `cand_wins / cand_seats - def_wins / def_seats` exactly and ranges over
  +-50pp (+50pp = the candidate side won every game).  In the 3-player rotation a game contributes
  +-0.5 (two candidate seats) or +-1 (one), so `delta` is the mean of those contributions
  (+75pp when the candidate side wins everything) and is not exactly the per-seat win-rate
  difference, which the report prints alongside;
* `se` = standard error of that mean over games and the 95% interval `delta +- 1.96 se`
  (clipped to +-100pp), using the unbiased sample variance of the paired differences.  In the
  2-vs-2 design this is the binomial standard error of the game-level side win rate
  (`side_win_rate`, `side_se` in the JSON) scaled to seat units, up to the `N / (N - 1)`
  small-sample factor; with `N` games it is about `0.5 / sqrt(N)`, i.e. about +-3.5pp (SE) and
  +-7pp (95% interval) at 200 games;
* average VP per side (more sensitive than wins in small samples);
* mean and p95 wall time per non-trivial decision (more than one legal action) per side
  (`inner.decide` only, see above), `extra_ms` = candidate - default, and `cost` (`costlier` /
  `cheaper` / `same cost` within 0.05 ms);
* `value_per_ms` = `delta / extra_ms`, only when the candidate is measurably costlier: the win
  rate bought per millisecond of decision time.  A cheaper candidate that is not worse needs no
  ratio, it is simply the better setting;
* a verdict: `candidate better` / `default better` when the 95% interval excludes zero with at
  least 30 games, otherwise `no significant difference` or `inconclusive (too few games)`.

Reading a row.  A strategy earns its place when its default beats the "off" value (or the
smaller weight) by a margin whose interval excludes zero **and** the extra decision time is
paid for: with the depth-1 search bot deciding in about 25 ms (C++ evaluator), a strategy that
costs 1 ms per decision must buy more than the search would gain from 4% more nodes.  A strategy whose off/low value is
not worse, or whose cost is high for a small gain, is a candidate for cutting or for a cheaper
implementation.

## The registry

`kind`: `weight` = a numeric constant; `flag` = an on/off strategy; `search` = a `SearchConfig`
field.  `pyeval` = needs the Python evaluator; `search bot` = has no effect on the heuristic bot;
"depth >= 2 only" = `requires_depth = 2` in the registry (the script uses the depth-2 spec).
Defaults are read from the modules at import time (the registry never hard-codes them).

<!-- ablate:registry:start -->
| name | kind | default | candidates | pyeval | search bot | description |
|---|---|---|---|---|---|---|
| `danger.TURNS_HALF` | weight | 3 | 1.5, 2, 4.5, 6 |  |  | turns-to-win at which danger = 0.5 (steeper = only near-winners count as dangerous) |
| `danger.BLOCK_FLOOR` | weight | 0.35 | 0, 0.2, 0.5, 0.7 |  |  | block_factor for a resource the target does not need |
| `danger.BLOCK_NEED` | weight | 2 | 0, 1, 3, 4 |  |  | extra block_factor for a fully needed resource |
| `danger.danger_multiplier` | flag | on | off |  |  | off: robber.target_weight uses the VP threat only (multiplier fixed at 1.0 instead of 0.4 + 1.6 x danger); implemented by monkeypatching danger.danger_multiplier and the copy robber.py imported |
| `heuristic.EXPOSURE_WEIGHT` | weight | 0.25 | 0, 0.1, 0.4, 0.6 | yes | yes | weight of robber.steal_exposure_fast in static_value (only the search bot evaluates; needs the Python evaluator) |
| `coalitions.SCALE` | weight | 0.5 | 0.25, 1, 2 |  |  | EV sacrifice (card-value units) that counts as one full signal |
| `coalitions.BLOC_THRESHOLD` | weight | 1 | 0.5, 2, 100 |  |  | pairwise strength above which two players are treated as a bloc (100 = never); also rewrites the def-time defaults of allies/blocs/against |
| `coalitions.DECAY` | weight | 0.97 | 0.9, 0.99 |  |  | per-turn decay of coalition signals (def-time default of CoalitionDetector.decay) |
| `politics.MAX_SLACK` | weight | 0.3 | 0, 0.15, 0.6 |  |  | cap on the favour slack a friend gets in should_accept |
| `politics.BASELINE` | weight | 0.1 | 0, 0.3 |  |  | political capital everyone starts with (def-time default of PoliticalState.__init__) |
| `politics.DECAY` | weight | 0.97 | 0.9, 0.99 |  |  | per-turn decay of political capital towards the baseline |
| `opponent_model.DECAY` | weight | 0.9 | 0.7, 0.97 |  |  | per-observation decay of the opponent statistics (also _EW.add's def-time default) |
| `opponent_model.VALUE_LR` | weight | 0.12 | 0, 0.06, 0.25 |  |  | learning rate of the opponents' implied resource valuations (0 = never learn) |
| `opponent_model.stage_late_drop` | weight | 0.7 | 0, 0.4, 0.85 |  |  | trade_stage_factor = 1 - drop x game_stage^1.5 (0 = trade willingness never drops late); implemented by replacing the function at its three import sites |
| `trading.accept_margin` | weight | 0.002 | 0, 0.01, 0.03 |  |  | base win-probability gain should_accept demands (the margin parameter's default) |
| `trading.feed_leader_guard` | flag | on | off |  |  | off: the don't-feed-the-leader rule never fires (offer_is_feeding_leader patched in trading and heuristic; search/opponent_model import it lazily from trading) |
| `placement.RESOURCE_DEMAND` | weight | 0.934579/0.934579/0.841121/1.16822/1.1215 | 1/1/1/1/1, 0.833333/0.833333/0.740741/1.2963/1.2963, 1.10577/1.10577/0.865385/0.961538/0.961538 | yes |  | per-resource demand weights wood/brick/sheep/wheat/ore (mean 1; values a/b/c/d/e); mutated in place, also read by static_value so it needs the Python evaluator |
| `placement.PLACEMENT_BLOCK_WEIGHT` | weight | 1 | 0, 0.5, 2 | yes |  | weight of the robber-exposure penalty in settlement scoring (also read by static_value, so it needs the Python evaluator) |
| `placement.PLACEMENT_ROBBER_Q` | weight | 0.35 | 0, 0.175, 0.7 | yes |  | probability scale of the robber landing on a strong hex (also read by static_value, so it needs the Python evaluator) |
| `devcards.KNIGHT_VALUE` | weight | 0.55 | 0.3, 0.8 |  |  | VP-equivalent value of a drawn knight in should_buy_dev |
| `devcards.MONOPOLY_BASE_VALUE` | weight | 0.6 | 0.3, 0.9 |  |  | base VP-equivalent value of a drawn monopoly |
| `search.trade_proposals` | search | 3 | 0, 1, 5 |  | yes | PROPOSE_TRADE candidates per node (0 = never propose) |
| `search.dump_candidates` | search | 3 | 0, 1, 5 |  | yes | surplus dumps tried with > 7 cards (0 = off) |
| `search.opponent_proposals` | search | 1 | 0, 2 |  | yes | proposals a simulated opponent may make per turn (0 = never); depth >= 2 only |
| `search.opponent_actions` | search | 4 | 2, 6 |  | yes | greedy actions per simulated opponent turn; depth >= 2 only |
| `search.opponent_expand` | search | 6 | 3, 10 |  | yes | candidates evaluated per simulated opponent decision; depth >= 2 only |
| `search.opp_roll_samples` | search | 4 | 1, 2, 6 |  | yes | sampled roll sequences for the opponents' turns; depth >= 2 only |
| `search.beam` | search | 4 | 2, 6, 8 |  | yes | partial sequences kept per level |
| `search.expand` | search | 8 | 4, 12, 16 |  | yes | actions tried per decision node |
| `search.depth` | search | 1 | 2 |  | yes | turns of lookahead (2 = + opponents' turns) |
<!-- ablate:registry:end -->

Implementation notes for the entries that are not plain module attributes (none of the source
files is edited; the registry only patches attributes at run time and restores them):

* `danger.danger_multiplier` off: `danger.danger_multiplier` **and** `robber.danger_multiplier`
  (the copy `robber.py` bound with `from .danger import danger_multiplier`) become
  `lambda wp: 1.0`, so `robber.target_weight` = VP threat only.  Patching only `danger` would
  change nothing in `robber`.
* `opponent_model.stage_late_drop`: `trade_stage_factor` is replaced at its three import sites
  (`opponent_model`, `search`, `trading`) by `1 - drop * game_stage(state) ** 1.5`; the
  default 0.7 reproduces the original function exactly.
* `trading.feed_leader_guard` off: `offer_is_feeding_leader` (in `trading` and `heuristic`)
  answers `(False, "")`; `search` and `opponent_model` import it lazily from `trading`.
* `trading.accept_margin`: the default of `should_accept`'s `margin` parameter.
* Constants that were bound at *definition* time as default arguments (`def decay(self,
  factor=DECAY)`) are rewritten in the function's `__defaults__` as well as in the module:
  `coalitions.BLOC_THRESHOLD` (`allies` / `blocs` / `against`), `coalitions.DECAY`,
  `politics.DECAY`, `politics.BASELINE`, `opponent_model.DECAY` (`_EW.add`).
* `placement.RESOURCE_DEMAND` is mutated in place (every `from .placement import
  RESOURCE_DEMAND` site shares the list) and normalised to mean 1.
* `placement.PLACEMENT_BLOCK_WEIGHT` / `PLACEMENT_ROBBER_Q` are registered only when present
  (they are being added by another change); the candidates are 0x / 0.5x / 2x the default.

## Smoke-run results

**These are smoke numbers: 2-4 games per candidate on a 4-core machine under heavy load from
other workflows.  Nothing here is significant (a 4-game interval is about +-50pp) and the
decision times include scheduling noise; they only demonstrate that the harness runs end to
end.  Do not tune anything from these tables.**

<!-- ablate:results:start -->
Sweep of 5 tunables: 4 paired games per candidate, seed 1, 4 players, workers 2, run 2026-09-25 05:49 (smoke run: tiny game counts on a loaded machine, nothing here is significant).

| tunable | kind | value | default | spec | mode | games | cand W / def W (seats) | delta win-rate +- SE | 95% CI | avg VP c / d | ms/decision c / d (p95) | extra ms | value / ms | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| danger.TURNS_HALF | weight | 1.5 | 3 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.75 / 8.38 | 1.21 / 1.29 (9.6 / 9.8) | -0.08 | n/a (cheaper) | inconclusive (too few games) |
| danger.TURNS_HALF | weight | 2 | 3 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 8.12 | 1.26 / 1.15 (9.7 / 8.3) | 0.12 | -2.1570 | inconclusive (too few games) |
| danger.danger_multiplier | flag | off | on | heuristic | c++ | 4 | 0 / 4 (8/8) | -50.0pp +- 0.0pp | [-50.0pp, -50.0pp] | 5.62 / 8.00 | 1.23 / 1.25 (9.7 / 9.6) | -0.02 | n/a (same cost) | inconclusive (too few games) |
| coalitions.SCALE | weight | 0.25 | 0.5 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 7.75 | 1.31 / 1.25 (10.2 / 9.8) | 0.07 | -3.6446 | inconclusive (too few games) |
| coalitions.SCALE | weight | 1 | 0.5 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 7.75 | 1.27 / 1.20 (9.6 / 9.7) | 0.07 | -3.6187 | inconclusive (too few games) |
| politics.MAX_SLACK | weight | 0 | 0.3 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 7.75 | 1.11 / 1.10 (9.4 / 9.5) | 0.01 | n/a (same cost) | inconclusive (too few games) |
| politics.MAX_SLACK | weight | 0.15 | 0.3 | heuristic | c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 7.75 | 1.40 / 1.19 (10.0 / 9.5) | 0.21 | -1.1664 | inconclusive (too few games) |
| trading.feed_leader_guard | flag | off | on | heuristic | c++ | 4 | 2 / 2 (8/8) | +0.0pp +- 28.9pp | [-56.6pp, +56.6pp] | 8.00 / 8.50 | 1.30 / 1.29 (9.7 / 9.7) | 0.02 | n/a (same cost) | inconclusive (too few games) |
<!-- ablate:results:end -->

<!-- ablate:smoke:start -->
Individual smoke runs (verification pass, `--workers 2 --seed 1`; decision times are `inner.decide` only,
override overhead excluded and shown separately; 4 games for the heuristic spec, 2 for the search specs):

| command | spec / mode | games | cand W / def W (seats) | delta +- SE | 95% CI | avg VP c / d | ms/decision c / d (p95) | extra ms | overhead c / d ms |
|---|---|---|---|---|---|---|---|---|---|
| `--tunable danger.BLOCK_NEED --values 0,4 --games 4 --players 4` value 0 | heuristic / c++ (catanbot_core) | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.62 / 7.88 | 1.07 / 0.98 (8.3 / 8.2) | +0.09 | 0.038 / 0.003 |
| `--tunable danger.BLOCK_NEED --values 0,4 --games 4 --players 4` value 4 | heuristic / c++ (catanbot_core) | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 7.25 / 8.00 | 1.13 / 1.19 (9.6 / 9.5) | -0.07 | 0.049 / 0.011 |
| `--tunable trading.feed_leader_guard --flag-off --games 4 --players 3` value off | heuristic / c++ (catanbot_core) | 4 | 1 / 3 (6/6) | -37.5pp +- 31.5pp | [-99.2pp, +24.2pp] | 7.00 / 8.83 | 0.97 / 1.02 (8.2 / 7.3) | -0.04 | 0.031 / 0.003 |
| `--tunable placement.RESOURCE_DEMAND --max-candidates 1 --games 4 --players 4` value 1/1/1/1/1 | heuristic / python (CATANBOT_NO_ACCEL=1), re-exec; game processes reported python too | 4 | 2 / 2 (8/8) | +0.0pp +- 28.9pp | [-56.6pp, +56.6pp] | 7.88 / 8.62 | 1.19 / 1.12 (9.5 / 9.3) | +0.07 | 0.042 / 0.006 |
| `--tunable search.opp_roll_samples --values 1 --games 2 --players 4` on the **depth-1** spec (before the `requires_depth` rule) | search depth 1 / c++ | 2 | 2 / 0 (4/4) | +50.0pp +- 0.0pp | [+50.0pp, +50.0pp] | 7.75 / 7.50 | 35.01 / 25.09 (123.2 / 100.7) | +9.91 | 0.024 / 0.006 |
| `--tunable search.opp_roll_samples --values 1 --games 2 --players 4 --max-turns 40` (depth-2 spec, the default now) | search depth 2 / c++ | 2 | 1 / 1 (4/4) | +0.0pp +- 50.0pp | [-98.0pp, +98.0pp] | 3.50 / 3.75 | 34.74 / 40.90 (142.4 / 138.0) | -6.16 | 0.033 / 0.022 |
| `--tunable danger.TURNS_HALF --values 2,4.5 --games 4 --workers 1 --seed 7` (pairing proof, both values) | heuristic / c++ | 4 | 1 / 3 (8/8) | -25.0pp +- 25.0pp | [-74.0pp, +24.0pp] | 5.62 / 7.12 (value 2) | 0.56 / 0.68 (2.4 / 3.7) | -0.12 | 0.015 / 0.007 |

What the smoke runs do show.  The harness runs both bots, both player counts, the flag path, the
in-place vector override, the search-knob path and the Python-evaluator re-exec (the game processes
report `python (CATANBOT_NO_ACCEL=1)` and `catanbot.accel.AVAILABLE` is false in them; on the C++ path
the Python `static_value` is never called).  The pairing-proof run (seed 7, `--workers 1`) has the
same four seeds `700021..700024` and the same seat patterns `CCDD DDCC CDCD DCDC` for both candidate
values, and its `delta`, `se`, interval, per-side VP and decision times recompute exactly from the
per-game `records` in the JSON.  The depth-1 `opp_roll_samples` row is the negative example that led
to the `requires_depth` rule: at depth 1 the knob is never read, both sides did identical work, and
the "+9.9 ms" is nothing but scheduling noise on this loaded machine (two workers on four shared
cores under other workflows) - which is also the size of noise to expect in any search-spec timing
here; the 0.05 ms `same cost` band is only meaningful on a quiet machine.  On the depth-2 spec the
same candidate is measurably cheaper, as it should be.  A heuristic decision costs about 1 ms here,
a depth-1 search decision about 25 ms with the C++ evaluator and about 130 ms with the Python one.
<!-- ablate:smoke:end -->

## Protocol for a full run (quiet machine)

1. Quiet machine, C++ extension built (`scripts/build_cpp.sh`), `--workers` = cores - 1.
2. Cheap pass, every tunable, heuristic bot, 4 players: `--sweep-all --games 200 --workers W
   --seed 1 --players 4 --out docs/ABLATIONS.md` (about 200 x 0.3 s x candidates per tunable, a
   few minutes per tunable).  Repeat with `--players 3 --seed 2`.  This screens weights that the
   rule-based bot reads (danger, coalitions, politics, trading, placement, dev cards).
3. Real pass, the search bot, per tunable: `--tunable NAME --base-spec
   search:depth=1,beam=4,expand=8,evaluator=heuristic --games 200 --workers W --players 4` and
   again with `--players 3`; at 10 s a game that is about 35 min of CPU per candidate per player
   count (about 9 min with 4 workers).  Tunables marked `pyeval` run the Python evaluator on both sides (3.5x slower);
   their decision times are comparable only with each other.
4. Search knobs: the same, with the candidate's cost in the table; keep a knob's larger value
   only when `value_per_ms` beats the alternative of spending the same milliseconds on
   `beam` / `expand` (run those first as the yardstick).  The four opponent knobs run on the
   depth-2 spec (several times slower per game; budget accordingly) and `search.depth=2` itself
   is the ablation of the opponents'-turns lookahead as a whole - run it first: if depth 2 does
   not pay for its cost, the opponent knobs are moot.
5. Decide per tunable: interval excludes zero at 200 games (about +-5pp) in both player counts
   -> adopt / cut; otherwise keep the default and the cheaper implementation.  Confirm a winner
   with a fresh `--seed` before changing the constant in the module.
6. Re-run the C++ bit-exactness tests (`tests/test_accel_heuristic.py`) after changing an
   evaluator constant, and mirror the value in `cpp/heuristic.cpp`.

## Sweep 1 (2026-09-25 11:30 UTC): 120 paired games per candidate, depth-1 search bot, 4 players

Base spec `search:depth=1,beam=4,expand=8,evaluator=heuristic`, seed 101,
2 workers on a loaded machine (decision times not reported for that reason).
A candidate is seated 2 vs 2 against the default in the same games.

| tunable | default | candidate | games | wins cand / default | delta (pp) | s.e. (pp) |
|---|---|---|---|---|---|---|
| coalitions.SCALE | 0.5 | 0.25 | 120 | None / None | +0.8 | 4.6 |
| coalitions.SCALE | 0.5 | 1.0 | 120 | None / None | +1.7 | 4.6 |
| danger.BLOCK_NEED | 2.0 | 0.0 | 120 | None / None | +1.7 | 4.6 |
| danger.BLOCK_NEED | 2.0 | 1.0 | 120 | None / None | +3.3 | 4.6 |
| danger.TURNS_HALF | 3.0 | 1.5 | 120 | None / None | -1.7 | 4.6 |
| danger.TURNS_HALF | 3.0 | 2.0 | 120 | None / None | +0.0 | 4.6 |
| danger.danger_multiplier | True | False | 120 | None / None | +2.5 | 4.6 |
| politics.MAX_SLACK | 0.3 | 0.0 | 120 | None / None | -3.3 | 4.6 |
| politics.MAX_SLACK | 0.3 | 0.15 | 120 | None / None | -1.7 | 4.6 |

Reading: none of the five terms moves the self-play win rate by more than
its standard error at 120 games (detectable effect about +-9 pp).  Switching
the distance-to-win multiplier off is +2.5 pp (noise), removing the favour
slack is -3.3 pp (noise).  Self-play between identical bots is a weak test
bed for targeting and political terms (the opponents share the same logic);
the next step is the same paired design against the Catanatron stand-ins
and 500+ games per candidate for the terms that matter for compute.

## Sweep 2 (2026-09-25 13:40 UTC): the remaining six tunables, 120 paired games each

Same design and base spec (seed 102).  Decision times are per searched
decision, measured under a load average of about 4 (another workflow
running), so only the relative cost is meaningful.

| tunable | default | candidate | games | wins cand / default | delta (pp) | s.e. (pp) | ms/decision cand / default |
|---|---|---|---|---|---|---|---|
| devcards.KNIGHT_VALUE | 0.55 | 0.3 | 120 | None / None | +0.8 | 4.6 | 13.3 / 13.2 |
| devcards.KNIGHT_VALUE | 0.55 | 0.8 | 120 | None / None | +0.0 | 4.6 | 13.1 / 12.9 |
| opponent_model.stage_late_drop | 0.7 | 0.0 | 120 | None / None | +4.2 | 4.6 | 16.2 / 12.5 |
| opponent_model.stage_late_drop | 0.7 | 0.4 | 120 | None / None | +5.0 | 4.6 | 14.5 / 12.8 |
| placement.PLACEMENT_BLOCK_WEIGHT | 1.0 | 0.0 | 120 | None / None | -1.7 | 4.6 | 13.3 / 13.1 |
| placement.PLACEMENT_BLOCK_WEIGHT | 1.0 | 0.5 | 120 | None / None | -2.5 | 4.6 | 13.6 / 13.4 |
| search.dump_candidates | 3 | 0 | 120 | None / None | +0.0 | 4.6 | 13.1 / 12.9 |
| search.dump_candidates | 3 | 1 | 120 | None / None | -0.8 | 4.6 | 13.1 / 13.0 |
| search.trade_proposals | 3 | 0 | 120 | None / None | -17.5 | 4.3 | 7.1 / 18.8 |
| search.trade_proposals | 3 | 1 | 120 | None / None | -5.0 | 4.6 | 10.7 / 14.2 |
| trading.feed_leader_guard | True | False | 120 | None / None | +0.8 | 4.6 | 12.8 / 13.0 |

Reading: one clear result - **trade proposals in the search are worth
+17.5 pp (s.e. 4.3) at about 12 ms per decision** (proposals=0 costs 17.5
pp, proposals=1 costs 5 pp): the single most valuable term measured so far
and cheap for what it buys.  Two weak signals worth a 400-game follow-up:
the late-game trade damping may be slightly too strong (stage drop 0.0 /
0.4 = +4.2 / +5.0 pp, one standard error) and the blockability weight is on
the right side (removing it -1.7 pp, halving it -2.5 pp).  Knight value,
dump candidates and the feed-the-leader guard are neutral at this sample
size in self-play.

Caveat (found 2026-09-25, see "Adversarial verification" at the end): the
`placement.PLACEMENT_BLOCK_WEIGHT` rows above ran with the C++ evaluator
(13 ms/decision), whose port of `static_value` hard-codes the constant, so the
override only reached the Python-side move priors, not the evaluator.  The
tunable is now flagged `needs_python_evaluator`; re-measure it (and
`PLACEMENT_ROBBER_Q`) before reading anything into the -1.7 / -2.5 pp.

## Paired ablations against Catanatron

Self-play tells us whether a term helps against *catanbot*; the question that matters next is
whether it helps against strong bots that share none of our code.  `scripts/ablate_catanatron.py`
runs the same paired ablations with catanbot in **one** seat against **three copies of a Catanatron
opponent**, and `scripts/campaign.py` strings many of them into a resumable campaign with thousands
of games per candidate.  Tests: `tests/test_ablate_catanatron.py`.

### Design

* **Arms and pairing.**  For a registry tunable (`--tunable`, same `--values` / `--flag-off`
  semantics as `scripts/ablate.py`) or two bot specs (`--cand-spec` / `--def-spec`, e.g. heuristic vs
  search bot, depth 2 vs depth 1; `--cand-set NAME=VALUE` adds overrides to the candidate), every
  seed `s` is played once per **arm**: the candidate catanbot and the default catanbot, each against
  the same three opponents (`--opponent`: a `scripts/bench_catanatron.py` preset - `value`,
  `alphabeta`, `sameturn` on the 3.3 engine, `vf`, `ab`, `vp`, `weighted`, `random` on both - with
  `--opponent-params` passed through), with the **same board / dev deck / dice seed** (`s + 1`,
  which is what `bench_catanatron.py --seed 0` uses for its game `s`) and the **same seat** (`s % 4`).
  The default arm is played once per seed and shared by every candidate value of the invocation
  (3 values cost 4 games per seed, not 6).  A worker process plays all arms of a seed back to back.
* **Determinism.**  catanatron iterates sets of `Color` enums, whose order depends on Python's
  string-hash seed, so the script re-executes itself with `PYTHONHASHSEED=0` when it is not set
  (like `ablate.py`'s `CATANBOT_NO_ACCEL` re-exec; forked workers inherit it).  Measured: with the
  seed pinned, a game is a deterministic function of (seed, arm) across processes, on 3.2.1 and 3.3,
  and even across the C++ / Python evaluators (bit-exact port); without it two runs of the same seed
  differ.  So the two arms of a pair are **identical up to the first catanbot decision that
  differs**, which every pair verifies (below).
* **Overrides.**  The catanbot seat is `CatanbotPlayer(bot=ParamBot(make_bot(spec), overrides))`:
  weights / flags through `tuning.apply`, search knobs through `tuning.apply_to_bot`, restored after
  every hook; a tunable read by the static evaluator (`needs_python_evaluator`) makes the script
  re-execute with `CATANBOT_NO_ACCEL=1`, so both arms run the Python evaluator.  Players are built
  exactly as `bench_catanatron.py` builds them: opponents through its `opponent_factory` and wrapped
  in the adapter's `BenchOpponent` (per-decision timing, trade-answer rule), `suppress_trades =
  (trades == "off")` for catanbot.  `--trades off|native|value|fair` (3.3 only) enables domestic
  trading for both arms; `--cand-adapter-opt trades=MODE` for the candidate only; `--adapter-opt
  KEY=VALUE` passes any other `CatanbotPlayer` keyword (an unknown one is an error, not a no-op).
* **Pairing check.**  Each game record carries a fingerprint of the action log (a hash every 16
  actions) and of catanbot's first 32 consulted decisions.  Per pair the report counts *identical*
  games (the change never altered a decision), *diverged / consistent* (the logs agree up to the
  first catanbot decision that differs, as they must) and *inconsistent* (the logs part before any
  catanbot decision differs: nondeterminism, e.g. an opponent's time limit - catanatron's
  `AlphaBetaPlayer` stops searching after 20 s).  The median first differing decision shows how
  early a change starts to matter.

### Statistics (per candidate, from the JSONL alone)

Only complete pairs count (both arms finished without error, played by the same code).  With
`c_s`, `d_s` = 1 if the candidate / default seat won seed `s`:

* wins and win rate per arm; `delta` = mean of `c_s - d_s` = the difference of the two win rates;
  `se` = standard deviation of the per-seed differences (unbiased) / `sqrt(n)`; 95% CI `delta +-
  1.96 se`; `mde80 = 2.8 se`, the difference this sample detects with 80% power;
* the same for catanbot's final VP (`vp_delta +- vp_se`): VP is the more sensitive statistic (every
  game contributes a graded value, not just the ~25-65% that are wins);
* the concordance table (both won / only candidate / only default / neither): only the discordant
  pairs carry information, which is why pairing helps: `se ~ sqrt(discordant fraction) / sqrt(n)`;
* per-seat breakdown (seat 0-3; seeds rotate seats, so use a multiple of 4 seeds);
* catanbot decision time per arm (`inner.decide` of consulted decisions, mean and p95 pooled over
  games from per-game log2 histograms), extra ms, the opponents' decision time, seconds per game,
  turns, games truncated at the turn cap, adapter statistics (errors, fallbacks, unmapped top actions,
  trade prompts / offers / confirmed trades) and the opponents' trade answers;
* verdict: `candidate better` / `candidate worse` when the 95% interval excludes 0 with at least 30
  pairs, else `no detectable difference at 95%` (or `inconclusive (< 30 pairs)`); the same for VP.

Sample size.  An unpaired comparison at win rate `p` has `se = sqrt(2 p (1 - p) / n)`: 1.5 pp at
`n = 2000` for `p = 0.64` (catanbot vs `value`), 1.4 pp at `p = 0.25`.  Pairing only lowers it (by
the fraction of pairs whose outcome cannot change), so **2000 seeds detect ~4 pp with 80% power**
and 1000 seeds ~6 pp.  A change that alters decisions in only a small fraction of games (see the
identical-pair counts) has a correspondingly small possible effect.

### JSONL, resume, reuse

`--out FILE.jsonl` is append-only; one line per event, each written with a single `write` and
fsynced:

* `{"kind": "run", "run_key", "cand_key", "def_key", "cand": {spec, overrides, adapter, label},
  "def": {...}, "ctx": {opponent, opponent_params, python, catanatron, evaluator, vps_to_win,
  discard_limit, hashseed}, "seeds", "seed_base", "code", ...}` declares a comparison.  An *arm key*
  hashes the arm's role, spec, overrides, adapter options and the context; the *run key* hashes the
  two arm keys.
* `{"kind": "game", "arm_key", "arm": "cand"|"def", "s", "game_seed", "seat", "status": "ok"|"error",
  "winner", "winner_seat", "won", "our_vp", "opp_vps", "vps", "turns", "actions", "truncated",
  "duration", "dec": {n, ms, mean, p95, max, h}, "dec_adapter", "opp_dec", "adapter": {...},
  "opp_trade_stats", "trace", "ours", "code", "evaluator", "hashseed", "hash_probe", "pid", "t", ...}`
  per finished game (candidate records also carry `run_key`); an `"error"` record holds the
  exception (a game that raises, exceeds `--game-timeout`, or kills its worker process).
* `{"kind": "stop", ...}` when a sequential stop ended a candidate.

Rerunning the same command skips every (arm, seed) already in the file, so a killed run resumes
where it stopped (a line cut short by the kill is skipped by the reader and terminated before the
next append); `--seeds` can be raised later to extend a run; error records count as played unless
`--retry-errors`.  `code` is a hash of every file that can change a game (the `catanbot` package
except vision / CLI / training, the C++ extension, `bench_catanatron.py`, the catanatron version): a
pair is only formed from two games with the same code, and a seed whose other arm was played by
older code is re-played as a whole pair.  `--reuse GLOB` copies default-arm games with the same arm
key and code from other files (marked `reused_from`) instead of replaying them; the campaign passes
its whole directory.  `--report --out FILE` prints the statistics from the file alone
(`--report-json` writes them).  `--max-minutes M` starts no new seed after M minutes (in-flight games
finish).  A worker that dies (a hard crash, no Python exception) is detected, the in-flight seeds are
re-played one game per fresh process and the game that kills its process again becomes an error
record.

`--stop-at-se X` (with `--stop-min-pairs`, default 100) ends a candidate once its paired s.e. is
at most `X` **and** `|delta| > 3 s.e.`.  Caveat: looking repeatedly and stopping on a large
difference biases the stopped estimate away from 0 (the winner's curse) and raises the false-positive
rate above the nominal level of one look; the 3-s.e. threshold (two-sided p ~ 0.003 per look) keeps
it small, but a stopped row is a screening result - confirm a surprising one with a fresh seed range
(`--seed-base`).

### Commands

```bash
export PYTHONPATH=/home/user/ClaudeTesting2
PY33=/home/user/venv_cat33/bin/python       # catanatron 3.3: value / alphabeta / sameturn, domestic trading
# a registry tunable vs 3 x catanatron's ValueFunctionPlayer, 2000 seeds (4000 games), 2 workers
$PY33 scripts/ablate_catanatron.py --tunable danger.TURNS_HALF --values 2,4.5 --opponent value \
    --seeds 2000 --workers 2 --out runs/turns_half@value.jsonl
# search vs the heuristic bot, and depth 2 vs depth 1 (3.2.1 stand-ins, fast)
python3 scripts/ablate_catanatron.py --cand-spec heuristic:temp=0 \
    --def-spec search:depth=1,beam=4,expand=8,evaluator=heuristic --opponent vf --seeds 1000 --workers 2 --out runs/heur@vf.jsonl
python3 scripts/ablate_catanatron.py --tunable search.depth --values 2 --opponent vf --seeds 1000 --workers 2 --out runs/depth2@vf.jsonl
# trade proposals need domestic trading (3.3) and an answer rule for the opponents
$PY33 scripts/ablate_catanatron.py --tunable search.trade_proposals --values 0 --opponent value --trades value \
    --seeds 2000 --workers 2 --out runs/trades@value.jsonl
python3 scripts/ablate_catanatron.py --report --out runs/turns_half@value.jsonl           # statistics only
python3 scripts/ablate_catanatron.py ... --plan                                            # keys and progress only
timeout 1200 $PY33 scripts/ablate_catanatron.py ... --max-minutes 15                      # a bounded chunk; rerun to resume
```

### Campaigns

`scripts/campaign.py --plan plan.json --dir DIR` runs a list of experiments one after another in
ascending `priority` (ties: plan order), each through the right interpreter (`py321` = system
`python3` with catanatron 3.2.1, `py330` = `/home/user/venv_cat33/bin/python`; override with
`"interpreters"` in the plan) with `PYTHONHASHSEED=0`, each appending to `DIR/<name>.jsonl` (its log
in `DIR/logs/<name>.log`, events in `DIR/campaign_events.log`) and reusing the default-arm games other
experiments in `DIR` already played.  After every experiment it rewrites the markdown summary
(`--summary PATH`, default `DIR/SUMMARY.md`): one row per candidate with games, both arms' win rates,
delta +- s.e., 95% CI, VP delta +- s.e., ms/decision of both arms, the opponents' ms/decision,
identical pairs, verdict and status.  It is idempotent and resumable: a complete experiment is
skipped, an incomplete one resumes, a failing experiment is logged and the campaign moves on, a
crashing game is an error record.  `--status` prints per experiment the games done / planned, errors,
reused games, measured throughput (games/hour from the record timestamps, idle gaps excluded) and
the ETA; `--max-minutes M` bounds one invocation (run it under `timeout` in chunks and rerun);
`--only a,b`, `--dry-run`, `--summary-only`, `--max-workers` (default 2 caps every experiment).

```json
{"defaults": {"seeds": {"count": 2000, "base": 0}, "workers": 2},
 "experiments": [
  {"name": "search_vs_heur@value", "interpreter": "py330", "opponent": "value", "priority": 1,
   "cand_spec": "heuristic:temp=0", "def_spec": "search:depth=1,beam=4,expand=8,evaluator=heuristic"},
  {"name": "trades@value", "interpreter": "py330", "opponent": "value", "priority": 1,
   "tunable": "search.trade_proposals", "values": [0], "trades": "value"},
  {"name": "depth2@value", "interpreter": "py330", "opponent": "value", "priority": 2,
   "tunable": "search.depth", "values": [2], "seeds": {"count": 1000, "base": 0}},
  {"name": "turns_half@value", "interpreter": "py330", "opponent": "value", "priority": 3,
   "tunable": "danger.TURNS_HALF", "values": [2, 4.5], "stop_at_se": 0.012}
 ]}
```

Other experiment fields: `flag_off`, `base_spec`, `cand_set` / `set` (`{"NAME": value}`),
`adapter_opts` / `cand_adapter_opts`, `trades`, `opponent_params`, `stop_at_se`, `stop_min_pairs`,
`game_timeout`, `vps_to_win`, `discard_limit`, `enabled`, `notes`, `extra_args` (passed through).
`"reuse": false` makes an experiment play its own default games: reused default games were played
at another time and load, so the decision-time comparison then only uses co-played pairs (the report
prints how many; a row whose default games were all reused is marked `(reused)` in the summary).

### Smoke and pilot results (2026-09-25; 4 shared cores, another benchmark on the other 2)

Throughput at `--workers 2` (both arms counted; the default bot `search:depth=1,beam=4,expand=8,evaluator=heuristic`):

| engine | opponent | games | wall | games / hour | s / game | 2000 seeds x 2 arms |
|---|---|---|---|---|---|---|
| 3.2.1 | `vf` (our stand-in) | 40 | 25 s | 5,800 | 1.2 | 0.7 h |
| 3.2.1 | `vf`, Python evaluator (pyeval tunables) | 40 | 74 s | 1,950 | ~3.5 | 2.1 h |
| 3.3.0 | `value` | 40 | 40 s | 3,560 | 2.0 | 1.1 h |
| 3.3.0 | `value`, depth-2 candidate | 16 | 25 s | 2,280 | 4.0 / 2.0 | 1.8 h |
| 3.3.0 | `sameturn` | 8 | 80 s | 360 | 17.5 | 11 h |
| 3.3.0 | `alphabeta` | 8 | 118 s | 244 | 21.6 | 16 h |

Each further candidate value of the same experiment (or any experiment sharing the default arm)
costs only its own games, i.e. half of the last column.

Pairing checks (all measured with this harness):
* A/A runs (`search.trade_proposals=0`: with trading off, catanbot can never propose, so the arms are
  the same bot): 20/20 identical games vs `value`, 4/4 vs `alphabeta`, 4/4 vs `sameturn`, 20/20 vs `vf`;
  every paired difference exactly 0.  catanatron's 20-s AlphaBeta deadline did not break determinism
  at these game lengths (21 s per whole game).
* Every non-identical pair was consistent (the action logs agree up to the first catanbot decision
  that differs): `danger.TURNS_HALF=2` vs `vf` 1/1 (decision #24, action 144, reproduced in a second
  process), `search.depth=2` vs `value` 8/8 (decision #1), trade proposals 0 vs 3 with `--trades value`
  on 3.3 8/8.
* The default arm reproduces `scripts/bench_catanatron.py --seed 0` game for game (8/8 identical
  VPs and turn counts vs `vf`).

Pilot: which terms change a decision at all against Catanatron?  21 tunables at one extreme value,
20 seeds each vs `vf` on 3.2.1 (trading off), run as one campaign (6.8 minutes; the default games
were played once and reused).  "Identical" = pairs whose two games were the same game:

| identical pairs | tunables |
|---|---|
| 20/20 (no decision ever changed) | `search.trade_proposals=0`, `coalitions.BLOC_THRESHOLD=100`, `coalitions.SCALE=2`, `politics.MAX_SLACK=0`, `opponent_model.VALUE_LR=0`, `opponent_model.stage_late_drop=0`, `trading.accept_margin=0.03`, `trading.feed_leader_guard=off`, `devcards.MONOPOLY_BASE_VALUE=0.9` |
| 15-19/20 | `danger.TURNS_HALF=6` 19, `danger.BLOCK_NEED=0` 19, `danger.danger_multiplier=off` 18, `devcards.KNIGHT_VALUE=0.3` 18, `danger.BLOCK_FLOOR=0` 17, `placement.PLACEMENT_BLOCK_WEIGHT=0` 15, `placement.PLACEMENT_ROBBER_Q=0` 15 |
| 0-12/20 | `search.dump_candidates=0` 12, `heuristic.EXPOSURE_WEIGHT=0` 7, `search.beam=2` 1, `search.expand=4` 1, `placement.RESOURCE_DEMAND=1/1/1/1/1` 1 |

Reading: the trading / politics / coalition / opponent-model terms only act through domestic trades,
so **against Catanatron they can only be tested on 3.3 with `--trades value` or `fair`** (our model of
how an opponent answers an offer: catanatron's own players answer degenerately - `value` always
rejects, `alphabeta` / `sameturn` raise and are counted as rejecting, see docs/BENCHMARKS.md), and
there they measure the term against that answer rule, not against Catanatron.  The danger / robber /
placement-block terms change a decision in only 5-25% of games; a term that changes a fraction `f`
of games can move the win rate by at most `f`, and its paired s.e. is at most `sqrt(f / n)` (0.7 pp
at `f = 0.1`, 2000 seeds), so thousands of seeds are needed and the possible effect is small.  The
search knobs, the evaluator weights and the placement demand vector change almost every game and are
where a 2000-seed run can find 3-4 pp.  (Win rates in this 20-seed pilot are noise: +-10 pp s.e.)

### Adversarial verification (2026-09-25, 4 cores, no other benchmark running)

Scratch data: `/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/campaign/verify/`.

**Pairing is real.**  A/A with a candidate equal to the default (`--tunable danger.TURNS_HALF --values 3.0`,
which still goes through ParamBot's apply/restore): 10/10 identical games (winner, VPs, turns, full
action-log hash) vs `value` on 3.3 and 10/10 vs `vf` on 3.2.1.  The same seeds replayed in a fresh
process with `--workers 1` and a different seed order: 5/5 identical per arm on both engines.  All
throughput runs below were played at `--workers 2` and again at `--workers 3`: 340/340 games identical,
alphabeta included.  With a real candidate an independent checker (full action logs, not the harness's
checkpoints) found the first differing log entry to be a catanbot action with a different action in
10/10 diverged pairs vs `value` (`search.beam=2`), 10/10 vs `vf` (`search.beam=2`) and 2/2 vs `value`
(`danger.BLOCK_FLOOR=0`).  No source of nondeterminism was found (3.3 draws dice from `game.state.random`;
3.2.1 from the global `random`, which no catanbot or stand-in code touches; catanbot draws from its own
seeded `rng`).

**Fixed: false "inconsistent" pairs.**  The tracer recorded only decisions in which the bot was consulted.
A knight whose victim differs between the arms logs the same `PLAY_KNIGHT_CARD` in both, and the games part
at the adapter's follow-up `MOVE_ROBBER` (played from `_pending_robber` without consulting the bot); the
3.3 per-card discard has the same shape.  Such pairs (1 in 10 vs `value`, 1 in 10 vs `vf`) were reported
"inconsistent" although only our own decision made them diverge.  The tracer now also records follow-ups
(`[index, hash, 1]`, records carry `ours_v: 2`; older records are compared on consulted decisions only),
and the same runs report 10/10 consistent.

**Statistics and resume.**  `--tunable search.expand --values 4,12` vs `vf`, 80 seeds: delta, paired
s.e. (also via the discordant-pair formula), 95% CI, VP delta / s.e. / CI, wins, concordance and the
per-seat deltas recomputed from the JSONL by hand match the report to the last digit.  The same command
killed with SIGTERM at 30 s, then with SIGKILL at 25 s, then finished: 240 records, no duplicate
(arm, seed), every game identical to the uninterrupted run and identical statistics.  Campaign: a
3-experiment plan (3.2.1 `vf`, 3.3 `value`, 3.3 `value` with a Python-evaluator tunable) run in a budgeted
chunk, `--status`, resumed, re-run (no new games), summary rows correct; a 4th experiment sharing the
default arm reused 10 default games.  Changed: `--status` borrowed an ETA rate across experiments with the
same interpreter/opponent even when one needs the Python evaluator (3-4x slower); it now borrows only
within the same interpreter, opponent, opponent params, trade mode and evaluator.

**Multiple comparisons (added to the campaign summary).**  Every row's verdict is a 95% test; a campaign
with ~60 candidate rows produces ~3 "better"/"worse" verdicts by chance.  SUMMARY.md now has a `Holm p`
column (two-sided normal p, Holm-adjusted over the rows with >= 30 pairs).  Budget for it: at 2000 seeds and
s.e. 1.3 pp a single test detects ~3.6 pp with 80% power, a Holm-corrected one over 60 rows ~5.4 pp; a
surviving row should still be confirmed on fresh seeds (`seeds.base`).

**Fixed: two tunables were silently half-applied.**  `static_value` reads `placement.PLACEMENT_BLOCK_WEIGHT`
and `PLACEMENT_ROBBER_Q` (via `score_settlement_spot`), but `cpp/heuristic.cpp` has `constexpr` copies, so
with the extension loaded the override reached only the Python-side move priors.  Measured: `=0` changed
14/20 games vs `vf` with the Python evaluator but 5/20 with the C++ one; the C++ and Python static values
differ under both overrides and agree bit for bit at the defaults.  Both are now `needs_python_evaluator`
(the scripts re-exec with `CATANBOT_NO_ACCEL=1`), and `tests/test_ablate.py::test_cpp_static_value_constants_are_flagged`
checks every non-search tunable against the C++ static value.  The self-play sweep-2 rows and the 15/20
pilot counts for these two were measured half-applied.  (`cpp/policy.cpp` also hard-codes the danger
constants, but only for the native opponent simulation, which runs at depth >= 2.)

**Trading.**  40 games vs 3x `value` with `--trades value` (seed 11): 0 adapter errors, 0 fallbacks,
0 unmapped actions; 1,680 offers played, every one accepted by the engine.  10 games with a checker after
every engine tick: 434 offers, all with cards the offerer held and at most 4 per turn; after each of the 92
`CONFIRM_TRADE`s catanbot's own engine applied to the pre-confirm state predicted exactly catanatron's hands
and bank; at all 1,636 catanbot decisions the adapter's shadow state and the converted state matched
catanatron's hands, bank and dev cards; 685 end-of-turn checks matched.  Re-measured on 100 fresh offers
(my positions and sampler): catanatron `value` rejected 100/100; `alphabeta` raised
`RuntimeError: Unknown ActionType REJECT_TRADE` on 100/100.  Our value rule accepted 33/100 (1:1 6/34,
2:1 21/33, 1:2 6/33) and my re-implementation of it agreed with `BenchOpponent.rule_accepts` on all 100.
Conclusion: trade-related tunables cannot be tested against Catanatron's own players - they never accept
(native mode can only waste compute) - and with `--trades value` / `fair` they are tested against our
myopic answer rule, which accepted 13.9% of catanbot's answerable offers (not a blanket accept, so not a
bug exploit, but also not Catanatron behaviour; `value` even accepts 18% of 1-for-2 offers against itself).
Report such results as "vs value-rule responders".  Side observation: catanbot cancels about a quarter of
the offers someone accepted (34 of 126 in the 10 games).

**Openings on the 3.3 adapter path.**  20 seeds x {search bot, heuristic bot} x {standin_book,
pips_diversity, denial, denial:2, setup_pick}, catanbot seat vs 3x `value`, ParamBot exactly as the harness
builds it: all 800 setup decisions equal `openings.choose(state, legal, policy)`; before and after every
decision (and after the games) nothing is patched (`installed() == "current"`, `heuristic.setup_pick` /
`setup_road_pick` / `SearchBot.decide` are the original objects).  Decisions that differ from the current
opening: search bot 44-55 of 80 per policy (setup_pick 46), heuristic bot 48-49 for standin_book /
pips_diversity, 7 for denial, 14 for denial:2, 0 for setup_pick (the null check).

**Throughput for the planner** (games/hour, both arms counted, default spec; `--seed-base 5000`; the
machine was otherwise idle, load ~1.9 at 2 workers and ~2.6-2.9 at 3):

| engine / opponent | experiment | 2 workers | 3 workers | s / game (2 w) | 2000 seeds x 2 arms at 2 w / 3 w |
|---|---|---|---|---|---|
| 3.3 `value` | danger.TURNS_HALF=2 (80 games) | 4,140 | 6,110 | 1.7 | 1.0 h / 0.65 h |
| 3.3 `value` | heuristic.EXPOSURE_WEIGHT=0, Python evaluator (32) | 1,320 | 1,960 | 5.0-5.5 | 3.0 h / 2.0 h |
| 3.3 `value` | search.trade_proposals=0 with `--trades value` (40) | 2,320 | 3,550 | 3.9 default / 2.1 cand | 1.7 h / 1.1 h |
| 3.3 `alphabeta` | danger.TURNS_HALF=2 (4 / 6 games) | 320 | 470 | 16-22 | 12.5 h / 8.5 h |
| 3.2.1 `vf` | danger.TURNS_HALF=2 (120) | 6,360 | 9,390 | 1.1 | 0.6 h / 0.4 h |
| 3.2.1 `vf` | heuristic.EXPOSURE_WEIGHT=0, Python evaluator (40) | 2,300 | 3,250 | 3.0 | 1.7 h / 1.2 h |
| 3.2.1 `ab` | danger.TURNS_HALF=2 (24) | 1,120 | 1,580 | 6.1 | 3.6 h / 2.5 h |

3 workers give ~1.45x the 2-worker rate on an idle 4-core machine; with the other benchmark on 2 cores,
use 2.  The alphabeta row rests on 4-6 games.

## Testing policy (user decision, 2026-09-26): thousands of games, priority queue, shelve the unprovable

This replaces any plan above that needed tens of thousands of games per
idea.

* **Budget:** a few thousand games per candidate (about 2,000 paired seeds
  against Catanatron, about 2,400 self-play games).  If proving an effect
  would take tens of thousands of games, the idea is **SHELVED**: its default
  stays as is and it is listed as shelved.
* **Early stopping (preemption):** each candidate is checked at interim
  looks.
  - **ADOPT** when it is clearly good; it then goes through the league gate
    before becoming the default.
  - **REJECT** when it is clearly bad, or clearly too small to matter.
  - **SHELVE** when neither is clear at the budget cap.
* **Priority queue** by how likely an idea is to move the needle per game
  spent.  A higher-priority item preempts the running one only when that is
  worth its switching cost.  Everything runs in batches that share their
  default arms.
* **Priorities** (regrouped below: diversification / expansion is now
  its own area, second after trading), in order:
  1. **Trading**: get what we need from whichever source is cheapest: the
     bank at 4:1, ports at 3:1 or 2:1, or other players.
  2. **Ports**: is a port spot worth giving up a three-tile spot for a two-
     or one-tile one, including the roads to reach it and whether others can
     and will block it, by what they need and a simple "are they the
     leader" check.
  3. **Robber**: block production and steal the right resources.
  4. **Card counting**: who holds what.
     - Resource odds for robber targets and our Monopoly.
     - Hurting the leader.
     - Reading held dev cards.  A card held a long time is most likely a
       VP card, so a secret leader.  Knights tend to be played for Largest
       Army or to move the robber, and Year of Plenty is usually used fast.
       A Monopoly becomes more plausible when the bank is low in what that
       player needs, because the other players then hold it.

  Everything else runs only if there is time.
* **Replace ideas that do not deliver:** an idea that does not deliver its
  promised benefit goes on ice and is replaced by a more moderate version of
  it.
* **2x2 interaction tests and joint tuning** only run within the same
  thousands-of-games budget, and only for the four priority areas.
* **Cost is measured in CPU time, not wall-clock** (user note: another
  project shares the workload, so wall-clock speed is not a clean signal).
  - Queue costs and "is it worth the compute" judgements use CPU seconds
    per game and positions evaluated per decision.
  - Hours are rough estimates only.
  - Win rates do not depend on speed.  Our search is fixed-size, with no
    time limit.  The one wall-clock limit in play is Catanatron's 20 s
    AlphaBeta cutoff, and every run reports the slowest opponent decision.
* **The politics rule below still applies.**

## Test queue (`scripts/run_queue.py`, 2026-09-26)

The policy above is implemented by the budgeted test queue. docs/QUEUE.md covers how to run it, read the verdicts, add rows, and use fallbacks and epochs. The plan is `scripts/queue_plan.json`.

In short:
- Every row goes through a sequential verdict engine (`scripts/seqtest.py`):
  - screen: ADOPT / REJECT / SHELVE;
  - knockout: REMOVE / KEEP;
  - estimate;
  - politics: a fixed-N screen, then SIGNIFICANT / INCONCLUSIVE by Holm within the row's tier;
  - confirm.
- A null row stops at about half its cap. False ADOPT is about 2% per row.
- Rows are ordered by area (harness, trades, diversification, ports, robber, counting, politics, other), then priority. Preemption is decided in CPU seconds (getrusage).
- Chunks run from pinned code snapshots, one epoch per area batch.
- A shelved idea unlocks its declared fallback row.

Verdicts are also written as `{"kind": "stop", "source": "queue"}` records into each row's JSONL. `ablate_catanatron.py` (and so campaign.py) no longer plays a candidate the queue has stopped. Other stop records keep their old meaning.

**Tooling exemption from the tuning.py rule.** Queue options are test tooling, not bot behaviour, so none of them is a `catanbot/tuning.py` tunable:
- the plan fields `design`, `polarity`, `estimator`, `crn`;
- the `ablate_catanatron.py` flags `--mech`, `--crn dice`, `--default-only`.

They are all off by default outside the queue, and arm keys are unchanged when they are off. The bot is untouched. A tuning.py entry would change the code fingerprint and void default-arm reuse.

## Queue after the strength proof (2026-09-26): every strategy the user asked to test

Everything below runs after T7-T11, one experiment at a time on 3 cores.
- Rows marked *campaign* are in `scripts/campaign_plan.json`.
  - Paired games against Catanatron: same seeds and seats, candidate vs
    default.
  - Run with `python3 scripts/campaign.py --plan scripts/campaign_plan.json
    --dir runs/campaign1`.
  - Tier 3 rows re-test against AlphaBeta only what tier 1 found
    significant (Holm).
- *Self-play* rows use `scripts/ablate.py` (2v2 paired self-play).  They
  test what Catanatron cannot measure: its bots never trade.
- A strategy becomes the default only after it passes the champion league
  gate (docs/LEAGUE.md).

| user's request | switch / tunable | where | status |
|---|---|---|---|
| is the search worth it at all | `heuristic:temp=0` vs the search bot | campaign t1 / t3 | queued |
| trade proposals | `search.trade_proposals` 0 | campaign t1 (vs value-rule responders) | queued |
| opening / setup placement policies | `openings.policy` (4 policies) | campaign t1 (value, vf) / t3 | queued |
| deeper search (all opponents' turns) | `search.depth` 2 | campaign t1 / t3 | queued |
| robber blockability in placement (2 buildings on one hex) | `placement.PLACEMENT_BLOCK_WEIGHT` 0 | campaign t1 (value, vf) / t3 | queued |
| rob the most dangerous player, not only the leader (distance to win) | `danger.danger_multiplier` off, `danger.TURNS_HALF` | campaign t2 | queued |
| block what they need (ports, holdings) | `danger.BLOCK_NEED` 0 | campaign t2 | queued |
| which resource to rob | `danger.steal_factor` off (new flag) | campaign t2 | queued |
| rob to break a near-winner's build | `danger.rob_break` off (new flag) | campaign t2 | queued |
| don't hold what they want to rob (steal exposure) | `heuristic.EXPOSURE_WEIGHT` 0 | campaign t2 (Python evaluator) | queued |
| politics: favour slack, coalitions, don't feed the leader | `politics.MAX_SLACK`, `coalitions.SCALE`, `trading.feed_leader_guard` | campaign t2 (vs value-rule responders) | queued |
| knight value / dev cards | `devcards.KNIGHT_VALUE` | campaign t2 | queued |
| discard / surplus handling | `search.dump_candidates` 0 | campaign t2 | queued |
| late-game trade willingness | `opponent_model.stage_late_drop` | campaign t2 | queued |
| cheaper search (compute worth it?) | `search.beam` 2, `search.expand` 4 | campaign t2 extra | queued |
| Colonist information (card counting) | `--info counted` vs full | campaign `t2_counted_info` (plus proof T7-T9) | queued |
| win-path portfolio with crowding | `search.paths` 1 | campaign `paths_main` + Stages 1-6 below | queued |
| counteroffers | `search.counters` 1 (`--counters` rules) | self-play, 1,200 games | queued |
| analyse the proposer's turn before answering an offer | `search.respond_lookahead` 1 | self-play, 1,200 games, then the league gate | queued |
| does the 6/8 rule matter / break the model | boards with adjacent 6/8 vs without | not built yet (needs a board generator switch) | to build |
| going to ports (4:1 without one) | port weights in placement and static value (new tunables) | campaign vs value and vf | to build (see below) |

Self-play commands:

```
PYTHONPATH=$PWD nice python3 scripts/ablate.py --tunable search.counters --values 1 --counters \
  --base-spec search:depth=1,beam=4,expand=8,evaluator=heuristic --games 1200 --players 4 --seed 11 --workers 3
PYTHONPATH=$PWD nice python3 scripts/ablate.py --tunable search.respond_lookahead --values 1 \
  --base-spec search:depth=1,beam=4,expand=8,evaluator=heuristic --games 1200 --players 4 --seed 12 --workers 3
```

Earlier self-play sweeps (Sweeps 1-2 above: 120 paired games per candidate)
found no significant effect for most tier 2 terms.  At that size the
standard error is about 4.6 points, so small effects cannot be seen.  These
runs use 1,000-2,000 seeds per row, with a standard error of about 1-1.5
points.

## Ports: a measured gap (2026-09-26)

The proof logs show how our bot uses ports:

| | T1 (vs 3x ValueFunction) | T2 (vs 3x AlphaBeta) |
|---|---|---|
| our bot has a port settlement | 59 % of games | 56 % |
| Catanatron's bots have one | 71 % | 68 % |
| our bank / port trades per game | 6.7 | 6.6 |
| Catanatron's, per seat | 5.5 | 6.8 |
| share of our bank trades at 4:1 | 75 % | 78 % |
| share of Catanatron's at 4:1 | 52 % | 60 % |

Our bot settles on ports less often, yet trades with the bank more, mostly
at the worst rate.  About 5 trades a game at 4:1 instead of 3:1 is roughly 5
cards a game, about one build.

The current weights:

| where | generic 3:1 port | 2:1 port |
|---|---|---|
| placement (spot score of typically 12-16) | flat +1.0, whatever we produce | 0.5 + 6 x production of that resource |
| static value (10 points per VP) | flat +0.2 | 0.15 + 4 x production of that resource |

Plan (off by default, then tested):
1. Make the port weights tunables.  The placement score and static_value
   are mirrored in C++, so the candidate arm needs the Python evaluator,
   like `PLACEMENT_BLOCK_WEIGHT`.
2. Add a candidate generic-port value that scales with our total
   production: a 3:1 port converts every surplus resource.
3. Run paired campaign rows vs value and vf, then the league gate.

## Regrouping (user decision, 2026-09-26): diversification is its own area; test the linked pieces together

**Areas, in priority order:**
1. **Trading:** get what we need from the cheapest source.
2. **Diversification / expansion:** grow onto new ground and new resource
   types.  This covers `ports.conversion_cost` (the cost of missing
   resources, weighted by our ports), opening diversity (`pips_diversity`,
   `standin_book`, `RESOURCE_DEMAND`) and expansion pace (a road/settlement
   plan credit).
   - Linked to **Longest Road**: the same roads serve expansion and the road
     race.
3. **Ports:** port access, meaning which port spots are worth a weaker land
   spot, and the roads and blocking race to reach them.
4. **Robber:** block production, steal the right cards, robber persistence
   and knight insurance.
   - Linked to **Largest Army**: the same knights move the robber and win the
     army race.
5. **Card counting:** resource odds, and reading held dev cards.

**The VP-path layer ties them together.**  The win-path portfolio
(`search.paths`, docs/ABLATIONS_WINPATHS.md) values the Longest Road and
Largest Army races by crowding.  Diversification and robber features must be
tested with it, not in isolation.
- Roads: expansion value (resource access) and race value (the Longest Road
  prize) are different things, so they add.  The risk is over-building roads.
- Knights: insurance says hold the knight, the army race says play it.  The
  risk is bad knight timing.

**How linked pieces are tested within the budget: bundle first, then knock
out.**
1. Play the bundle (both pieces on) against the default, e.g.
   `conv=1,paths=1` or `robber_corr=1,paths=1`.
2. If the bundle is clearly good: knock out one piece at a time to see which
   carries the gain, and whether the pair beats each piece alone.
3. If the bundle is unclear at the cap: shelve both.  Do not spend a full
   2x2.
4. Mechanism metrics decide what went wrong in a failed bundle:
   - roads built, Longest Road held, extra settlements, resource types;
   - knights held vs played, Largest Army held, time the robber sits on us.

## Politics rule (user decision, 2026-09-26): inconclusive -> deferred to human testing

**Scope: politics and table-social terms.**
- `politics.MAX_SLACK`, `politics.BASELINE`, `politics.DECAY`;
- `coalitions.*`;
- `trading.feed_leader_guard`;
- `opponent_model.stage_late_drop` and the other trade-willingness terms of
  the opponent model;
- counteroffers (`search.counters`) and the out-of-turn offer analysis
  (`search.respond_lookahead`);
- the unbuilt coalition-splitting trades.

Their value depends on how *people* react.  Catanatron's bots do not react
at all, and self-play copies react exactly as our model predicts.

**Rule.**  Each such term gets its one pre-planned screen of thousands of
games: the campaign row (1,000-2,000 seeds, paired) or the self-play
ablation (1,200 games).
- **Significant** (Holm p < 0.05 within its tier): it continues to
  factorial and joint tuning like any other term.
- **Otherwise it is INCONCLUSIVE:**
  - no more games: no follow-up rows, no factorial pairs, not in the SPSA
    trade group;
  - its default stays exactly as it is (no evidence either way);
  - it goes on the deferred list below, for the human test sessions with
    the advisor.
- Inconclusive means "no clear effect against bots", not "no effect".

The coalition-splitting trade idea is not built until human testing gives
a reason to.

**Deferred to human testing** (filled in as the screens finish):

| term | screen | result | status |
|---|---|---|---|
| (none yet) | | | |

## After screening: interactions and joint tuning (user's point, 2026-09-26)

The terms are not independent.  Examples:
- a robber on our hex hurts less while we hold a knight;
- a leader who trades a little generously to break up a coalition may beat
  refusing to trade at all.

The one-at-a-time rows above measure each term's value *given everything
else at its current setting*.  They say nothing about re-tuning the others
around it.  So the screening is followed by three steps:

1. **Factorial tests for suspected pairs** (2x2 on the same seeds).  They
   report each main effect and the interaction (AB - A - B + base).  An
   interaction needs about 4x the games of a main effect.  First pairs:

   | pair | why |
   |---|---|
   | `devcards.KNIGHT_VALUE` x `heuristic.EXPOSURE_WEIGHT` | knight as robber insurance |
   | `danger.danger_multiplier` x `danger.steal_factor` | who to rob x what to take |
   | `politics.MAX_SLACK` x `coalitions.SCALE` | slack toward a bloc (only if both screened significant, politics rule) |
   | `search.trade_proposals` x `trading.feed_leader_guard` | trading as leader (only if the guard screened significant, politics rule) |

2. **Joint tuning (SPSA)** of groups of weights at once, in self-play and
   against Catanatron (`scripts/tune_joint.py`, docs/TUNING.md: being
   built).  SPSA moves all weights of a group together from paired games,
   which is how game engines tune their evaluation.  Groups:
   - robber: danger, knight, exposure, placement robber terms;
   - trade: margins, slack, coalitions, late drop, counter margin, but only
     the terms that screened significant (politics rule).
3. **Confirmation through the league gate:** a tuned set is a candidate like
   any other and must beat the current champion.

Interactions the code does not model yet (candidates to build, off by
default, then test):
- **Knight as robber insurance.**  A held knight counts a flat 0.7 in the
  static value.  The search already sees that *playing* it frees a blocked
  hex, but the value of *holding* one does not rise with the robber damage
  it can undo.
- **Trading to split a coalition.**  Coalitions currently steer robber
  targets and who we expect to accept our offers.  As the leader, we do not
  yet aim trades at one bloc member to break the bloc.  Deferred to human
  testing (politics rule): not built unless the human sessions show a need.

## Planned: win-path races with crowding (`search.paths`)

`catanbot/winpaths.py` (docs/STRATEGY.md "Win-path races") replaces static_value's permanent-award credit and flat
Longest Road / Largest Army progress credit with a race-aware expected value (win probability over projected
levels, a waste cost for crowded races, a passive floor).  Off by default; switch it on with the tunable
`search.paths` (values `1`) or the spec key `paths=1`; its knobs are `search.paths_w`, `search.paths_crowd`,
`search.paths_priors`, `search.paths_spots` and the `winpaths.*` constants (only with `paths=1` in the base spec).
The staged, pre-registered plan (calibration, shadow diagnostics, E1-E3 main runs, crowding / weight sweeps,
knock-outs incl. a seat-rotated PLACEBO, held-out confirmation), its gates and decision rules and the Stage 0
results are in docs/ABLATIONS_WINPATHS.md.  Nothing beyond Stage 0 runs until the strength proof has finished.
