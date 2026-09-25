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
  `placement.RESOURCE_DEMAND`; `needs_python_evaluator` in the registry) only takes effect on the
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
| `placement.PLACEMENT_BLOCK_WEIGHT` | weight | 1 | 0, 0.5, 2 |  |  | weight of the robber-exposure penalty in settlement scoring |
| `placement.PLACEMENT_ROBBER_Q` | weight | 0.35 | 0, 0.175, 0.7 |  |  | probability scale of the robber landing on a strong hex |
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
