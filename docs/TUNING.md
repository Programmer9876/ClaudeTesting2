# Interactions and joint tuning: factorial tests and SPSA

One-at-a-time ablations (`scripts/ablate.py` in paired self-play,
`scripts/ablate_catanatron.py` / `scripts/campaign.py` in paired games against
Catanatron; docs/ABLATIONS.md) measure what one strategy term is worth *with
every other term at its current setting*.  The terms interact:
- a robber on our hex hurts less while we hold a knight;
- a leader who trades a little generously to split a coalition may beat one
  who refuses to trade at all.

So a screened term can look useless on its own and still matter together
with another one, and a set of weights tuned one at a time is not jointly
tuned.  Two tools follow the screening:

1. **Factorial tests** (2x2, or 2^k up to 4 factors) of a *suspected pair*:
   every combination on the same seeds; each main effect and each
   interaction with a paired 95 % interval.
2. **Joint tuning with SPSA** of a *group* of numeric weights at once, the
   way game engines tune their evaluation weights.

Neither tool changes a default.  A factorial result or a tuned weight set is
a *candidate*, and it becomes the default only through the champion league
gate (docs/LEAGUE.md).

| tool | what | code |
|---|---|---|
| `scripts/ablate.py --factorial` | 2^k factorial, paired self-play (2 seats vs 2) | `catanbot/factorial.py` (design, arithmetic) |
| `scripts/factorial_catanatron.py` | 2^k factorial, 1 seat vs 3 Catanatron bots | ablate_catanatron's store / pool / resume, unchanged |
| `scripts/tune_joint.py` | SPSA over numeric registry tunables, self-play or vs Catanatron | ParamBot seats; `tune=` bot spec for the gate |

Tests: `tests/test_factorial.py`, `tests/test_tune_joint.py`.

## Ground rules (user decisions, 2026-09-26)

**Budget.**  Every test stays within a few thousand games:
- about **2,000 paired seeds** against Catanatron (4,000 games);
- about **2,400 games** in self-play.

Anything that would need tens of thousands of games is shelved.
- `tune_joint.py` needs `--max-games` for every new run.  The number of
  iterations follows from it, the planned total is printed before the first
  game, and no iteration starts past the cap.
- The factorial scripts print the planned total before playing.  Size them
  from the tables below.
- 2x2 factorials only in the **priority areas**: trading, ports, robber and
  card counting.

**Politics rule** (docs/ABLATIONS.md, "Politics rule").  The politics and
table-social terms enter a factorial or an SPSA run only if their one
pre-planned screen was significant (Holm p < 0.05).  Otherwise they are
INCONCLUSIVE and deferred to human testing.  The terms:
- `politics.MAX_SLACK`, `politics.BASELINE`, `politics.DECAY`;
- `coalitions.*`;
- `trading.feed_leader_guard`;
- `opponent_model.stage_late_drop`;
- `search.counters`, `search.counter_margin`, `search.respond_lookahead`.

The tools mark these terms (`[politics rule]` in `--list-groups`, a
`POLITICS RULE` note in `--plan`) but do not look the screens up.  Checking
is the operator's job: restrict a group with `--exclude`, or replace it with
`--params`.

## Factorial tests

### Design

Each factor is a registry tunable at a *test value*; its other level is its
registry default.  A flag given without a value is tested switched off.
Cells are the 2^k combinations: "base" (every factor at its default), "A",
"B", "A + B", ...  Every cell is played on the **same seeds**, so board and
dice luck cancels in the paired contrasts.

For an effect E (a set of factors), the per-unit contrast (a unit is a game
or a seed) is

    c_u(E) = 2^-(k-|E|) * sum over cells S of  prod_{i in E} (+1 if i in S else -1) * y_S(u)

- **Main effect** of A: "A on minus A off", averaged over the levels of the
  other factors.  In a 2x2: `((A + AB) - (base + B)) / 2`.
- **Two-factor interaction:** `AB - A - B + base`, the difference of
  differences.  It says how much more A is worth when B is on.  In a 2^3 it
  is averaged over the third factor.
- **Cells vs base:** each single-factor cell is exactly the one-at-a-time
  ablation on these seeds.  The *additivity* line compares "A + B" with the
  sum of the single-factor cells.
- **Paired s.e.:** the unbiased sample variance of `c_u` over units, divided
  by n.  95 % CI = mean +- 1.96 s.e.
- A reading needs >= 30 units; smaller runs print "inconclusive".
- An interaction sums four cells.  Its s.e. is about twice a main effect's,
  so it needs about **4x the games** for the same precision.

Per-unit values `y`:
- **Self-play (`ablate.py --factorial`):** each non-base cell plays 2 seats
  against 2 base seats.
  - Same seed formula and seat rotation as `ablate.py --tunable`, so cell "A"
    reproduces an `ablate.py --tunable A` run with the same `--seed`.
  - `y` = the paired per-seat win-rate difference of the game (+-1/2).  The
    final-VP difference is reported too.
  - The base cell is 0 by construction and is not played.
- **Against Catanatron (`factorial_catanatron.py`):** every cell is a
  candidate arm of `ablate_catanatron.py` against **one shared base arm**.
  - Each seed is played once per cell and once by the base: same board,
    dice and seat (`s % 4`).
  - `y` = win (0/1) and final VP of catanbot's seat.
  - Everything that plays and stores games is ablate_catanatron's, unchanged.
    That covers the append-only JSONL, `PYTHONHASHSEED=0`, crash isolation,
    `--game-timeout`, resume by rerunning, and `--reuse` of base games that
    a campaign already played with the same arm key and code.
  - The design is stored in each run line's `value`, so `--report`
    recomputes everything from the JSONL alone.
  - `ablate_catanatron.py --report` on the same file shows each cell as an
    ordinary candidate-vs-base row.

### Cost and precision at the budget (2x2)

The standard errors below are conservative.  They treat the cells as
independent; the common seeds make them positively correlated, which
lowers the contrasts' s.e.

| design | games | per cell | s.e. main effect | s.e. interaction | detectable at 80 % power (main / interaction) |
|---|---|---|---|---|---|
| self-play 2x2 | 2,400 | 800 games | ~1.5 pp | ~3.1 pp | ~4 pp / ~9 pp |
| vs Catanatron 2x2, base games reused | ~3,900 new | 1,300 seeds | ~1.2 pp | ~2.4 pp | ~3.4 pp / ~7 pp |
| vs Catanatron 2x2, base played | 4,000 | 1,000 seeds | ~1.4 pp | ~2.7 pp | ~3.8 pp / ~7.7 pp |

Units: self-play = per-seat win rate (a seat's share of the table); vs
Catanatron = catanbot's win rate with p ~ 0.25.

Within the budget a 2x2 finds only **large** interactions (8-9 pp, about a
third of a seat's baseline win rate).  A "not significant" interaction at
this size rules out only large effects.

### Commands for the first pairs

The base spec is the shipped bot, `search:depth=1,beam=4,expand=8,evaluator=heuristic`.
In self-play it has to be given explicitly: `ablate.py` otherwise uses the
cheap heuristic bot for terms the heuristic bot also reads.

```bash
cd /home/user/ClaudeTesting2 && export PYTHONPATH=$PWD
SPEC=search:depth=1,beam=4,expand=8,evaluator=heuristic

# 1. knight as robber insurance (robber area)
#    Static-evaluator weight: re-executes with CATANBOT_NO_ACCEL=1 (~5x slower in self-play, see Cost).
nice python3 scripts/ablate.py --factorial devcards.KNIGHT_VALUE=0.8,heuristic.EXPOSURE_WEIGHT=0 \
    --base-spec $SPEC --games 800 --workers 3 --seed 21 --json runs/fx/knight_x_exposure_sp.json
nice /home/user/venv_cat33/bin/python scripts/factorial_catanatron.py \
    --factorial devcards.KNIGHT_VALUE=0.8,heuristic.EXPOSURE_WEIGHT=0 --opponent value \
    --seeds 1300 --workers 3 --out runs/fx/knight_x_exposure_value.jsonl --reuse 'runs/campaign1/*.jsonl'

# 2. who to rob x what to take (robber area)
nice python3 scripts/ablate.py --factorial danger.danger_multiplier=off,danger.steal_factor=off \
    --base-spec $SPEC --games 800 --workers 3 --seed 22 --json runs/fx/dm_x_sf_sp.json
nice /home/user/venv_cat33/bin/python scripts/factorial_catanatron.py \
    --factorial danger.danger_multiplier=off,danger.steal_factor=off --opponent value \
    --seeds 1300 --workers 3 --out runs/fx/dm_x_sf_value.jsonl --reuse 'runs/campaign1/*.jsonl'

# 3. slack toward a bloc (trading area) - ONLY if both terms screened significant (politics rule)
nice python3 scripts/ablate.py --factorial politics.MAX_SLACK=0,coalitions.SCALE=1 \
    --base-spec $SPEC --games 800 --workers 3 --seed 23 --json runs/fx/slack_x_scale_sp.json

# 4. trading as leader (trading area) - ONLY if the guard screened significant (politics rule)
nice python3 scripts/ablate.py --factorial search.trade_proposals=0,trading.feed_leader_guard=off \
    --base-spec $SPEC --games 800 --workers 3 --seed 24 --json runs/fx/trades_x_guard_sp.json
nice /home/user/venv_cat33/bin/python scripts/factorial_catanatron.py \
    --factorial search.trade_proposals=0,trading.feed_leader_guard=off --opponent value --trades value \
    --seeds 1300 --workers 3 --out runs/fx/trades_x_guard_value.jsonl --reuse 'runs/campaign1/*.jsonl'

# statistics again from the files alone
python3 scripts/factorial_catanatron.py --report --out runs/fx/dm_x_sf_value.jsonl
```

Notes on the pairs:
- **Pairs 1 and 2**: 800 self-play games per cell = 2,400 games; 1,300 seeds
  vs Catanatron = 3,900 new games, with the base games reused from the
  campaign.
- **Pair 1, reuse:** `--reuse` finds base games only with the same arm key,
  and the evaluator mode is part of the key.  With the Python evaluator the
  matching base games are those of the campaign's `EXPOSURE_WEIGHT` row
  (also run with `CATANBOT_NO_ACCEL=1`).
- **Pair 3:** politics terms are not measurable against Catanatron: its bots
  do not react to coalitions or favours.  Self-play only, if at all.
- **Pair 4:** trade proposals only act against Catanatron with domestic
  trading on (`--trades value`, catanatron 3.3).
- **Pair 4, reuse:** base games are reused only from experiments with the
  same `--trades` mode.
- **A 2^3** (e.g. adding `danger.rob_break=off` to pair 2) costs 7 cells in
  self-play: at 2,400 games that is ~340 games per cell.  Only for effects
  far larger than the table above.

### Acting on a result

- **A significant interaction** is a reason to re-tune the pair jointly
  (SPSA below).  It can also motivate a new strategy term: the knight as
  robber insurance is a candidate in docs/ABLATIONS.md.
- **A cell that beats the base** (e.g. "A + B") is not adopted from the
  factorial.  It is a candidate for the league gate, as a spec with
  `tune=` (see below).

## Joint tuning with SPSA (`scripts/tune_joint.py`)

### How it works

Simultaneous perturbation stochastic approximation (Spall 1992) estimates a
gradient in n dimensions from **two** noisy measurements, whatever n is.
Each iteration plays one batch of paired games.

- **Normalised units.**  Each parameter `x` is tuned as `u = (x - default) / scale`.
  - `scale` is the span of the registry's default and candidate values
    (`catanbot/tuning.py`).
  - The default bounds are that span, so `u` covers an interval of width 1
    around 0.
  - The screened range is where the evidence is; values beyond it were never
    tested.  `--bounds NAME=LO:HI`, `--scale NAME=X` and `--start NAME=V`
    change these.
- **Iteration k:**
  - `Delta` = +-1 per parameter (Rademacher).  It comes from
    `random.Random("spsa:<seed>:<k>")`, so it is reproducible.
  - `theta+- = clip(theta +- c_k Delta)`.  An integer knob is perturbed by
    at least half a unit, so its two rounded values always differ.
  - A batch of paired games theta+ vs theta- gives `D`, the estimate of
    f(theta+) - f(theta-).  f is the per-seat win probability
    (`--objective vp`: final VP / 10, less noisy but not the target).
  - Gradient estimate: `g_i = D / (u+_i - u-_i)`, using the *installed*
    values after clipping and rounding.  For an unclipped real parameter
    that is `D / (2 c_k Delta_i)`.
  - Update: `theta += a_k g`.  Each coordinate's step is capped at
    `--max-step` (0.1), then clipped to the bounds.
- **Gains:** `a_k = a / (A + k + 1)^0.602`, `c_k = c / (k + 1)^0.101`
  (Spall's exponents, `--alpha`, `--gamma`).
  - `A` = 10 % of the planned iterations (`--A`).
  - `c` = 0.25 (`--c`): theta+ and theta- differ by half the screened span
    at k = 0.
  - `a` is automatic unless given (`--a`).  It is set so that a
    one-standard-error estimate of D moves a parameter by `--first-step` =
    0.05 of its span at k = 0:
    `a = first_step (A + 1)^0.602 * 2c sqrt(games) / sigma`, where sigma =
    0.5 is the rough s.d. of one game's D.
  - `--plan` prints all of these, plus the noise per iteration.
- **Output: the tuned vector** is the average of the iterates over the
  second half of the run (Polyak-Ruppert averaging; less noisy than the
  last iterate, which is also printed).  Values are rounded to 4
  significant digits, far below the tuning noise.
- **Game modes:**
  - `--mode selfplay` (default): 4-player games, 2 seats theta+ vs 2 seats
    theta-.  Both sides are `ParamBot`s around the base spec.  Each seat
    installs its own constants only inside its own decision hooks, so the
    two sides never see each other's values; a test checks this in a real
    game.  Seat patterns rotate over the 6 arrangements.  The mirrored
    arrangements (CCDD / DDCC, CDCD / DCDC, CDDC / DCCD) share one seed,
    i.e. one board and dice; a pair whose two parameter sets make the same
    decisions cancels exactly.
  - `--mode catanatron --opponent value`: theta+ and theta- each play 1
    seat vs 3 Catanatron bots on the same seeds and seat, through
    ablate_catanatron's game code (`play_job`).  3.3 presets need
    `/home/user/venv_cat33/bin/python`.  `--trades`, `--opponent-params`.
  - `--games`: 2v2 games per iteration (self-play, default 36 = 6 rotations
    x 3 boards) or seeds per iteration (vs Catanatron, default 48 = 96
    games).
- **Seeds, determinism, files:**
  - Seeds are fresh every iteration and disjoint from the screening and
    league seeds.
  - The process pins `PYTHONHASHSEED=0` and re-executes itself to set it.
    It also sets `CATANBOT_NO_ACCEL=1` when a parameter is read by the
    Python static evaluator.
  - A run is therefore a deterministic function of its state file.
  - After every iteration the state JSON is rewritten atomically (temp file,
    fsync, rename).  It holds theta, k, and per iteration Delta, the
    installed theta+-, D +- s.e., the gradient estimate, the step, seeds and
    the code fingerprint.  Games played, the environment and the current
    recommendation are kept too.
  - The iteration's game records go to `<state>.games.jsonl`, including
    error records.  A game that raises or exceeds `--game-timeout` (1800 s)
    is an error record and is left out of D.  An iteration where every game
    failed stops the run without an update.
  - `--resume` continues from the last finished iteration, and the
    trajectory is identical to an uninterrupted run (tested with a kill in
    the middle of an iteration).  Configuration changes on resume are
    ignored, except `--iterations` / `--max-games` (raise explicitly),
    `--workers`, `--max-minutes` and `--game-timeout`.
  - A change of code, evaluator or interpreter since the start is warned
    about: from there on the run optimises another objective.
- **Parallelism:** `--workers N` runs a fork pool per batch.  A worker that
  dies is replayed alone in a fresh process; a second death is recorded as
  an error.  Results are the same for any N.

Only numeric tunables are tuned.  Flags, on/off integer knobs
(`search.counters`, `search.paths`, ...), `search.depth`, resource vectors
and `openings.policy` are refused, with a pointer to the factorial tool.

### Groups

```bash
python3 scripts/tune_joint.py --list-groups
```

| group | parameters | setting |
|---|---|---|
| `robber` | `danger.TURNS_HALF`, `danger.BLOCK_NEED`, `danger.BLOCK_FLOOR`, `devcards.KNIGHT_VALUE`, `heuristic.EXPOSURE_WEIGHT`, `placement.PLACEMENT_ROBBER_Q` | search bot; Python evaluator (EXPOSURE_WEIGHT, PLACEMENT_ROBBER_Q) |
| `trade` | `trading.accept_margin`, `politics.MAX_SLACK`, `politics.BASELINE`, `coalitions.SCALE`, `opponent_model.stage_late_drop`, `search.counter_margin` | **politics rule**; counter_margin adds `counter=1` and the counter-offer rules |

The robber group:
- It needs the Python evaluator: ~5x slower in self-play, ~3x against
  Catanatron (see Cost).
- `--exclude heuristic.EXPOSURE_WEIGHT,placement.PLACEMENT_ROBBER_Q` gives
  four parameters on the C++ evaluator.
- A tuned static-evaluator weight can only be adopted as a default together
  with its `constexpr` copy in `cpp/heuristic.cpp`.  `--status` says so.

The trade group:
- Only `trading.accept_margin` is outside the politics rule.  Every other
  member may be tuned only if its screen was significant.
- Restrict it, e.g.
  `--group trade --exclude politics.BASELINE,coalitions.SCALE,opponent_model.stage_late_drop,search.counter_margin`,
  or `--params trading.accept_margin,politics.MAX_SLACK`.
- `search.counter_margin` only acts with `counter=1` and the counter-offer
  rules.  The tuner adds both, unless `--base-spec` is given, where it
  refuses instead.  The whole run then plays the counter-offer variant,
  not the base game; exclude it to tune for the base game.
- Against Catanatron the trade terms need `--trades value` (3.3), and the
  politics terms cannot be measured there at all.

`--params A,B,...` replaces a group; `--exclude` removes members.  A
parameter that cannot act in the chosen setting is an error rather than a
silent random walk:
- a search knob with a heuristic spec;
- a depth-2 knob at depth 1;
- a win-path constant without `paths=1`;
- a counter knob against Catanatron.

### Budget, noise and what SPSA can find

One iteration's `D` has s.e. ~0.5 / sqrt(games): 8.3 pp at 36 self-play
games, 7.2 pp at 48 Catanatron seeds.  That is far more than any single
weight is worth, so SPSA moves in many small, noisy steps.  With a few
thousand games it can only find an edge that is large in total.

Synthetic calibration (tests/test_tune_joint.py style mock runner):
- the robber group's 6 parameters, optimum placed 0.15-0.3 span units from
  the defaults;
- 12 seeds per row;
- defaults (c 0.25, first step 0.05);
- "edge" = per-seat win-rate gain of the optimum over the defaults.

| budget | edge 2 pp: recovered (worse than default) | edge 5 pp: recovered (worse than default) |
|---|---|---|
| 2,400 games = 66 x 36 | +0.6 pp on average (4 of 12 runs worse, down to -1.2 pp) | +3.8 pp (0 of 12; min +2.1 pp) |
| (for comparison, 8 seeds) 7,200 games = 200 x 36, c 0.15 | +0.5 pp | +3.9 pp |

The same grid chose the defaults: at 2,400 games c 0.25-0.3 beat c 0.15
(+0.6 / +3.8 pp vs +0.0 / +2.8 pp), and first step 0.05 beat 0.1.

Reading:
- At the budget, SPSA turns a **large** joint edge (~5 pp per seat) into
  a clearly better weight set.  That is what a coordinated re-tune of a
  whole group could be worth.
- A **small** total edge (~2 pp) is within its noise.  A third of such
  runs end *worse* than the defaults.
- This is exactly why the result goes to the league gate and never
  straight to the defaults.
- The gate's detectable edge (~55 % of decisive 2v2 games, docs/LEAGUE.md)
  is itself ~5 pp of side share, so both instruments see the same size of
  effect.
- **A parameter nothing depends on drifts** as far as one that matters.
  In a synthetic robber run with three planted and three flat parameters,
  `BLOCK_NEED` (flat) ended 0.23 span units off its default.  Two
  per-parameter statistics were tried at 2,400 games; neither separates
  planted from flat parameters:
  - displacement / the random-walk s.d. of the steps (12 seeds x edges
    2 and 5 pp): mean |z| 0.67-0.78 for planted, 0.61-0.68 for flat
    parameters, never above 2;
  - mean gradient estimate over the run / its s.e. (the run above): |z| <=
    0.85 for the planted parameters, up to 1.24 for the flat ones.

  So the tuner reports no per-parameter significance.  The vector is
  judged as a whole, by the gate.  Put only parameters with a reason in a
  run: screened significant, or part of an interaction.

Budgets:
- **Self-play:** `--max-games 2400` = 66 iterations x 36 games = 2,376
  games.
- **Against Catanatron:** `--max-games 4000` = 41 iterations x 48 seeds x 2
  games = 3,936 games.
- Raising `--games` per iteration (fewer, quieter iterations) or lowering
  it (more, noisier ones) matters less than the total at this size.  Keep
  it a multiple of 6 (self-play) or of 4 (Catanatron, balances our seat).

### Cost

Measured 2026-09-26: one `tune_joint.py` iteration per row, one worker
under `nice -n 10`, while the proof benchmark kept the other 3 cores busy
(load ~4).  Samples are tiny (6-8 games) and game length varies, so read
the rows as +-30 %.

| setting | measured | per game | one iteration (default size) at 3 workers | whole budget at 3 workers |
|---|---|---|---|---|
| self-play, robber (6 params, Python evaluator) | 6 games 375 s | 62 s | 36 games: ~12.5 min | 2,376 games: ~14 h |
| self-play, robber minus EXPOSURE_WEIGHT / PLACEMENT_ROBBER_Q (C++) | 6 games 69 s | 11.5 s | ~2.3 min | ~2.5 h |
| vs Catanatron `value` 3.3, robber (Python evaluator) | 4 seeds = 8 games 29 s | 3.6 s | 48 seeds = 96 games: ~2 min | 3,936 games: ~1.3 h |
| vs Catanatron `value` 3.3, robber minus the two (C++) | 8 games 9 s | 1.1 s | ~0.6 min | ~0.4 h |

How to read the cost:
- **Self-play runs 4 search bots per game** and is ~10x the cost of a
  game against Catanatron, where only one seat is ours.  At ~11.5 s per
  game, 3 workers play ~1,000 self-play games per hour.  The ~6,000
  games/hour quoted for 3 cores is the Catanatron-mode rate: 1.1-1.7 s
  per game, and docs/ABLATIONS.md measured 6,110/h for `value` at 3
  workers.
- **The Python evaluator** costs ~5x in self-play here (all four seats
  run it) and ~3x against Catanatron.
- **The full robber group in self-play** is the one expensive run (~14 h).
  Options: the 4-parameter C++ variant (~2.5 h), the Catanatron mode
  (~1.3 h), or run it overnight in `--max-minutes` chunks with
  `--resume`.
- **Factorials in self-play** cost the same per game as SPSA self-play in
  the same setting.  2,400 games of pair 1 (Python evaluator) is ~14 h;
  pair 2 (C++) ~2.5 h.
- **`--plan`** prints the planned games; multiply by the per-game time of
  the matching row.

### Commands

```bash
cd /home/user/ClaudeTesting2 && export PYTHONPATH=$PWD

# robber group, self-play, 2,400 games (66 iterations x 36), 3 workers
python3 scripts/tune_joint.py --group robber --max-games 2400 --state runs/tune/robber_sp.json --plan
nice python3 scripts/tune_joint.py --group robber --max-games 2400 --state runs/tune/robber_sp.json --workers 3
nice python3 scripts/tune_joint.py --state runs/tune/robber_sp.json --resume --workers 3     # after a stop
python3 scripts/tune_joint.py --state runs/tune/robber_sp.json --status                     # anytime

# robber group against Catanatron's value bots (3.3), 4,000 games (41 x 48 seeds x 2)
nice /home/user/venv_cat33/bin/python scripts/tune_joint.py --group robber --mode catanatron \
    --opponent value --max-games 4000 --state runs/tune/robber_value.json --workers 3

# trade group, restricted by the politics rule (here: only accept_margin and MAX_SLACK screened significant)
nice python3 scripts/tune_joint.py --group trade --params trading.accept_margin,politics.MAX_SLACK \
    --max-games 2400 --state runs/tune/trade_sp.json --workers 3
# ... or by exclusion; keeping search.counter_margin plays every game under the counter-offer rules
nice python3 scripts/tune_joint.py --group trade \
    --exclude politics.BASELINE,coalitions.SCALE,opponent_model.stage_late_drop,search.counter_margin \
    --max-games 2400 --state runs/tune/trade_sp.json --workers 3

# bounded chunks: stop starting iterations after 60 minutes, resume later
nice python3 scripts/tune_joint.py --state runs/tune/robber_sp.json --resume --workers 3 --max-minutes 60
```

### From a tuned vector to the league gate

`--status` (also printed at the end of every run) shows:
- the parameter table (default, bounds, scale, start, last, average);
- the trajectory table, one row per iteration or sampled for long runs;
- the tuned vector as a ParamBot override dict and as a bot spec;
- the gate command.

Excerpt of a **synthetic** robber run (`--fake-optimum`, planted optimum
TURNS_HALF 3.9, KNIGHT_VALUE 0.65, EXPOSURE_WEIGHT 0.15, the other three
flat; not a result):

```
  parameter                            default            bounds    scale     start      last   average
  danger.TURNS_HALF                          3          [1.5, 6]      4.5         3      3.78     3.588
  danger.BLOCK_NEED                          2            [0, 4]        4         2     3.178      2.92
  ...
     k     a_k    c_k  ok/err   D (pp)   +-se     TURNS_HALF     BLOCK_NEED    BLOCK_FLOOR   KNIGHT_VALUE ...
     0  0.3000  0.250   36/0      -8.3    8.3          3.225            1.8          0.385          0.575 ...
    33  0.1122  0.175   36/0      +5.6    8.4          4.042          2.874         0.4336          0.676 ...
    65  0.0793  0.164   36/0     -13.9    8.1           3.78          3.178          0.277         0.6762 ...
  tuned vector (average of the last 33 iterates) as ParamBot overrides (parameters that moved off their default):
    {"danger.TURNS_HALF": 3.588, "danger.BLOCK_NEED": 2.92, "danger.BLOCK_FLOOR": 0.3452, "devcards.KNIGHT_VALUE": 0.7003, ...}
  bot spec: search:depth=1,beam=4,expand=8,evaluator=heuristic,tune=danger.TURNS_HALF:3.588;danger.BLOCK_NEED:2.92;...
    PYTHONPATH=$PWD python3 scripts/league.py gate --candidate-commit <sha of the commit that ran the tuner> \
        --candidate-spec 'search:depth=1,beam=4,expand=8,evaluator=heuristic,tune=danger.TURNS_HALF:3.588;...' \
        --env CATANBOT_NO_ACCEL=1 --allow-no-accel \
        --notes 'SPSA robber (6 params), 66 iterations x 36 games (selfplay), state runs/tune/robber_sp.json'
```

The flat `BLOCK_NEED` drifted from 2 to 2.92, as described above.

- **`tune=` spec key** (`catanbot/agents/param_bot.py`, `selfplay.make_bot`):
  - `tune=NAME:VALUE;NAME:VALUE` wraps any bot in a `ParamBot` with those
    registry overrides.
  - Search knobs that have a spec key become that key (`trades=`,
    `counter_margin=`, ...).
  - A spec without the key builds exactly the bot it always did.
  - So a tuned set is an ordinary spec for the league's bot servers, for
    `bench_catanatron.py --spec` and for the gate's external hook.
  - The gate materialises the candidate commit, which must contain this
    `make_bot` support.
- **`--env CATANBOT_NO_ACCEL=1 --allow-no-accel`** is added when a tuned
  static-evaluator weight moved.  The candidate's servers then run the
  Python evaluator; the champions keep the C++ one.
- **`--formats 4p2v2,3p1v2`** is added for trading terms (docs/LEAGUE.md:
  in 2v2 a trading change can trade with its own copy).
- **Pre-registration:** commit the code first and gate once.  Add the
  external Catanatron hook (`--external-key catanatron --external-cmd ...`,
  docs/LEAGUE.md) for weights tuned in self-play.
- **Tuned vs Catanatron:** a set tuned against Catanatron still has to pass
  the self-play gate.
- **Seeds:** the tuner's seeds are disjoint from the gate's, so the gate is
  a fresh confirmation.  Do not re-tune on gate outcomes.

## Recommended protocol

1. **Screen** every term one at a time (campaign / self-play rows,
   docs/ABLATIONS.md).  Apply the politics rule to the results.
2. **Factorial** only for suspected pairs in the priority areas (trading,
   ports, robber, card counting), within the budget above.  Read the
   interaction's interval, not only its sign.
3. **SPSA** for a group whose members screened significant, or that a
   factorial showed to interact.  Use one run within the budget; `--plan`
   first.
4. **League gate** for the tuned vector, once.  On FAIL the defaults stay.
   A new attempt is a new pre-registered run, not a re-tune on the same
   evidence.

## Limitations

- **Self-play objective.**  In self-play SPSA optimises strength *between
  nearby variants of itself*, not against the champion or people.  Moves
  that only exploit a copy of itself are possible.  That is why the gate
  and the external hook both apply.
- **Catanatron objective.**  Against Catanatron, the objective is a fixed
  opponent pool that never trades or reacts politically.
- **Normalisation.**  Scales come from the registry's candidate spans, which
  are wide for some knobs (`coalitions.BLOC_THRESHOLD` spans 0.5-100).
  Give `--scale` / `--bounds` for such parameters.  Tuning is linear in raw
  units (no log scale).
- **Bounds.**  The default bounds are the screened range.  An optimum
  outside it is reported as a parameter stuck at its bound (visible in the
  trajectory); widen the bounds only deliberately.
- **Integer knobs.**  Perturbed by at least one unit after rounding.  Their
  gradient is the secant over that unit.
- **Compute knobs.**  SPSA sees strength only, not cost: `search.beam`,
  `search.expand` or `opp_roll_samples` would drift toward more compute.
  Keep them out, or judge the result on value per ms (ablate.py).
- **Factorial scope.**  Up to 4 factors, full designs only (no fractional
  designs).
- **Correlation.**  The CIs are paired per unit.  They do not model the
  mild correlation between the two games of a mirrored self-play pair (the
  same holds for `ablate.py`).
- **Factorials and the campaign runner.**  `factorial_catanatron.py`
  factorials are not yet a campaign row type: `scripts/campaign.py` expects
  a tunable or two specs.  Run them directly; their base arm reuses campaign
  games through `--reuse`.
