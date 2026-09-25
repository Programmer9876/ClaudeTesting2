# Win-path races with crowding: pre-registered experiments

Feature: `catanbot/winpaths.py` (model: docs/STRATEGY.md "Win-path races"; search hooks: docs/DESIGN.md
section 13).  **Off by default**; nothing here changes the default bot.  Status: implemented and unit-tested,
Stage 0 done; Stages 1-6 are pre-registered below and must wait until the strength proof has finished and
freed its three cores.

## How to switch it on

| tool | how |
|---|---|
| bot spec (`make_bot`, adapter `CatanbotPlayer(spec=...)`, `--cand-spec`, campaign `cand_spec`) | `search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=1` (full: `...,paths=1,paths_w=1.0,paths_crowd=1.0,paths_priors=1,paths_spots=0`) |
| registry tunable (ablate.py, ablate_catanatron.py, campaign `tunable`, `ParamBot` overrides) | `search.paths` = 1 (default 0); also `search.paths_w`, `search.paths_crowd`, `search.paths_priors`, `search.paths_spots` |
| model constants | `winpaths.KAPPA`, `BETA`, `H_B`, `HAND_W`, `ESC`, `TIE_LR`, `LIVE_LR`, `LIVE_LA`, `PRIOR_SCALE`, `PLACEBO` - only with `paths=1` in the base / candidate spec (`--cand-set winpaths.KAPPA=0.25`, or `--base-spec "...,paths=1" --tunable winpaths.BETA`) |
| CLI | `python -m catanbot recommend ... --paths 1` (the "Win paths" advice section is printed either way) |

```bash
# self-play paired ablation (2 vs 2 seat patterns)
python3 scripts/ablate.py --base-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --tunable search.paths --values 1 \
    --players 4 --games 2000 --workers 3 --seed 1 --json data/ablations/paths_selfplay.json
# a model constant, both arms with the term on
python3 scripts/ablate.py --base-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=1" --tunable winpaths.KAPPA \
    --values 0.25,1.0 --players 4 --games 2000 --workers 3 --seed 1
# vs Catanatron (1 seat vs 3), resumable JSONL
$PY33 scripts/ablate_catanatron.py --tunable search.paths --values 1 --opponent value --seeds 2000 --workers 3 \
    --out runs/paths_main@value.jsonl
$PY33 scripts/ablate_catanatron.py --cand-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=1,paths_crowd=0" \
    --def-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --opponent value --seeds 2000 --workers 3 \
    --out runs/paths_crowd0@value.jsonl
$PY33 scripts/ablate_catanatron.py --cand-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=1" \
    --cand-set winpaths.KAPPA=0.25 --def-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --opponent value \
    --seeds 2000 --workers 3 --out runs/paths_kappa025@value.jsonl
```

## Seed rules

* Never the proof seeds (900001, 900101) or the proof smoke seeds (424201, 424301).
* Exploration: seed base 0 (the campaign's 1v3 seat rotation, seat = s % 4).  Confirmation: held-out base 100000.
* Always `PYTHONHASHSEED=0`, via `scripts/campaign.py`.
* Code fingerprint: the new module and the edits change it, so no pre-change default arm can be reused.  E1's
  default arm is the new reference; later `cand_spec` runs reuse it through the same default-arm key.

## Stage 0 - done (1 process, `nice -n 10`, 2026-09-25, 3 of 4 cores busy with the proof)

* `pytest tests/test_winpaths.py tests/test_search.py tests/test_ablate.py tests/test_selfplay.py tests/test_cli.py`:
  all pass (30 + 58 tests).  The action-hash test (`test_default_unchanged_six_games_and_thirty_searches`) is
  the proof that the default is unchanged: 6 seeded 4-seat games of the default spec and 30 root searches are
  identical in a fresh interpreter that never imported `catanbot.winpaths` and in one that did (default spec,
  `paths=0` spelled out, ParamBot with overridden winpaths constants).  One-off extra check: three 60-70-turn
  default games give identical action hashes under the frozen proof snapshot and the edited tree.
* Smoke: `scripts/ablate.py --base-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --tunable search.paths
  --values 1 --games 1 --workers 1 --players 4`: no crash; candidate 18.6 ms vs default 10.4 ms per decision in that
  one game (1.78x; the two sides face different positions, so a single game is a noisy cost estimate).
* Paired cost on identical positions (depth 1, beam 4, expand 8, process time, best of 3):

  | measurement | default | paths=1 | ratio |
  |---|---|---|---|
  | 30 mid-game positions (turns 20-81 of 3 heuristic games), mean ms / decision | 21.6 | 27.2 | 1.26x (p95 1.27x) |
  | same, awards only (`paths_priors=0`) | 21.6 | 26.8 | 1.24x |
  | same, with `paths_spots=1` | 21.6 | 33.8 | 1.56x (p95 1.49x) |
  | `winpaths_shadow.py --games 1` (281 main-phase roots of a default game) | 16.6 | 19.6 | 1.18x (p95 1.25x) |

  Per leaf: 11 us (batched C++ heuristic) vs 37 us (C++ static values + the correction); the context costs
  ~0.12 ms per decision; race-solve memo hit rate 96 %.  Budget targets (<= 1.35x awards + priors, <= 1.6x with
  spots, hard limit 2.0x) are met.  Independent review re-measurement after the review fixes (30 other positions,
  turns 16-110 of 3 heuristic games, best of 3): default 18.7 ms, `paths=1` 23.3 ms (1.25x, p95 1.46-1.51x),
  `paths_priors=0` 1.23x, `paths_spots=1` 30.7 ms (1.64x, p95 2.1x - slightly over the 1.6x spots target, under
  the 2.0x hard limit); single positions reach 2.5x because the term changes the tree, not the per-leaf cost.  The shadow game changed 11.4 % of the first actions (inside the Stage 2 band).

## Stage 1 - calibration (1 process, ~1-2 min)

```bash
python3 scripts/winpaths_calibrate.py --bot heuristic:temp=0.15 --games 80 --seed 7000
python3 scripts/winpaths_calibrate.py --bot "search:depth=1,beam=4,expand=8,evaluator=heuristic" --games 20 --seed 7100
```

Gates: on both bot types the Longest Road and the Largest Army log-loss of the model's P beats both baselines
("the holder keeps it with 0.85" and a level-only softmax) by at least 0.1 nats; the model horizon's MAE is at
most 3.5 rounds.  If the search-bot numbers miss: refit only `H_B`, `BETA` and `TIE_*` on seeds 7100-7199, then
freeze every constant and write the frozen values into this file before any strength game.  (A 2-game smoke of
the script ran for testing only; its numbers are not evidence.)

## Stage 2 - shadow diagnostics (1 process, ~5 min)

```bash
python3 scripts/winpaths_shadow.py --games 4 --seed 7200
```

Gates: 3-20 % of the main-phase first actions change (below 2 %: E1 uses `paths_w=2`; above 30 %: `paths_w=0.5`);
ms ratio at most 1.5 mean and 2.0 at p95.

## Stage 3 - main strength (3 workers)

Campaign entries (add to `scripts/campaign_plan.json` with `"enabled": false` until the proof ends; not added
yet because the plan file is outside this change):

```json
{"name": "paths_main@value", "priority": 60, "interpreter": "py330", "opponent": "value",
 "tunable": "search.paths", "values": [1], "seeds": {"count": 2000, "base": 0}, "enabled": false,
 "notes": "E1 win-path races: Catanatron 3.3 ValueFunctionPlayer, 1v3, ~0.8 h; its default arm is the new reference"},
{"name": "paths_main@vf", "priority": 61, "interpreter": "py321", "opponent": "vf",
 "tunable": "search.paths", "values": [1], "seeds": {"count": 2000, "base": 0}, "enabled": false,
 "notes": "E2 win-path races vs the 3.2.1 stand-in"}
```

E3 self-play: `scripts/ablate.py --base-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" --tunable search.paths
--values 1 --players 4 --games 2000 --workers 3 --seed 1 --json data/ablations/paths_selfplay.json` (paired 2v2).

Report per experiment: paired win-rate delta with 95 % CI, paired VP delta, per-seat breakdown, ms per decision
(ours and the opponents'), and the mechanism metrics: share of games our seat ends holding Longest Road / Largest
Army; roads built while `N_close_LR >= 1.5` vs `< 1.5`; dev cards bought while Largest Army is open; VP by source.

## Stage 4 - crowding and weight sweeps (vs value, 2000 seeds, base 0; `cand_spec` vs the default `def_spec`)

| arm | candidate |
|---|---|
| crowd 0 / 0.5 / 2 | `search:depth=1,beam=4,expand=8,evaluator=heuristic,paths=1,paths_crowd=X` (crowd 1 = E1) |
| weight 0.5 / 2 | `...,paths=1,paths_w=X` |
| prize weight 0.25 / 1.0 | `...,paths=1` with `"cand_set": {"winpaths.KAPPA": X}` |

An arm clearly worse than E1 may stop early (`stop_at_se 0.012`, `stop_min_pairs 1000`).  The best two settings
are repeated vs `vf` (2000 seeds each).

## Stage 5 - mechanism knock-outs at the best setting (vs value, 2000 seeds each)

`paths_priors=0`; `paths_spots=1`; `winpaths.ESC=0`; `winpaths.LIVE_LR=0` with `winpaths.LIVE_LA=0`;
`winpaths.PLACEBO=1` (the seat-rotated correction - the real one must beat it); a time-matched control (the default
spec with `beam=5,expand=10` at about the same ms per decision) to rule out a compute effect.

## Stage 6 - confirmation (held-out seeds)

The chosen setting vs the default, seed base 100000: 2000 seeds vs `value` and 2000 vs `vf`.

## Decision rules

* **PROVE** (the feature becomes a candidate for a later, separate default change with its own proof run) - all of:
  E1 delta > 0 with the 95 % CI lower bound > 0; E2 and E3 point estimates >= 0; the Stage 6 value delta > 0
  (one-sided p < 0.05) and the vf delta >= 0; mean ms ratio at most 1.5.
* **EXTEND**: 0 < delta but the CI contains 0 after 2000 seeds -> resume that JSONL to 4000 seeds.
* **KILL**, either: after 4000 paired seeds delta <= 0 at every combination of `paths_w` in {0.5, 1} and
  `paths_crowd` in {0, 1}; or the value CI upper bound is below +0.5 pp.
* **Crowding verdict**: crowd=1 beating crowd=0 on the same seeds supports the "crowded path EV is cooked"
  mechanism; if it is <= 0, set the `paths_crowd` default to 0 and report that the probability-share effect alone
  carries the result.
* **PLACEBO >= real**: the gain is noise or regularisation - do not adopt.
* Sweeps are exploratory; only Stage 6 counts as confirmatory evidence.

Budget (about 6000 games/h with 3 workers, +20 % candidate cost): Stage 3 ~2.5 h, Stage 4 ~3 h, Stage 5 ~2.5 h,
Stage 6 ~1.5 h - about 10 h of 3-core time after the proof; Stages 1-2 ~7 min on one core.

## Deviations from the implementation spec (with reasons)

* **No 1e-4 rounding of the race inputs.**  Every input is a deterministic function of the state and every memo
  is a pure function of its key, so exact-float keys already make cold and warm caches bit-identical (tested) and
  the rounding cost ~5 us per leaf.  The solve memo hit rate is unchanged (96 %).
* **Memo keys (review fix).**  Our road room, the reach sets and the spot scores are keyed on *every* seat's
  roads (the spec keyed our room on our roads, the reach of seat j on j's and our roads): another seat's road
  can occupy our frontier edges or cut a reach path, which happens at depth >= 2 leaves (the simulated
  opponents build).  The spot-score key also holds which opponents static's blockability term counts as strong
  (`robber.threat >= PLACEMENT_STRONG_THREAT`): with hidden opponent dev cards their estimated VP moves with the
  dev pool, i.e. with our own dev buy or knight play inside the turn - the implementation's key (our roads +
  award owners) returned a stale score there (warm vs cold spot correction -1.116 vs -1.143 in
  `test_memo_keys_cover_other_roads_and_the_dev_pool`).  Only the spot part matters at depth 1, and only with
  `paths_spots=1` and hidden dev cards.
* **Bundle rates (review fix).**  The spec's `road_rate = min(s_wood, s_brick, ...)` /
  `dev_rate = min(s_sheep, s_wheat, s_ore, ...)` with `s_r = inc_r + TAU (sum conv - conv_r)` credits the same
  converted cards to every missing resource, so a wood / brick-only seat got about the dev rate of an ore / wheat
  seat without sheep (0.18-0.25 vs 0.21 per round).  `bundle_rate` shares one conversion budget (`TAU` of the
  surplus left after the bundle's own cards, at port ratios): 0.06 vs 0.18.  For a seat missing one resource
  the rate is close to the spec's; `s_r` is kept for the contested-spot timing.
* **Won-path limit** (test 13): one more knight is worth 0.12 points there (P is already ~0.98; the softmax
  tails leave the rivals ~2 %), just above the spec's 0.1-point bound; the test asserts < 0.15 points (0.015 VP)
  and that the BUY_DEV prior moves by < 0.5 and is not promoted.
* **Uncrowded Largest Army** (test 11): with the spec's scenario (2 played + 1 held) `should_buy_dev` already
  recommends the buy (prior 80 -> 83.6); the DEV_FLOOR promotion itself is tested on the same scenario with
  0 held (P(knight) x DeltaLA = 1.22 >= 1: prior 20 -> 45).
* **Longest Road scenarios** (tests 10, 12, 14) are built on a fresh board with vertex-disjoint simple paths
  instead of extending chains of a played game: played boards could not grow a chain past 6 roads reliably.
* **USAGE.md and `scripts/campaign_plan.json` are not edited** (outside this change's allowed files, and a plan
  entry is only useful once the proof ends); the spec keys, `--paths` and the campaign entries are documented here
  and in docs/STRATEGY.md.
