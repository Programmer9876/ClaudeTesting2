# Results log

Living document, updated by the overnight check-ins.  Newest entries first.
Everything here was measured on the cloud container (4 shared cores, 15 GB);
numbers vary with the load from concurrent jobs.

## 2026-09-25 06:20 UTC - hourly check-in

* **Landed and verified**: the strategy ablation harness (`scripts/ablate.py`,
  30 tunables, paired games, per-decision cost; docs/ABLATIONS.md) and the
  blockability placement term (robber concentration penalty; C++ parity
  bit-exact; neutral within noise over 240 heuristic games, to be re-tested
  with the search bot in the full sweep).  Smoke hint from the harness: trade
  proposals are about half of the search bot's decision time at depth 1.
* **In progress**: value-net fix (training-side; agent is fitting a net on the
  saved buffer and running tournaments), native C++ search port (agent is
  building cpp/search.* and the MLP forward), Catanatron ladders on both
  versions (verification stage).
* Training stays paused until the fix is verified.  Load average ~10 on 4
  cores from the agents' experiments; the full ablation sweep waits for a
  quiet window.
* Opus switch notice: not yet sent (two implementation pieces still open).

## 2026-09-25 05:20 UTC - state before the overnight run

**Value net.**  Two candidates trained on the accelerated self-play data
(1.35M positions from 750-game iterations with trading-randomised bots) both
generalise as classifiers (validation AUC 0.80 / 0.78, loss 0.47 / 0.50, game-level
split) but lose the promotion tournament badly when used as the search evaluator:

| iteration | candidate wins | heuristic-search wins | avg VP candidate / heuristic |
|---|---|---|---|
| 0 | 11 % | 39 % | 6.5 / 8.3 |
| 1 | 13 % | 37 % | 6.3 / 7.9 |

Diagnosis in progress (workflow): the working hypothesis is that an
outcome-regression net learns cross-game correlations (a fat hand marks a
rich player) rather than the within-game effect of an action (spending the
hand on a settlement), so the search under-builds.  Training is paused until
the fix is verified; the replay buffer is kept for `--resume`.

**Speed.**  C++ extension (features, heuristic evaluator, engine) verified
bit-identical to Python: features 38-47x, evaluator 68x, self-play games
about 4 s instead of 14 s, depth-2 search 0.34 s (heuristic) / 0.61 s (net)
per position.  A native port of the search's opponent simulation is in
progress to make depth 3-4 practical.

**Review.**  58 findings (3 critical, 23 major, 32 minor) from the multi-lens
review were verified and fixed (engine Longest Road edge case, screenshot
parser robustness, CLI validation, arbitrage / kingmaker / card counting
wired into the search, self-play trading randomisation); full suite 385
passed.

**Benchmarks.**  Stock Catanatron bots: 99-100 % wins (control).  Strong
opponents (our in-engine ValueFunction / AlphaBeta stand-ins, catanatron 3.3's
own players) ladder in progress; a compute-fair (time-matched) ladder is
queued because the current ladder does not equalise thinking time.

**Pending overnight.**  Value-net fix and training restart; native search;
blockability placement term; ablation harness and full sweep; fair ladder.
