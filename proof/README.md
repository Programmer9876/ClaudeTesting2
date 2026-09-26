# Strength-proof evidence

Everything the pre-registered strength proof produced (docs/PROOF_PROTOCOL.md):
- the per-chunk results;
- a replayable action log of every game;
- the console output;
- the replay checks;
- the analysis.

The verdict and the reading of it are in docs/PROOF.md.  A skeptic's
questions are in docs/SCRUTINY.md.  The files were copied here from the run's
scratch directory by `scripts/archive_proof.py`, which checks that each test
covers exactly games 0..N-1 once, in both results and logs.

## Layout

```
proof/
  <ID>/                         one directory per test (T1-T11, R1-R2)
    results/<ID>_g<a>-<b>.json  bench results of games [a, b) of the test, unchanged
    logs/<ID>_<opponent>_<format>_seed<base>_g<a>-<b>.jsonl.gz
                                action logs of the same games, one game per line, sorted by game number
    console/<ID>_g<a>-<b>.txt   the bench's console output for the chunk
    replay_check_<ID>.txt       replay --check of the original logs, run by run_proof.sh
    interrupted/                T7 and T9 only: console output of a chunk cut off by a container
                                restart (discarded and replayed; the finished games match exactly)
  run/run_proof.log             what ran when (the runner's own log, both runs)
  run/queue_t7_t11.log          when T7-T11 started and the two restarts
  run/proof.txt, proof.json, PROOF.md
                                the final analysis of claims 1-4 (end of the T7-T11 run)
  run/claims1-2_first_analysis/ the analysis of claims 1-2 made at the end of the T1-R2 run
  MANIFEST.sha256               sha256 of every file above (sha256sum -c MANIFEST.sha256, from proof/)
```

**A result file** (`results/*.json`) holds three things:
- the run metadata the analysis checks against the protocol: spec, opponent
  and its class, parameters, engine version, format, seats, base seed, hash
  seed, trades, VP to win and discard limit;
- per-game results: `game`, `seed`, `seat`, winner, VPs, turns, crashes;
- per-decision timing for both sides.

**A log line** is one game:
- its base seed and game index;
- the players, with their class and our bot's seed;
- the board and the dev-deck order;
- every action with its chance outcome (dice, stolen card, drawn card);
- a final-state fingerprint.

## Re-check

From the repository root.  `$PY33` is a Python with Catanatron 3.3.0 (T1-T11);
`python3` has Catanatron 3.2.1 (R1-R2).

```
# checksums
(cd proof && sha256sum -c MANIFEST.sha256)

# replay every game and verify every action, the final VPs, winner, turns and fingerprint
$PY33 scripts/replay_catanatron.py proof/T1/logs --check        # likewise T2 ... T11
python3 scripts/replay_catanatron.py proof/R1/logs --check      # likewise R2

# rebuild one position (board, hands, buildings) after a given turn or action
$PY33 scripts/replay_catanatron.py proof/T2/logs --game 17 --turn 40

# the registered analysis of all four claims (scripts/prove_strength.py, sha256 prefix 3010fd6ee4063937,
# the file named in protocol amendment 5)
ARGS=""; for t in T1 T2 T3 T4 T5 T6 R1 R2 T7 T8 T9 T10 T11; do ARGS="$ARGS --test $t=proof/$t/results"; done
python3 scripts/prove_strength.py $ARGS
# claims 1-2 with the script of the first run
git show 9984181:scripts/prove_strength.py > /tmp/prove_9984181.py
ARGS=""; for t in T1 T2 T3 T4 T5 T6 R1 R2; do ARGS="$ARGS --test $t=proof/$t/results"; done
python3 /tmp/prove_9984181.py $ARGS

# do two tests that share a seed really differ (T2 AlphaBeta vs T3 SameTurnAlphaBeta)?
python3 scripts/audit_opponents.py --run proof T2 T3
```

Result on 2026-09-26:
- all 7,600 archived games replay with 0 mismatches;
- the analysis of the archived copy is identical to the run's own
  (`run/proof.txt`, and `run/claims1-2_first_analysis/` for the first run);
- claims 1-4 all PASS (docs/PROOF.md).

**Hash seed:** games are reproducible only with `PYTHONHASHSEED=0`, because
Catanatron iterates over sets.  `replay --check` does not depend on it: it
re-applies the logged actions and outcomes.

## Code that played

| tests | code | Catanatron |
|---|---|---|
| T1-T6 | commit `9984181` (frozen worktree) | 3.3.0, unmodified GitHub checkout `ecf9311` |
| R1-R2 | commit `9984181` (frozen worktree) | 3.2.1 (PyPI) |
| T7-T11 | commit `9599eed` plus four analysis / runner files from `855bfd0` (protocol amendment 5) | 3.3.0 |
