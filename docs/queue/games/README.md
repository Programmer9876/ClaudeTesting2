# Test-queue games

*Started 2026-09-26 (UTC) and extended at every check-in.  `index.json` and `epochs.json` record, per row,
the code epoch and the commit its games ran on.*

The per-game results of every finished test-queue row (docs/QUEUE.md), one file per row.  The verdicts made
from them are in `../ledger.jsonl` and `../QUEUE.md`.  `scripts/archive_queue.py` copies a row here once its
last record is `stop`.  Rows still running are copied when they finish, or with `--all` before a session ends.

| file | what |
|---|---|
| `<row>.jsonl.gz` | the row's file, unchanged, gzip with no timestamp |
| `index.json` | per row: finished or partial, games, runs, code fingerprints, sha256 of the uncompressed file |
| `epochs.json` | per code epoch: the commit that introduced the code the games ran on (the snapshot's files equal it) |
| `selfplay/<row>.json.gz` | a self-play row's chunk files, one JSON object mapping each chunk's path to its content; these hold per-seat results of each chunk, not one record per game |
| `MANIFEST.sha256` | `sha256sum -c MANIFEST.sha256`, run from this directory |

A row file holds three kinds of record:
- **`run`**: one per run, with both arms' specs and overrides, the opponent, the seeds, the Catanatron
  version and the hash seed;
- **`game`**: one per game, with the arm, seed, seat, winner, VPs and turns, our decisions and timing, trades,
  and the mechanics metrics (first city and settlement round, bank trades at 4:1, 3:1 and 2:1, cards lost
  to the robber, robber moves, dev cards, and so on);
- **`stop`**: the row's end, with the verdict and the reason.

These are results, not replay logs: queue games keep no action log.  A game is reproduced by running the same
arm at the same seed with the epoch's code (`epochs.json`), its Catanatron version and `PYTHONHASHSEED=0`.
The strength proof's games do keep full replay logs, in `proof/`.

```
zcat docs/queue/games/t2_dump0@value.jsonl.gz | head -2
```
