#!/usr/bin/env bash
# Run the pre-registered 1v1 benchmark (docs/BENCH_1V1_PROTOCOL.md) in resumable chunks, check and analyse it,
# and archive the evidence into the repository.
#
# Usage:
#   scripts/run_1v1.sh [TEST ...]        # default: H1 H2, then replay --check of every log and the analysis
#   scripts/run_1v1.sh --analyze         # only the replay checks and the analysis of what exists
#   scripts/run_1v1.sh --archive [DEST]  # copy the finished run into DEST (default: bench_1v1/ in the repository):
#                                        # results, sorted replay logs, console output, replay checks, analysis,
#                                        # README and MANIFEST.sha256 (scripts/archive_proof.py does the copying)
#   BENCH1V1_SMOKE=1 scripts/run_1v1.sh  # smoke test of the tooling (see below)
#
# Tests (catanatron 3.3.0, the venv interpreter; catanbot in seat g % 2 against ONE opponent, --players 2):
#   H1  primary    AlphaBetaPlayer (defaults)       400 games, base seed 910501
#   H2  secondary  ValueFunctionPlayer (defaults)   400 games, base seed 910601
# Every chunk runs scripts/bench_catanatron.py --players 2 --opponent OPP --spec SPEC --seed SEED --game-range A:B
# --workers W --trades off --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE under
# `timeout $BENCH1V1_TIMEOUT` (default 1200 s).  Chunks are consecutive games [a, b) of the SAME base seed, so the
# per-game seeds and seats are those of one uninterrupted run.  A chunk whose JSON exists is skipped (start the script
# again after an interruption); a chunk that times out is split in two (a `.split` marker records it) down to single
# games; a chunk writes its JSON and action log under temporary names and moves them into place only when it finished.
#
# Layout under $BENCH1V1_OUT (default .../scratchpad/bench1v1/run):
#   json/<ID>/g<a>-<b>.json      bench results of one chunk          logs/<ID>/*.jsonl.gz   action logs (replayable)
#   out/<ID>_g<a>-<b>.txt        bench output of one chunk            replay_check_<ID>.txt  replay --check
#   run_1v1.log                  what ran when                        result_1v1.txt / RESULT_1V1.md / result_1v1.json
#
# Environment: BENCH1V1_OUT, BENCH1V1_WORKERS (1: the machine is shared), BENCH1V1_TIMEOUT (1200), PY33
# (/home/user/venv_cat33/bin/python), PY (python3, for the analysis), BENCH1V1_CHUNK_<ID> (games per chunk).
# BENCH1V1_SMOKE=1 plays 4 games per test in chunks of 2 under NON-protocol seeds (424701 / 424801, so no registered
# game is played before the registered run) into .../scratchpad/bench1v1/smoke, and --archive then defaults to
# .../smoke/bench_1v1 (never the repository); its analysis reports "not the registered data" by construction.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH=/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/bench1v1
PY33="${PY33:-/home/user/venv_cat33/bin/python}"
PY="${PY:-python3}"
SPEC="search:depth=1,beam=4,expand=8,evaluator=heuristic"
SMOKE="${BENCH1V1_SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    OUT="${BENCH1V1_OUT:-$SCRATCH/smoke}"
else
    OUT="${BENCH1V1_OUT:-$SCRATCH/run}"
fi
WORKERS="${BENCH1V1_WORKERS:-1}"
TIMEOUT="${BENCH1V1_TIMEOUT:-1200}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# id  opponent  games  chunk  seed  smoke-seed   (docs/BENCH_1V1_PROTOCOL.md; a 1v1 alpha-beta game takes about
# 2 s and a value-player game under 1 s, so a chunk takes a few minutes even at 1 worker)
TABLE="H1 alphabeta 400 50 910501 424701
H2 value 400 100 910601 424801"
ALL_IDS=$(echo "$TABLE" | awk '{ printf "%s ", $1 }')

ANALYZE_ONLY=0
ARCHIVE=0
DEST=""
TESTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --analyze) ANALYZE_ONLY=1 ;;
        --archive) ARCHIVE=1; if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then DEST=$2; shift; fi ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) TESTS+=("$1") ;;
    esac
    shift
done
[ ${#TESTS[@]} -eq 0 ] && read -r -a TESTS <<< "$ALL_IDS"

mkdir -p "$OUT/json" "$OUT/logs" "$OUT/out"
MASTER="$OUT/run_1v1.log"
FAILED=0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

row() {  # row ID -> "opponent games chunk seed smoke-seed"
    echo "$TABLE" | awk -v id="$1" '$1 == id { print $2, $3, $4, $5, $6 }'
}

# run_range ID OPPONENT SEED A B : play games [A, B) as one chunk, splitting it on a timeout.
run_range() {
    local id=$1 opp=$2 seed=$3 a=$4 b=$5
    local tag
    tag=$(printf "g%05d-%05d" "$a" "$b")
    local json="$OUT/json/$id/$tag.json"
    local marker="$OUT/json/$id/$tag.split"
    [ -f "$json" ] && return 0
    if [ ! -f "$marker" ]; then
        local part="$OUT/logs/$id/.partial-$tag"
        rm -rf "$part" "$json.tmp"
        mkdir -p "$OUT/json/$id" "$OUT/logs/$id" "$part"
        say "$id games $a..$((b - 1)) ($opp, 1v1, seed $seed) starting"
        local t0=$SECONDS rc=0
        timeout "$TIMEOUT" "$PY33" "$ROOT/scripts/bench_catanatron.py" --players 2 --opponent "$opp" --spec "$SPEC" \
            --seed "$seed" --game-range "$a:$b" --workers "$WORKERS" --trades off --hash-seed 0 --rerun-crashes \
            --log-actions "$part" --verbose --json "$json.tmp" > "$OUT/out/${id}_$tag.txt" 2>&1 || rc=$?
        if [ "$rc" = 0 ] && [ -f "$json.tmp" ]; then
            mv "$part"/*.jsonl.gz "$OUT/logs/$id/" && rmdir "$part"
            mv "$json.tmp" "$json"
            say "$id games $a..$((b - 1)) done in $((SECONDS - t0)) s: $(grep -m1 '  wins' "$OUT/out/${id}_$tag.txt" | sed 's/^ *//')"
            return 0
        fi
        rm -rf "$part" "$json.tmp"
        if [ "$rc" != 124 ]; then
            say "$id games $a..$((b - 1)) FAILED (exit $rc), see $OUT/out/${id}_$tag.txt"
            FAILED=1
            return 1
        fi
        if [ $((b - a)) -le 1 ]; then
            say "$id game $a timed out alone after $TIMEOUT s - giving up on it"
            FAILED=1
            return 1
        fi
        say "$id games $a..$((b - 1)) timed out after $TIMEOUT s: splitting the chunk"
        touch "$marker"
    fi
    local m=$(((a + b) / 2))
    run_range "$id" "$opp" "$seed" "$a" "$m"
    run_range "$id" "$opp" "$seed" "$m" "$b"
}

run_test() {
    local id=$1
    local spec_row
    spec_row=$(row "$id")
    if [ -z "$spec_row" ]; then
        say "unknown test $id (known: $ALL_IDS)"
        FAILED=1
        return
    fi
    # shellcheck disable=SC2086
    set -- $spec_row
    local opp=$1 games=$2 chunk=$3 seed=$4 smoke_seed=$5
    local var="BENCH1V1_CHUNK_$id"
    chunk="${!var:-$chunk}"
    if [ "$SMOKE" = "1" ]; then games=4; chunk=2; seed=$smoke_seed; fi
    local a=0
    while [ "$a" -lt "$games" ]; do
        local b=$((a + chunk))
        [ "$b" -gt "$games" ] && b=$games
        run_range "$id" "$opp" "$seed" "$a" "$b"
        a=$b
    done
}

analyze() {
    for id in $ALL_IDS; do
        ls "$OUT/logs/$id"/*.jsonl.gz > /dev/null 2>&1 || continue
        if timeout "$TIMEOUT" "$PY33" "$ROOT/scripts/replay_catanatron.py" --check "$OUT/logs/$id" \
                > "$OUT/replay_check_$id.txt" 2>&1; then
            say "replay --check $id: $(tail -n 1 "$OUT/replay_check_$id.txt")"
        else
            say "replay --check $id FAILED, see $OUT/replay_check_$id.txt"
            FAILED=1
        fi
    done
    local args=()
    for id in $ALL_IDS; do
        if ls "$OUT/json/$id"/*.json > /dev/null 2>&1; then
            args+=(--test "$id=$OUT/json/$id")
        fi
    done
    if [ ${#args[@]} -eq 0 ]; then
        say "no results to analyse yet"
        FAILED=1
        return
    fi
    "$PY" "$ROOT/scripts/analyze_1v1.py" "${args[@]}" --markdown "$OUT/RESULT_1V1.md" --json "$OUT/result_1v1.json" \
        > "$OUT/result_1v1.txt" 2>&1 || FAILED=1
    cat "$OUT/result_1v1.txt"
    say "analysis: $OUT/result_1v1.txt, $OUT/RESULT_1V1.md, $OUT/result_1v1.json"
}

archive() {
    if [ -z "$DEST" ]; then
        if [ "$SMOKE" = "1" ]; then DEST="$OUT/bench_1v1"; else DEST="$ROOT/bench_1v1"; fi
    fi
    local ids=()
    for id in $ALL_IDS; do
        ls "$OUT/json/$id"/*.json > /dev/null 2>&1 && ids+=("$id")
    done
    if [ ${#ids[@]} -eq 0 ]; then
        say "nothing to archive in $OUT"
        exit 1
    fi
    mkdir -p "$DEST/run"
    for f in run_1v1.log result_1v1.txt result_1v1.json RESULT_1V1.md; do
        [ -f "$OUT/$f" ] && cp "$OUT/$f" "$DEST/run/$f"
    done
    cat > "$DEST/README.md" <<'EOF'
# 1v1 benchmark evidence

Everything the pre-registered 1v1 benchmark (docs/BENCH_1V1_PROTOCOL.md) produced: catanbot against
catanatron 3.3.0's AlphaBetaPlayer (H1, primary) and ValueFunctionPlayer (H2, secondary), head to head.
Copied from the run's scratch directory by `scripts/run_1v1.sh --archive` (which uses
`scripts/archive_proof.py`: every test must cover exactly games 0..N-1 once, in results and logs).

```
bench_1v1/
  <ID>/results/<ID>_g<a>-<b>.json   bench results of games [a, b) (run metadata, per-game results, timing)
  <ID>/logs/<ID>_<opponent>_1v1_seed<base>_g<a>-<b>.jsonl.gz
                                    replayable action logs of the same games, one game per line, sorted
  <ID>/console/<ID>_g<a>-<b>.txt    the bench's console output of the chunk
  <ID>/replay_check_<ID>.txt        replay --check of the original logs
  run/run_1v1.log                   what ran when
  run/result_1v1.txt, RESULT_1V1.md, result_1v1.json   the analysis (scripts/analyze_1v1.py)
  MANIFEST.sha256                   sha256 of every file here: (cd bench_1v1 && sha256sum -c MANIFEST.sha256)
```

Re-check from the repository root (`$PY33` = a Python with catanatron 3.3.0):

```
(cd bench_1v1 && sha256sum -c MANIFEST.sha256)
$PY33 scripts/replay_catanatron.py bench_1v1/H1/logs --check      # likewise H2
python3 scripts/analyze_1v1.py --test H1=bench_1v1/H1/results --test H2=bench_1v1/H2/results
$PY33 scripts/replay_catanatron.py bench_1v1/H1/logs --game 17 --turn 40   # any position of any game
```
EOF
    "$PY" "$ROOT/scripts/archive_proof.py" --run "$OUT" --dest "$DEST" "${ids[@]}" || exit 1
    say "archived ${ids[*]} into $DEST (manifest $DEST/MANIFEST.sha256)"
}

say "run_1v1.sh: out $OUT, workers $WORKERS, timeout $TIMEOUT s per chunk, smoke=$SMOKE, tests ${TESTS[*]}"
if [ "$ARCHIVE" = 1 ]; then
    archive
    exit 0
fi
if [ "$ANALYZE_ONLY" = 0 ]; then
    for id in "${TESTS[@]}"; do
        run_test "$id"
    done
fi
analyze
exit "$FAILED"
