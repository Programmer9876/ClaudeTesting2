#!/usr/bin/env bash
# Run the pre-registered multi-seat benchmark (docs/BENCH_MULTI_PROTOCOL.md) in resumable chunks, check and analyse
# it, and archive the evidence into the repository.
#
# Usage:
#   scripts/run_multi.sh [TEST ...]        # default: M1 M2 M3, then replay --check of every log and the analysis
#   scripts/run_multi.sh --analyze         # only the replay checks and the analysis of what exists
#   scripts/run_multi.sh --archive [DEST]  # copy the finished run into DEST (default: bench_multi/ in the repository):
#                                          # results, sorted replay logs, console output, replay checks, analysis,
#                                          # README and MANIFEST.sha256 (scripts/archive_proof.py does the copying)
#   BENCHMULTI_SMOKE=1 scripts/run_multi.sh  # smoke test of the tooling (see below)
#
# Tests (catanatron 3.3.0, the venv interpreter; four seats, three or two of them catanbot):
#   M1  3v1        three catanbot seats vs AlphaBetaPlayer (defaults) in seat g % 4          400 games, seed 910701
#   M2  3v1        three catanbot seats vs ValueFunctionPlayer (defaults) in seat g % 4      400 games, seed 910801
#   M3  2v2-mixed  two catanbot seats vs one ValueFunctionPlayer + one AlphaBetaPlayer,
#                  placement g % 12                                                          400 games, seed 910901
# Every chunk runs scripts/bench_catanatron.py --our-seats 3 --opponent OPP (M1, M2) or --our-seats 2
# --mixed-opponents value,alphabeta (M3) --spec SPEC --seed SEED --game-range A:B --workers W --trades off
# --hash-seed 0 --rerun-crashes --log-actions DIR --verbose --json FILE under `timeout $BENCHMULTI_TIMEOUT`
# (default 1200 s).  Chunks are consecutive games [a, b) of the SAME base seed, so the per-game seeds and seatings
# are those of one uninterrupted run.  A chunk whose JSON exists is skipped (start the script again after an
# interruption); a chunk that times out is split in two (a `.split` marker records it) down to single games; a
# chunk writes its JSON and action log under temporary names and moves them into place only when it finished.
#
# Layout under $BENCHMULTI_OUT (default .../scratchpad/benchmulti/run):
#   json/<ID>/g<a>-<b>.json      bench results of one chunk          logs/<ID>/*.jsonl.gz   action logs (replayable)
#   out/<ID>_g<a>-<b>.txt        bench output of one chunk            replay_check_<ID>.txt  replay --check
#   run_multi.log                what ran when                        result_multi.txt / RESULT_MULTI.md / .json
#
# Environment: BENCHMULTI_OUT, BENCHMULTI_WORKERS (1: the machine is shared), BENCHMULTI_TIMEOUT (1200), PY33
# (/home/user/venv_cat33/bin/python), PY (python3, for the analysis), BENCHMULTI_CHUNK_<ID> (games per chunk).
# BENCHMULTI_SMOKE=1 plays 4 games per test in chunks of 2 under NON-protocol seeds (424901 / 425001 / 425101, so no
# registered game is played before the registered run) into .../scratchpad/benchmulti/smoke, and --archive then
# defaults to .../smoke/bench_multi (never the repository); its analysis reports "not the registered data" by
# construction.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH=/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/benchmulti
PY33="${PY33:-/home/user/venv_cat33/bin/python}"
PY="${PY:-python3}"
SPEC="search:depth=1,beam=4,expand=8,evaluator=heuristic"
SMOKE="${BENCHMULTI_SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    OUT="${BENCHMULTI_OUT:-$SCRATCH/smoke}"
else
    OUT="${BENCHMULTI_OUT:-$SCRATCH/run}"
fi
WORKERS="${BENCHMULTI_WORKERS:-1}"
TIMEOUT="${BENCHMULTI_TIMEOUT:-1200}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# id  format  opponent(s)  games  chunk  seed  smoke-seed   (docs/BENCH_MULTI_PROTOCOL.md; in the smoke a game took
# about 5 s in M1, 2 s in M2 and 7 s in M3 at 1 worker, so a chunk takes about 4-5 minutes, a quarter of the timeout)
TABLE="M1 3v1 alphabeta 400 50 910701 424901
M2 3v1 value 400 100 910801 425001
M3 2v2-mixed value,alphabeta 400 40 910901 425101"
ALL_IDS=$(echo "$TABLE" | awk '{ printf "%s ", $1 }')

ANALYZE_ONLY=0
ARCHIVE=0
DEST=""
TESTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --analyze) ANALYZE_ONLY=1 ;;
        --archive) ARCHIVE=1; if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then DEST=$2; shift; fi ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) TESTS+=("$1") ;;
    esac
    shift
done
[ ${#TESTS[@]} -eq 0 ] && read -r -a TESTS <<< "$ALL_IDS"

mkdir -p "$OUT/json" "$OUT/logs" "$OUT/out"
MASTER="$OUT/run_multi.log"
FAILED=0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

row() {  # row ID -> "format opponent games chunk seed smoke-seed"
    echo "$TABLE" | awk -v id="$1" '$1 == id { print $2, $3, $4, $5, $6, $7 }'
}

# seat_args FORMAT OPPONENT -> the bench options that pick the format and the opponent(s)
seat_args() {
    if [ "$1" = "3v1" ]; then
        echo "--our-seats 3 --opponent $2"
    else
        echo "--our-seats 2 --mixed-opponents $2"
    fi
}

# run_range ID FORMAT OPPONENT SEED A B : play games [A, B) as one chunk, splitting it on a timeout.
run_range() {
    local id=$1 fmt=$2 opp=$3 seed=$4 a=$5 b=$6
    local tag
    tag=$(printf "g%05d-%05d" "$a" "$b")
    local json="$OUT/json/$id/$tag.json"
    local marker="$OUT/json/$id/$tag.split"
    [ -f "$json" ] && return 0
    if [ ! -f "$marker" ]; then
        local part="$OUT/logs/$id/.partial-$tag"
        rm -rf "$part" "$json.tmp"
        mkdir -p "$OUT/json/$id" "$OUT/logs/$id" "$part"
        say "$id games $a..$((b - 1)) ($fmt, $opp, seed $seed) starting"
        local t0=$SECONDS rc=0
        # shellcheck disable=SC2046
        timeout "$TIMEOUT" "$PY33" "$ROOT/scripts/bench_catanatron.py" $(seat_args "$fmt" "$opp") --spec "$SPEC" \
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
    run_range "$id" "$fmt" "$opp" "$seed" "$a" "$m"
    run_range "$id" "$fmt" "$opp" "$seed" "$m" "$b"
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
    local fmt=$1 opp=$2 games=$3 chunk=$4 seed=$5 smoke_seed=$6
    local var="BENCHMULTI_CHUNK_$id"
    chunk="${!var:-$chunk}"
    if [ "$SMOKE" = "1" ]; then games=4; chunk=2; seed=$smoke_seed; fi
    local a=0
    while [ "$a" -lt "$games" ]; do
        local b=$((a + chunk))
        [ "$b" -gt "$games" ] && b=$games
        run_range "$id" "$fmt" "$opp" "$seed" "$a" "$b"
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
    "$PY" "$ROOT/scripts/analyze_multi.py" "${args[@]}" --markdown "$OUT/RESULT_MULTI.md" \
        --json "$OUT/result_multi.json" > "$OUT/result_multi.txt" 2>&1 || FAILED=1
    cat "$OUT/result_multi.txt"
    say "analysis: $OUT/result_multi.txt, $OUT/RESULT_MULTI.md, $OUT/result_multi.json"
}

archive() {
    if [ -z "$DEST" ]; then
        if [ "$SMOKE" = "1" ]; then DEST="$OUT/bench_multi"; else DEST="$ROOT/bench_multi"; fi
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
    for f in run_multi.log result_multi.txt result_multi.json RESULT_MULTI.md; do
        [ -f "$OUT/$f" ] && cp "$OUT/$f" "$DEST/run/$f"
    done
    cat > "$DEST/README.md" <<'EOF'
# Multi-seat benchmark evidence

Everything the pre-registered multi-seat benchmark (docs/BENCH_MULTI_PROTOCOL.md) produced: three catanbot seats
against one catanatron 3.3.0 AlphaBetaPlayer (M1) or ValueFunctionPlayer (M2) in 3v1 games, and two catanbot seats
against one ValueFunctionPlayer and one AlphaBetaPlayer in 2v2-mixed games (M3).  Copied from the run's scratch
directory by `scripts/run_multi.sh --archive` (which uses `scripts/archive_proof.py`: every test must cover exactly
games 0..N-1 once, in results and logs).

```
bench_multi/
  <ID>/results/<ID>_g<a>-<b>.json   bench results of games [a, b) (run metadata, per-game results, timing)
  <ID>/logs/<ID>_<opponent>_<format>_seed<base>_g<a>-<b>.jsonl.gz
                                    replayable action logs of the same games, one game per line, sorted
  <ID>/console/<ID>_g<a>-<b>.txt    the bench's console output of the chunk
  <ID>/replay_check_<ID>.txt        replay --check of the original logs
  run/run_multi.log                 what ran when
  run/result_multi.txt, RESULT_MULTI.md, result_multi.json   the analysis (scripts/analyze_multi.py)
  MANIFEST.sha256                   sha256 of every file here: (cd bench_multi && sha256sum -c MANIFEST.sha256)
```

Re-check from the repository root (`$PY33` = a Python with catanatron 3.3.0):

```
(cd bench_multi && sha256sum -c MANIFEST.sha256)
$PY33 scripts/replay_catanatron.py bench_multi/M1/logs --check      # likewise M2, M3
python3 scripts/analyze_multi.py --test M1=bench_multi/M1/results --test M2=bench_multi/M2/results \
    --test M3=bench_multi/M3/results
$PY33 scripts/replay_catanatron.py bench_multi/M1/logs --game 17 --turn 40   # any position of any game
```
EOF
    "$PY" "$ROOT/scripts/archive_proof.py" --run "$OUT" --dest "$DEST" "${ids[@]}" || exit 1
    say "archived ${ids[*]} into $DEST (manifest $DEST/MANIFEST.sha256)"
}

say "run_multi.sh: out $OUT, workers $WORKERS, timeout $TIMEOUT s per chunk, smoke=$SMOKE, tests ${TESTS[*]}"
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
