#!/usr/bin/env bash
# Run the pre-registered strength proof (docs/PROOF_PROTOCOL.md) in resumable chunks, then analyse it.
#
# Usage:
#   scripts/run_proof.sh [TEST ...]      # default: T1 T2 T3 T4 T5 T6 R1 R2, then replay checks + prove_strength.py
#   scripts/run_proof.sh --analyze       # only the replay checks and the analysis of what exists
#   PROOF_SMOKE=1 scripts/run_proof.sh   # smoke test of the tooling (see below)
#
# Every test plays exactly the protocol's opponent, format, games and base seed (T1-T6: catanatron 3.3.0,
# the venv interpreter, seed 900001; R1-R2: catanatron 3.2.1, system python3, seed 900101) with the default
# bot spec, --trades off, --hash-seed 0 (PYTHONHASHSEED=0), --rerun-crashes (a crashed game is re-played once
# with the same seed; a second crash is recorded as a loss) and --log-actions (replayable action logs).
# A test is split into chunks of consecutive games [a, b) of the SAME base seed (bench --game-range a:b),
# so the per-game seeds and seats are those of one uninterrupted run; each chunk runs under
# `timeout $PROOF_TIMEOUT` (default 1200 s = 20 min).  A chunk whose JSON exists is skipped, so the script
# can simply be started again after an interruption.  A chunk that times out is split in two (recorded by a
# `.split` marker so a later start goes straight to the halves) down to single games.  A chunk writes its
# JSON and action log under temporary names and moves them into place only when it finished.
#
# Layout under $PROOF_OUT (default /tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof/run):
#   json/<ID>/g<a>-<b>.json      bench results of one chunk          logs/<ID>/*.jsonl.gz   action logs
#   out/<ID>_g<a>-<b>.txt        bench output of one chunk            replay_check_<ID>.txt  replay --check
#   run_proof.log                what ran when                        proof.txt / PROOF.md / proof.json  analysis
#
# Environment: PROOF_OUT, PROOF_WORKERS (3), PROOF_TIMEOUT (1200), PY33 (/home/user/venv_cat33/bin/python),
# PY321 (python3), PROOF_CHUNK_<ID> (games per chunk, e.g. PROOF_CHUNK_T2=40), PROOF_CHECK_LOGS (1: replay
# every action log with --check before the analysis).
# PROOF_SMOKE=1 plays 4 games per test in chunks of 2 with 2 workers under NON-protocol seeds (424201 /
# 424301, so no proof game is played before the proof) into .../scratchpad/proof/smoke; its verdict is
# meaningless (it fails the protocol-conformance checks by construction).
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH=/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad/proof
PY33="${PY33:-/home/user/venv_cat33/bin/python}"
PY321="${PY321:-python3}"
SPEC="search:depth=1,beam=4,expand=8,evaluator=heuristic"
SMOKE="${PROOF_SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    OUT="${PROOF_OUT:-$SCRATCH/smoke}"
    WORKERS="${PROOF_WORKERS:-2}"
    SEED_T=424201
    SEED_R=424301
else
    OUT="${PROOF_OUT:-$SCRATCH/run}"
    WORKERS="${PROOF_WORKERS:-3}"
    SEED_T=900001
    SEED_R=900101
fi
TIMEOUT="${PROOF_TIMEOUT:-1200}"
CHECK_LOGS="${PROOF_CHECK_LOGS:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# id  engine  opponent   our-seats  games  chunk   (docs/PROOF_PROTOCOL.md; chunk sizes fit 20 min at 3 workers)
TABLE="T1 33 value 1 1000 250
T2 33 alphabeta 1 400 40
T3 33 sameturn 1 400 50
T4 33 value 2 1000 250
T5 33 alphabeta 2 400 40
T6 33 sameturn 2 400 50
R1 321 vf 1 1000 250
R2 321 ab 1 400 80"

mkdir -p "$OUT/json" "$OUT/logs" "$OUT/out"
MASTER="$OUT/run_proof.log"
FAILED=0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

row() {  # row ID -> "engine opponent seats games chunk"
    echo "$TABLE" | awk -v id="$1" '$1 == id { print $2, $3, $4, $5, $6 }'
}

# run_range ID PY OPPONENT SEATS SEED A B : play games [A, B) as one chunk, splitting it on a timeout.
run_range() {
    local id=$1 py=$2 opp=$3 seats=$4 seed=$5 a=$6 b=$7
    local tag
    tag=$(printf "g%05d-%05d" "$a" "$b")
    local json="$OUT/json/$id/$tag.json"
    local marker="$OUT/json/$id/$tag.split"
    [ -f "$json" ] && return 0
    if [ ! -f "$marker" ]; then
        local part="$OUT/logs/$id/.partial-$tag"
        rm -rf "$part" "$json.tmp"
        mkdir -p "$OUT/json/$id" "$OUT/logs/$id" "$part"
        say "$id games $a..$((b - 1)) ($opp, $([ "$seats" = 2 ] && echo 2v2 || echo 1v3), seed $seed) starting"
        local t0=$SECONDS rc=0
        timeout "$TIMEOUT" "$py" "$ROOT/scripts/bench_catanatron.py" --opponent "$opp" --spec "$SPEC" \
            --our-seats "$seats" --seed "$seed" --game-range "$a:$b" --workers "$WORKERS" --trades off \
            --hash-seed 0 --rerun-crashes --log-actions "$part" --verbose --json "$json.tmp" \
            > "$OUT/out/${id}_$tag.txt" 2>&1 || rc=$?
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
    run_range "$id" "$py" "$opp" "$seats" "$seed" "$a" "$m"
    run_range "$id" "$py" "$opp" "$seats" "$seed" "$m" "$b"
}

run_test() {
    local id=$1
    local spec_row
    spec_row=$(row "$id")
    if [ -z "$spec_row" ]; then
        say "unknown test $id"
        FAILED=1
        return
    fi
    # shellcheck disable=SC2086
    set -- $spec_row
    local engine=$1 opp=$2 seats=$3 games=$4 chunk=$5
    local var="PROOF_CHUNK_$id"
    chunk="${!var:-$chunk}"
    local py=$PY33 seed=$SEED_T
    if [ "$engine" = 321 ]; then py=$PY321; seed=$SEED_R; fi
    if [ "$SMOKE" = "1" ]; then games=4; chunk=2; fi
    local a=0
    while [ "$a" -lt "$games" ]; do
        local b=$((a + chunk))
        [ "$b" -gt "$games" ] && b=$games
        run_range "$id" "$py" "$opp" "$seats" "$seed" "$a" "$b"
        a=$b
    done
}

ANALYZE_ONLY=0
TESTS=()
for arg in "$@"; do
    case "$arg" in
        --analyze) ANALYZE_ONLY=1 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) TESTS+=("$arg") ;;
    esac
done
[ ${#TESTS[@]} -eq 0 ] && TESTS=(T1 T2 T3 T4 T5 T6 R1 R2)

say "run_proof.sh: out $OUT, workers $WORKERS, timeout $TIMEOUT s per chunk, smoke=$SMOKE, tests ${TESTS[*]}"
if [ "$ANALYZE_ONLY" = 0 ]; then
    for id in "${TESTS[@]}"; do
        run_test "$id"
    done
fi

if [ "$CHECK_LOGS" = 1 ]; then
    for id in T1 T2 T3 T4 T5 T6 R1 R2; do
        [ -d "$OUT/logs/$id" ] || continue
        ls "$OUT/logs/$id"/*.jsonl.gz > /dev/null 2>&1 || continue
        py=$PY33
        case "$id" in R*) py=$PY321 ;; esac
        if timeout "$TIMEOUT" "$py" "$ROOT/scripts/replay_catanatron.py" --check "$OUT/logs/$id" \
                > "$OUT/replay_check_$id.txt" 2>&1; then
            say "replay --check $id: $(tail -n 1 "$OUT/replay_check_$id.txt")"
        else
            say "replay --check $id FAILED, see $OUT/replay_check_$id.txt"
            FAILED=1
        fi
    done
fi

ARGS=()
for id in T1 T2 T3 T4 T5 T6 R1 R2; do
    if ls "$OUT/json/$id"/*.json > /dev/null 2>&1; then
        ARGS+=(--test "$id=$OUT/json/$id")
    fi
done
if [ ${#ARGS[@]} -eq 0 ]; then
    say "no results to analyse yet"
    exit 1
fi
"$PY321" "$ROOT/scripts/prove_strength.py" "${ARGS[@]}" --markdown "$OUT/PROOF.md" --json "$OUT/proof.json" \
    > "$OUT/proof.txt" 2>&1 || FAILED=1
cat "$OUT/proof.txt"
say "analysis: $OUT/proof.txt, $OUT/PROOF.md (body for docs/PROOF.md), $OUT/proof.json"
exit "$FAILED"
