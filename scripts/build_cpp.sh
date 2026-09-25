#!/usr/bin/env bash
# Build the C++ extension catanbot.catanbot_core in place (see docs/CPP.md).
#
#   scripts/build_cpp.sh            # regenerate tables, build, smoke-test
#   scripts/build_cpp.sh --clean    # remove build artefacts first
#
# Requirements: g++ (C++17), python3 with numpy, setuptools and pybind11.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf build catanbot/catanbot_core*.so catanbot_core*.so
    shift
fi

echo "== regenerating cpp/board_tables.hpp and cpp/feature_layout.hpp"
"$PY" scripts/gen_board_tables.py

echo "== building catanbot.catanbot_core"
"$PY" setup_cpp.py build_ext --inplace "$@"

echo "== smoke test"
CATANBOT_NO_ACCEL= "$PY" - <<'EOF'
import numpy as np
from catanbot import accel, features as F
from catanbot.state import new_game
assert accel.AVAILABLE, "extension built but catanbot.accel could not load it"
core = accel.load_core()
assert core.num_features() == F.NUM_FEATURES, (core.num_features(), F.NUM_FEATURES)
assert list(core.feature_names()) == list(F.FEATURE_NAMES)
s = new_game(4)
x = core.extract_batch([s] * 4, [0, 1, 2, 3])
assert x.shape == (4, F.NUM_FEATURES) and x.dtype == np.float32
print("catanbot_core OK:", core.__file__)
EOF
