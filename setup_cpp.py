"""Build the optional C++ extension ``catanbot.catanbot_core`` (see docs/CPP.md).

    python3 setup_cpp.py build_ext --inplace

produces ``catanbot/catanbot_core.<abi>.so``.  ``scripts/build_cpp.sh`` wraps
this (regenerating the C++ tables first).  Flags: C++17, -O3, -march=native
(the binary is tuned to the build machine), strict floating point
(-ffp-contract=off keeps the results bit-identical to the numpy reference).
"""
from __future__ import annotations

import os
import sys

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pragma: no cover
    sys.exit("pybind11 is required to build the extension: pip install pybind11")

ROOT = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(ROOT, "cpp")

EXTRA_FLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-fvisibility=hidden", "-DNDEBUG", "-Wall"]
if os.environ.get("CATANBOT_CPP_PORTABLE"):  # build without -march=native
    EXTRA_FLAGS.remove("-march=native")

ext = Pybind11Extension(
    "catanbot.catanbot_core",
    sources=[os.path.join("cpp", "module.cpp"), os.path.join("cpp", "features.cpp"), os.path.join("cpp", "heuristic.cpp"),
             os.path.join("cpp", "engine.cpp"), os.path.join("cpp", "policy.cpp"), os.path.join("cpp", "evaluator.cpp"),
             os.path.join("cpp", "search.cpp")],
    include_dirs=[CPP],
    depends=[os.path.join(CPP, f) for f in ("state.hpp", "features.hpp", "heuristic.hpp", "engine.hpp", "board_tables.hpp",
                                             "feature_layout.hpp", "policy.hpp", "evaluator.hpp", "search.hpp",
                                             "search_bindings.inc")],
    cxx_std=17,
    extra_compile_args=EXTRA_FLAGS,
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
