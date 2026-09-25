#!/usr/bin/env python3
"""Champion league and promotion gate - see docs/LEAGUE.md.

    PYTHONPATH=. python3 scripts/league.py status
    PYTHONPATH=. python3 scripts/league.py materialize champion-0
    PYTHONPATH=. python3 scripts/league.py gate --candidate-spec "search:depth=1,beam=4,expand=8,model=models/x.npz"
    PYTHONPATH=. python3 scripts/league.py promote GATE_ID

The engine of the tree this script is imported from referees every game; for a
gate that runs for days, run this script from a frozen worktree of the runner
commit so concurrent edits of the working tree cannot change the rules midway.
"""
from __future__ import annotations

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":   # pin str hashing (set iteration order) for reproducible runs
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanbot.league.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
