"""Champion league: a ladder of frozen champions and a sequential promotion gate (see ``docs/LEAGUE.md``).

* ``registry``   - ``league/champions.json``, materialising a champion's exact commit (worktree + C++ build)
* ``serve``      - bot server run with ``PYTHONPATH`` at a champion's tree (line-delimited JSON)
* ``match``      - the current engine playing seats that are bot servers (or in-process bots)
* ``stats``      - exact binomial tests, Clopper-Pearson, Holm, Fisher
* ``sequential`` - exact alpha-spending boundaries (O'Brien-Fleming / Pocock type) for batched looks
* ``gate``       - candidate vs every champion, resumable JSONL, verdict and promotion
* ``cli``        - ``scripts/league.py`` (materialize, gate, status, evaluate, promote, external-baseline)
"""
