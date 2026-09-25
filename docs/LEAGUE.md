# Champion league and promotion gate

A self-play sanity check that runs for days or weeks: **every new training run
or new strategy must beat the current champion and must not be worse than any
earlier champion before it becomes the new default.** Each champion plays with
its *own* code (its git commit, its C++ build, its weights), never with
today's code, so later changes cannot silently weaken an old baseline.

```
 candidate (commit + spec [+ weights])
      |
      v
 gate: 2v2 tables vs EACH champion, rotated seats, fixed seeds,     ----> FAIL (clearly worse / futility /
       batches of 198 games, exact alpha-spending boundaries              not significant at max games)
      |
      v
 (c) external non-regression hook (Catanatron benchmark) vs the champion's recorded result
      |
      v
 promote: append to league/champions.json with the gate evidence
```

Code: `catanbot/league/` (`registry.py`, `serve.py`, `match.py`, `stats.py`,
`sequential.py`, `gate.py`, `cli.py`), CLI `scripts/league.py`, tests
`tests/test_league.py`, registry `league/champions.json`.

## Quick start

```bash
cd /home/user/ClaudeTesting2
export PYTHONPATH=$PWD

python3 scripts/league.py status                      # ladder + every gate (running ones with ETA)
python3 scripts/league.py materialize champion-0      # worktree of 9984181 under /home/user/league/champion-0/tree + C++ build

# gate a committed candidate (default --candidate-commit HEAD; materialised as its own worktree)
python3 scripts/league.py gate --candidate-commit <sha> \
    --candidate-spec "search:depth=1,beam=4,expand=8,evaluator=heuristic" \
    --name champion-1 --notes "win-path portfolio"
# ... or a trained net (the file is copied into the gate and hashed; model= is rewritten to the copy)
python3 scripts/league.py gate --candidate-commit <sha> \
    --candidate-spec "search:depth=1,beam=4,expand=8" --weights models/value_net.npz

python3 scripts/league.py gate --resume <GATE_ID>     # after an interruption (Ctrl-C, reboot, --stop-after)
python3 scripts/league.py evaluate <GATE_ID>          # recompute looks / verdict from the records, no games
python3 scripts/league.py promote <GATE_ID>           # only if the verdict is PASS and the ladder is unchanged
```

`scripts/league.py` pins `PYTHONHASHSEED=0` (it re-executes itself if needed).
Global options: `--registry` (default `league/champions.json`), `--gates-dir`
(default `league/gates`), `--league-home` (default `$CATANBOT_LEAGUE_HOME` or
`/home/user/league`), `--repo` (the git repository of the champions' commits).

**Run long gates from a frozen runner.** The *engine* that referees the games
is the one of the tree `scripts/league.py` is imported from. The working tree
changes under your feet (other agents, your own edits), so for a gate that runs
for days, run the script from a worktree of a fixed commit, e.g.
`git worktree add --detach /home/user/league/runner <sha>` then
`python3 /home/user/league/runner/scripts/league.py --registry $PWD/league/champions.json --gates-dir $PWD/league/gates ...`.
Every game record keeps the engine it ran on (`"engine": "<sha12>[+dirty]"`),
and a resume warns when the engine differs from the gate's start.

## Champions and the registry

`league/champions.json` is append-only; the last entry is the current champion.

| field | meaning |
| --- | --- |
| `name` | `champion-<N>` |
| `commit` | full sha whose code the champion plays with |
| `spec` | bot spec for *that commit's* `selfplay.make_bot` |
| `weights` | `null` or `{"path", "sha256"}` (repo-relative path, e.g. `league/weights/champion-3.npz`); `model=` in the spec is replaced by the materialised copy |
| `created`, `notes` | when / what |
| `gate` | the evidence that promoted it: gate id, verdict, stop look and reason, the rule, per-comparison wins / shares / intervals / p-values, the external result, the engine, the path and sha256 of the game records. `{"kind": "seed"}` for champion-0 |
| `env` (optional) | environment variables for its bot servers |
| `shim` (optional) | compatibility shim file (see *Version skew*) |
| `external` (optional) | recorded external benchmark results, e.g. `{"catanatron": {"wins", "games", "win_rate", "command"}}` |

Seed: **champion-0** = commit `9984181` (the frozen proof snapshot), spec
`search:depth=1,beam=4,expand=8,evaluator=heuristic`, no weights, "default bot
at the strength proof".

## Materialisation

`league.py materialize NAME` (idempotent; `--all` for every champion) creates

```
/home/user/league/<name>/tree/             git worktree add --detach <tree> <commit>
/home/user/league/<name>/weights/<file>    copy of the registered weights, sha256 verified
/home/user/league/<name>/build.log         the tree's own scripts/build_cpp.sh (single process)
/home/user/league/<name>/materialized.json commit, method, build info (.so sha256), finish time
```

A finished `materialized.json` is reused (and the weights re-verified); a tree
holding another commit is an error. `--method archive` exports the commit with
`git archive` instead (an exact copy with no worktree metadata; the repository
is not modified), `--method copy --source DIR` copies a directory (tests; the
commit is then recorded as unverified). `--no-build` skips the C++ build; the
gate then refuses seats without the extension unless `--allow-no-accel`
(results are meant to be identical, only slower, but a champion should run as
it ran when promoted). A candidate given by commit is materialised the same
way under `<league home>/candidates/candidate-<sha12>/`. Clean-up:
`git worktree remove /home/user/league/<name>/tree` then `git worktree prune`.

## Bot servers and the match runner

Every seat, the candidate's included, is a **bot server**: a subprocess
`python3 <gate dir>/serve.py --spec SPEC` started with
`PYTHONPATH=<that seat's tree>` only, `PYTHONHASHSEED=0`, one BLAS thread,
and none of the operator's `CATANBOT_*` switches (a champion runs with its
defaults unless its registry entry has `env`). `serve.py` imports nothing from
the league package, so it runs against any commit; the runner copies it into
the gate directory at gate start, so a gate keeps one protocol all its life.
The server's `hello` reports where `catanbot` was imported from (the runner
refuses a server whose `catanbot` does not come from its tree), whether the
C++ extension and the native search are active, and the Python version; the
first hello of each side is kept in `servers.json`.

Line-delimited JSON, one reply per request:

| request | reply |
| --- | --- |
| `{"cmd":"new_game","seat":i,"seed":s,"num_players":n,"max_turns":400}` | fresh `make_bot(spec)`, `reset()`, private `random.Random(s)`; `{"ok":true,"bot":..,"wants_observe":bool}` |
| `{"cmd":"decide","state_id":k,"state":<GameState.to_dict()>,"legal":[actions]}` | `{"action":[...],"ms":<time in bot.decide>,"parse_ms":..}` |
| `{"cmd":"observe","state_id":k,"state":..,"action":[...],"player":p}` | `bot.observe(state_before, action, player)`; `{"ok":true}` |
| `{"cmd":"ping"}` / `{"cmd":"quit"}` | `{"ok":true}` |
| error | `{"error":msg,"kind":"state_format"|"bot"|"protocol"}` |

Actions use `actions.to_json` / `from_json`; `state` may be omitted when
`state_id` is the one the server received last (the actor observing its own
action). The runner (`match.play_match_game`) mirrors `selfplay.play_game`:
the engine of the runner's tree applies the actions with its own
`random.Random(block seed)`; the acting seat decides; the action is validated
against the engine's legal list (trade proposals outside the engine's bounded
candidate list are accepted when the engine accepts them, as in `play_game`);
every seat whose bot overrides `observe` observes it in the pre-action state
(observations are fanned out to all servers, then the acks collected). Seats
without an `observe` hook are not sent observations (pure optimisation).

* **Illegal action** or unreadable reply: counted (`illegal`), replaced by the
  first legal action, logged in the game's `error_log`. **Bot exception**:
  counted (`errors`), same fallback. Both count against the gate's
  `--max-errors` (default 0: any error makes the verdict INVALID).
* **Server crash / timeout** (`--decide-timeout`, default 600 s): the game is
  replayed once with the same seeds after restarting the server; a second
  failure records the game as void (not scored). More than `--max-voids`
  (default 3) voids stops the gate.
* **Exactness.** Tests play the same game with server seats and in-process
  seats and get the identical action sequence (a search bot with an observe
  hook included). A seat's bot rng depends only on the block seed and the seat,
  not on the side, so **a candidate identical to the champion wins exactly 3 of
  every 6 arrangements of a block** (the 2v2 smoke of the current search bot vs
  champion-0 indeed played the identical game in both arrangements).
* Per seat and game the record keeps decisions, mean round-trip ms (all and
  non-trivial), time inside `bot.decide`, overhead (round trip minus bot time),
  server-side parse ms, max ms, observations and error counts, plus the runner's
  own serialisation time.

### Version skew

The engine sends states in the *current* `GameState.to_dict` format; an old
champion reads them with *its* `from_dict`. Every received state is
round-tripped by the champion's own `from_dict` + `to_dict`; any field the
champion drops (a newer format) or reads differently is an error of kind
`state_format` naming the paths, e.g.

```
VersionSkewError: champion-0#0 (tree /home/user/league/champion-0/tree) cannot read the engine's state:
state format skew: vp_to_win: field unknown to this commit (dropped by from_dict/to_dict). ...
```

and the gate stops - never a silent default. (A `from_dict` that raises, e.g.
`KeyError('hexes')`, is reported the same way.) To keep such a champion in the
league, write a **shim** for its commit, e.g. `league/shims/champion-0.py`, and
set `"shim": "league/shims/champion-0.py"` in its registry entry; the gate
copies it and passes it to that champion's servers. A shim may define

```python
IGNORE_FIELDS = {"vp_to_win"}          # fields verified to be irrelevant to this commit's play

def adapt_state(d: dict) -> dict:      # rewrite a new-format state for the old from_dict
    d = dict(d); d.pop("vp_to_win", None); return d

def adapt_legal(legal: list) -> list:  # rewrite the legal action list (e.g. a renamed action kind)
    return legal

def adapt_action(a: list, legal_json: list) -> list:   # map the returned action back
    return a
```

Only shim what you have checked: a field the old code cannot represent (a rule
it does not know) is a real incompatibility, and the right fix is then to
retire the champion from future gates by a documented registry decision, not to
hide the field. `--no-roundtrip-check` disables the check (not recommended).
Not transported: `GameState.rolls_history_len` (unused by bots); `max_turns`
comes with `new_game`. Bots must not rely on the identity of the state object
across calls (each message is a fresh object in the server).

## The gate

`league.py gate` plays the candidate against **each** champion:

* **Formats** (`--formats`, first = primary): `4p2v2` (default: 2 candidate
  seats, 2 champion seats, 4 players; null win share 0.5) and optionally
  `3p1v2` (1 candidate vs 2 champion seats; null 1/3). Generic `NpKvM`.
* **Rotation**: game `i` of a comparison uses arrangement `i mod C(N,K)` of the
  candidate seats (turn order), so every block of 6 (2v2) or 3 (3p1v2) games
  gives every seat to each side equally often, on one board + dice seed.
* **Seeds**: block seed = hash(`--seed-base`, champion, format, block) - fixed
  per champion, independent of the candidate. Gates are reproducible, and all
  candidates meet a champion on the same boards (a paired, lower-variance
  comparison between gates). Bot seeds = hash(block seed, seat).
* **Comparisons**: (champion, format) for every champion; the *primary*
  comparison is the current champion in the primary format.
* **Batches / looks**: `--batch-games` (default 198 = 33 blocks) per
  comparison per look, up to `--max-games` (default 1188 = 6 looks). All
  comparisons advance together; a look is evaluated when all its games exist.

A game counts for the binomial when it has a winner; a turn-cap game (no
winner) is a draw and excluded (reported as `draws`); a void is excluded.
`vp_diff` = mean candidate-seat VP minus mean champion-seat VP per game.

### The sequential rule

Testing "p < 0.01" after every batch would inflate the false-promotion rate
(six looks at a nominal 0.01 give about 0.03; the test suite checks this). The
gate uses **Lan-DeMets alpha spending** with exact binomial boundaries
(`catanbot/league/sequential.py`):

* `obf` (default): O'Brien-Fleming type, alpha(t) = 2(1 - Phi(z_{1-alpha/2} / sqrt t));
  `pocock`: alpha(t) = alpha ln(1 + (e - 1) t). t = decisive games / `--max-games`, 1 at the last look.
* At each look the null distribution of the cumulative win count over the
  paths that have not crossed yet is propagated by convolution with the
  batch's Binomial(m, p0), and the critical count is the smallest one whose
  crossing probability fits in alpha(t) minus what earlier looks spent. So
  P_null(ever crossing) = total spent <= alpha exactly, for any batch sizes
  (including batch sizes reduced by draws). Tests: the exact spent alpha, 2000
  simulated null gates per spending function (<= alpha + 3 SE), 400 simulated
  gates with random draws for both boundaries, and power.
* The lower ("clearly worse") boundary is the mirror image on the loss count at
  `alpha_fail / #comparisons` (Bonferroni). Each boundary is computed as if the
  other did not exist; stopping for the other reason only removes paths, so both
  error rates stay below nominal (non-binding boundaries).

Decision after each look, in this order: **fail** if any comparison crossed its
lower boundary; **success** if the primary crossed its upper boundary;
**futility** (non-binding, `--futility 0.0 --futility-min-t 0.5`; `none`
disables) if the primary's share <= null + margin once half the games are
played; **max_games** at the last look; otherwise continue.

Default design (2v2, alpha 0.01 one-sided, `obf`, 198 x 6):

| look | games | success if wins >= | (share) | nominal one-sided level | fail if wins <= (1 champion; 2; 4) |
| --- | --- | --- | --- | --- | --- |
| 1 | 198 | 143 | 0.722 | 1.6e-10 | 66; 61; 56 |
| 2 | 396 | 242 | 0.611 | 5.7e-6 | 165; 160; 156 |
| 3 | 594 | 340 | 0.572 | 2.4e-4 | 265; 260; 255 |
| 4 | 792 | 439 | 0.554 | 1.3e-3 | 364; 359; 355 |
| 5 | 990 | 537 | 0.542 | 4.2e-3 | 464; 459; 454 |
| 6 | 1188 | 636 | 0.535 | 8.0e-3 | 563; 558; 553 |

Exact type-I error 0.0095 (simulated 0.0093). Operating characteristics with
the futility rule (20 000 simulated gates against one champion):

| true share | P(promote) | P(fail boundary) | P(futility) | expected games |
| --- | --- | --- | --- | --- |
| 0.45 | 0 | 0.44 | 0.56 | 576 |
| 0.50 | 0.009 | 0.005 | 0.64 | 838 |
| 0.53 | 0.38 | 0 | 0.10 | 1066 |
| 0.55 | 0.86 | 0 | 0.01 | 930 |
| 0.57 | 0.99 | 0 | 0 | 732 |
| 0.60 | 1.00 | 0 | 0 | 541 |

Detectable edge: ~55 % of decisive 2v2 games (86 % power). For smaller edges
raise `--max-games` (and keep `--batch-games` a multiple of 6).

### Verdict

When the gate stops, `verdict.json` says **PASS** iff

* **(a)** the primary comparison crossed its success boundary, its exact
  one-sided p (share > null) < `--alpha` (0.01) and the lower end of its
  `--ci` (95 %) Clopper-Pearson interval > the null share;
* **(b)** no other comparison (every earlier champion, and other formats vs
  the current one) is significantly worse: exact one-sided p (share < null),
  **Holm-adjusted** across those comparisons, all > `--alpha-fail` (0.05), and
  no comparison crossed its interim failure boundary. Holm keeps the chance of
  blocking an equally strong candidate by a false "worse" call <= 0.05 across
  the ladder; `--no-holm` uses raw p-values (stricter for the candidate);
* **(c)** the external hook passed, when configured (below);
* and the gate is valid: illegal actions + bot exceptions <= `--max-errors`
  (0), voids <= `--max-voids` (3). Otherwise INVALID.

Else FAIL, with every reason listed. The verdict also reports, per comparison,
games, decisive games, draws, wins, share, CI, one- and two-sided exact p,
VP difference +- SE, per-arrangement wins, errors by side, decision ms and
overhead by side, and Holm-adjusted "better" p-values over all comparisons
(information only).

### External non-regression hook (c)

```bash
# once per champion: record its baseline
python3 scripts/league.py external-baseline champion-0 --key catanatron --cmd \
  '{python} scripts/bench_catanatron.py --spec {spec} --opponent value --our-seats 2 --games 400 --seed 900001 --hash-seed 0 --json {out}'
# in the gate: the same template for the candidate
python3 scripts/league.py gate ... --external-key catanatron --external-cmd '<same template>'
```

Placeholders `{spec}` (the resolved spec, weights path included), `{tree}`,
`{weights}`, `{out}`, `{commit}`, `{python}`; the command runs in the seat's
tree with `PYTHONPATH` = that tree (so it measures that commit's code) and must
write JSON with `wins` and `games` (or `win_rate` and `games`) to `{out}`
(the `bench_catanatron.py --json` summary qualifies). It runs only after (a)
and (b) passed; its result is cached in `external.json`. Comparison: one-sided
Fisher exact test "candidate rate < champion's recorded rate" must have
p > `--external-alpha` (0.05); `--external-tolerance X` additionally requires
rate >= champion rate - X. No recorded baseline -> FAIL with a hint. On
promotion the candidate's result becomes its own recorded baseline.

## Records, resume, status

```
league/gates/<GATE_ID>/gate.json       frozen configuration (candidate, champions + trees, rule, seeds, engine)
                      games.jsonl      one line per game (~1.7 KB): seeds, seats, specs, sides, winner, VPs,
                                       vp_diff, turns, actions, per-seat timing and errors, error log, engine
                      looks.jsonl      the evaluation after each look (boundaries, shares, decision)
                      progress.json    pid, host, games done, games/h, ETA to the next look and to max games
                      verdict.json     the verdict (only once stopped)
                      servers.json     first hello of each side; servers/*.log: server stderr
                      serve.py, shims/, weights/, external/
```

Everything is recomputed from `games.jsonl`, so a gate interrupted at any
point (a line cut mid-write is ignored and the file repaired) resumes exactly
(`gate --resume ID`; `--stop-after N` interrupts on purpose; `--workers N`
plays N games at once, each worker with its own servers; outcomes depend only
on the seeds, so the records are the same for any N). `league.py status`
prints the ladder, each gate's verdict or its state (RUNNING if its process is
alive and updating, else STOPPED with the resume command), current shares, and
the ETA from this run's games/h.

## Promotion

`league.py promote GATE_ID [--name] [--notes]` appends the candidate only if the
verdict is PASS, the ladder is exactly the one gated (same champions, same
commits, same current champion - otherwise re-gate), and the candidate is a
clean commit (a dirty or copied tree needs `--allow-uncommitted`, and could not
be materialised exactly later). Weights are copied to `league/weights/<name>.<ext>`
and hashed. Commit `league/champions.json`, `league/weights/` and the gate
directory together (the registry records the records' sha256).

## Compute (smoke measurements)

Smoke gate, 1 runner process + 4 bot servers in lockstep (about one core),
machine fully loaded by the running proof benchmark, runner niced:

| seat mix (4 players) | games | game time | games/h | actions/game | decide ms (bot) | overhead / decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2 x `heuristic:temp=0.3` (candidate, current tree) + 2 x champion-0 (commit 9984181, `git archive` + own C++ build) | 4 | 33.7 s | 427 | 740 | 0.40 / 15.8 | 0.77 / 0.84 ms |
| 2 x champion-0 spec from the current tree + 2 x champion-0 | 2 | 25.1 s | 287 | 783 | 12.6 / 10.6 | 0.69 / 0.74 ms |

Overhead per decision (round trip minus time in `bot.decide`) is ~0.7-0.8 ms,
of which ~0.3 ms is the server's `from_dict` + round-trip check; the runner's
`to_dict` + JSON costs 0.18 ms per action (once, shared by all servers); search
seats additionally receive every action as an observation. For search-vs-search
seats that is ~6 % of the decision time. Server start-up ~1 s, champion-0 C++
build 45 s. Budget: a gate against one champion at ~290 games/h per core needs
at most 1188 games (~4 h), about 840 on average for an equally strong
candidate; multiply by the number of champions; use `--workers` or separate
gate processes for more cores.

## Recommended protocol for new strategies and training runs

1. **Screen cheaply first** (paired ablations, `scripts/ablate.py`, short
   tournaments). Screens never promote anything.
2. **Freeze the candidate**: commit the code; keep the weights file (the gate
   copies and hashes it, so later retraining cannot change a running gate).
   Write down the spec and gate settings before starting (pre-registration;
   defaults unless there is a reason).
3. **Gate it against all champions** from a frozen runner worktree:
   `league.py gate --candidate-commit <sha> --candidate-spec ... [--weights ...] --name champion-<N> --notes "..."`.
4. **External non-regression** (`--external-cmd`, Catanatron benchmark) so a
   league-internal gain is not a self-play artefact.
5. **Promote on PASS** and commit the registry, weights and gate directory.
   On FAIL, do not re-run the same candidate with other seeds or settings until
   it passes: a change is a new commit and a new gate (and each gate has a 1 %
   false-promotion rate, so keep the number of gated candidates in mind).

Per kind of change:

* **New strategy module** (e.g. the win-path portfolio): the spec may stay the
  same - the commit is what differs. Primary format 2v2.
* **Trading changes** (e.g. counteroffers): in 2v2 the candidate's two seats
  can trade with a like-minded copy of themselves, which may flatter a trading
  change. Add `--formats 4p2v2,3p1v2`: in 3-player 1v2 the candidate faces two
  champions alone, and criterion (b) requires it not to be worse there either.
* **Information handling** (e.g. counted information): bots receive the full
  engine state, exactly as in `play_game`; an information-restricted strategy
  is only measured as far as the bot itself restricts what it reads. Back it
  with the external hook in the relevant information mode of the Catanatron
  benchmark (`bench_catanatron.py --info ...`).
* **Training runs**: `--weights` (or `model=` in the spec) with the training
  commit; one gate per run you intend to ship, not per checkpoint (screen
  checkpoints with cheaper tools).

## Limitations

* The referee engine is the runner's tree: a rule change applies to every
  champion's games, while each champion's internal simulations keep its own
  rules. Run long gates from a frozen runner (above).
* Worktree creation (`--method worktree`) is implemented but not exercised by
  the tests (they must not create worktrees); `archive` and `copy` are.
* Games of a block share the board and dice seed; the rotation makes their
  outcomes negatively correlated under the null, so the binomial test is
  conservative rather than exact. Draws are excluded from the binomial.
* Seeds are fixed per champion across gates: good for comparability, but do
  not tune a candidate on gate outcomes (use `--seed-base` for a fresh
  confirmatory gate if that happened).
* Holm in (b) is lenient toward the candidate by design (family-wise); the
  interim failure boundary uses Bonferroni over the comparisons.
* Lockstep servers use about one core per runner / worker; `--workers` was
  tested with in-process seats (deterministic records), not at scale with
  servers.
* Bot specs with a wall-clock budget (`time_limit`) would make results depend
  on machine load; the current `make_bot` specs are node-budgeted.
* The external comparison is against the champion's recorded result (another
  time, possibly another benchmark version); record baselines with the same
  command and seeds.
