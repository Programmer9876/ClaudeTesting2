# Using catanbot

Install (Python 3.10+):

```bash
pip install -e .            # numpy + pillow
pip install opencv-python-headless   # faster / more robust screenshot parsing (recommended)
pip install anthropic                # optional: Claude-vision screenshot parser
```

Run everything through `python -m catanbot <command>` (or `catanbot` after
`pip install -e .`).

## 1. Analyse a Colonist.io screenshot

Take a screenshot of the whole game window: the full board with number
tokens, the player list on the left and your hand at the bottom.

```bash
python -m catanbot analyze shot.png --me red
```

What happens:

1. **Parse** - the computer-vision parser (`--parser cv`, default) finds the
   board, reads tiles / numbers / robber / roads / settlements / cities /
   ports, and best-effort reads the player panel (VP, cards, dev cards,
   knights, Longest Road / Largest Army badges, whose turn it is), your hand
   bar and the dice.  With `ANTHROPIC_API_KEY` set and the `anthropic`
   package installed, `--parser auto` uses Claude vision instead
   (`--parser llm` forces it).  `--no-ui` skips the panel / hand bar: seat
   order, hands, VP, dev cards, dice and whose turn it is are then guesses
   (a warning says so) and should be supplied with `--fix`.  An image in
   which no board can be found is reported as `error: could not parse ...`
   (exit code 2), never as a stack trace.
2. **Check** - the board is printed as ASCII with hex indices, plus a players
   table and parse warnings / low-confidence fields.  `--debug out.png`
   writes an overlay with every detected element plus catanbot's vertex
   ids (`v0`..`v53`) and edge ids (`e0`..`e71`), so you can see what was
   read and name pieces / ports in fixes; `--ids` prints the same numbering
   as a table (the six corner vertices and six edges of every hex).
3. **Fix** anything wrong with `--fix` (repeatable) and re-run, or start
   from the saved JSON:

   ```
   --fix "hex 4=wheat 8"           tile 4 is a wheat 8      (row by row, top-left = 0)
   --fix "hex 9=desert"
   --fix "robber=13"
   --fix "me=red"  --fix "current=blue"  --fix "dice=8"
   --fix "red.hand=wood:2,brick:1,ore:3"     your exact cards
   --fix "red.devs=knight:1,vp:1"            your exact dev cards (screenshots only show the count)
   --fix "blue.cards=6" --fix "blue.dev=1" --fix "blue.knights=2" --fix "blue.vp=5"
   --fix "blue.lr=1" --fix "green.la=0"      Longest Road / Largest Army flags
   --fix "red.settlements=+12,-5"  --fix "red.cities=+7"  --fix "red.roads=+13,-40"
   --fix "port 66=3:1"  --fix "port 6=ore"  --fix "port 6=none"   (ports under the hand bar are invisible)
   --fix "port 3-7=ore"                      the same port named by its two vertices
   --fix "bank.ore=3"   --fix "deck=14"
   --fix "players=red,blue,orange,green"
   --fix "blue.name=Kelsey"                  the name the game log uses (card counting, below)
   ```

   Vertex and edge ids are catanbot's fixed numbering (`--ids` prints it,
   `--debug` draws it).  `red.cities=+7` upgrades the settlement on vertex 7
   (the settlement is removed automatically).  A fix that cannot be honoured
   - an impossible number token (`hex 4=wheat 13`), an id out of range, a
   colour that is not in the game, a non-coastal port edge, `dice=x` - is
   rejected with a one-line `error: bad --fix ...` and exit code 2; players
   are only ever added through `players=`.  `--save-state parsed.json` keeps
   the parse so you can iterate with `recommend --state parsed.json --fix ...`
   without re-parsing.
4. **Recommend** - an expectimax search (`--depth` turns of lookahead, 1 by
   default: our whole turn with exact dice / draw / steal / trade odds.  `--depth 2`
   adds every opponent's simulated turn (about 5x the time); measured over 1200
   benchmark games it is not stronger than depth 1 with the heuristic evaluator,
   because the sampled opponents' turns are noisier than the decision margins, so
   use it for the explanation of what opponents can do to you, not for strength;
   `--depth 3` needs a large `--time` budget and is an analysis option only.  `--beam` width; `--samples` determinizations of the hidden
   opponent hands) ranks your moves with the trained value net
   (`models/value_net.npz` when it exists - otherwise the heuristic
   evaluator is used and the output says `heuristic evaluator (no trained
   value net found)`; `--model heuristic` selects it explicitly and a
   `--model PATH` that does not exist or cannot be loaded is an error).
   Every action comes with a reason and the principal line.  Then the
   situational sections:
   trading plan (bank/port vs players, exploitable deals, arbitrage), 7 risk
   (including how likely you are to be robbed before your next roll and by
   whom), knight / robber advice with a threat board (per player: VP, loaded
   / building / overextended, turns to win, the path, the missing cards and
   the rolls that produce them), dev cards (deck odds, monopoly, year of plenty),
   politics (target pressure, coalition, kingmaker-lite options) and the
   opponent profiles.

Other knobs:

* `--offer "blue:give=wood:1;get=ore:1"` - evaluate an incoming offer (blue
  gives you 1 wood for your 1 ore): accept or reject, with the reason.
  Both sides must be non-empty and disjoint and the proposer must be an
  opponent; if your hand cannot cover what they ask for, a warning says so
  and only rejecting is possible (fix a misread hand with `--fix`).
* `--phase roll|main` - force whether you still have to roll (knight before
  the roll is considered) or are in the build phase.
* `--time 5` - search time budget in seconds, shared by all `--samples`
  determinizations (no new sample starts once it is spent; the note
  `--time ... ran out` tells you how many were searched).  The deadline is
  checked between search levels, so a very deep search can still overrun it
  slightly - lower `--depth` / `--beam` for hard limits.
* Which `--depth` / `--time` (measured on 4 shared cores with the C++
  extension built, `scripts/build_cpp.sh`; see docs/CPP.md "Native lookahead"):
  the opponents' turns and the deeper lookahead now run natively, so with the
  heuristic evaluator a depth-2 search costs ~0.03 s per determinization
  sample and depth 3 ~0.1 s (before: 1.2 s and 6 s), i.e. `--depth 3
  --samples 4 --time 5` is comfortable.  With a trained value net
  (`models/value_net.npz`, 256/128 hidden) the net's own evaluation dominates:
  ~0.1 s per sample at depth 2 and ~1 s at depth 3, so keep `--depth 2` there
  or give `--time`.  Depth 3 is *affordable* but not yet *stronger*: in
  same-table tournaments the depth-3 search bot does not beat depth 2
  (see docs/CPP.md "Strength"), because the reduced sub-search that scores
  the deeper leaves is noisier than the depth-2 mean shift; until that is
  tuned, `--depth 2` remains the recommendation and `--depth 3` is for
  analysis.  Without the extension (`CATANBOT_NO_ACCEL=1` or not built) the
  old costs apply: depth 2 ~1 s per sample, depth 3 ~6 s.
* `--seed N` - seed for the sampled opponent hands; `--json` - machine
  readable output (`actions`, `advice`, `state` as given, `decision` = the
  situation that was actually searched, `search` = time / sample budget).

### Opponent profiles and politics

Keep one profile file per group you play with and pass it every time:

```bash
python -m catanbot analyze shot.png --me red --profiles friends.json \
    --event "blue accepted give ore get wood" \
    --event "orange rejected give 2 wheat get brick" \
    --event "green proposed give sheep get ore" \
    --event "green robbed red" \
    --event "blue bank 4 wood for 1 ore"
```

Events update the opponent's implied resource valuations, acceptance
tendencies and robber habits (with exponential decay), the political
capital between players (`"blue robbed red"`, `"blue traded red"`,
`"blue rejected red"`, `"blue blocked red"`, `"blue monopolized red"`,
`"blue helped red"`) and the coalition detector: give the deal detail
(`"blue traded orange give 2 ore get 1 wood"`, i.e. blue gave 2 ore and got
1 wood) and every trade is scored by how much value the parties sacrificed
versus their best alternative; blatant favours count quadratically more
than subtle ones, and blocs are reported in the Politics section.  `python -m catanbot profiles friends.json` shows
what has been learned (acceptance habits, political capital, recent events,
blocs and coalition signals).  `--profiles` only accepts files written by
catanbot (a JSON object with a `profiles` key) or a path that does not exist
yet; any other file is refused rather than overwritten.  During self-play
the bot maintains the same profiles automatically from the actions it
observes.

### Card counting from the game log (`--session`, `--game-log`)

A single screenshot shows every player's hand *size* but not which cards
they hold.  Colonist prints everything else that is public in its log panel
(dice and production, builds, development-card buys and plays, bank / port
and player trades, offers, Monopoly, Year of Plenty, robber steals, 7
discards), so the advisor can count cards like a strong human: exactly,
except for what is hidden from you - the card of a steal between two other
players, the types of cards discarded on a 7 (only the count is shown) and
unplayed development cards - which become probabilities.  Off by default:
without these two options nothing changes.

```bash
# every screenshot during the game, read by the Claude-vision parser (it also transcribes the log panel)
python -m catanbot analyze shot1.png --parser llm --me red --session game1.json
python -m catanbot analyze shot2.png --parser llm --me red --session game1.json
# log text you paste or type (any parser, or a saved / hand-written state)
python -m catanbot analyze shot.png --me red --session game1.json --game-log log.txt
python -m catanbot recommend --state saved.json --session game1.json --game-log - < log.txt
```

* `--session FILE` keeps the count across calls (created on the first call,
  one file per game; a file of another game or another seat is refused, and
  so is any JSON that is not a session).  With `--parser llm` the model also
  transcribes the visible log entries into the parse (`log` field, saved by
  `--save-state` and read again by `recommend --state`).
* `--game-log FILE` reads log text, one entry per line, oldest first (`-` =
  standard input).  It works with or without `--session`.  (The option is
  not called `--log`: that one appends win estimates for `calibrate`.)
* Every call sees a *window* of the log - the lines visible on this
  screenshot, or whatever you pasted.  Windows may overlap, repeat, or
  contain the whole log from the start: each is aligned with the entries
  already counted (longest overlap of the window's head with the counted
  tail; when repeated lines make that ambiguous, the alignment that agrees
  with the hand sizes on screen wins) and only the new entries are counted.
  A window that shares no entry with the counted log is counted whole with a
  warning that entries may be missing; one that lies inside the counted log
  (you scrolled up) adds nothing.
* Start the session at the beginning of the game (the setup placements and
  "received starting resources" lines in the first window): every hand then
  starts empty and the count is exact.  A session started mid-game begins
  from a production-weighted prior fitted to the hand sizes on screen (the
  output says so: "Session started mid-game", hands marked "estimated") and
  turns exact as the opponents spend their cards; with small hands (or a
  visible bank that leaves few possibilities) the prior already covers every
  possible hand and the count is exact from there.
* After the new entries, the count is checked against the screenshot: your
  own hand, every hand size and, when the parse carries it, the bank.  A
  mismatch means a missed or misread log entry: it is reported and the count
  is corrected without losing what it knows (a difference in hand size is
  branched in as cards of unknown type drawn from the production prior, or
  removed like a hidden discard; a payment the count says a player cannot
  make is explained by the smallest unrecorded gain).
* Player names: Colonist's log writes user names.  The Claude-vision parser
  reports players by colour; the computer-vision parser only knows colours,
  so tell it the names once with `--fix 'blue.name=Kelsey'` (the session
  remembers them).  "You" / "you" is you.  An unknown name is reported and
  its entries are skipped (the hand sizes on screen then correct the count).

The advisor then samples the opponents' hands for the search from the count
(`--samples` determinizations - the same code as the benchmark's
`--info counted` mode) and prints a "Card count" section right after the
recommended actions:

```
== Card count ==

 Counting from the start of the game: 157 log entries (12 new), 2 hand hypothesis(es).
 The search samples the opponents' hands from this count (4 determinization(s), --samples).
  blue (Bob): no cards
  orange (Carol): 1 wood, 2 ore certain; 1 card uncertain from hidden steals with green (Dave), blue (Bob): wood 88% / sheep 12%; most likely 2 wood, 2 ore (88%)
  green (Dave): 2 sheep, 1 ore (exact, 3 cards)
  ! 1 log line(s) not understood (skipped; extend the phrase table or retype them, see docs/USAGE.md): 'Happy settling'
```

"certain" is what the player holds in every hand still possible, the
uncertain cards come with the probability of each resource, "most likely" is
the single most probable hand.  `--json` adds a `card_count` object (the
same per player, plus the warnings and counters).

**The phrase table.**  Pasted text is read line by line through one table of
regular expressions, `PHRASES` in `catanbot/colonist_log.py` (first match
wins; extend or correct it there).  Written from the best available knowledge
of Colonist's wording - it has not been checked against a live game, so
compare it with yours:

| event | accepted lines (examples) |
|---|---|
| setup | `Alice placed a Settlement` / `Alice placed a Road` (free), `Alice received starting resources wood brick ore` |
| roll | `Alice rolled 5 3`, `Alice rolled 8`, `Alice rolled` (the value does not matter for counting) |
| production | `Bob got 2 wood, 1 ore` (also `received`) |
| build | `Alice built a Road` / `Settlement` / `City` (roads after Road Building are free) |
| development card | `Bob bought Development Card`; `Bob used Knight` / `Road Building` / `Year of Plenty` / `Monopoly` / `Victory Point` (also `played`) |
| Monopoly | `Bob stole 5 ore` (after `used Monopoly`), or `Bob used Monopoly and stole 5 ore` - the total; the split between the victims is inferred from the hand sizes |
| Year of Plenty | `Bob took from bank wood ore`, or `Bob used Year of Plenty and took wood ore` |
| bank / port trade | `Bob gave bank 4 wood and took 1 ore`, `Bob gave 3 wool and got 1 ore from bank`, `Bob traded 2 wool for 1 ore with bank` |
| player trade | `Alice traded 1 wood for 1 ore with Bob` (Alice gave the wood, got the ore) |
| offer / counter-offer | `Alice wants to give 1 wood for 1 ore`, `Bob counter-offered 1 ore for 2 wood` (hard evidence they hold what they offer; soft that they lack what they ask for) |
| steal | `Carol stole a card from Bob` (hidden), `You stole ore from Bob`, `Bob stole wood from you` |
| 7 discard | `Bob discarded 4 cards` (hidden), `You discarded 2 wood, 2 ore`; no cards at all = half the hand |
| robber / turn | `Bob moved Robber to 6 wheat`; `Bob ended their turn`, `Bob's turn` |
| ignored | Largest Army / Longest Road, `won the game`, `No player gets resources`, `accepted` / `rejected` / `cancelled`, `is selecting ...`, joined / left |

Cards may be written as counts (`2 wood, 1 brick`), repeated words (`wood
wood brick`), with multipliers (`2x wood`, `wood x2`), Colonist's names
(`lumber`, `brick`, `wool`, `grain`, `ore`), or letters (upper case: `W`/`L`
wood, `B`/`C` brick, `S` sheep, `G` wheat, `O` ore, `?` unknown: `WWB`);
`a card` / `2 cards` / `?` are face-down cards.  Colonist draws cards as
icons, so a copied log may lose them ("Bob got"): such lines are reported as
unreadable and the hand sizes on screen correct the count; retype them
(`Bob got 2 wood`) for an exact count.  Lines that match no phrase are
reported (quoted) in the Card count section, once per session.

**Assumptions to check against a real Colonist game** (the table is a best
guess): the exact verbs (`got` vs `received`, `built a` vs `placed a` after
setup, `bought Development Card`, `used <card>`, `took from bank`, `gave bank
... and took ...`, `traded ... for ... with ...`, `wants to give ... for
...`, the counter-offer wording, `moved Robber to`, `discarded`); whether a
Monopoly is logged as one line or two and whether it shows the total or the
amount per victim; whether Year of Plenty has its own "took" line; whether a
third-party steal shows a card back and whether discards show their cards
(if Colonist shows them, they are simply counted exactly); whether turn ends
are logged at all; whether the log says "You" for you; whether a free Road
Building road is logged differently; whether colons follow the verbs (all
optional in the table).

**Limitations.**  The count is only as good as the log it sees: a missed
line is detected and repaired from the hand sizes, but the repair is a guess
(production-weighted).  Several large hidden discards on one 7 can exceed
the hypothesis cap (4096 joint hands) when the bank is not visible - the
least likely hands are dropped; a visible bank resolves them exactly.  A
Monopoly's split between victims is taken from the hand sizes on screen when
the log only gives the total (ambiguous only if another unreadable entry
follows in the same window).  The Claude-vision transcription can misread an
icon: the alignment of two windows needs identical entries, so a misread
line in the overlap may look like a gap (reported, then resynchronised).
Development-card types stay a probability (the public pool: the 25-card deck
minus every played card and your own); played types other than knights are
only known from the log, so a mid-game session counts only the knights shown
on screen.  The session does not validate turn order or legality.  Only the
search uses the count (its determinizations); the Trading, Knight / robber
and offer-response sections keep their own hand estimates, and `watch`
prints its compact summary without the Card count section (the session is
still updated).

### Live advisor while you play (`watch`)

```bash
pip install mss
python -m catanbot watch --me red --interval 6 --profiles friends.json      # whole primary monitor
python -m catanbot watch --me red --region 0,0,1600,1000 --depth 2         # a screen region
```

Every few seconds the screen is captured, parsed, and (when the position
changed) the top three moves with one-line reasons are printed, plus the
usual notes.  You still make every move yourself: this automates the
screenshot loop, not the play.  `--from-dir DIR` replays saved screenshots
instead of capturing (used by the tests).

### Validating the bot on your own games

Human games are the real test, and you do not need an app for it: log the
positions you analyse, record who won, and score the win estimates.

```bash
python -m catanbot analyze shot1.png --me red --log mygames.jsonl --game 2026-09-25a
python -m catanbot analyze shot2.png --me red --log mygames.jsonl --game 2026-09-25a
python -m catanbot outcome mygames.jsonl --game 2026-09-25a --winner blue
python -m catanbot calibrate --log mygames.jsonl          # Brier score + reliability table
python -m catanbot calibrate --selfplay 20                # same check on fresh self-play games
```

`outcome` checks that the game id has logged positions and that the winner
is one of its players; `calibrate` needs `--log` and/or `--selfplay`
(`--model` and `--seed` apply to the self-play games).  A well-calibrated
evaluator's predicted win probabilities match the
observed win rates bin by bin; the trajectory printed per game shows
whether the estimate moved the right way as the game unfolded.  Log games
against strong players in particular - that is where the value net and
the political model are tested, not against average opponents.

## 2. Recommend from a JSON state

```bash
python -m catanbot recommend --state game.json --me red --depth 2
```

`game.json` may be a full `GameState` (`GameState.to_dict()` format) or a
parsed-screenshot dict (the format `--save-state` writes, documented in
`catanbot/vision/schema.py`).  `--fix` corrections apply to the
parsed-screenshot format only; with a full `GameState` they are refused
(edit that JSON directly).  Every `analyze` option except the parser ones
(`--me`, `--offer`, `--profiles`, `--event`, `--ids`, `--json`, `--log`,
`--session`, `--game-log`, ...) works here too; a parsed state saved with a
`log` field (the Claude-vision parser's transcription) feeds `--session`.

## 3. Simulate, evaluate, train

```bash
python -m catanbot play --bots search,heuristic,heuristic,random --verbose
python -m catanbot eval --bots "search:depth=1,model=models/value_net.npz,search:depth=1,evaluator=heuristic,heuristic,random" --games 40 --workers 4
python -m catanbot train --iters 6 --games 150 --eval-games 40 --workers 4
```

Bot specs: `random[:end=0.3]`, `heuristic[:temp=0.3,eps=0.05]`,
`search[:depth=1,beam=4,expand=8,model=PATH|evaluator=heuristic,eps=0.05,temp=0.3,rolls=11,trades=3]`.
Specs are separated by commas, semicolons or spaces; a comma-separated
`key=value` always belongs to the preceding spec, so the `eval` line above
is four bots (`model=...` needs a trained net at that path).  An unknown
bot name or option is reported as `error: bad bot spec ...`.  Games have 3
or 4 players (`--players`; `play` pads a shorter list by repeating the last
spec and refuses more specs than seats).

Training generates self-play games (seats drawn from the current best net
with different exploration settings, heuristic bots and older nets, 3 or 4
players), fits a new value net on the replay buffer, evaluates it against
the current best in a tournament and promotes it if it wins more.  Output:
`models/value_net.npz`, `models/value_net_log.json`, `models/value_net_train.log`.
Use `--resume` to continue.  Besides `--games` (search-bot games per
iteration) every iteration also plays `--heur-games` cheap heuristic-bot
games (default 600) - lower it for a quick smoke run, e.g.
`train --iters 1 --games 2 --heur-games 2 --eval-games 2 --workers 1 --epochs 1`.
`train --help` lists the remaining knobs (`--depth`, `--beam`, `--expand`,
`--epochs`, `--hidden`, `--blend`, `--buffer`, `--max-turns`, ...).
`--depth` for self-play: with the extension built a depth-2 search bot costs
~0.02-0.03 s per decision with the heuristic evaluator and ~0.1 s with the
net (Python: 0.07 s / 0.3 s), so `--depth 2` is now the sensible default for
self-play data; `--depth 3` costs 0.03-0.05 s (heuristic) / ~1 s (net) per
decision and does not play better yet (docs/CPP.md "Strength"), so it is not
worth its games.  Bots seated with `CATANBOT_NO_NATIVE_SEARCH=1` or
`CATANBOT_NO_ACCEL=1` take the Python path (same strength, slower).

## 4. Render a state

```bash
python -m catanbot render game.json shot.png --size 1280x800 --me red
```

Produces a synthetic Colonist.io-style screenshot (used for parser tests).
`--size` is `WIDTHxHEIGHT` in pixels; `--me` picks whose hand is drawn.

## 5. Replay archive (game picker and "what our bot knew")

One static page lists every logged game of the strength proof and the
benchmarks. Any game opens in a step-through replay. In the public-information
tests (T7, T8, T9, T11) it shows each opponent's real hand next to what our
bot believed at that moment. The design is in `docs/designs/replay_archive.md`.

```bash
S=/tmp/claude-0/-home-user-ClaudeTesting2/e59cf40d-e496-56e7-a6ac-661eab3c04d1/scratchpad
# export (nice -n 10, at most 2 workers); writes $S/replay_archive/{index.html,index.json.gz,shards/}
python3 scripts/replay_export.py --tests T2,T8,H1 --workers 2      # omit --tests for all 18
# rewrite index.html only (after editing the page or the core), from the existing parts
python3 scripts/replay_export.py --tests T2,T8,H1 --merge-only
# validate the bundle against the original logs, results files and manifests
node scripts/check_replay_bundle.mjs --bundle $S/replay_archive           # add --expect-all for the full 18 tests
```

The page template is `scripts/replay_archive_template.html`; the exporter
inlines `scripts/replay_core.js` between its `BEGIN`/`END replay_core.js`
markers, so the bundle has no `.js` file. Publish `index.html` as the page, with
`index.json.gz` and `shards/*.json.gz` as supporting files. Deep links are bare
tokens: `#T8-123` (a game), `#T8-123.57` (position 57), `#T8~won` (a filtered
list), `#selftest` (decode check against the host).

## Notes on real screenshots

The colour references in `catanbot/vision/colonist.py::Calibration` match
Colonist.io's default board.  If your theme differs, parse once, correct
with `--fix`, then call `calibrate_from_image(image, state)` from Python
and pass the returned `Calibration` to `parse_image` (or open an issue with
the screenshot).  Anything the parser cannot read is reported as a warning
and can be supplied with `--fix`.

## Exit codes

`0` success; `2` for any input problem (unreadable or wrong-format file, an
unknown colour, a bad `--fix` / `--offer` / bot spec, an option out of
range), reported as a single `error: ...` line on stderr.
