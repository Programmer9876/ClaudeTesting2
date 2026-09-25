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
   (`--parser llm` forces it).
2. **Check** - the board is printed as ASCII with hex indices, plus a players
   table and parse warnings / low-confidence fields.  `--debug out.png`
   writes an overlay with every detected element so you can see what was
   read.
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
   --fix "bank.ore=3"   --fix "deck=14"
   --fix "players=red,blue,orange,green"
   ```

   Vertex and edge ids are catanbot's fixed numbering (see the debug
   overlay); `--save-state parsed.json` keeps the parse so you can iterate
   with `recommend --state parsed.json --fix ...` without re-parsing.
4. **Recommend** - an expectimax search (`--depth` turns of lookahead, 2 by
   default; `--beam` width; `--samples` determinizations of the hidden
   opponent hands) ranks your moves with the trained value net
   (`models/value_net.npz`, or `--model heuristic`).  Every action comes
   with a reason and the principal line.  Then the situational sections:
   trading plan (bank/port vs players, exploitable deals, arbitrage), 7 risk,
   knight / robber advice, dev cards (deck odds, monopoly, year of plenty),
   politics (target pressure, coalition, kingmaker-lite options) and the
   opponent profiles.

Other knobs:

* `--offer "blue:give=wood:1;get=ore:1"` - evaluate an incoming offer (blue
  gives you 1 wood for your 1 ore): accept or reject, with the reason.
* `--phase roll|main` - force whether you still have to roll (knight before
  the roll is considered) or are in the build phase.
* `--time 5` - cap the search time; `--json` - machine readable output.

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
what has been learned.  During self-play the bot maintains the same
profiles automatically from the actions it observes.

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

A well-calibrated evaluator's predicted win probabilities match the
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
`catanbot/vision/schema.py`).

## 3. Simulate, evaluate, train

```bash
python -m catanbot play --bots search,heuristic,heuristic,random --verbose
python -m catanbot eval --bots "search:depth=1,model=models/value_net.npz,search:depth=1,evaluator=heuristic,heuristic,random" --games 40 --workers 4
python -m catanbot train --iters 6 --games 150 --eval-games 40 --workers 4
```

Bot specs: `random[:end=0.3]`, `heuristic[:temp=0.3,eps=0.05]`,
`search[:depth=1,beam=4,expand=8,model=PATH|evaluator=heuristic,eps=0.05,temp=0.3,rolls=11,trades=3]`.

Training generates self-play games (seats drawn from the current best net
with different exploration settings, heuristic bots and older nets, 3 or 4
players), fits a new value net on the replay buffer, evaluates it against
the current best in a tournament and promotes it if it wins more.  Output:
`models/value_net.npz`, `models/value_net_log.json`, `models/value_net_train.log`.
Use `--resume` to continue.

## 4. Render a state

```bash
python -m catanbot render game.json shot.png --size 1280x800
```

Produces a synthetic Colonist.io-style screenshot (used for parser tests).

## Notes on real screenshots

The colour references in `catanbot/vision/colonist.py::Calibration` match
Colonist.io's default board.  If your theme differs, parse once, correct
with `--fix`, then call `calibrate_from_image(image, state)` from Python
and pass the returned `Calibration` to `parse_image` (or open an issue with
the screenshot).  Anything the parser cannot read is reported as a warning
and can be supplied with `--fix`.
