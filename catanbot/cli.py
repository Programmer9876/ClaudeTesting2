"""Command line interface: ``python -m catanbot <command>``.

    analyze IMAGE      parse a Colonist.io screenshot and recommend a move
    recommend          recommend a move from a JSON state (GameState or parsed-screenshot format)
    play               simulate a game between bots
    eval               tournament between bot specs
    train              self-play training of the value net
    render             render a JSON state as a synthetic screenshot
    profiles           show a saved opponent-profile file
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from . import engine as E
from .state import (GameState, PHASE_MAIN, PHASE_ROLL, PHASE_TRADE_RESPONSE, PLAYER_COLORS, TradeOffer)

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "value_net.npz")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class UsageError(Exception):
    pass


def _res_index(name: str) -> int:
    r = B.RESOURCE_ALIASES.get(name.strip().lower())
    if r is None or r == B.DESERT:
        raise UsageError(f"unknown resource '{name}'")
    return r


def _counts_from_text(text: str) -> List[int]:
    """'wood:2,brick:1' or 'wood 2 brick 1' -> 5 counts."""
    out = [0] * 5
    toks = text.replace(":", " ").replace(",", " ").replace("=", " ").split()
    i = 0
    while i < len(toks):
        name = toks[i]
        n = 1
        if i + 1 < len(toks) and toks[i + 1].isdigit():
            n = int(toks[i + 1])
            i += 1
        out[_res_index(name)] += n
        i += 1
    return out


def _player_entry(parsed: dict, color: str) -> dict:
    color = color.lower()
    for p in parsed.get("players", []):
        if p.get("color", "").lower() == color or (p.get("name") or "").lower() == color:
            return p
    if color not in PLAYER_COLORS:
        raise UsageError(f"unknown player '{color}' (players: {[p['color'] for p in parsed.get('players', [])]})")
    p = {"color": color, "name": color, "vp": 0, "cards": 0, "dev_cards": 0, "knights": 0,
         "longest_road": False, "largest_army": False, "settlements": [], "cities": [], "roads": []}
    parsed.setdefault("players", []).append(p)
    return p


def apply_fix(parsed: dict, fix: str) -> None:
    """Apply one ``--fix`` correction to a parsed-screenshot dict (see docs/USAGE.md)."""
    text = fix.strip()
    if "=" not in text:
        raise UsageError(f"fix '{fix}' must contain '='")
    key, value = [t.strip() for t in text.split("=", 1)]
    kl = key.lower()
    vl = value.lower()
    if kl.startswith("hex "):
        h = int(kl.split()[1])
        parts = vl.split()
        res = parts[0]
        if _res_index(res) if res != "desert" else True:
            pass
        num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if res == "desert":
            parsed["hexes"][h] = {"resource": "desert", "number": None}
        else:
            parsed["hexes"][h] = {"resource": B.RESOURCE_NAMES[_res_index(res)], "number": num}
        return
    if kl == "robber":
        parsed["robber"] = int(vl)
        return
    if kl == "me":
        parsed["me"] = vl
        return
    if kl == "current":
        parsed["current_player"] = vl
        return
    if kl == "dice":
        parsed["dice"] = int(vl)
        return
    if kl in ("deck", "dev_deck", "dev_deck_remaining"):
        parsed["dev_deck_remaining"] = int(vl)
        return
    if kl == "players":
        for c in vl.replace(",", " ").split():
            _player_entry(parsed, c)
        return
    if kl.startswith("port "):
        e = int(kl.split()[1])
        ports = [p for p in (parsed.get("ports") or []) if int(p["edge"]) != e]
        if vl not in ("none", "remove", ""):
            ports.append({"edge": e, "type": "3:1" if vl in ("3:1", "generic", "any") else B.RESOURCE_NAMES[_res_index(vl)]})
        parsed["ports"] = sorted(ports, key=lambda p: p["edge"])
        return
    if kl.startswith("bank."):
        parsed.setdefault("bank", {B.RESOURCE_NAMES[r]: 19 for r in range(5)})
        parsed["bank"][B.RESOURCE_NAMES[_res_index(kl[5:])]] = int(vl)
        return
    if "." in kl:
        color, attr = kl.split(".", 1)
        p = _player_entry(parsed, color)
        if attr in ("hand", "resources"):
            counts = _counts_from_text(value)
            p["resources"] = {B.RESOURCE_NAMES[r]: counts[r] for r in range(5)}
            p["cards"] = sum(counts)
            return
        if attr == "cards":
            p["cards"] = int(vl)
            return
        if attr in ("dev", "dev_cards"):
            p["dev_cards"] = int(vl)
            return
        if attr in ("devs", "dev_types"):
            # exact dev card types, e.g. 'red.devs=knight:1,vp:1,monopoly:1'
            counts: Dict[str, int] = {}
            toks = value.replace(":", " ").replace(",", " ").split()
            i = 0
            names = {"knight": "knight", "knights": "knight", "vp": "victory_point", "victory_point": "victory_point",
                     "rb": "road_building", "road_building": "road_building", "roads": "road_building",
                     "yop": "year_of_plenty", "year_of_plenty": "year_of_plenty", "plenty": "year_of_plenty",
                     "monopoly": "monopoly", "mono": "monopoly"}
            while i < len(toks):
                name = names.get(toks[i].lower())
                if name is None:
                    raise UsageError(f"unknown dev card type '{toks[i]}'")
                n = 1
                if i + 1 < len(toks) and toks[i + 1].isdigit():
                    n = int(toks[i + 1])
                    i += 1
                counts[name] = counts.get(name, 0) + n
                i += 1
            p["dev_cards"] = counts
            return
        if attr == "knights":
            p["knights"] = int(vl)
            return
        if attr == "vp":
            p["vp"] = int(vl)
            return
        if attr in ("lr", "longest_road"):
            p["longest_road"] = vl in ("1", "true", "yes")
            return
        if attr in ("la", "largest_army"):
            p["largest_army"] = vl in ("1", "true", "yes")
            return
        if attr in ("settlements", "cities", "roads"):
            cur = list(p.get(attr, []))
            for tok in value.replace(",", " ").split():
                if tok.startswith("-"):
                    cur = [x for x in cur if x != int(tok[1:])]
                else:
                    x = int(tok.lstrip("+"))
                    if x not in cur:
                        cur.append(x)
            p[attr] = sorted(cur)
            return
        raise UsageError(f"unknown player attribute '{attr}' in fix '{fix}'")
    raise UsageError(f"unrecognised fix '{fix}'")


def load_state_file(path: str) -> Tuple[GameState, Optional[dict]]:
    """Load either a GameState JSON or a parsed-screenshot JSON. Returns (state, parsed_or_None)."""
    with open(path) as f:
        d = json.load(f)
    if "parsed" in d and isinstance(d["parsed"], dict):
        d = d["parsed"]
    if "phase" in d or "version" in d:
        return GameState.from_dict(d), None
    from .vision.schema import parsed_to_state
    return parsed_to_state(d), d


def load_evaluator(model_path: Optional[str]) -> Tuple[Any, str]:
    if model_path and model_path.lower() in ("heuristic", "none"):
        from .heuristic import HeuristicEvaluator
        return HeuristicEvaluator(), "heuristic evaluator"
    path = model_path or DEFAULT_MODEL
    if os.path.exists(path):
        try:
            from .model import ValueNet
            return ValueNet.load(path), f"value net {path}"
        except Exception as ex:  # pragma: no cover
            print(f"could not load value net {path}: {ex}; using the heuristic evaluator", file=sys.stderr)
    from .heuristic import HeuristicEvaluator
    return HeuristicEvaluator(), "heuristic evaluator (no trained value net found)"


def load_profiles(path: Optional[str], state: GameState):
    from .opponent_model import OpponentModel
    from .politics import PoliticalState
    model = OpponentModel(state)
    politics = PoliticalState(state.num_players)
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        model = OpponentModel.from_dict(d)
        model.attach(state)
        if d.get("politics"):
            politics = PoliticalState.from_named_dict(d["politics"], state)
    return model, politics


def save_profiles(path: str, model, politics, state: GameState) -> None:
    d = model.to_dict()
    d["politics"] = politics.to_named_dict(state)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
_RES_ABBR = {B.WOOD: "WO", B.BRICK: "BR", B.SHEEP: "SH", B.WHEAT: "WH", B.ORE: "OR", B.DESERT: "DE"}


def board_ascii(state: GameState) -> str:
    lines = []
    for row in B.HEX_ROWS:
        cells = []
        for h in row:
            res, num = state.hexes[h]
            label = f"{h:2d}:{_RES_ABBR[res]}{num if num else '':<2}"
            if h == state.robber:
                label += "*"
            else:
                label += " "
            cells.append(label)
        indent = " " * (3 * (5 - len(row)))
        lines.append(indent + "  ".join(cells))
    lines.append("(* = robber; ids are hex indices)")
    ports = {}
    for v, t in state.ports.items():
        ports.setdefault(B.PORT_NAMES[t], []).append(v)
    if ports:
        lines.append("Ports: " + "; ".join(f"{t} at vertices {sorted(vs)}" for t, vs in sorted(ports.items())))
    return "\n".join(lines)


def players_table(state: GameState, me: int) -> str:
    rows = ["  player      VP  cards dev  kn  LR LA  S/C/R   hand"]
    for i, p in enumerate(state.players):
        name = (p.name or p.color)[:10]
        vp = state.total_vp(i) if i == me else state.public_vp(i)
        cards = p.total_resources if p.hand_known else p.hand_size
        dev = p.total_dev if p.dev_known else p.dev_count
        hand = ", ".join(f"{p.resources[r]} {B.RESOURCE_NAMES[r]}" for r in range(5) if p.resources[r]) if p.hand_known else "?"
        mark = "*" if i == me else (">" if i == state.current else " ")
        rows.append(f"{mark} {name:<10} {vp:>3} {cards:>5} {dev:>4} {p.played_knights:>3}   "
                    f"{'x' if state.longest_road_owner == i else '-'}  {'x' if state.largest_army_owner == i else '-'}  "
                    f"{len(p.settlements)}/{len(p.cities)}/{len(p.roads):<3}  {hand}")
    rows.append("(* = you, > = player to move)")
    return "\n".join(rows)


def _section(title: str) -> str:
    return f"\n== {title} ==\n"


# ---------------------------------------------------------------------------
# core: recommend
# ---------------------------------------------------------------------------
def recommend_for_state(state: GameState, me: int, args, parsed: Optional[dict] = None) -> Dict[str, Any]:
    """Run the search + advice modules; returns a JSON-able report dict."""
    from .devcards import dev_card_advice
    from .discard import explain_seven_risk
    from .inference import is_fully_known
    from .politics import political_trade_options, runway_advice
    from .robber import best_robber_move, should_play_knight
    from .search import SearchConfig, Searcher, search_determinized
    from .trading import trade_advice

    report: Dict[str, Any] = {"warnings": [], "notes": []}
    evaluator, ev_name = load_evaluator(getattr(args, "model", None))
    report["evaluator"] = ev_name
    model, politics = load_profiles(getattr(args, "profiles", None), state)
    for ev in getattr(args, "event", None) or []:
        err = model.observe_event(state, ev)
        err2 = politics.observe_event(state, ev)
        if err and err2:
            report["warnings"].append(f"event '{ev}': {err}")
    # Which decision are we making?
    offer = getattr(args, "offer", None)
    if offer:
        proposer_color, spec = offer.split(":", 1)
        give = get = [0] * 5
        for part in spec.split(";"):
            k, v = part.split("=", 1)
            if k.strip().lower() == "give":
                give = _counts_from_text(v)
            elif k.strip().lower() == "get":
                get = _counts_from_text(v)
        state.pending_trade = TradeOffer(state.player_index(proposer_color.strip().lower()), give, get)
        state.current = state.pending_trade.proposer
        state.phase = PHASE_TRADE_RESPONSE
        state.trade_responder = me
        state.dice = state.dice or 8
        report["notes"].append(f"Evaluating the offer from {proposer_color}: you receive "
                               f"{A._counts_str(give)} and pay {A._counts_str(get)}.")
    elif state.current != me:
        report["notes"].append(f"It is {state.players[state.current].name or state.players[state.current].color}'s turn; "
                               "the recommendation is for your next turn with your current hand.")
        state.current = me
        state.phase = PHASE_ROLL
        state.dice = 0
        state.pending_trade = None
        state.discard_queue = []
    elif state.phase == PHASE_MAIN and not state.dice:
        state.phase = PHASE_ROLL
    if getattr(args, "phase", None):
        state.phase = args.phase
        if args.phase == PHASE_MAIN and not state.dice:
            state.dice = 8
    cfg = SearchConfig(depth=args.depth, beam=args.beam, expand=max(6, args.beam + 4),
                       time_limit=getattr(args, "time", None), opp_roll_samples=4)
    rng = random.Random(getattr(args, "seed", 0) or 0)
    t0 = time.time()
    if is_fully_known(state):
        results = Searcher(evaluator, cfg, model, None, politics).search(state, me, rng)
    else:
        results = search_determinized(state, me, evaluator, cfg, samples=max(1, args.samples), rng=rng,
                                      model=model, politics=politics)
    report["search_seconds"] = round(time.time() - t0, 2)
    report["actions"] = [r.to_dict(state) for r in results[:8]]
    if any(r.action[0] == A.BUY_DEV for r in results[:5]):
        report["notes"].append("Dev cards bought this turn cannot be played until next turn (VP cards count immediately).")
    mep = state.players[me]
    if not mep.dev_known and mep.dev_count > 0:
        report["notes"].append(f"You hold {mep.dev_count} dev card(s) of unknown type; tell me with "
                               f"--fix '{mep.color}.devs=knight:1,vp:1' for exact knight / monopoly advice.")
    # advice sections
    advice: Dict[str, List[str]] = {}
    try:
        advice["trading"] = trade_advice(state, me, model=model)
    except Exception as ex:  # pragma: no cover
        advice["trading"] = [f"(trade advice unavailable: {ex})"]
    advice["seven_risk"] = [explain_seven_risk(state, me)]
    ok, why = should_play_knight(state, me)
    h, victim, reason = best_robber_move(state, me, target_weights=politics.robber_target_weights(state, me))
    advice["robber"] = [("Play a knight now: " if ok else "Hold the knight: ") + why,
                        f"Best robber target: {A.describe((A.MOVE_ROBBER, h, victim), state)} - {reason}"]
    advice["dev_cards"] = dev_card_advice(state, me)
    advice["opponents"] = model.summary(state, me=me)
    pol = runway_advice(state, me, politics)
    try:
        for opt in political_trade_options(state, me, evaluator, politics, model=model)[:2]:
            pol.append("Option: " + A.describe(opt["action"], state) + " - " + opt["reason"])
    except Exception:
        pass
    advice["politics"] = pol
    report["advice"] = advice
    report["me"] = me
    report["state"] = state.to_dict()
    if parsed is not None:
        report["parsed"] = parsed
    profiles = getattr(args, "profiles", None)
    if profiles:
        try:
            save_profiles(profiles, model, politics, state)
        except OSError as ex:  # pragma: no cover
            report["warnings"].append(f"could not save profiles: {ex}")
    return report


def print_report(state: GameState, me: int, report: Dict[str, Any], parse_warnings: Sequence[str] = (),
                 confidence: Optional[Dict[str, float]] = None) -> None:
    print(_section("Board"))
    print(board_ascii(state))
    print()
    print(players_table(state, me))
    if parse_warnings or confidence:
        print(_section("Parse warnings"))
        for w in parse_warnings:
            print(" - " + w)
        if confidence:
            low = [f"{k} {v:.0%}" for k, v in confidence.items() if v < 0.7]
            if low:
                print(" - low confidence: " + ", ".join(low))
        if not parse_warnings:
            print(" (none)")
    for n in report.get("notes", []):
        print("NOTE: " + n)
    for w in report.get("warnings", []):
        print("WARNING: " + w)
    print(_section(f"Recommended actions ({report['evaluator']}, {report['search_seconds']}s)"))
    for k, a in enumerate(report["actions"][:5]):
        line = " -> ".join(a["line"][:4])
        print(f"{k + 1}. [{a['value']:.3f}] {a['text']}")
        if a["explanation"]:
            print(f"     why: {a['explanation']}")
        if len(a["line"]) > 1:
            print(f"     line: {line}")
    titles = [("trading", "Trading"), ("seven_risk", "7-protection"), ("robber", "Knight / robber"),
              ("dev_cards", "Development cards"), ("politics", "Politics"), ("opponents", "Opponents")]
    for key, title in titles:
        lines = report["advice"].get(key) or []
        if not lines:
            continue
        print(_section(title))
        for l in lines:
            print(l if l.startswith("  ") else " " + l)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_analyze(args) -> int:
    parser_kind = args.parser
    if parser_kind == "auto":
        parser_kind = "cv"
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                parser_kind = "llm"
            except ImportError:
                pass
    if parser_kind == "llm":
        try:
            from .vision.llm import parse_with_claude
            result = parse_with_claude(args.image, me=args.me)
        except RuntimeError as ex:
            print(f"LLM parser unavailable: {ex}\nFalling back to the computer-vision parser.", file=sys.stderr)
            parser_kind = "cv"
    if parser_kind == "cv":
        try:
            from .vision.colonist import parse_image
        except ImportError as ex:
            print(f"computer-vision parser needs opencv-python-headless / pillow: {ex}", file=sys.stderr)
            return 2
        result = parse_image(args.image, me=args.me, read_ui=not args.no_ui)
    parsed = result.parsed
    for fx in args.fix or []:
        try:
            apply_fix(parsed, fx)
        except (UsageError, ValueError, IndexError, KeyError) as ex:
            print(f"bad --fix '{fx}': {ex}", file=sys.stderr)
            return 2
    from .vision.schema import parsed_to_state, validate
    warnings = list(result.warnings)
    if args.fix:
        warnings = [w for w in warnings if not w.startswith("expected")] + validate(parsed)
    state = parsed_to_state(parsed)
    if parsed.get("dice"):
        state.phase = PHASE_MAIN
        state.dice = int(parsed["dice"])
    me = state.player_index(parsed.get("me") or state.players[0].color)
    if args.debug:
        try:
            from .vision.colonist import draw_debug
            draw_debug(args.image, result).save(args.debug)
            print(f"debug overlay written to {args.debug}", file=sys.stderr)
        except Exception as ex:  # pragma: no cover
            print(f"could not write debug overlay: {ex}", file=sys.stderr)
    if args.save_state:
        with open(args.save_state, "w") as f:
            json.dump({"parsed": parsed, "state": state.to_dict(), "warnings": warnings}, f, indent=1)
    report = recommend_for_state(state, me, args, parsed)
    report["parse_warnings"] = warnings
    report["confidence"] = {k: float(v) for k, v in result.confidence.items()}
    report["parser"] = parser_kind
    if args.json:
        print(json.dumps(report, indent=1, default=float))
    else:
        print(f"parser: {parser_kind}")
        print_report(state, me, report, warnings, result.confidence)
    return 0


def cmd_recommend(args) -> int:
    state, parsed = load_state_file(args.state)
    if parsed is not None and args.fix:
        for fx in args.fix:
            apply_fix(parsed, fx)
        from .vision.schema import parsed_to_state
        state = parsed_to_state(parsed)
        if parsed.get("dice"):
            state.phase = PHASE_MAIN
            state.dice = int(parsed["dice"])
    if args.me:
        me = state.player_index(args.me.lower())
    elif parsed is not None and parsed.get("me"):
        me = state.player_index(parsed["me"])
    else:
        me = E.acting_player(state) if state.phase != "game_over" else 0
    report = recommend_for_state(state, me, args, parsed)
    if args.json:
        print(json.dumps(report, indent=1, default=float))
    else:
        print_report(state, me, report)
    return 0


def cmd_play(args) -> int:
    from .selfplay import make_bot, play_game
    specs = [s.strip() for s in args.bots.split(",") if s.strip()]
    n = args.players or len(specs)
    while len(specs) < n:
        specs.append(specs[-1] if specs else "heuristic")
    specs = specs[:n]
    rng = random.Random(args.seed)
    for g in range(args.games):
        bots = [make_bot(s) for s in specs]

        def on_action(state, action, player):
            if args.verbose:
                p = state.players[player]
                text = A.describe(action, state)
                expl = bots[player].explain(state) if hasattr(bots[player], "explain") else None
                print(f"t{state.turn:3d} {p.color:<7} {text}" + (f"   [{expl}]" if expl and action[0] != A.ROLL else ""))

        r = play_game(bots, rng=random.Random(rng.random()), max_turns=args.max_turns, specs=specs, on_action=on_action)
        winner = r.specs[r.winner] if r.winner >= 0 else "nobody"
        print(f"game {g + 1}: winner seat {r.winner} ({winner}) after {r.turns} turns, VP {r.vps}, "
              f"{r.actions} actions, {r.duration:.1f}s")
    return 0


def cmd_eval(args) -> int:
    from .selfplay import tournament
    specs = [s.strip() for s in args.bots.split(",") if s.strip()]
    res = tournament(specs, games=args.games, workers=args.workers, seed=args.seed, num_players=args.players,
                     max_turns=args.max_turns)
    print(f"{args.games} games, avg {res['avg_turns']:.0f} turns")
    print(f"{'bot':<50} games  win%   avgVP")
    for s, st in res["summary"].items():
        print(f"{s:<50} {st['games']:>5}  {st['win_rate'] * 100:5.1f}  {st['avg_vp']:5.2f}")
    return 0


def cmd_render(args) -> int:
    from .vision.synth import render_to_file
    state, parsed = load_state_file(args.state)
    w, h = [int(x) for x in args.size.lower().split("x")]
    render_to_file(state, args.out, size=(w, h), seed=args.seed, me=args.me or 0, jitter=False)
    print(f"wrote {args.out}")
    return 0


def cmd_profiles(args) -> int:
    from .opponent_model import OpponentModel
    if not os.path.exists(args.file):
        print(f"no such file: {args.file}")
        return 2
    with open(args.file) as f:
        d = json.load(f)
    model = OpponentModel.from_dict(d)
    for name, prof in model.profiles.items():
        print(f"{name}: {prof.style_summary()} (observations {prof.observations})")
    pol = d.get("politics")
    if pol and pol.get("colors"):
        cols = pol["colors"]
        print("political capital (row = how the column player views the row player):")
        print("           " + " ".join(f"{c[:6]:>6}" for c in cols))
        for i, c in enumerate(cols):
            print(f"{c[:10]:<10} " + " ".join(f"{pol['capital'][i][j]:6.2f}" for j in range(len(cols))))
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def _add_recommend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--me", help="your colour (default: from the screenshot / state)")
    p.add_argument("--depth", type=int, default=2, help="search depth in turns (1-3)")
    p.add_argument("--beam", type=int, default=6)
    p.add_argument("--samples", type=int, default=4, help="determinizations of hidden hands")
    p.add_argument("--model", help="value net path, or 'heuristic'")
    p.add_argument("--profiles", help="JSON file with opponent profiles + political capital (read and updated)")
    p.add_argument("--event", action="append", help="observed event, e.g. 'blue accepted give ore get wood' or 'blue robbed red'")
    p.add_argument("--fix", action="append", help="correct the parse, e.g. 'hex 4=wheat 8', 'red.hand=wood:2,ore:1'")
    p.add_argument("--offer", help="evaluate an incoming offer: 'blue:give=wood:1;get=ore:1' (blue gives you wood for your ore)")
    p.add_argument("--phase", choices=[PHASE_ROLL, PHASE_MAIN], help="force the decision phase")
    p.add_argument("--time", type=float, help="search time limit in seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="machine readable output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catanbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="parse a Colonist.io screenshot and recommend a move")
    a.add_argument("image")
    a.add_argument("--parser", choices=["auto", "cv", "llm"], default="auto")
    a.add_argument("--save-state", help="write the parsed state JSON here")
    a.add_argument("--debug", help="write a debug overlay PNG here")
    a.add_argument("--no-ui", action="store_true", help="skip reading the player panel / hand bar")
    _add_recommend_args(a)
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("recommend", help="recommend a move from a JSON state")
    r.add_argument("--state", required=True)
    _add_recommend_args(r)
    r.set_defaults(func=cmd_recommend)

    g = sub.add_parser("play", help="simulate a game")
    g.add_argument("--players", type=int, default=0)
    g.add_argument("--bots", default="search,heuristic,heuristic,heuristic")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-turns", type=int, default=400)
    g.add_argument("--games", type=int, default=1)
    g.add_argument("--verbose", action="store_true")
    g.set_defaults(func=cmd_play)

    ev = sub.add_parser("eval", help="tournament between bot specs")
    ev.add_argument("--bots", default="search,heuristic,random")
    ev.add_argument("--games", type=int, default=20)
    ev.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ev.add_argument("--players", type=int, default=4)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-turns", type=int, default=400)
    ev.set_defaults(func=cmd_eval)

    from .train import build_parser as build_train_parser, train as train_fn
    t = build_train_parser(sub)
    t.set_defaults(func=lambda args: (train_fn(args), 0)[1])

    rd = sub.add_parser("render", help="render a JSON state as a synthetic screenshot")
    rd.add_argument("state")
    rd.add_argument("out")
    rd.add_argument("--size", default="1280x800")
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--me")
    rd.set_defaults(func=cmd_render)

    pr = sub.add_parser("profiles", help="show a saved opponent profile file")
    pr.add_argument("file")
    pr.set_defaults(func=cmd_profiles)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except UsageError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
