#!/usr/bin/env python3
"""Zero-game dev-reading shadow (shadow.dev_oracle, docs/PRIORITY_PLAN.md step 6): how much do our counted-mode
decisions lose from not knowing the opponents' development cards, and how much of it does the held-age posterior
(devbelief) recover?

    # positions from archived proof games (our seat followed by the counted tracker; the logged actions are replayed)
    /home/user/venv_cat33/bin/python scripts/belief_shadow.py --logs 'proof/T4/logs/*.jsonl.gz' --max-positions 2000 \\
        --json shadow_dev_oracle.json
    # or live games against three default catanbot opponents (a dev-heavy table; our seat plays the U arm)
    python3 scripts/belief_shadow.py --games 2 --opponents selfbot --json smoke.json

At every searched decision of our seat (more than one distinct playable action) the counted decision is made
three ways, each over ``--k`` determinizations aggregated as the counted adapter does (mean value over the samples
that ranked an action, actions ranked by at least half of them first):

* ``U``: the uniform dev deal (``devbelief.ENABLED=0``, the default bot);
* ``H``: the held-age posterior (``devbelief.ENABLED=1``; RNG-neutral dealing, so the hand samples are U's);
* ``O``: the true opponent development cards (an oracle tracker with ``reveal_devs``: only the dev draws are
  revealed - steals and discards stay hidden, so its hand samples are U's too).

Sample ``k`` draws its determinization from ``Random(det_k)`` and its search from ``Random(search_k)``, the same
seeds in every arm, so a disagreement can only come from the development cards.  The U arm is run twice (A/A:
``aa_ok`` = identical choices everywhere) and the hand samples of U / H / O must coincide (``crn_ok``).

Scoring (paired value regret): where ``a_U != a_H`` (and on a ``--sub`` share of the other positions, for the
headroom), ``V_O(a)`` = mean over ``--kref`` oracle determinizations (their own fixed stream) of the searched value
of ``a`` with the root restricted to the arms' distinct choices; ``regret_X = max_a V_O(a) - V_O(a_X)``.

Output: the change rate P(a_U != a_H) overall and per decision class; headroom E[regret_U] (subsample);
captured E[regret_U - regret_H] over all positions (concordant positions contribute 0) with a paired t; the
implied game effect (captured x searched decisions per game, win-probability points: evaluator based, optimistic,
for gating only); the sanity bound regret_O <= regret_U; process time.  ``pass`` (gate B) = captured > 0 at
one-sided p < 0.05 (t > 1.645) with aa_ok and crn_ok.  Pre-registered readouts: ``advisor_only`` (change rate
below 1%), ``queue_d`` (implied effect x 3 >= 4 pp).  Observation only; one process.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()[:8], "big")


def aggregate(results: Sequence[Tuple[Any, List[Tuple[Any, float]]]], k: int):
    """The counted adapter's aggregation: ``results`` = per sample (decision, [(action, value)])."""
    values: Dict[Any, List[float]] = {}
    votes: Dict[Any, int] = {}
    for d, ranked in results:
        votes[d] = votes.get(d, 0) + 1
        for a, v in ranked:
            values.setdefault(a, []).append(float(v))
    if not values:
        return max(votes, key=lambda a: votes[a])
    need = (k + 1) // 2
    return max(values, key=lambda a: (len(values[a]) >= need, sum(values[a]) / len(values[a]), votes.get(a, 0)))


class forced_root:
    """Within the block the search's root legal actions are ``actions`` (a forced evaluation: every listed action
    gets its searched value); every other state keeps the engine's list."""

    def __init__(self, root, actions):
        self.root, self.actions = root, list(actions)

    def __enter__(self):
        from catanbot import engine as E
        self._E, self._real = E, E.legal_actions
        real, root, acts = self._real, self.root, self.actions
        E.legal_actions = lambda s, *a, **kw: list(acts) if s is root else real(s, *a, **kw)
        return self

    def __exit__(self, *exc):
        self._E.legal_actions = self._real


def regret_stats(rows: Sequence[Dict[str, Any]], per_game: float) -> Dict[str, Any]:
    """Headroom / captured / sanity from the per-position rows (see the module docstring)."""
    n = len(rows)
    d = [r["regret_u"] - r["regret_h"] if r.get("evaluated") else 0.0 for r in rows]
    mean = sum(d) / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mean > 0 else 0.0)
    sub = [r for r in rows if r.get("sub")]
    head = sum(r["regret_u"] for r in sub) / len(sub) if sub else None
    orc = sum(r["regret_o"] for r in sub) / len(sub) if sub else None
    return {"positions": n, "captured": mean, "captured_t": t, "captured_sd": sd, "headroom": head,
            "regret_o": orc, "oracle_sane": (orc is None or head is None or orc <= head + 1e-12),
            "implied_pp": 100.0 * mean * per_game, "sub_positions": len(sub)}


class Shadow:
    """The three arms at every searched counted decision of one CatanbotPlayer (``player``)."""

    def __init__(self, k: int, kref: int, sub: float, rows: List[Dict[str, Any]], max_positions: Optional[int]):
        self.k, self.kref, self.sub, self.rows, self.max_positions = k, kref, sub, rows, max_positions
        self.game_tag = ""

    def arm(self, player, tracker, pub, legal, enabled: int, seeds) -> Tuple[Any, list]:
        from catanbot import tuning
        from catanbot.robber_eval import knight_hint
        tok = tuning.apply({"devbelief.ENABLED": enabled})
        try:
            dealt = tracker.unknown_dev_holders()
            pk = tracker.knight_posterior([pub.public_vp(j) for j in range(len(pub.players))])
            res, hands = [], []
            for ds, ss in seeds:
                det = tracker.determinize(pub, random.Random(ds))
                hands.append([list(p.resources) for p in det.players])
                with knight_hint(dealt=dealt, p_knight=pk):
                    d = player.bot.decide(det, list(legal), random.Random(ss))
                res.append((d, [(r.action, r.value) for r in (getattr(player.bot, "last_results", None) or [])]))
        finally:
            tuning.restore(tok)
        return aggregate(res, len(seeds)), hands

    def reference(self, player, oracle, pub, actions, tag) -> Dict[Any, float]:
        vals: Dict[Any, List[float]] = {}
        for i in range(self.kref):
            det = oracle.determinize(pub, random.Random(_seed(tag, "ref", i)))
            with forced_root(det, actions):
                player.bot.decide(det, list(actions), random.Random(_seed(tag, "refsearch", i)))
            for r in getattr(player.bot, "last_results", None) or []:
                if r.action in actions:
                    vals.setdefault(r.action, []).append(float(r.value))
        return {a: sum(v) / len(v) for a, v in vals.items()}

    def position(self, player, pub, legal):
        """Shadow one decision; returns the U arm's choice (what the default counted bot plays), or ``None``
        once ``max_positions`` is reached."""
        from decision_shadow import classes_of
        if self.max_positions is not None and len(self.rows) >= self.max_positions:
            return None
        idx = len(self.rows)
        tag = f"{self.game_tag}:{player._counted_calls}"
        player._counted_calls += 1
        seeds = [(_seed(tag, "det", k), _seed(tag, "search", k)) for k in range(self.k)]
        t0 = time.process_time()
        tr, orc = player.tracker, player.oracle
        a_u, hands_u = self.arm(player, tr, pub, legal, 0, seeds)
        a_u2, _ = self.arm(player, tr, pub, legal, 0, seeds)
        a_h, hands_h = self.arm(player, tr, pub, legal, 1, seeds)
        row: Dict[str, Any] = {"i": idx, "tag": tag, "classes": list(classes_of(pub, legal)), "u": repr(a_u),
                               "h": repr(a_h), "changed": a_u != a_h, "aa": a_u == a_u2,
                               "crn": hands_u == hands_h, "sub": (_seed(tag, "sub") % 10 ** 6) < self.sub * 10 ** 6}
        if row["changed"] or row["sub"]:
            a_o, hands_o = self.arm(player, orc, pub, legal, 0, seeds)
            row["crn"] = row["crn"] and hands_o == hands_u
            acts = []
            for a in (a_u, a_h, a_o):
                if a not in acts:
                    acts.append(a)
            if len(acts) >= 2:
                v = self.reference(player, orc, pub, acts, tag)
                if all(a in v for a in acts):
                    best = max(v.values())
                    row.update({"evaluated": True, "o": repr(a_o), "regret_u": best - v[a_u],
                                "regret_h": best - v[a_h], "regret_o": best - v[a_o]})
            else:
                row.update({"evaluated": True, "o": repr(a_o), "regret_u": 0.0, "regret_h": 0.0, "regret_o": 0.0})
            if row["sub"] and not row.get("evaluated"):
                row["sub"] = False          # no reference value: out of the headroom sample
        row["cpu_s"] = time.process_time() - t0
        self.rows.append(row)
        return a_u


def shadow_player_class():
    from catanbot.bench import catanatron_adapter as ad
    from catanbot.bench.public_info import PublicInfoTracker

    class ShadowPlayer(ad.CatanbotPlayer):
        """Counted CatanbotPlayer whose searched decisions are shadowed (it plays the U arm's choice) and which
        also follows the game with a dev-card oracle tracker."""

        shadow: Shadow = None
        oracle = None

        def _begin(self, game):
            super()._begin(game)
            self.oracle = PublicInfoTracker(self.color, discards_public=self.discards_public,
                                            vps_to_win=int(getattr(game, "vps_to_win", 10)), reveal_devs=True)
            self.oracle.start(game.state)

        def _follow_tracker(self, st):
            super()._follow_tracker(st)
            try:
                self.oracle.follow(st)
            except Exception:  # noqa: BLE001
                self.oracle.resync(st)

        def _decide_counted(self, pub, legal):
            a = self.shadow.position(self, pub, legal)
            if a is None:
                return super()._decide_counted(pub, legal)
            return a, [a]

    return ShadowPlayer


def run_live(args, shadow: Shadow) -> Dict[str, Any]:
    from catanbot.bench import catanatron_adapter as ad
    Shadowed = shadow_player_class()
    games, decisions = 0, 0
    for g in range(args.games):
        seat = g % 4
        gseed = 1 + _seed("belief_shadow", args.seed, g) % (2 ** 31 - 2)
        me = Shadowed(ad.COLORS[seat], spec=args.spec, info="counted", seed=gseed, seeded_samples=True,
                      discards_public=args.discards_public)
        me.shadow = shadow
        shadow.game_tag = f"live{args.seed}:{g}"
        players = []
        for i, c in enumerate(ad.COLORS):
            if i == seat:
                players.append(me)
            elif args.opponents == "selfbot":
                players.append(ad.CatanbotPlayer(c, spec=args.spec, seed=_seed(gseed, i) % 2 ** 31))
            else:
                from catanatron.players.weighted_random import WeightedRandomPlayer
                players.append(WeightedRandomPlayer(c))
        game = ad.make_game(players, seed=gseed)
        game.play()
        games += 1
        decisions += int(me.stats.get("searched", 0))
        print(f"game {g}: {len(shadow.rows)} positions so far", file=sys.stderr, flush=True)
        if args.max_positions is not None and len(shadow.rows) >= args.max_positions:
            break
    return {"games": games, "searched": decisions}


def run_logs(args, shadow: Shadow) -> Dict[str, Any]:
    from catanatron.models.player import Color
    from catanbot.bench import catanatron_adapter as ad
    Shadowed = shadow_player_class()
    api = "3.3" if ad.API_33 else "3.2"
    games, decisions = 0, 0
    for p in sorted(glob.glob(args.logs)):
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("api") != api:
                    raise SystemExit(f"error: {p} was written by catanatron {doc.get('catanatron')} "
                                     f"({doc.get('api')} API); run the shadow with the interpreter of that engine")
                seats = json.loads(doc["our_seats"]) if isinstance(doc["our_seats"], str) else doc["our_seats"]
                players = doc["players"]
                if isinstance(players, str):
                    import ast
                    players = ast.literal_eval(players)
                seat = int(seats[0])
                color = Color(doc["colors"][seat])
                game = ad.rebuild_game(doc)
                me = Shadowed(color, spec=args.spec, info="counted", seed=int(players[seat].get("bot_seed", 0)),
                              seeded_samples=True, discards_public=args.discards_public)
                me.shadow = shadow
                shadow.game_tag = f"{os.path.basename(p)}:{doc.get('game')}"
                for item in doc["actions"]:
                    if item[0] == color.value:
                        try:
                            me.decide(game, ad.playable_actions_of(game))
                        except Exception:  # noqa: BLE001  (the logged action is replayed anyway)
                            pass
                    ad.replay_log_action(game, item, check=False)
                games += 1
                decisions += int(me.stats.get("searched", 0))
                print(f"{shadow.game_tag}: {len(shadow.rows)} positions so far", file=sys.stderr, flush=True)
                if (args.max_games is not None and games >= args.max_games) or \
                        (args.max_positions is not None and len(shadow.rows) >= args.max_positions):
                    return {"games": games, "searched": decisions}
    return {"games": games, "searched": decisions}


def summarize(rows: Sequence[Dict[str, Any]], src: Dict[str, Any]) -> Dict[str, Any]:
    per_game = src["searched"] / src["games"] if src["games"] else 0.0
    out = regret_stats(rows, per_game)
    n = len(rows)
    out["change_rate"] = sum(r["changed"] for r in rows) / n if n else 0.0
    classes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        for c in r["classes"]:
            e = classes.setdefault(c, {"n": 0, "changed": 0})
            e["n"] += 1
            e["changed"] += r["changed"]
    for c, e in classes.items():
        e["rate"] = e["changed"] / e["n"]
        e.update({k: v for k, v in regret_stats([r for r in rows if c in r["classes"]], per_game).items()
                  if k in ("captured", "captured_t")})
    out["classes"] = classes
    out["aa_ok"] = all(r["aa"] for r in rows)
    out["crn_ok"] = all(r["crn"] for r in rows)
    out["searched_per_game"] = per_game
    out["advisor_only"] = out["change_rate"] < 0.01
    out["queue_d"] = out["implied_pp"] * 3 >= 4.0
    out["pass"] = bool(out["aa_ok"] and out["crn_ok"] and out["captured"] > 0 and out["captured_t"] > 1.645)
    out.update(src)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", help="positions from these catanbot-actionlog/1 files (glob) instead of live games")
    ap.add_argument("--max-games", type=int, default=None, help="--logs: at most this many games")
    ap.add_argument("--games", type=int, default=2, help="live: games to play (our seat rotates)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponents", choices=("selfbot", "random"), default="selfbot",
                    help="live: three default catanbot seats (full information; dev-heavy) or WeightedRandom")
    ap.add_argument("--spec", default=None, help="our bot spec (default: the adapter's DEFAULT_SPEC)")
    ap.add_argument("--k", type=int, default=4, help="determinizations per arm (the counted adapter's 4)")
    ap.add_argument("--kref", type=int, default=8, help="oracle determinizations of the reference values")
    ap.add_argument("--sub", type=float, default=0.25, help="share of concordant positions scored (headroom)")
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--discards-hidden", dest="discards_public", action="store_false",
                    help="hide the discarded cards (default: public, as on Colonist)")
    ap.add_argument("--keep-rows", action="store_true")
    ap.add_argument("--json", help="write the result here (key 'pass')")
    args = ap.parse_args(argv)
    from catanbot.bench import catanatron_adapter as ad
    args.spec = args.spec or ad.DEFAULT_SPEC
    t0 = time.process_time()
    rows: List[Dict[str, Any]] = []
    shadow = Shadow(args.k, args.kref, args.sub, rows, args.max_positions)
    src = run_logs(args, shadow) if args.logs else run_live(args, shadow)
    out = summarize(rows, src)
    out["cpu_s"] = round(time.process_time() - t0, 1)
    out["args"] = {k: v for k, v in vars(args).items() if k != "json"}
    if args.keep_rows:
        out["rows"] = rows
    print(f"{out['positions']} positions ({out['games']} games): change rate {out['change_rate']:.3%}, "
          f"captured {out['captured']:+.5f} (t {out['captured_t']:.2f}), headroom {out['headroom']}, "
          f"regret_O {out['regret_o']}, implied {out['implied_pp']:+.2f} pp/game, aa_ok {out['aa_ok']}, "
          f"crn_ok {out['crn_ok']}, pass {out['pass']}, {out['cpu_s']} CPU s")
    if args.json:
        tmp = args.json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        os.replace(tmp, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
