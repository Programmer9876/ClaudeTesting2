"""Game runner, tournaments and self-play dataset generation.

Bots are described by *specs* (strings) so games can be farmed out to worker
processes::

    "random"                          uniform random legal actions
    "heuristic" / "heuristic:temp=0.5,eps=0.05"
    "search:depth=1,beam=4,expand=8,eps=0.05,temp=0.3,model=models/value_net.npz"
    "search:depth=1,evaluator=heuristic"
    "search:depth=1,model=models/value_net.npz,blend=0.5"   # 50/50 net + heuristic

``make_bot(spec)`` builds the bot.  ``play_game`` runs one game and can
record training samples (feature vectors from every player's perspective at
every decision, labelled with the eventual winner).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import actions as A
from . import engine as E
from .agents.base import Bot
from .agents.heuristic_bot import HeuristicBot
from .agents.random_bot import RandomBot
from .agents.search_bot import SearchBot
from .heuristic import HeuristicEvaluator
from .search import SearchConfig
from .state import GameState, PHASE_GAME_OVER, PHASE_ROLL, new_game


# ---------------------------------------------------------------------------
# Bot specs
# ---------------------------------------------------------------------------
def parse_spec(spec: str) -> Tuple[str, Dict[str, str]]:
    name, _, rest = spec.partition(":")
    kw: Dict[str, str] = {}
    if rest:
        for part in rest.split(","):
            if not part:
                continue
            k, _, v = part.partition("=")
            kw[k.strip()] = v.strip()
    return name.strip(), kw


_MODEL_CACHE: Dict[str, object] = {}


class BlendedEvaluator:
    """``alpha * net + (1 - alpha) * heuristic`` - robust when the net is under-trained."""

    name = "blend"

    def __init__(self, net, alpha: float = 0.5, heuristic: Optional[HeuristicEvaluator] = None):
        self.net = net
        self.alpha = float(alpha)
        self.heuristic = heuristic or HeuristicEvaluator()

    def evaluate(self, states, players) -> np.ndarray:
        a = self.alpha
        vn = np.asarray(self.net.evaluate(states, players), dtype=np.float64)
        if a >= 1.0:
            return vn
        vh = np.asarray(self.heuristic.evaluate(states, players), dtype=np.float64)
        return a * vn + (1.0 - a) * vh

    __call__ = evaluate


def load_evaluator(path: Optional[str], blend: Optional[float] = None):
    """Value net from ``path`` (cached per process) or the heuristic evaluator.

    ``blend`` in (0, 1) mixes the net with the heuristic evaluator.
    """
    if not path or path in ("heuristic", "none"):
        return HeuristicEvaluator()
    try:
        st = os.stat(path)
        key = f"{path}:{st.st_mtime_ns}:{st.st_size}"   # a rewritten file must not hit the cache
    except OSError:
        key = path
    ev = _MODEL_CACHE.get(key)
    if ev is None:
        from .model import ValueNet
        ev = ValueNet.load(path)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = ev
    if blend is not None and 0.0 < blend < 1.0:
        return BlendedEvaluator(ev, blend)
    return ev


def make_bot(spec: str) -> Bot:
    name, kw = parse_spec(spec)
    if name == "random":
        return RandomBot(end_turn_bias=float(kw.get("end", 0.0)))
    if name == "heuristic":
        return HeuristicBot(temperature=float(kw.get("temp", 0.0)), epsilon=float(kw.get("eps", 0.0)))
    if name == "search":
        cfg = SearchConfig(
            depth=int(kw.get("depth", 1)),
            beam=int(kw.get("beam", 4)),
            expand=int(kw.get("expand", 8)),
            max_actions_per_turn=int(kw.get("actions", 6)),
            roll_samples=int(kw.get("rolls", 11)),
            opp_roll_samples=int(kw.get("opprolls", 4)),
            finished_lookahead=int(kw.get("lookahead", 3)),
            trade_proposals=int(kw.get("trades", 3)),
            max_nodes=int(kw.get("nodes", 20000)),
        )
        ev = load_evaluator(kw.get("model") or kw.get("evaluator"),
                            float(kw["blend"]) if "blend" in kw else None)
        bot = SearchBot(ev, cfg, epsilon=float(kw.get("eps", 0.0)), temperature=float(kw.get("temp", 0.0)))
        bot.name = spec
        return bot
    raise ValueError(f"unknown bot spec: {spec}")


def _valid_extra_action(state: GameState, a, legal) -> bool:
    """Trade proposals outside the engine's bounded candidate list are still legal if well formed."""
    if not a or a[0] != A.PROPOSE_TRADE or len(a) != 3:
        return False
    if not any(x[0] == A.PROPOSE_TRADE for x in legal):
        return False
    give, get = a[1], a[2]
    if len(give) != 5 or len(get) != 5 or sum(give) == 0 or sum(get) == 0:
        return False
    if any((give[r] and get[r]) or give[r] < 0 or get[r] < 0 for r in range(5)):
        return False
    p = state.players[state.current]
    return all(p.resources[r] >= give[r] for r in range(5))


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------
@dataclass
class GameResult:
    winner: int
    vps: List[int]
    turns: int
    num_players: int
    specs: List[str]
    seed: int
    duration: float
    actions: int
    X: Optional[np.ndarray] = None       # (N, NUM_FEATURES) float16 samples
    players: Optional[np.ndarray] = None  # (N,) perspective player of each sample
    y: Optional[np.ndarray] = None       # (N,) 1.0 if that player won
    vp_frac: Optional[np.ndarray] = None  # (N,) final VP / 10 of that player (auxiliary)


def play_game(bots: Sequence[Bot], state: Optional[GameState] = None, rng: Optional[random.Random] = None,
              record: bool = False, max_turns: int = 400, num_players: Optional[int] = None,
              sample_every: int = 1, seed: int = 0, specs: Optional[List[str]] = None,
              on_action: Optional[Callable[[GameState, A.Action, int], None]] = None) -> GameResult:
    """Play one full game.  ``bots[i]`` controls seat ``i``."""
    rng = rng or random.Random(seed)
    n = num_players or len(bots)
    if n > len(bots):
        raise ValueError(f"{n} players but only {len(bots)} bots")
    if state is None:
        state = new_game(n, rng=rng)
    state.max_turns = max_turns
    for b in bots:
        b.reset()
    feats: List[np.ndarray] = []
    who: List[int] = []
    t0 = time.time()
    n_actions = 0
    extract = None
    if record:
        from .features import extract_batch
        extract = extract_batch
    while state.phase != PHASE_GAME_OVER:
        i = E.acting_player(state)
        legal = E.legal_actions(state)
        if not legal:
            break
        # Record decision states and every roll-phase state: the depth-1 search evaluates
        # exactly the start-of-turn (roll phase) states, so the net must see them in training.
        if record and n_actions % sample_every == 0 and (len(legal) > 1 or state.phase == PHASE_ROLL):
            X = extract([state] * n, list(range(n)))
            feats.append(X.astype(np.float16))
            who.extend(range(n))
        a = bots[i].decide(state, legal, rng)
        if a not in legal and not _valid_extra_action(state, a, legal):
            a = legal[0]
        for b in bots:
            b.observe(state, a, i)
        if on_action is not None:
            on_action(state, a, i)
        state = E.apply_inplace(state, a, rng)
        n_actions += 1
        if n_actions > 20000:
            break
    vps = [state.total_vp(i) for i in range(n)]
    winner = state.winner
    res = GameResult(winner, vps, state.turn, n, list(specs or [b.name for b in bots]), seed,
                     time.time() - t0, n_actions)
    if record and feats:
        res.X = np.concatenate(feats, axis=0)
        res.players = np.array(who, dtype=np.int8)
        if winner >= 0:
            res.y = (res.players == winner).astype(np.float32)
        else:
            # No winner (turn cap): soft target from final VP share.
            tot = float(sum(vps)) or 1.0
            res.y = np.array([vps[p] / tot for p in res.players], dtype=np.float32)
        res.vp_frac = np.array([min(1.0, vps[p] / 10.0) for p in res.players], dtype=np.float32)
    return res


# ---------------------------------------------------------------------------
# Parallel games
# ---------------------------------------------------------------------------
def _worker(args) -> GameResult:
    specs, seed, record, max_turns, sample_every = args
    bots = [make_bot(s) for s in specs]
    rng = random.Random(seed)
    return play_game(bots, rng=rng, record=record, max_turns=max_turns, sample_every=sample_every,
                     seed=seed, specs=list(specs))


def run_games(jobs: Sequence[Tuple[List[str], int]], workers: int = 1, record: bool = False, max_turns: int = 400,
              sample_every: int = 1, progress: Optional[Callable[[int, int, GameResult], None]] = None) -> List[GameResult]:
    """Run ``jobs`` = [(specs_per_seat, seed), ...] possibly in parallel."""
    args = [(list(specs), seed, record, max_turns, sample_every) for specs, seed in jobs]
    results: List[GameResult] = []
    if workers <= 1 or len(args) <= 1:
        for k, a in enumerate(args):
            r = _worker(a)
            results.append(r)
            if progress:
                progress(k + 1, len(args), r)
        return results
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, args)):
            results.append(r)
            if progress:
                progress(k + 1, len(args), r)
    return results


def tournament(specs: Sequence[str], games: int = 20, workers: int = 1, seed: int = 0,
               num_players: int = 4, max_turns: int = 400,
               progress: Optional[Callable[[int, int, GameResult], None]] = None) -> Dict[str, object]:
    """Round-robin-ish tournament: every game seats a random rotation of ``specs``.

    With more specs than seats a random subset is used each game.  Returns
    win rates, average VP and games played per spec.
    """
    rng = random.Random(seed)
    jobs = []
    for g in range(games):
        pool = list(specs)
        rng.shuffle(pool)
        seats = pool[:num_players]
        while len(seats) < num_players:
            seats.append(rng.choice(list(specs)))
        jobs.append((seats, seed * 100003 + g))
    results = run_games(jobs, workers=workers, record=False, max_turns=max_turns, progress=progress)
    stats: Dict[str, Dict[str, float]] = {s: {"games": 0, "wins": 0, "vp": 0.0} for s in specs}
    for r in results:
        for i, s in enumerate(r.specs):
            st = stats[s]
            st["games"] += 1
            st["vp"] += r.vps[i]
            if r.winner == i:
                st["wins"] += 1
    summary = {}
    for s, st in stats.items():
        g = max(1, st["games"])
        summary[s] = {"games": st["games"], "win_rate": st["wins"] / g, "avg_vp": st["vp"] / g}
    return {"summary": summary, "results": results, "avg_turns": sum(r.turns for r in results) / max(1, len(results))}


def generate_dataset(spec_pool: Sequence[str], games: int, workers: int = 1, seed: int = 0,
                     num_players_choices: Sequence[int] = (3, 4), max_turns: int = 400, sample_every: int = 2,
                     progress: Optional[Callable[[int, int, GameResult], None]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[GameResult]]:
    """Self-play games with seats drawn from ``spec_pool``; returns (X, y, vp_frac, results)."""
    rng = random.Random(seed)
    jobs = []
    for g in range(games):
        n = rng.choice(list(num_players_choices))
        seats = [rng.choice(list(spec_pool)) for _ in range(n)]
        jobs.append((seats, seed * 7919 + g))
    results = run_games(jobs, workers=workers, record=True, max_turns=max_turns, sample_every=sample_every,
                        progress=progress)
    Xs = [r.X for r in results if r.X is not None]
    ys = [r.y for r in results if r.y is not None]
    vs = [r.vp_frac for r in results if r.vp_frac is not None]
    if not Xs:
        from .features import NUM_FEATURES
        return np.zeros((0, NUM_FEATURES), np.float16), np.zeros(0, np.float32), np.zeros(0, np.float32), results
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs), results
