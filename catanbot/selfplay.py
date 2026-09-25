"""Game runner, tournaments and self-play dataset generation.

Bots are described by *specs* (strings) so games can be farmed out to worker
processes::

    "random"                          uniform random legal actions
    "heuristic" / "heuristic:temp=0.5,eps=0.05"
    "search:depth=1,beam=4,expand=8,eps=0.05,temp=0.3,model=models/value_net.npz"
    "search:depth=1,evaluator=heuristic"
    "search:depth=1,model=models/value_net.npz,blend=0.5"   # 50/50 net + heuristic

Trading-style keys (heuristic and search bots, DESIGN section 11):
``accept_bias=0.3`` (per-game acceptance bias drawn from U(-0.3, 0.3)),
``offer_temp=0.5`` (temperature over the proposal ranking) and
``trade_eps=0.05`` (random accept / reject or random proposal).

``make_bot(spec)`` builds the bot.  ``play_game`` runs one game and can
record training samples (feature vectors from every player's perspective at
every decision, labelled with the eventual winner); ``GameResult.bias``
keeps the acceptance bias of each sample's bot as metadata.

With ``sibling_rate > 0`` ``play_game`` also records **sibling afterstates**
at a random share of the current player's main-phase decisions: for every
legal action the state reached by playing it (one sampled chance outcome, as
the search sees it) *and then ending the turn*, so that all siblings sit at
the horizon the depth-1 search compares (the next player's roll; END_TURN
itself is "end the turn now" and a winning action is its game-over state),
with the heuristic evaluator's value and the action kind.  The value net is
trained to *order* these siblings like the heuristic (``ValueNet.fit(pairs=
...)``): the Monte-Carlo labels alone never say that holding an affordable
build is worse than making it, because the behaviour policy never holds.
Each sibling is also kept at its *mid-turn* horizon (the immediate
afterstate, or the decision state itself for END_TURN): the search values
lines pruned by its beam by such mid-turn statics next to fully expanded
lines valued at the END_TURN horizon, so the net is trained to give both
the same value (``ValueNet.fit(consistency=...)``).

**Setup placements** are recorded the same way (``sibling_setup_rate``, every
setup-settlement decision by default when siblings are recorded at all): for
the best ``setup_candidates`` spots of ``placement.setup_pick`` the state after
the settlement (the setup-road phase, the mid-turn twin the search's beam
ranks) and after the placement module's road (the finished afterstate the
search compares).  Setup states are 2 % of the outcome samples and their
labels carry almost no signal about *which* spot was better, so a net fitted
on outcomes alone picks the heuristic's best spot in a quarter of the setup
nodes and starts every game with a weaker economy - the heuristic evaluator's
production term orders spots well, and the ranking term hands that ordering
to the net.  Setup decision states themselves are always recorded as ordinary
samples (``sample_every`` is not applied to them): setup settlements and roads
strictly alternate, so an even ``sample_every`` would never record a
setup-road state although the search ranks exactly those.

**Incoming offers, robber placement and discards** (``sibling_other_rate``,
``SIBLING_OTHER_PHASES``) are recorded the same way: the candidates the
search would expand (the best spots by ``heuristic.action_priors`` plus a
few random others), each as one sampled chance outcome, robber moves
followed by END_TURN like the main-phase siblings.  The search bot decides
every one of these with its evaluator, and a net whose ordering is right
only in the main phase still loses: with the rejected candidate the bot at
parity when the net decides the main phase alone (0.29 vs 0.21 wins in 12
games) fell to 0.12 vs 0.38 when it also answered offers, placed the robber
and discarded.
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
from .state import (GameState, PHASE_DISCARD, PHASE_GAME_OVER, PHASE_MAIN, PHASE_ROBBER, PHASE_ROLL,
                    PHASE_SETUP_ROAD, PHASE_SETUP_SETTLEMENT, PHASE_TRADE_RESPONSE, new_game)

_SETUP_PHASES = (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD)


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


def _trade_style(kw: Dict[str, str]) -> Dict[str, float]:
    """The trading-style keys of a spec (``accept_bias`` / ``bias``, ``offer_temp``, ``trade_eps``)."""
    return {"accept_bias": float(kw.get("accept_bias", kw.get("bias", 0.0))),
            "offer_temp": float(kw.get("offer_temp", 0.0)),
            "trade_eps": float(kw.get("trade_eps", 0.0))}


def make_bot(spec: str) -> Bot:
    name, kw = parse_spec(spec)
    if name == "random":
        return RandomBot(end_turn_bias=float(kw.get("end", 0.0)))
    if name == "heuristic":
        return HeuristicBot(temperature=float(kw.get("temp", 0.0)), epsilon=float(kw.get("eps", 0.0)),
                            **_trade_style(kw))
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
        bot = SearchBot(ev, cfg, epsilon=float(kw.get("eps", 0.0)), temperature=float(kw.get("temp", 0.0)),
                        **_trade_style(kw))
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
# Sibling afterstates (training targets for the net's action ordering)
# ---------------------------------------------------------------------------
SIBLING_KINDS: List[str] = [A.END_TURN, A.BUILD_ROAD, A.BUILD_SETTLEMENT, A.BUILD_CITY, A.BUY_DEV, A.BANK_TRADE,
                            A.PROPOSE_TRADE, A.PLAY_KNIGHT, A.PLAY_ROAD_BUILDING, A.PLAY_YEAR_OF_PLENTY,
                            A.PLAY_MONOPOLY,
                            # appended later (codes are stored in replay buffers, so the order above is fixed)
                            A.SETUP_SETTLEMENT, A.ACCEPT_TRADE, A.REJECT_TRADE, A.MOVE_ROBBER, A.DISCARD]
_SIBLING_KIND_ID = {k: i for i, k in enumerate(SIBLING_KINDS)}
SIBLING_KIND_OTHER = len(SIBLING_KINDS)
SIBLING_KIND_SETUP = _SIBLING_KIND_ID[A.SETUP_SETTLEMENT]
# Decision phases outside the main phase whose candidate afterstates are recorded (``sibling_other_rate``):
# incoming offers (accept / reject), robber placement and discards.  The search bot decides all of them
# with the same evaluator, and a net that orders them badly loses the game elsewhere than in the main phase.
SIBLING_OTHER_PHASES = (PHASE_TRADE_RESPONSE, PHASE_ROBBER, PHASE_DISCARD)
SIBLING_OTHER_KINDS = tuple(_SIBLING_KIND_ID[k] for k in (A.ACCEPT_TRADE, A.REJECT_TRADE, A.MOVE_ROBBER, A.DISCARD))


def sibling_kind_id(action) -> int:
    """Small integer code of an action kind (``SIBLING_KINDS`` index, or ``SIBLING_KIND_OTHER``)."""
    return _SIBLING_KIND_ID.get(action[0], SIBLING_KIND_OTHER)


class _SiblingRecorder:
    """Collects one concrete end-of-turn afterstate per legal action at sampled main-phase decisions."""

    def __init__(self, rate: float, seed: int, max_proposals: int = 6, setup_rate: float = 1.0,
                 setup_candidates: int = 12, other_rate: float = 0.0, other_candidates: int = 12):
        from .search import Searcher
        self.rate = float(rate)
        self.setup_rate = float(setup_rate)
        self.setup_candidates = int(setup_candidates)
        self.other_rate = float(other_rate)
        self.other_candidates = int(other_candidates)
        self.rng = random.Random(seed * 7 + 12345)
        self.max_proposals = int(max_proposals)
        self.heuristic = HeuristicEvaluator()
        self.searcher = Searcher(self.heuristic, SearchConfig(depth=1, beam=4, expand=8))
        self.searcher._rng = random.Random(seed * 11 + 777)
        self.states: List[GameState] = []      # END_TURN horizon (next player's roll / game over)
        self.mid_states: List[GameState] = []  # mid-turn horizon (immediate afterstate; the decision state for END_TURN)
        self.players: List[int] = []
        self.node: List[int] = []
        self.kind: List[int] = []
        self.n_nodes = 0

    def _add_node(self, rows, me: int) -> None:
        for s2, mid, k in rows:
            self.states.append(s2)
            self.mid_states.append(mid)
            self.players.append(me)
            self.node.append(self.n_nodes)
            self.kind.append(k)
        self.n_nodes += 1

    def maybe_record_setup(self, state: GameState, legal, me: int) -> None:
        """Setup placement: the best ``setup_candidates`` spots, each followed by the placement module's road."""
        if state.phase != PHASE_SETUP_SETTLEMENT or state.current != me or len(legal) < 2:
            return
        if self.setup_rate <= 0 or self.rng.random() >= self.setup_rate:
            return
        from .placement import setup_pick, setup_road_pick
        legal_set = set(legal)
        rows = []
        for v, _score in setup_pick(state, me, k=self.setup_candidates):
            a = (A.SETUP_SETTLEMENT, v)
            if a not in legal_set:
                continue
            try:
                mid = E.apply(state, a, self.searcher._rng)          # setup-road phase: what the beam ranks
                road = setup_road_pick(mid, me, v)[0]
                end = E.apply(mid, (A.SETUP_ROAD, road), self.searcher._rng) if road >= 0 else mid
            except E.IllegalActionError:
                continue
            rows.append((end, mid, SIBLING_KIND_SETUP))
        if len(rows) >= 2:
            self._add_node(rows, me)

    def _sample_outcome(self, state: GameState, a, me: int):
        """One concrete afterstate of ``a`` (a sampled chance outcome, as the search sees it) or ``None``."""
        try:
            outs = self.searcher._outcomes(state, a, me)
        except E.IllegalActionError:
            return None
        if not outs:
            return None
        r = self.rng.random()
        acc = 0.0
        for p, s2 in outs:
            acc += p
            if r < acc:
                return s2
        return outs[-1][1]

    def maybe_record_other(self, state: GameState, legal, me: int) -> None:
        """Incoming offer / robber / discard decisions: the candidates the search would expand (the best
        ``other_candidates`` - 4 by prior plus 4 random others), each at one common horizon per node."""
        if state.phase not in SIBLING_OTHER_PHASES or len(legal) < 2 or E.acting_player(state) != me:
            return
        if self.other_rate <= 0 or self.rng.random() >= self.other_rate:
            return
        from .heuristic import action_priors
        acts = list(legal)
        if len(acts) > self.other_candidates:
            priors = action_priors(state, acts, me)
            order = sorted(range(len(acts)), key=lambda i: -priors[i])
            n_top = max(1, self.other_candidates - 4)
            top = order[:n_top]
            rest = order[n_top:]
            self.rng.shuffle(rest)
            acts = [acts[i] for i in top + rest[:self.other_candidates - n_top]]
        mids = []
        for a in acts:
            s2 = self._sample_outcome(state, a, me)
            if s2 is not None:
                mids.append((s2, sibling_kind_id(a)))
        if len(mids) < 2:
            return
        # Robber moves land in our main phase: like the main-phase siblings they are compared after
        # END_TURN, but only when every sibling can end the turn (one common horizon per node).
        ends = [m for m, _ in mids]
        if state.phase == PHASE_ROBBER and all(
                m.phase == PHASE_MAIN and m.current == me and (A.END_TURN,) in E.legal_actions(m) for m in ends):
            try:
                ends = [E.apply(m, (A.END_TURN,), self.searcher._rng) for m in ends]
            except E.IllegalActionError:
                ends = [m for m, _ in mids]
        self._add_node([(end, mid, k) for end, (mid, k) in zip(ends, mids)], me)

    def maybe_record(self, state: GameState, legal, me: int) -> None:
        if state.phase == PHASE_SETUP_SETTLEMENT:
            self.maybe_record_setup(state, legal, me)
            return
        if state.phase in SIBLING_OTHER_PHASES:
            self.maybe_record_other(state, legal, me)
            return
        if state.phase != PHASE_MAIN or state.current != me or len(legal) < 3 or (A.END_TURN,) not in legal:
            return
        # nodes with a settlement / city affordable (the decisive build-vs-hold counterfactual) are rarer
        # than ordinary road / trade / dev decisions and are oversampled
        rate = self.rate
        if any(a[0] in (A.BUILD_SETTLEMENT, A.BUILD_CITY) for a in legal):
            rate = min(1.0, 6.0 * rate)
        if self.rng.random() >= rate:
            return
        root = state.copy()   # the game mutates ``state`` in place afterwards
        n_prop = 0
        rows = []
        for a in legal:
            if a[0] == A.PROPOSE_TRADE:
                n_prop += 1
                if n_prop > self.max_proposals:
                    continue
            # one sampled chance outcome (dev card drawn, card stolen, offer answered): a concrete
            # afterstate, ordered by the heuristic's value of exactly that state
            chosen = self._sample_outcome(state, a, me)
            if chosen is None:
                continue
            mid = chosen
            if a[0] == A.END_TURN:
                mid = root
            elif chosen.phase != PHASE_GAME_OVER:
                # then end the turn: every sibling is compared at the same horizon as END_TURN itself
                # (otherwise "still my turn" alone would separate the pairs)
                if chosen.phase != PHASE_MAIN or chosen.current != me or (A.END_TURN,) not in E.legal_actions(chosen):
                    continue
                try:
                    chosen = E.apply(chosen, (A.END_TURN,), self.searcher._rng)
                except E.IllegalActionError:
                    continue
            rows.append((chosen, mid, sibling_kind_id(a)))
        if len(rows) < 2:
            return
        self._add_node(rows, me)

    def arrays(self):
        """``(Xs float16, node int32, h float32, kind int8, Xm float16)`` or ``None`` when nothing was recorded."""
        if not self.states:
            return None
        from .features import extract_batch
        X = extract_batch(self.states, self.players).astype(np.float16)
        Xm = extract_batch(self.mid_states, self.players).astype(np.float16)
        h = np.asarray(self.heuristic.evaluate(self.states, self.players), dtype=np.float32)
        return X, np.asarray(self.node, np.int32), h, np.asarray(self.kind, np.int8), Xm


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
    bias: Optional[np.ndarray] = None    # (N,) per-game acceptance bias of that player's bot (trading-style metadata)
    # sibling afterstates (``sibling_rate > 0``): one row per legal action of a sampled decision node
    Xs: Optional[np.ndarray] = None      # (M, NUM_FEATURES) float16, from the acting player's perspective
    s_node: Optional[np.ndarray] = None  # (M,) node id within this game
    s_h: Optional[np.ndarray] = None     # (M,) heuristic evaluator's win probability of the afterstate
    s_kind: Optional[np.ndarray] = None  # (M,) int8 action kind (``SIBLING_KINDS`` index)
    Xm: Optional[np.ndarray] = None      # (M, NUM_FEATURES) float16, the same siblings at the mid-turn horizon


def play_game(bots: Sequence[Bot], state: Optional[GameState] = None, rng: Optional[random.Random] = None,
              record: bool = False, max_turns: int = 400, num_players: Optional[int] = None,
              sample_every: int = 1, seed: int = 0, specs: Optional[List[str]] = None,
              on_action: Optional[Callable[[GameState, A.Action, int], None]] = None,
              sibling_rate: float = 0.0, sibling_setup_rate: Optional[float] = None,
              sibling_other_rate: Optional[float] = None) -> GameResult:
    """Play one full game.  ``bots[i]`` controls seat ``i``.

    ``sibling_rate`` (with ``record``) is the share of the current player's
    main-phase decisions at which the afterstates of every legal action are
    recorded (``GameResult.Xs`` / ``s_node`` / ``s_h`` / ``s_kind``);
    ``sibling_setup_rate`` the share of setup-settlement decisions and
    ``sibling_other_rate`` the share of incoming-offer / robber / discard
    decisions recorded the same way (defaults: all setup decisions and
    ``5 * sibling_rate`` of the others whenever ``sibling_rate > 0``, none
    otherwise; see the module docstring).
    """
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
    siblings: Optional[_SiblingRecorder] = None
    if record:
        from .features import extract_batch
        extract = extract_batch
        if sibling_setup_rate is None:
            sibling_setup_rate = 1.0 if sibling_rate > 0 else 0.0
        if sibling_other_rate is None:
            sibling_other_rate = min(1.0, 5.0 * sibling_rate)
        if sibling_rate > 0 or sibling_setup_rate > 0 or sibling_other_rate > 0:
            siblings = _SiblingRecorder(sibling_rate, seed, setup_rate=sibling_setup_rate,
                                        other_rate=sibling_other_rate)
    while state.phase != PHASE_GAME_OVER:
        i = E.acting_player(state)
        legal = E.legal_actions(state)
        if not legal:
            break
        # Record decision states and every roll-phase state: the depth-1 search evaluates
        # exactly the start-of-turn (roll phase) states, so the net must see them in training.
        # Setup states are always recorded (settlements and roads alternate, so an even
        # ``sample_every`` would never record a setup-road state, which the search ranks).
        if record and (n_actions % sample_every == 0 or state.phase in _SETUP_PHASES) \
                and (len(legal) > 1 or state.phase == PHASE_ROLL):
            X = extract([state] * n, list(range(n)))
            feats.append(X.astype(np.float16))
            who.extend(range(n))
        if siblings is not None:
            siblings.maybe_record(state, legal, i)
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
        seat_bias = [float(getattr(b, "trade_bias", 0.0) or 0.0) for b in bots[:n]]
        res.bias = np.array([seat_bias[p] for p in res.players], dtype=np.float32)
    if siblings is not None:
        arrs = siblings.arrays()
        if arrs is not None:
            res.Xs, res.s_node, res.s_h, res.s_kind, res.Xm = arrs
    return res


# ---------------------------------------------------------------------------
# Parallel games
# ---------------------------------------------------------------------------
def _worker(args) -> GameResult:
    specs, seed, record, max_turns, sample_every = args[:5]
    sibling_rate = args[5] if len(args) > 5 else 0.0
    sibling_setup_rate = args[6] if len(args) > 6 else None
    sibling_other_rate = args[7] if len(args) > 7 else None
    bots = [make_bot(s) for s in specs]
    rng = random.Random(seed)
    return play_game(bots, rng=rng, record=record, max_turns=max_turns, sample_every=sample_every,
                     seed=seed, specs=list(specs), sibling_rate=sibling_rate, sibling_setup_rate=sibling_setup_rate,
                     sibling_other_rate=sibling_other_rate)


def run_games(jobs: Sequence[Tuple[List[str], int]], workers: int = 1, record: bool = False, max_turns: int = 400,
              sample_every: int = 1, progress: Optional[Callable[[int, int, GameResult], None]] = None,
              sibling_rate: float = 0.0, sibling_setup_rate: Optional[float] = None,
              sibling_other_rate: Optional[float] = None) -> List[GameResult]:
    """Run ``jobs`` = [(specs_per_seat, seed), ...] possibly in parallel."""
    args = [(list(specs), seed, record, max_turns, sample_every, sibling_rate, sibling_setup_rate, sibling_other_rate)
            for specs, seed in jobs]
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
                     progress: Optional[Callable[[int, int, GameResult], None]] = None,
                     sibling_rate: float = 0.0, sibling_setup_rate: Optional[float] = None,
                     sibling_other_rate: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[GameResult]]:
    """Self-play games with seats drawn from ``spec_pool``; returns (X, y, vp_frac, results).

    With ``sibling_rate > 0`` (or ``sibling_setup_rate > 0``) every result also
    carries sibling afterstates (see :func:`play_game`); :func:`sibling_arrays`
    stacks them.
    """
    rng = random.Random(seed)
    jobs = []
    for g in range(games):
        n = rng.choice(list(num_players_choices))
        seats = [rng.choice(list(spec_pool)) for _ in range(n)]
        jobs.append((seats, seed * 7919 + g))
    results = run_games(jobs, workers=workers, record=True, max_turns=max_turns, sample_every=sample_every,
                        progress=progress, sibling_rate=sibling_rate, sibling_setup_rate=sibling_setup_rate,
                        sibling_other_rate=sibling_other_rate)
    Xs = [r.X for r in results if r.X is not None]
    ys = [r.y for r in results if r.y is not None]
    vs = [r.vp_frac for r in results if r.vp_frac is not None]
    if not Xs:
        from .features import NUM_FEATURES
        return np.zeros((0, NUM_FEATURES), np.float16), np.zeros(0, np.float32), np.zeros(0, np.float32), results
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs), results


def sibling_arrays(results: Sequence[GameResult], base_id: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack the sibling afterstates of ``results``: ``(Xs, node, h, kind, Xm)`` with node ids made unique
    across games (``base_id + 1000 * game_index + local id``)."""
    Xs, nodes, hs, kinds, Xm = [], [], [], [], []
    for k, r in enumerate(results):
        if r.Xs is None or len(r.Xs) == 0:
            continue
        Xs.append(r.Xs)
        nodes.append(r.s_node.astype(np.int64) + base_id + 1000 * k)
        hs.append(r.s_h)
        kinds.append(r.s_kind)
        Xm.append(r.Xm if r.Xm is not None else r.Xs)
    if not Xs:
        from .features import NUM_FEATURES
        return (np.zeros((0, NUM_FEATURES), np.float16), np.zeros(0, np.int64), np.zeros(0, np.float32),
                np.zeros(0, np.int8), np.zeros((0, NUM_FEATURES), np.float16))
    return np.concatenate(Xs), np.concatenate(nodes), np.concatenate(hs), np.concatenate(kinds), np.concatenate(Xm)
