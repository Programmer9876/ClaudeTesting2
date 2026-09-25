"""Card counting: bank stock, development deck and opponent hand beliefs.

Two usage modes:

* **Live play** - :class:`CardCounter` follows every public event from the
  start of a game (production, builds, bank / player trades, discards,
  monopolies, steals) as an *exact* weighted mixture of joint hand
  hypotheses: it is exact while nothing hidden happened, and after a hidden
  steal (or a discard whose cards were not shown) it holds precisely the
  hands that are still possible, weighted by their probability, consistent
  with every public hand size and the bank.  ``catanbot.bench.public_info``
  drives it from catanatron's action log ("counted" information mode).
* **Screenshot** - only hand sizes are known: :class:`HandBelief` is the cheap
  *expected-count* model (one float per resource per player), seeded by
  :func:`expected_opponent_hands`'s production prior.

:class:`CardCounter` is a :class:`HandBelief` (same ``expected`` / ``size`` /
``observe_*`` / ``probability_has`` interface), so either can be handed to the
heuristics that take a ``belief``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import board as B
from .placement import player_production
from .state import GameState

TOTAL_RESOURCE_CARDS = B.BANK_PER_RESOURCE * 5


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------
def bank_remaining(state: GameState) -> List[int]:
    """Cards left in the bank per resource (as tracked in the state)."""
    return list(state.bank)


def bank_is_short(state: GameState, res: int, need: int = 1) -> bool:
    """True if the bank cannot supply ``need`` of ``res`` (relevant for trades)."""
    return state.bank[res] < need


def resources_in_play(state: GameState) -> List[int]:
    """Cards outside the bank per resource (known hands only; unknown hands add to totals)."""
    out = [B.BANK_PER_RESOURCE - b for b in state.bank]
    return out


# ---------------------------------------------------------------------------
# Development deck
# ---------------------------------------------------------------------------
def dev_pool(state: GameState) -> List[int]:
    """Cards of each type that are *not accounted for publicly*.

    pool = standard deck - knights played - dev cards held by players whose
    hands are known.  Unknown opponent hands and the undrawn deck are both
    drawn from this pool.
    """
    pool = list(B.DEV_DECK_COUNTS)
    for p in state.players:
        pool[B.DEV_KNIGHT] -= p.played_knights
        if p.dev_known:
            for t in range(5):
                pool[t] -= p.dev_cards[t] + p.dev_cards_new[t]
    return [max(0, x) for x in pool]


def dev_draw_probabilities(state: GameState) -> List[float]:
    """Probability that the next bought card is of each type."""
    deck = state.dev_deck
    total = sum(deck)
    if total <= 0:
        pool = dev_pool(state)
        total = sum(pool)
        if total <= 0:
            return [0.0] * 5
        return [x / total for x in pool]
    return [x / total for x in deck]


def dev_cards_remaining(state: GameState) -> int:
    return sum(state.dev_deck)


def expected_hidden_vp(state: GameState, player: int) -> float:
    """Expected number of VP cards held by ``player`` (exact if known)."""
    p = state.players[player]
    if p.dev_known:
        return float(p.vp_cards)
    pool = dev_pool(state)
    total = sum(pool)
    if total <= 0 or p.dev_count <= 0:
        return 0.0
    return p.dev_count * pool[B.DEV_VP] / total


# ---------------------------------------------------------------------------
# Opponent hands
# ---------------------------------------------------------------------------
def hand_prior_weights(state: GameState, player: int, smoothing: float = 0.04) -> List[float]:
    """Relative likelihood of each resource in an opponent's unknown hand.

    Production-weighted (cards come from rolls) with smoothing so trades,
    steals and starting resources are covered.  Sums to 1.
    """
    prod = player_production(state, player, ignore_robber=True)
    w = [prod[r] + smoothing for r in range(5)]
    # Cards the bank has run out of are proportionally more likely to be in hands.
    for r in range(5):
        w[r] *= 1.0 + 0.5 * (1.0 - state.bank[r] / B.BANK_PER_RESOURCE)
    tot = sum(w)
    return [x / tot for x in w]


def expected_opponent_hands(state: GameState, me: Optional[int] = None) -> List[List[float]]:
    """Expected resource counts for every player (exact where the hand is known)."""
    out: List[List[float]] = []
    for i, p in enumerate(state.players):
        if p.hand_known or i == me:
            out.append([float(x) for x in p.resources])
        else:
            w = hand_prior_weights(state, i)
            out.append([w[r] * p.hand_size for r in range(5)])
    return out


class HandBelief:
    """Expected-count belief over every player's hand, updated from public events.

    ``expected[i][r]`` is the expected number of cards of resource ``r`` in
    player ``i``'s hand; ``size[i]`` is the exact hand size (always public);
    ``exact[i]`` is True while player ``i``'s hand is known exactly (known at
    construction or through :meth:`set_exact`, and no hidden event touched it
    since): every update of an exact hand is pure integer bookkeeping.

    This is the cheap model for partially observed (screenshot) states.  It
    keeps one expectation per resource, so it cannot represent *which* card a
    hidden steal moved (the thief's and the victim's hands are correlated);
    :class:`CardCounter` does, exactly, when the game was followed from its start.
    """

    def __init__(self, state: GameState):
        self.n = state.num_players
        self.expected: List[List[float]] = []
        self.size: List[int] = []
        self.exact: List[bool] = []
        for i, p in enumerate(state.players):
            if p.hand_known:
                self.expected.append([float(x) for x in p.resources])
                self.size.append(sum(p.resources))
                self.exact.append(True)
            else:
                w = hand_prior_weights(state, i)
                self.expected.append([w[r] * p.hand_size for r in range(5)])
                self.size.append(p.hand_size)
                self.exact.append(p.hand_size <= 0)

    @classmethod
    def from_expected(cls, expected: Sequence[Sequence[float]], size: Sequence[int],
                      exact: Optional[Sequence[bool]] = None) -> "HandBelief":
        """A belief with the given per-player expectations / sizes (e.g. a :class:`CardCounter` snapshot)."""
        b = cls.__new__(cls)
        b.n = len(size)
        b.expected = [[float(x) for x in e] for e in expected]
        b.size = [int(x) for x in size]
        b.exact = [bool(x) for x in exact] if exact is not None else [sz <= 0 for sz in b.size]
        return b

    # --- helpers ------------------------------------------------------------
    def _renormalise(self, i: int) -> None:
        e = self.expected[i]
        for r in range(5):
            if e[r] < 0:
                e[r] = 0.0
                self.exact[i] = False      # an exact hand never goes negative: the belief was wrong
        tot = sum(e)
        n = self.size[i]
        if n <= 0:
            self.expected[i] = [0.0] * 5
            self.exact[i] = True
        elif tot <= 1e-9:
            self.expected[i] = [n / 5.0] * 5
            self.exact[i] = False
        elif abs(tot - n) > 1e-9:
            # Proportional rescale = the deficit is taken from the other resources in proportion to
            # their expectations (the cards we credited to them must have been this resource).
            f = n / tot
            self.expected[i] = [x * f for x in e]
            self.exact[i] = False

    def set_exact(self, i: int, resources: Sequence[int]) -> None:
        self.expected[i] = [float(x) for x in resources]
        self.size[i] = int(sum(resources))
        self.exact[i] = True

    def is_exact(self, i: int) -> bool:
        return bool(self.exact[i])

    def _single_type(self, i: int) -> int:
        """The only resource an exact hand holds (-1 if it is not exact or holds several)."""
        if not self.exact[i]:
            return -1
        held = [r for r in range(5) if self.expected[i][r] > 0.5]
        return held[0] if len(held) == 1 else -1

    # --- public events --------------------------------------------------------
    def observe_gain(self, i: int, res: int, n: int = 1) -> None:
        self.expected[i][res] += n
        self.size[i] += n

    def observe_spend(self, i: int, counts: Sequence[int]) -> None:
        for r in range(5):
            self.expected[i][r] -= counts[r]
            self.size[i] -= counts[r]
        self._renormalise(i)

    def observe_bank_trade(self, i: int, give_res: int, ratio: int, get_res: int) -> None:
        self.expected[i][give_res] -= ratio
        self.size[i] -= ratio
        self._renormalise(i)
        self.observe_gain(i, get_res, 1)

    def observe_player_trade(self, a: int, give: Sequence[int], b: int, get: Sequence[int]) -> None:
        """``a`` gives ``give`` to ``b`` and receives ``get`` from ``b``."""
        self.observe_spend(a, give)
        self.observe_spend(b, get)
        for r in range(5):
            if give[r]:
                self.observe_gain(b, r, give[r])
            if get[r]:
                self.observe_gain(a, r, get[r])

    def observe_discard(self, i: int, counts: Optional[Sequence[int]] = None, n: Optional[int] = None) -> None:
        """Player ``i`` discarded ``counts`` (shown) or ``n`` cards whose types were not shown
        (expected removal of a uniformly random discard)."""
        if counts is not None:
            self.observe_spend(i, counts)
            return
        k = int(n or 0)
        if k <= 0 or self.size[i] <= 0:
            return
        single = self._single_type(i)
        if single >= 0:
            self.observe_spend(i, [k if r == single else 0 for r in range(5)])
            return
        k = min(k, self.size[i])
        frac = k / self.size[i]
        self.expected[i] = [x * (1.0 - frac) for x in self.expected[i]]
        self.size[i] -= k
        self.exact[i] = self.size[i] <= 0
        self._renormalise(i)

    def observe_steal(self, victim: int, thief: int, res: Optional[int] = None) -> None:
        """A random card moved from victim to thief; ``res`` if we saw it.  An unseen card
        moves the victim's expected composition (the mean of the mixture over the stolen card)."""
        if self.size[victim] <= 0:
            return
        if res is None:
            res = self._single_type(victim)     # an exact one-resource hand: the card is known
            if res < 0:
                res = None
        if res is not None:
            self.expected[victim][res] -= 1
            self.size[victim] -= 1
            self._renormalise(victim)
            self.observe_gain(thief, res, 1)
            return
        n = self.size[victim]
        probs = [self.expected[victim][r] / n for r in range(5)]
        for r in range(5):
            self.expected[victim][r] -= probs[r]
            self.expected[thief][r] += probs[r]
        self.size[victim] -= 1
        self.size[thief] += 1
        self.exact[thief] = False
        self.exact[victim] = self.size[victim] <= 0
        self._renormalise(victim)

    def observe_monopoly(self, i: int, res: int, taken: Optional[int] = None,
                         taken_by: Optional[Dict[int, int]] = None) -> None:
        """Player ``i`` monopolised ``res``.  ``taken_by`` = cards taken from each victim (public:
        their hand sizes drop by it); else ``taken`` = the public total, split by the expected
        shares; if neither is known the expected total is used.  Card counts are conserved exactly.
        Each victim held exactly its take of ``res``: the rest of its hand is rescaled to the new size."""
        ints = [0] * self.n
        if taken_by is not None:
            for j, k in taken_by.items():
                if j != i:
                    ints[j] = max(0, min(int(k), self.size[j]))
        else:
            shares = [self.expected[j][res] if j != i else 0.0 for j in range(self.n)]
            total = sum(shares)
            if taken is None:
                taken = int(round(total))
                if any(not self.exact[j] for j in range(self.n) if j != i):
                    self.exact[i] = False
            taken = max(0, min(taken, sum(self.size[j] for j in range(self.n) if j != i)))
            # integer shares per victim by largest remainder, bounded by their hand sizes
            if taken > 0 and total > 0:
                raw = [taken * s / total for s in shares]
                ints = [min(int(r), self.size[j]) for j, r in enumerate(raw)]
                rem = taken - sum(ints)
                order = sorted((j for j in range(self.n) if j != i), key=lambda j: -(raw[j] - int(raw[j])))
                k = 0
                while rem > 0 and order:
                    j = order[k % len(order)]
                    if ints[j] < self.size[j]:
                        ints[j] += 1
                        rem -= 1
                    k += 1
                    if k > 4 * self.n:
                        break
        for j in range(self.n):
            if j == i:
                continue
            if self.exact[j] and abs(self.expected[j][res] - ints[j]) > 1e-9:
                self.exact[j] = False
            self.size[j] -= ints[j]
            self.expected[j][res] = 0.0
            self._renormalise(j)
        self.expected[i][res] += sum(ints)
        self.size[i] += sum(ints)

    def observe_hand_size(self, i: int, size: int) -> None:
        """Resynchronise with the public hand size (e.g. after an unobserved event)."""
        if size != self.size[i] and size > 0:
            self.exact[i] = False
        self.size[i] = size
        self._renormalise(i)

    def observe_bank(self, bank: Sequence[int], total: int = B.BANK_PER_RESOURCE, iters: int = 60) -> bool:
        """Make the inexact hands consistent with an *exactly known* bank: per resource the
        cards outside the bank (``total - bank[r]``) are held by the players, so the inexact
        hands jointly hold ``total - bank[r] - sum(exact hands)``.  Iterative proportional
        fitting of their expectations to those column totals and to their sizes.  Only for a
        bank that is really known (not the screenshot convention where unknown hands are still
        counted in ``state.bank``); returns False when the totals are inconsistent."""
        rows = [i for i in range(self.n) if not self.exact[i] and self.size[i] > 0]
        if not rows:
            return True
        col = [total - bank[r] - sum(self.expected[i][r] for i in range(self.n) if i not in rows) for r in range(5)]
        if any(c < -1e-6 for c in col) or abs(sum(col) - sum(self.size[i] for i in rows)) > 1e-6:
            return False
        m = [[max(0.0, self.expected[i][r]) if col[r] > 1e-9 else 0.0 for r in range(5)] for i in rows]
        for k, i in enumerate(rows):   # a hand needing mass on a resource with none left keeps a floor
            if sum(m[k]) <= 1e-12:
                m[k] = [1.0 if col[r] > 1e-9 else 0.0 for r in range(5)]
        for _ in range(iters):
            for r in range(5):
                s = sum(m[k][r] for k in range(len(rows)))
                if s > 1e-12:
                    f = col[r] / s
                    for k in range(len(rows)):
                        m[k][r] *= f
            for k, i in enumerate(rows):
                s = sum(m[k])
                if s > 1e-12:
                    f = self.size[i] / s
                    m[k] = [x * f for x in m[k]]
        for k, i in enumerate(rows):
            self.expected[i] = m[k]
        return True

    def sync(self, state: GameState) -> None:
        """Snap known hands to their exact contents and sizes to the state."""
        for i, p in enumerate(state.players):
            if p.hand_known:
                self.set_exact(i, p.resources)
            else:
                self.observe_hand_size(i, p.hand_size)

    # --- queries -------------------------------------------------------------
    def expected_hand(self, i: int) -> List[float]:
        return list(self.expected[i])

    def probability_has(self, i: int, res: int, at_least: int = 1) -> float:
        """P(player i holds >= ``at_least`` of ``res``): exact for an exact hand, else a binomial
        approximation."""
        n = self.size[i]
        if n <= 0 or at_least > n:
            return 0.0
        if self.exact[i]:
            return 1.0 if self.expected[i][res] >= at_least - 1e-9 else 0.0
        p = min(1.0, max(0.0, self.expected[i][res] / n))
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        # 1 - P(X < at_least), X ~ Binomial(n, p)
        from math import comb
        below = sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(at_least))
        return max(0.0, min(1.0, 1.0 - below))


# ---------------------------------------------------------------------------
# Exact card counting (live play)
# ---------------------------------------------------------------------------
Hand = Tuple[int, int, int, int, int]
Joint = Tuple[Hand, ...]


def _sub_multisets(hand: Sequence[int], k: int) -> List[Tuple[Hand, float]]:
    """Every ``d <= hand`` with ``sum(d) == k`` and its probability under a uniformly random
    ``k``-card discard (multivariate hypergeometric: ``prod C(h_r, d_r) / C(n, k)``)."""
    n = sum(hand)
    if k < 0 or k > n:
        return []
    denom = math.comb(n, k)
    out: List[Tuple[Hand, float]] = []
    cur = [0] * 5

    def rec(r: int, rem: int, num: int) -> None:
        if r == 4:
            if rem <= hand[4]:
                cur[4] = rem
                out.append((tuple(cur), num * math.comb(hand[4], rem) / denom))
            return
        rest = sum(hand[r + 1:])
        for x in range(max(0, rem - rest), min(hand[r], rem) + 1):
            cur[r] = x
            rec(r + 1, rem - x, num * math.comb(hand[r], x))
        cur[r] = 0

    rec(0, k, 1)
    return out


class CardCounter(HandBelief):
    """Exact card counting: a weighted mixture of *joint* hand hypotheses (every player's hand).

    Started from exactly known hands (the start of a game, or any position whose hands are
    public knowledge), every public event is applied to every hypothesis:

    * a public change of a hand (production, build costs, bank / player trades, Year of
      Plenty, a shown discard or steal) is added to that player's hand, and hypotheses in
      which the player could not have paid are dropped (conditioning on the event);
    * an unseen steal branches every hypothesis over the victim's cards, weighted by the
      victim's hand in that hypothesis (the card is uniformly random);
    * an unseen discard of ``k`` cards branches over the ``k``-card sub-hands with their
      hypergeometric probability (catanatron 3.2.1 discards uniformly at random; for a
      player that chooses its discard this is the neutral prior);
    * a Monopoly with the public per-victim amounts (hand-size drops) keeps the hypotheses in
      which every victim held exactly that many cards of the resource;
    * :meth:`observe_bank` keeps the hypotheses whose per-resource totals match the bank.

    Identical hypotheses are merged, so with no hidden event there is exactly one - the true
    hands - and the counter is exact.  Hand sizes are identical in every hypothesis and
    tracked in ``size``.  ``expected`` is the marginal expectation (a :class:`HandBelief`
    view), :meth:`sample` draws a joint hypothesis for a determinization.

    If the mixture grows beyond ``max_hypotheses`` the least likely hypotheses are dropped
    (``stats["pruned_mass"]``); if an event contradicts every hypothesis (only possible after
    such a drop, or from a caller's bookkeeping error) the counter restarts from one integer
    hand assignment closest to the marginals (``stats["resets"]``).
    """

    MAX_HYPOTHESES = 4096

    def __init__(self, hands: Sequence[Sequence[int]], max_hypotheses: int = MAX_HYPOTHESES,
                 total_per_resource: int = B.BANK_PER_RESOURCE):
        joint: Joint = tuple(tuple(int(x) for x in h) for h in hands)   # type: ignore[misc]
        if any(len(h) != 5 or min(h) < 0 for h in joint):
            raise ValueError(f"bad hands {hands!r}")
        self.n = len(joint)
        self.total = int(total_per_resource)
        self.max_hypotheses = int(max_hypotheses)
        self.hyps: Dict[Joint, float] = {joint: 1.0}
        self.size: List[int] = [sum(h) for h in joint]
        self.stats: Dict[str, float] = {"events": 0, "branching_events": 0, "max_hypotheses": 1,
                                        "pruned_mass": 0.0, "contradictions": 0, "resets": 0}
        self.last_bank: Optional[List[int]] = None
        self.last_reset: Optional[str] = None
        self._marg: Optional[List[List[float]]] = None

    @classmethod
    def from_state(cls, state: GameState, **kw) -> "CardCounter":
        """Counter started from a state whose hands are all known (``hand_known``)."""
        if not all(p.hand_known for p in state.players):
            raise ValueError("CardCounter.from_state needs every hand known (start it from the game's beginning)")
        c = cls([p.resources for p in state.players], **kw)
        c.last_bank = list(state.bank)
        return c

    # --- mixture plumbing ---------------------------------------------------
    @property
    def expected(self) -> List[List[float]]:   # type: ignore[override]
        if self._marg is None:
            m = [[0.0] * 5 for _ in range(self.n)]
            for joint, w in self.hyps.items():
                for i, h in enumerate(joint):
                    row = m[i]
                    for r in range(5):
                        if h[r]:
                            row[r] += w * h[r]
            self._marg = m
        return self._marg

    @property
    def exact(self) -> List[bool]:   # type: ignore[override]
        return [self.is_exact(i) for i in range(self.n)]

    def _update(self, fn, what: str) -> bool:
        """Replace the mixture by ``{h2: w2 for h, w in hyps for h2, w2 in fn(h, w)}`` (merged,
        renormalised, capped); on a contradiction restart from the marginals."""
        self.stats["events"] += 1
        new: Dict[Joint, float] = {}
        for joint, w in self.hyps.items():
            for j2, w2 in fn(joint, w):
                if w2 > 0.0:
                    new[j2] = new.get(j2, 0.0) + w2
        tot = sum(new.values())
        if not new or tot <= 0.0:
            self.stats["contradictions"] += 1
            return False
        if len(new) > self.max_hypotheses:
            keep = sorted(new.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_hypotheses]
            kept = sum(w for _, w in keep)
            self.stats["pruned_mass"] += (tot - kept) / tot
            new = dict(keep)
            tot = kept
        self.hyps = {j: w / tot for j, w in new.items()}
        if len(self.hyps) > self.stats["max_hypotheses"]:
            self.stats["max_hypotheses"] = len(self.hyps)
        self._marg = None
        return True

    def _reset(self, reason: str, hands_hint: Optional[Dict[int, Sequence[int]]] = None) -> None:
        """Contradiction fallback: one integer hypothesis closest to the current marginals, with
        the current sizes (and the last bank's column totals when known)."""
        marg = [list(x) for x in self.expected]
        cols = ([self.total - b for b in self.last_bank] if self.last_bank is not None else None)
        fixed = dict(hands_hint or {})
        hands: List[List[int]] = [[0] * 5 for _ in range(self.n)]
        if cols is not None:
            for i, h in fixed.items():
                for r in range(5):
                    cols[r] -= int(h[r])
        for i in range(self.n):
            if i in fixed:
                hands[i] = [int(x) for x in fixed[i]]
                continue
            want = self.size[i]
            m = marg[i]
            tot = sum(m)
            m = [x * want / tot for x in m] if tot > 1e-12 else [want / 5.0] * 5
            h = [int(math.floor(x)) for x in m]
            if cols is not None:
                h = [max(0, min(h[r], cols[r])) for r in range(5)]
            order = sorted(range(5), key=lambda r: -(m[r] - math.floor(m[r])))
            guard = 0
            while sum(h) < want and guard < 200:
                for r in order + list(range(5)):
                    if sum(h) >= want:
                        break
                    if cols is None or h[r] < cols[r]:
                        h[r] += 1
                guard += 1
            while sum(h) > want:
                r = max(range(5), key=lambda x: h[x])
                h[r] -= 1
            if cols is not None:
                for r in range(5):
                    cols[r] -= h[r]
            hands[i] = h
        self.hyps = {tuple(tuple(h) for h in hands): 1.0}   # type: ignore[dict-item]
        self._marg = None
        self.stats["resets"] += 1
        self.last_reset = reason

    @staticmethod
    def _replace(joint: Joint, i: int, hand: Sequence[int]) -> Joint:
        return joint[:i] + (tuple(hand),) + joint[i + 1:]   # type: ignore[return-value]

    # --- public events --------------------------------------------------------
    def observe_delta(self, i: int, counts: Sequence[int]) -> None:
        """A public signed change of player ``i``'s hand (gains positive, payments negative)."""
        d = [int(x) for x in counts]
        if not any(d):
            return
        rep = self._replace

        def f(joint, w):
            h = joint[i]
            new = [h[r] + d[r] for r in range(5)]
            if min(new) < 0:
                return ()
            return ((rep(joint, i, new), w),)

        self.size[i] += sum(d)
        if not self._update(f, "delta"):
            self._reset(f"player {i} paid {d} which no hypothesis could")

    def observe_gain(self, i: int, res: int, n: int = 1) -> None:
        self.observe_delta(i, [n if r == res else 0 for r in range(5)])

    def observe_spend(self, i: int, counts: Sequence[int]) -> None:
        self.observe_delta(i, [-int(c) for c in counts])

    def observe_bank_trade(self, i: int, give_res: int, ratio: int, get_res: int) -> None:
        d = [0] * 5
        d[give_res] -= ratio
        d[get_res] += 1
        self.observe_delta(i, d)

    def observe_player_trade(self, a: int, give: Sequence[int], b: int, get: Sequence[int]) -> None:
        """``a`` gives ``give`` to ``b`` and receives ``get`` from ``b``."""
        self.observe_delta(a, [get[r] - give[r] for r in range(5)])
        self.observe_delta(b, [give[r] - get[r] for r in range(5)])

    def observe_discard(self, i: int, counts: Optional[Sequence[int]] = None, n: Optional[int] = None) -> None:
        """Player ``i`` discarded ``counts`` (shown; weighted by its random-discard likelihood) or
        ``n`` cards whose types were not shown (branches over the possible sub-hands)."""
        rep = self._replace
        if counts is not None:
            c = [int(x) for x in counts]
            k = sum(c)
            if k <= 0:
                return

            def f(joint, w):
                h = joint[i]
                if any(h[r] < c[r] for r in range(5)):
                    return ()
                like = math.prod(math.comb(h[r], c[r]) for r in range(5)) / math.comb(sum(h), k)
                return ((rep(joint, i, [h[r] - c[r] for r in range(5)]), w * like),)

            self.size[i] -= k
            if not self._update(f, "discard"):
                self._reset(f"player {i} discarded {c} which no hypothesis held")
            return
        k = int(n or 0)
        if k <= 0:
            return

        def g(joint, w):
            h = joint[i]
            return [(rep(joint, i, [h[r] - d[r] for r in range(5)]), w * p) for d, p in _sub_multisets(h, k)]

        self.size[i] -= k
        self.stats["branching_events"] += 1
        if not self._update(g, "hidden discard"):
            self._reset(f"player {i} discarded {k} cards but holds fewer")

    def observe_steal(self, victim: int, thief: int, res: Optional[int] = None) -> None:
        """A uniformly random card moved from ``victim`` to ``thief``; ``res`` when we saw it
        (then weighted by its likelihood ``hand[res] / size``), else every possibility."""
        if self.size[victim] <= 0:
            return
        nv = self.size[victim]

        def move(joint, r):
            hv = list(joint[victim])
            ht = list(joint[thief])
            hv[r] -= 1
            ht[r] += 1
            out = list(joint)
            out[victim] = tuple(hv)
            out[thief] = tuple(ht)
            return tuple(out)

        if res is not None:
            def f(joint, w):
                k = joint[victim][res]
                return ((move(joint, res), w * k / nv),) if k > 0 else ()
        else:
            self.stats["branching_events"] += 1

            def f(joint, w):
                h = joint[victim]
                return [(move(joint, r), w * h[r] / nv) for r in range(5) if h[r] > 0]

        self.size[victim] -= 1
        self.size[thief] += 1
        if not self._update(f, "steal"):
            self._reset(f"steal of {res} from player {victim}")

    def observe_monopoly(self, i: int, res: int, taken: Optional[int] = None,
                         taken_by: Optional[Dict[int, int]] = None) -> None:
        """Player ``i`` monopolised ``res``: ``taken_by[j]`` cards from each victim ``j`` (public:
        the victims' hand sizes drop by it) - every victim held exactly that many.  With only the
        total ``taken`` the victims' amounts must be the same in every surviving hypothesis
        (else pass ``taken_by``: the hand sizes must stay exact)."""
        victims = [j for j in range(self.n) if j != i]
        if taken_by is not None:
            want = {j: int(taken_by.get(j, 0)) for j in victims}

            def ok(joint):
                return all(joint[j][res] == want[j] for j in victims)
        elif taken is not None:
            def ok(joint):
                return sum(joint[j][res] for j in victims) == int(taken)
        else:
            raise ValueError("CardCounter.observe_monopoly needs the public take (taken_by or taken)")

        def f(joint, w):
            if not ok(joint):
                return ()
            out = list(joint)
            got = 0
            for j in victims:
                h = list(joint[j])
                got += h[res]
                h[res] = 0
                out[j] = tuple(h)
            h = list(joint[i])
            h[res] += got
            out[i] = tuple(h)
            return ((tuple(out), w),)

        if not self._update(f, "monopoly"):
            self._reset(f"monopoly of {res} inconsistent with every hypothesis")
            if taken_by is not None:   # apply it to the restarted hypothesis (clamped)
                joint = next(iter(self.hyps))
                out = [list(h) for h in joint]
                for j in victims:
                    out[j][res] = max(0, out[j][res] - want[j]) if out[j][res] < want[j] else 0
                out[i][res] += sum(want.values())
                self.hyps = {tuple(tuple(h) for h in out): 1.0}   # type: ignore[dict-item]
                self._marg = None
        amounts = {j: next(iter(self.hyps))[j][res] for j in victims}   # 0 after the event
        sizes_before = list(self.size)
        if taken_by is not None:
            for j in victims:
                self.size[j] = sizes_before[j] - want[j]
            self.size[i] = sizes_before[i] + sum(want.values())
        else:
            # the per-victim amounts must agree across hypotheses to keep sizes exact
            per = {j: set() for j in victims}
            del amounts
            self.size[i] = sizes_before[i] + int(taken)   # type: ignore[arg-type]
            for joint in self.hyps:
                for j in victims:
                    per[j].add(sum(joint[j]))
            for j in victims:
                if len(per[j]) != 1:
                    raise ValueError("monopoly amounts per victim differ across hypotheses: pass taken_by")
                self.size[j] = per[j].pop()

    def observe_hand_size(self, i: int, size: int) -> None:
        """Consistency check against the public hand size (restart if it disagrees)."""
        if int(size) != self.size[i]:
            self.stats["contradictions"] += 1
            self.size[i] = int(size)
            self._reset(f"hand size of player {i} is {size}")

    def observe_hand(self, i: int, resources: Sequence[int]) -> None:
        """Player ``i``'s hand is known exactly (e.g. our own): keep the hypotheses that agree."""
        want = tuple(int(x) for x in resources)
        self.size[i] = sum(want)
        if all(joint[i] == want for joint in self.hyps):
            return
        if not self._update(lambda joint, w: ((joint, w),) if joint[i] == want else (), "own hand"):
            self._reset(f"hand of player {i} is {list(want)}", {i: want})

    def observe_bank(self, bank: Sequence[int], total: Optional[int] = None) -> bool:   # type: ignore[override]
        """Keep the hypotheses whose per-resource totals match the (exactly known) bank."""
        tot = self.total if total is None else int(total)
        self.last_bank = [int(x) for x in bank]
        want = [tot - int(b) for b in bank]

        def ok(joint):
            for r in range(5):
                if sum(h[r] for h in joint) != want[r]:
                    return False
            return True

        if all(ok(j) for j in self.hyps):
            return True
        if not self._update(lambda joint, w: ((joint, w),) if ok(joint) else (), "bank"):
            self._reset(f"bank {list(bank)} matches no hypothesis")
            return False
        return True

    def sync(self, state: GameState) -> None:
        for i, p in enumerate(state.players):
            if p.hand_known:
                self.observe_hand(i, p.resources)
            else:
                self.observe_hand_size(i, p.hand_size)

    # --- queries -------------------------------------------------------------
    @property
    def num_hypotheses(self) -> int:
        return len(self.hyps)

    def is_exact(self, i: int) -> bool:
        it = iter(self.hyps)
        first = next(it)[i]
        return all(joint[i] == first for joint in it)

    def marginal(self, i: int) -> Dict[Hand, float]:
        """Distribution of player ``i``'s hand."""
        out: Dict[Hand, float] = {}
        for joint, w in self.hyps.items():
            out[joint[i]] = out.get(joint[i], 0.0) + w
        return out

    def weight_of(self, hands: Sequence[Sequence[int]]) -> float:
        """Probability of the exact joint assignment ``hands`` (0 if it is not in the support)."""
        return self.hyps.get(tuple(tuple(int(x) for x in h) for h in hands), 0.0)   # type: ignore[arg-type]

    def probability_has(self, i: int, res: int, at_least: int = 1) -> float:
        return sum(w for joint, w in self.hyps.items() if joint[i][res] >= at_least)

    def most_likely(self) -> Joint:
        """The most probable joint hypothesis (ties: the smallest tuple, so it is deterministic)."""
        return min(self.hyps.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def sample(self, rng) -> Joint:
        """One joint hypothesis drawn with its probability."""
        if len(self.hyps) == 1:
            return next(iter(self.hyps))
        x = rng.random()
        acc = 0.0
        last = None
        for joint, w in sorted(self.hyps.items()):
            acc += w
            last = joint
            if x <= acc:
                return joint
        return last   # type: ignore[return-value]

    def hand_belief(self) -> HandBelief:
        """An expected-count :class:`HandBelief` snapshot (marginals, sizes, exactness)."""
        return HandBelief.from_expected(self.expected, self.size, self.exact)


def monopoly_expected_take(state: GameState, player: int, res: int,
                           belief: Optional[HandBelief] = None) -> float:
    """Expected number of cards a Monopoly on ``res`` would collect."""
    if belief is None:
        hands = expected_opponent_hands(state, me=player)
        return sum(hands[j][res] for j in range(state.num_players) if j != player)
    return sum(belief.expected[j][res] for j in range(state.num_players) if j != player)
