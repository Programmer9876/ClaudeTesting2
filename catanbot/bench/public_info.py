"""What a Colonist.io player knows, for catanbot inside catanatron games ("counted" information mode).

catanatron exposes every hand, so :func:`catanbot.bench.catanatron_adapter.state_to_catanbot`
converts its state with every card known (the ``"full"`` information mode).  This module gives
:class:`~catanbot.bench.catanatron_adapter.CatanbotPlayer` exactly the information a human card
counter has at a Colonist table instead (``CatanbotPlayer(info="counted")``):

* **Always known**: the board, robber, buildings, roads, the bank per resource, the size of the
  development deck, every player's hand *size* and development-card *count*, every played
  development card (type), awards and public VP; our own hand and development cards exactly.
* **Public events with their content**: dice and the production each player received, builds
  (their costs), bank / port trades, domestic trades, Monopoly (resource and the amount each
  victim lost - their hand sizes drop by it), Year of Plenty, Road Building, development-card
  purchases (count and cost only).
* **Hidden to third parties**: the card of a robber steal (known to the thief and the victim
  only), the cards discarded on a 7 (the count is public; the types are hidden unless
  ``discards_public=True`` or the discard is ours) and the type of every development card an
  opponent drew and still holds.  Discards of one 7 are *simultaneous* (as on Colonist): the
  bank only reveals their total per resource once the last discarder is done, so with a
  single hidden discarder the bank still pins the cards down, with several only their split
  between the discarders stays unknown.
* Opponents' unplayed development types: only the public pool (the 25-card deck minus every
  played card minus our own cards) is known, :func:`catanbot.counting.dev_pool` semantics.

:class:`PublicInfoTracker` follows catanatron's action log on its own shadow copy of the game
and turns every entry into the public events above (:meth:`PublicInfoTracker._observe_entry` is
the only place that looks at the true hands, and for a hidden event it uses nothing but the
public hand sizes; a chance result - stolen card, drawn development card, discarded cards - is
read only when the model allows it).  A :class:`catanbot.counting.CardCounter` holds the exact
posterior over every player's hand (one hypothesis while nothing hidden happened); the
development-card bookkeeping (counts, played types, cards bought this turn) is kept alongside.

The player then decides on a public view (:meth:`PublicInfoTracker.public_view`: every opponent
``hand_known=False`` with the exact ``hand_size``, ``dev_known=False`` with the exact
``dev_count`` and zeroed cards, the deck as the public pool's expectation) and on ``K``
determinizations sampled from the tracker (:meth:`PublicInfoTracker.determinize`: a joint hand
hypothesis drawn from the counter, development cards dealt from the public pool, conditioned on
no opponent already holding enough VP cards to have won).  Our own legal actions are computed
on :meth:`PublicInfoTracker.canonical_view` (the most likely hypothesis): they only depend on our
own cards, the board, the bank and the opponents' hand *sizes* (robber victims), all public, so
every determinization offers the same list.  These views live in
:class:`catanbot.public_belief.PublicBelief` (catanatron-free, shared with the advisor's
Colonist-log card counting in :mod:`catanbot.colonist_log`).

Both catanatron generations are supported (3.3 ``ActionRecord`` results / 3.2.1 fully specified
logged actions).
"""
from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from catanatron.models.enums import ActionType
from catanatron.state import State

from .. import board as B
from ..counting import CardCounter
from ..public_belief import (  # noqa: F401 - apportion / redact_state re-exported
    PublicBelief,
    _public_pool_estimate,
    _weighted_pick,
    apportion,
    redact_state,
)
from .catanatron_adapter import (  # noqa: F401 - INFO_MODES / DEFAULT_INFO_SAMPLES re-exported
    API_33,
    DEFAULT_INFO_SAMPLES,
    INFO_MODES,
    ActionRecord,
    CB_TO_DEV,
    CB_TO_RESOURCE,
    DEV_TO_CB,
    DISCARD_TYPES,
    RESOURCE_TO_CB,
    action_log,
    log_action,
    log_result,
    replay_entry,
)

__all__ = [
    "INFO_MODES",
    "DEFAULT_INFO_SAMPLES",
    "PublicInfoTracker",
    "TrackerError",
    "Snapshot",
    "snapshot",
    "initial_state_like",
    "redact_state",
    "apportion",
]

_PLAY_TYPES = {
    ActionType.PLAY_KNIGHT_CARD: B.DEV_KNIGHT,
    ActionType.PLAY_YEAR_OF_PLENTY: B.DEV_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY: B.DEV_MONOPOLY,
    ActionType.PLAY_ROAD_BUILDING: B.DEV_ROAD_BUILDING,
}


class TrackerError(RuntimeError):
    """The tracker's shadow game or bookkeeping disagrees with the live game's public information."""


@dataclass
class Snapshot:
    """What the tracker reads from a (shadow) catanatron state around one log entry.

    ``hands`` are every player's true resources: :meth:`PublicInfoTracker._observe_entry` uses
    them only for events whose content is public (and the sizes of every hand, which are).
    Development cards are read as *counts* only, except ``my_dev`` (our own cards).
    """

    hands: List[List[int]]
    dev_count: List[int]
    played: List[List[int]]
    bank: List[int]
    deck: int
    my_dev: List[int]


def snapshot(st: State, me: int) -> Snapshot:
    ps = st.player_state
    n = len(st.colors)
    hands = [[int(ps[f"P{i}_{r}_IN_HAND"]) for r in CB_TO_RESOURCE] for i in range(n)]
    dev_count = [sum(int(ps[f"P{i}_{d}_IN_HAND"]) for d in CB_TO_DEV) for i in range(n)]
    played = [[int(ps.get(f"P{i}_PLAYED_{d}", 0)) for d in CB_TO_DEV] for i in range(n)]
    my_dev = [int(ps[f"P{me}_{d}_IN_HAND"]) for d in CB_TO_DEV]
    return Snapshot(hands, dev_count, played, [int(x) for x in st.resource_freqdeck],
                    len(st.development_listdeck), my_dev)


def initial_state_like(st: State) -> State:
    """A fresh catanatron state with ``st``'s board, seating and rules and no action taken (its
    development deck order is random - replays take the drawn cards from the log).  The global
    ``random`` state is left untouched (3.2.1 shuffles with it)."""
    params = inspect.signature(State.__init__).parameters
    kwargs = {}
    if "rng" in params:
        kwargs["rng"] = random.Random(0)
    if "friendly_robber" in params:
        kwargs["friendly_robber"] = bool(getattr(st, "friendly_robber", False))
    saved = random.getstate()
    try:
        st0 = State(list(st.players), catan_map=st.board.map, discard_limit=st.discard_limit, **kwargs)
    finally:
        random.setstate(saved)
    st0.players = list(st.players)
    st0.colors = tuple(st.colors)
    st0.color_to_index = {c: i for i, c in enumerate(st0.colors)}
    return st0


class PublicInfoTracker(PublicBelief):
    """Follows a catanatron game's action log with Colonist's public information for seat ``me``.

    ``start(st)`` replays the whole log from a rebuilt initial state (so a player that joins
    late still only uses public events), ``follow(st)`` processes the entries logged since.
    ``counter`` (:class:`~catanbot.counting.CardCounter`) holds the posterior over every hand;
    ``dev_count`` / ``played`` / ``bought_this_turn`` the public development-card bookkeeping.

    ``discards_public`` makes the cards of every discard public (Colonist shows them in some
    modes); ``reveal_hidden`` (diagnostics / tests only) treats every steal, discard and
    development draw as public - the counter must then equal the true hands at every step.

    The views handed to the bot (:meth:`public_view`, :meth:`canonical_view`, :meth:`determinize`)
    and the belief queries come from :class:`catanbot.public_belief.PublicBelief`, shared with the
    advisor's Colonist-log tracker (:class:`catanbot.colonist_log.ColonistLogTracker`).
    """

    #: 3.3 follows the official rule (a card bought this turn is not playable yet), 3.2.1 does not
    new_devs_unplayable = API_33

    def __init__(self, me_color, discards_public: bool = False, reveal_hidden: bool = False,
                 vps_to_win: int = 10, max_hypotheses: int = CardCounter.MAX_HYPOTHESES):
        self.me_color = me_color
        self.discards_public = bool(discards_public)
        self.reveal_hidden = bool(reveal_hidden)
        self.vps_to_win = int(vps_to_win)
        self.max_hypotheses = int(max_hypotheses)
        self.me = -1
        self.n = 0
        self.color_to_index: Dict = {}
        self.counter: Optional[CardCounter] = None
        self.dev_count: List[int] = []
        self.played: List[List[int]] = []
        self.bought_this_turn: List[int] = []
        self.known_dev: List[Optional[List[int]]] = []   # exact types (reveal_hidden only; ours via my_dev)
        self.my_dev: List[int] = [0] * 5
        self.deck = 0
        self.bank: List[int] = [B.BANK_PER_RESOURCE] * 5
        self.stats: Dict[str, int] = {"entries": 0, "hidden_steals": 0, "seen_steals": 0, "hidden_discards": 0,
                                      "hidden_dev_draws": 0, "resyncs": 0}
        #: hidden discards of the 7 being resolved (seat -> cards); applied jointly with the bank
        #: once the last discarder is done (Colonist's simultaneous discards)
        self.pending_discards: Dict[int, int] = {}
        self._shadow: Optional[State] = None
        self._observed = 0

    # --- lifecycle -----------------------------------------------------------
    def _init_from(self, st: State) -> None:
        self.color_to_index = dict(st.color_to_index)
        self.n = len(st.colors)
        self.me = self.color_to_index[self.me_color]
        snap = snapshot(st, self.me)
        if any(any(h) for h in snap.hands) or any(snap.dev_count):
            raise TrackerError("the tracker must start from a state whose hands are public (the game's start)")
        self.counter = CardCounter(snap.hands, max_hypotheses=self.max_hypotheses)
        self.counter.last_bank = list(snap.bank)
        self.dev_count = list(snap.dev_count)
        self.played = [list(x) for x in snap.played]
        self.bought_this_turn = [0] * self.n
        self.known_dev = [[0] * 5 if (self.reveal_hidden and j != self.me) else None for j in range(self.n)]
        self.my_dev = list(snap.my_dev)
        self.deck = snap.deck
        self.bank = list(snap.bank)
        self.pending_discards = {}

    def start(self, st: State) -> None:
        """Start following ``st``'s game: replay its whole log from the initial position."""
        log = action_log(st)
        if len(log) == 0:
            self._init_from(st)
        else:
            for _ in self.replay_log(st):
                pass
            self._check_public(st)
        self._shadow = st.copy()
        self._observed = len(log)

    def replay_log(self, st: State):
        """Restart from ``st``'s initial position and process its log one entry at a time; yields
        ``(entry, shadow state after it)`` (the shadow is the tracker's own true replay - for
        tests and diagnostics: the counter never reads it except through the public rules)."""
        st0 = initial_state_like(st)
        self._init_from(st0)
        self._shadow = st0
        self._observed = 0
        for entry in list(action_log(st)):
            self.step(entry)
            yield entry, self._shadow

    def follow(self, st: State) -> None:
        """Process the entries logged since the last call (raises :class:`TrackerError` /
        catanatron's exception if the shadow game cannot replay them; see :meth:`resync`)."""
        if self._shadow is None:
            self.start(st)
            return
        log = action_log(st)
        n = len(log)
        while self._observed < n:
            self.step(log[self._observed])

    def step(self, entry) -> None:
        """Apply one log entry to the shadow game and feed its public events to the counter."""
        pre = snapshot(self._shadow, self.me)
        replay_entry(self._shadow, entry)
        post = snapshot(self._shadow, self.me)
        self._observe_entry(entry, pre, post)
        self._observed += 1

    def resync(self, st: State) -> None:
        """Recover after the shadow diverged: continue from ``st``'s public information (the
        counter restarts from its marginals fitted to the public sizes and bank)."""
        self.stats["resyncs"] += 1
        snap = snapshot(st, self.me)
        c = self.counter
        c.size = [sum(h) for h in snap.hands]      # public hand sizes
        c.last_bank = list(snap.bank)
        self.pending_discards = {}
        c._reset("resync")
        c.observe_hand(self.me, snap.hands[self.me])
        self._public_bookkeeping(snap)
        self._shadow = st.copy()
        self._observed = len(action_log(st))

    def _check_public(self, st: State) -> None:
        live = snapshot(st, self.me)
        sh = snapshot(self._shadow, self.me)
        if ([sum(h) for h in live.hands] != [sum(h) for h in sh.hands] or live.bank != sh.bank
                or live.deck != sh.deck or live.dev_count != sh.dev_count or live.played != sh.played
                or live.hands[self.me] != sh.hands[self.me]):
            raise TrackerError("replaying the log from the initial position does not reach the live game")

    # --- the public events of one log entry ------------------------------------
    def _revealed_result(self, entry, index: Optional[int] = None):
        """The chance result of an entry (3.3 ``ActionRecord.result``, 3.2.1 the logged value) -
        only called where the information model allows it."""
        if ActionRecord is not None:
            return log_result(entry)
        v = log_action(entry).value
        if index is None:
            return v
        return v[index] if v is not None and len(v) > index else None

    def _observe_entry(self, entry, pre: Snapshot, post: Snapshot) -> None:
        a = log_action(entry)
        t = a.action_type
        actor = self.color_to_index[a.color]
        me = self.me
        c = self.counter
        n = self.n
        self.stats["entries"] += 1
        if self.pending_discards and t not in DISCARD_TYPES:
            # the 7's discards are over: the bank now shows their total per resource
            c.last_bank = list(pre.bank)
            c.observe_discards(self.pending_discards, pre.bank)
            self.pending_discards = {}
        c.last_bank = list(post.bank)
        size0 = [sum(h) for h in pre.hands]
        size1 = [sum(h) for h in post.hands]
        if t == ActionType.MOVE_ROBBER:
            who = a.value[1] if a.value is not None and len(a.value) > 1 else None
            if who is not None:
                v = self.color_to_index[who]
                if size1[v] < size0[v]:                       # a card changed hands (sizes are public)
                    res = None
                    if me in (actor, v) or self.reveal_hidden:
                        card = self._revealed_result(entry, 2)
                        res = RESOURCE_TO_CB[card] if card is not None else None
                    if res is None:
                        self.stats["hidden_steals"] += 1
                    else:
                        self.stats["seen_steals"] += 1
                    c.observe_steal(v, actor, res)
        elif t in DISCARD_TYPES:
            k = size0[actor] - size1[actor]
            if k > 0:
                if actor == me or self.discards_public or self.reveal_hidden:
                    cards = a.value if isinstance(a.value, (list, tuple)) else [a.value]
                    counts = [0] * 5
                    for card in cards:
                        counts[RESOURCE_TO_CB[card]] += 1
                    c.observe_discard(actor, counts)
                else:
                    self.stats["hidden_discards"] += k      # cards (3.2.1 logs one action per discarder)
                    self.pending_discards[actor] = self.pending_discards.get(actor, 0) + k
        elif t == ActionType.PLAY_MONOPOLY:
            res = RESOURCE_TO_CB[a.value]
            c.observe_monopoly(actor, res, taken_by={j: size0[j] - size1[j] for j in range(n) if j != actor})
        else:
            # Every other entry is public with its content: production, build costs, trades,
            # Year of Plenty, the cost of a development card.
            for j in range(n):
                d = [post.hands[j][r] - pre.hands[j][r] for r in range(5)]
                if any(d):
                    c.observe_delta(j, d)
        c.observe_hand(me, post.hands[me])
        for j in range(n):
            if j not in self.pending_discards:
                c.observe_hand_size(j, size1[j])
        if t not in DISCARD_TYPES:
            # The bank is public: after a 7's (simultaneous) discards, and after every other entry.
            c.observe_bank(post.bank)
        # development cards: counts, played types, purchases this turn
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            self.bought_this_turn[actor] += 1
            if actor != me:
                if self.reveal_hidden:
                    card = self._revealed_result(entry)
                    if card is not None and self.known_dev[actor] is not None:
                        self.known_dev[actor][DEV_TO_CB[card]] += 1
                else:
                    self.stats["hidden_dev_draws"] += 1
        elif t in _PLAY_TYPES and actor != me and self.known_dev[actor] is not None:
            kd = self.known_dev[actor]
            kd[_PLAY_TYPES[t]] = max(0, kd[_PLAY_TYPES[t]] - 1)
        elif t == ActionType.END_TURN:
            self.bought_this_turn = [0] * n
        self._public_bookkeeping(post)

    def _public_bookkeeping(self, snap: Snapshot) -> None:
        self.dev_count = list(snap.dev_count)
        self.played = [list(x) for x in snap.played]
        self.my_dev = list(snap.my_dev)
        self.deck = snap.deck
        self.bank = list(snap.bank)
