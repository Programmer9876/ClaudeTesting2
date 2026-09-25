"""Exploitative opponent modelling for 3-4 player games.

Nash equilibrium play is intractable (and not even well defined) for 3-4
players, so instead of trying to be unexploitable we try to *exploit*: every
opponent gets a :class:`OpponentProfile` that records how they deviate from
our own model of sensible play, with exponential decay so recent behaviour
dominates.  The profile feeds:

* **Acceptance prediction** - how likely a given offer is to be accepted by
  that player (overall generosity, per-resource tendencies, implied
  resource valuations, stage of the game).
* **Arbitrage** - offers where the counterpart values what we give more
  than what we get (by *their* implied valuation) while *we* value it the
  other way round, either directly or as an intermediary between two
  opponents.
* **Robber / target tendencies** - who they rob, whether they hit the
  leader, how much risk (> 7 cards) they run.
* **Surprise** - how often their action differs from what our heuristic
  predicts; a high surprise player is modelled with more weight on the
  observed statistics and less on the heuristic prior.

Profiles are keyed by player *name* so they can persist across games
(``save`` / ``load``) and across screenshots of the same game.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .actions import Action
from .counting import HandBelief, expected_hidden_vp, expected_opponent_hands
from .placement import RESOURCE_DEMAND, player_production, resource_scarcity
from .state import GameState, PHASE_GAME_OVER, TradeOffer

DECAY = 0.9          # per observation of the same statistic
VALUE_LR = 0.12      # learning rate for implied valuations
SURPRISE_DECAY = 0.85


# ---------------------------------------------------------------------------
# Game stage
# ---------------------------------------------------------------------------
def game_stage(state: GameState) -> float:
    """0.0 = just started, 1.0 = somebody is about to win."""
    if state.phase == PHASE_GAME_OVER:
        return 1.0
    best = 0.0
    for i in range(state.num_players):
        vp = state.public_vp(i) + expected_hidden_vp(state, i)
        best = max(best, vp)
    by_vp = max(0.0, min(1.0, (best - 2.0) / 8.0))
    by_turn = max(0.0, min(1.0, state.turn / (30.0 * max(1, state.num_players))))
    return max(by_vp, 0.6 * by_turn + 0.4 * by_vp)


def trade_stage_factor(state: GameState) -> float:
    """Multiplier on trade willingness: early ~1.0, late ~0.3.

    Early on every trade grows both economies and the game is long enough to
    cash in; late in the game a trade mostly accelerates whoever is closer
    to 10 VP, so the bar for accepting/proposing rises.
    """
    s = game_stage(state)
    return 1.0 - 0.7 * (s ** 1.5)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
class _EW:
    """Exponentially weighted ratio num/den."""

    __slots__ = ("num", "den")

    def __init__(self, num: float = 0.0, den: float = 0.0):
        self.num = num
        self.den = den

    def add(self, x: float, decay: float = DECAY) -> None:
        self.num = self.num * decay + x
        self.den = self.den * decay + 1.0

    def mean(self, prior: float = 0.5, prior_weight: float = 2.0) -> float:
        return (self.num + prior * prior_weight) / (self.den + prior_weight)

    @property
    def weight(self) -> float:
        return self.den

    def to_list(self) -> List[float]:
        return [self.num, self.den]

    @staticmethod
    def from_list(v) -> "_EW":
        return _EW(float(v[0]), float(v[1]))


class OpponentProfile:
    """Decayed statistics about one player's behaviour."""

    def __init__(self, name: str = ""):
        self.name = name
        self.accept = _EW()                          # accepts offers made to them
        self.accept_get = [_EW() for _ in range(5)]  # accept when they would receive r
        self.accept_give = [_EW() for _ in range(5)]  # accept when they would pay r
        self.value = [1.0] * 5                       # implied relative valuation of each resource
        self.proposes = _EW()                        # proposes a trade on their turn
        self.robs_leader = _EW()                     # robber goes to the current leader
        self.robbed: Dict[str, float] = {}           # decayed count of robber hits per victim name
        self.traded_with: Dict[str, float] = {}      # decayed count of executed trades per partner name
        self.risk_over7 = _EW()                      # ends turn with > 7 cards
        self.builds = {"road": 0.0, "settlement": 0.0, "city": 0.0, "dev": 0.0}
        self.surprise = _EW()                        # action != our heuristic's top pick
        self.observations = 0

    # --- valuation updates -------------------------------------------------
    def _shift_value(self, up: Sequence[int], down: Sequence[int], lr: float) -> None:
        for r in range(5):
            if up[r]:
                self.value[r] += lr * up[r]
            if down[r]:
                self.value[r] -= lr * down[r]
        # keep mean 1 and bounded
        for r in range(5):
            self.value[r] = min(3.0, max(0.3, self.value[r]))
        m = sum(self.value) / 5.0
        self.value = [v / m for v in self.value]

    def note_accept(self, receives: Sequence[int], pays: Sequence[int], accepted: bool) -> None:
        self.accept.add(1.0 if accepted else 0.0)
        for r in range(5):
            if receives[r]:
                self.accept_get[r].add(1.0 if accepted else 0.0)
            if pays[r]:
                self.accept_give[r].add(1.0 if accepted else 0.0)
        if accepted:
            # They value what they receive at least as much as what they pay.
            self._shift_value(receives, pays, VALUE_LR)
        else:
            self._shift_value(pays, receives, VALUE_LR * 0.5)
        self.observations += 1

    def note_proposal(self, gives: Sequence[int], wants: Sequence[int]) -> None:
        self.proposes.add(1.0)
        self._shift_value(wants, gives, VALUE_LR * 0.8)
        self.observations += 1

    def note_bank_trade(self, give_res: int, ratio: int, get_res: int) -> None:
        up = [0] * 5
        down = [0] * 5
        up[get_res] = 1
        down[give_res] = 1
        self._shift_value(up, down, VALUE_LR * 0.6)
        self.observations += 1

    # --- queries ------------------------------------------------------------
    def acceptance_rate(self) -> float:
        return self.accept.mean(prior=0.45)

    def gives_easily(self, r: int) -> float:
        return self.accept_give[r].mean(prior=self.acceptance_rate(), prior_weight=1.5)

    def wants(self, r: int) -> float:
        return self.accept_get[r].mean(prior=self.acceptance_rate(), prior_weight=1.5)

    def confidence(self) -> float:
        """0..1: how much the statistics (vs. the heuristic prior) should be trusted."""
        return 1.0 - math.exp(-self.accept.weight / 3.0)

    def surprise_rate(self) -> float:
        return self.surprise.mean(prior=0.3)

    def style_summary(self) -> str:
        parts = []
        acc = self.acceptance_rate()
        parts.append(f"accepts {acc:.0%} of offers" + (" (few observations)" if self.accept.weight < 2 else ""))
        cheap = [B.RESOURCE_NAMES[r] for r in range(5) if self.value[r] < 0.8]
        dear = [B.RESOURCE_NAMES[r] for r in range(5) if self.value[r] > 1.25]
        if cheap:
            parts.append("undervalues " + "/".join(cheap))
        if dear:
            parts.append("overvalues " + "/".join(dear))
        if self.robs_leader.weight >= 1:
            parts.append(f"robs the leader {self.robs_leader.mean():.0%} of the time")
        if self.risk_over7.weight >= 1 and self.risk_over7.mean(prior=0.2) > 0.4:
            parts.append("often holds > 7 cards (7s hurt them)")
        b = self.builds
        if sum(b.values()) > 0:
            fav = max(b, key=b.get)
            parts.append(f"favours {fav}s")
        if self.surprise.weight >= 3:
            parts.append(f"deviates from expected play {self.surprise_rate():.0%}")
        return "; ".join(parts)

    # --- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "accept": self.accept.to_list(),
            "accept_get": [e.to_list() for e in self.accept_get],
            "accept_give": [e.to_list() for e in self.accept_give],
            "value": list(self.value),
            "proposes": self.proposes.to_list(),
            "robs_leader": self.robs_leader.to_list(),
            "robbed": dict(self.robbed),
            "traded_with": dict(self.traded_with),
            "risk_over7": self.risk_over7.to_list(),
            "builds": dict(self.builds),
            "surprise": self.surprise.to_list(),
            "observations": self.observations,
        }

    @staticmethod
    def from_dict(d: dict) -> "OpponentProfile":
        p = OpponentProfile(d.get("name", ""))
        p.accept = _EW.from_list(d.get("accept", [0, 0]))
        p.accept_get = [_EW.from_list(x) for x in d.get("accept_get", [[0, 0]] * 5)]
        p.accept_give = [_EW.from_list(x) for x in d.get("accept_give", [[0, 0]] * 5)]
        p.value = [float(x) for x in d.get("value", [1.0] * 5)]
        p.proposes = _EW.from_list(d.get("proposes", [0, 0]))
        p.robs_leader = _EW.from_list(d.get("robs_leader", [0, 0]))
        p.robbed = {k: float(v) for k, v in d.get("robbed", {}).items()}
        p.traded_with = {k: float(v) for k, v in d.get("traded_with", {}).items()}
        p.risk_over7 = _EW.from_list(d.get("risk_over7", [0, 0]))
        p.builds = {k: float(v) for k, v in d.get("builds", p.builds).items()}
        p.surprise = _EW.from_list(d.get("surprise", [0, 0]))
        p.observations = int(d.get("observations", 0))
        return p


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _pname(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


def _leader(state: GameState, exclude: int) -> int:
    best, best_vp = -1, -1.0
    for i in range(state.num_players):
        if i == exclude:
            continue
        vp = state.public_vp(i) + expected_hidden_vp(state, i)
        if vp > best_vp:
            best, best_vp = i, vp
    return best


def our_resource_values(state: GameState, me: int, needed: Optional[Sequence[int]] = None) -> List[float]:
    """Marginal value of one more card of each resource for ``me`` (mean ~1)."""
    p = state.players[me]
    prod = player_production(state, me, ignore_robber=True)
    scarcity = resource_scarcity(state)
    if needed is None:
        from .discard import needed_vector
        needed = needed_vector(state, me)
    vals = []
    for r in range(5):
        missing = max(0, needed[r] - p.resources[r])
        surplus = max(0, p.resources[r] - needed[r])
        v = RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) * (1.0 + 1.5 * missing) / (1.0 + 4.0 * prod[r] + 0.6 * surplus)
        vals.append(v)
    m = sum(vals) / 5.0
    return [v / m for v in vals]


class OpponentModel:
    """Profiles for every seat plus prediction / exploitation helpers."""

    def __init__(self, state: Optional[GameState] = None, track_surprise: bool = False):
        self.profiles: Dict[str, OpponentProfile] = {}
        self.track_surprise = track_surprise
        if state is not None:
            self.attach(state)

    # --- bookkeeping ---------------------------------------------------------
    def attach(self, state: GameState) -> None:
        for i in range(state.num_players):
            self.profile(_pname(state, i))

    def profile(self, name: str) -> OpponentProfile:
        prof = self.profiles.get(name)
        if prof is None:
            prof = OpponentProfile(name)
            self.profiles[name] = prof
        return prof

    def profile_of(self, state: GameState, i: int) -> OpponentProfile:
        return self.profile(_pname(state, i))

    def observe(self, state_before: GameState, action: Action, player: int,
                predicted: Optional[Action] = None) -> None:
        """Update the acting player's profile from a public action.

        ``state_before`` is the state the action was taken in (needed for
        the pending offer and hand sizes).  ``predicted`` is our model's top
        choice for that decision, if known (feeds the surprise statistic).
        """
        prof = self.profile_of(state_before, player)
        kind = action[0]
        if predicted is not None and kind not in (A.ROLL,):
            prof.surprise.add(0.0 if predicted == action else 1.0, SURPRISE_DECAY)
        if kind == A.PROPOSE_TRADE:
            prof.note_proposal(action[1], action[2])
        elif kind in (A.ACCEPT_TRADE, A.REJECT_TRADE):
            offer = state_before.pending_trade
            if offer is not None:
                prof.note_accept(offer.give, offer.get, kind == A.ACCEPT_TRADE)
        elif kind == A.EXECUTE_TRADE:
            partner = _pname(state_before, action[1])
            prof.traded_with[partner] = prof.traded_with.get(partner, 0.0) * DECAY + 1.0
            pp = self.profile(partner)
            me = _pname(state_before, player)
            pp.traded_with[me] = pp.traded_with.get(me, 0.0) * DECAY + 1.0
        elif kind == A.BANK_TRADE:
            ratio = state_before.port_ratio(player, action[1])
            prof.note_bank_trade(action[1], ratio, action[2])
        elif kind in (A.MOVE_ROBBER, A.PLAY_KNIGHT):
            victim = action[2]
            if victim is not None and victim >= 0:
                vname = _pname(state_before, victim)
                prof.robbed[vname] = prof.robbed.get(vname, 0.0) * DECAY + 1.0
                prof.robs_leader.add(1.0 if victim == _leader(state_before, player) else 0.0)
            prof.observations += 1
        elif kind == A.BUILD_ROAD:
            prof.builds["road"] = prof.builds["road"] * DECAY + 1.0
        elif kind == A.BUILD_SETTLEMENT:
            prof.builds["settlement"] = prof.builds["settlement"] * DECAY + 1.0
        elif kind == A.BUILD_CITY:
            prof.builds["city"] = prof.builds["city"] * DECAY + 1.0
        elif kind == A.BUY_DEV:
            prof.builds["dev"] = prof.builds["dev"] * DECAY + 1.0
        elif kind == A.END_TURN:
            p = state_before.players[player]
            n = p.total_resources if p.hand_known else p.hand_size
            prof.risk_over7.add(1.0 if n > 7 else 0.0)

    def observe_event(self, state: GameState, text: str) -> Optional[str]:
        """Parse a human-typed event such as::

            blue accepted give ore get wood      (blue received ore, paid wood)
            blue rejected give 2 wood get brick
            blue proposed give wheat get ore
            blue robbed red
            blue bank 4 wood for 1 ore

        Returns an error message or None.
        """
        toks = text.strip().lower().replace(",", " ").split()
        if len(toks) < 2:
            return "empty event"
        name = toks[0]
        try:
            idx = next(i for i in range(state.num_players)
                       if _pname(state, i).lower() == name or state.players[i].color.lower() == name)
        except StopIteration:
            return f"unknown player '{name}'"
        prof = self.profile_of(state, idx)
        verb = toks[1]

        def counts(words: List[str]) -> List[int]:
            out = [0] * 5
            n = 1
            for w in words:
                if w.isdigit():
                    n = int(w)
                    continue
                r = B.RESOURCE_ALIASES.get(w)
                if r is None or r == B.DESERT:
                    continue
                out[r] += n
                n = 1
            return out

        if verb in ("accepted", "rejected", "proposed"):
            if "give" not in toks or "get" not in toks:
                return "expected 'give ... get ...'"
            gi, ge = toks.index("give"), toks.index("get")
            give = counts(toks[gi + 1:ge])
            get = counts(toks[ge + 1:])
            if verb == "proposed":
                prof.note_proposal(give, get)
            else:
                prof.note_accept(give, get, verb == "accepted")
            return None
        if verb == "robbed" and len(toks) >= 3:
            vname = toks[2]
            try:
                vidx = next(i for i in range(state.num_players)
                            if _pname(state, i).lower() == vname or state.players[i].color.lower() == vname)
            except StopIteration:
                return f"unknown victim '{vname}'"
            prof.robbed[_pname(state, vidx)] = prof.robbed.get(_pname(state, vidx), 0.0) * DECAY + 1.0
            prof.robs_leader.add(1.0 if vidx == _leader(state, idx) else 0.0)
            return None
        if verb == "bank" and "for" in toks:
            fi = toks.index("for")
            give = counts(toks[2:fi])
            get = counts(toks[fi + 1:])
            gr = max(range(5), key=lambda r: give[r])
            tr = max(range(5), key=lambda r: get[r])
            prof.note_bank_trade(gr, max(1, give[gr]), tr)
            return None
        return f"unknown event verb '{verb}'"

    # --- prediction ----------------------------------------------------------
    def predict_accept(self, state: GameState, j: int, receives: Sequence[int], pays: Sequence[int],
                       proposer: Optional[int] = None, belief: Optional[HandBelief] = None,
                       politics=None) -> float:
        """Probability that player ``j`` accepts receiving ``receives`` for ``pays``.

        ``politics`` (a ``politics.PoliticalState``) adds a bounded favour
        slack: friends accept slightly unfavourable deals, the visible leader
        gets a premium demanded - never enough to make an unfair deal pass.
        """
        p = state.players[j]
        # Can they pay at all?
        if p.hand_known:
            if any(p.resources[r] < pays[r] for r in range(5)):
                return 0.0
            can_pay = 1.0
        elif belief is not None:
            can_pay = 1.0
            for r in range(5):
                if pays[r] > 0:
                    can_pay *= belief.probability_has(j, r, pays[r])
            if can_pay <= 0.0:
                return 0.0
        else:
            hands = expected_opponent_hands(state, me=proposer)
            n = p.hand_size
            can_pay = 1.0
            for r in range(5):
                if pays[r] > 0:
                    if n <= 0:
                        return 0.0
                    q = min(1.0, hands[j][r] / n)
                    can_pay *= max(0.0, 1.0 - (1.0 - q) ** n) if pays[r] == 1 else max(0.0, q ** pays[r])
            if can_pay <= 0.0:
                return 0.0
        prof = self.profile_of(state, j)
        # Value of the deal for them by their implied valuation, plus their needs.
        vals = list(prof.value)
        # Needs: resources they are short of for a city / settlement / dev
        prod = player_production(state, j, ignore_robber=True)
        gain = 0.0
        for r in range(5):
            need_boost = 1.0 + 0.6 / (1.0 + 8.0 * prod[r])
            gain += vals[r] * need_boost * (receives[r] - pays[r])
        gain += 0.15 * (sum(receives) - sum(pays))  # card count
        if politics is not None and proposer is not None:
            gain += politics.favor_slack(state, j, proposer)
        stage = trade_stage_factor(state)
        logit = 1.8 * gain - 0.4 + 1.2 * (stage - 0.65)
        # Personal generosity (deviation from the 45% prior), weighted by confidence.
        acc = prof.acceptance_rate()
        logit += 2.5 * (acc - 0.45) * (0.3 + 0.7 * prof.confidence())
        for r in range(5):
            if receives[r]:
                logit += 1.0 * (prof.wants(r) - acc) * prof.confidence()
            if pays[r]:
                logit += 1.0 * (prof.gives_easily(r) - acc) * prof.confidence()
        # Don't-feed-the-leader: if the proposer is the leader they accept less.
        if proposer is not None:
            lead = _leader(state, j)
            if lead == proposer:
                pvp = state.public_vp(proposer) + expected_hidden_vp(state, proposer)
                logit -= 0.6 * max(0.0, pvp - 5.0)
        prob = 1.0 / (1.0 + math.exp(-logit))
        return max(0.0, min(1.0, prob * can_pay))

    # --- exploitation ----------------------------------------------------------
    def rank_offers(self, state: GameState, me: int, offers: Sequence[Action], needed: Optional[Sequence[int]] = None,
                    belief: Optional[HandBelief] = None) -> List[dict]:
        """Score PROPOSE_TRADE actions by expected gain = P(accept) * our value gain."""
        our_vals = our_resource_values(state, me, needed)
        out = []
        for a in offers:
            give, get = a[1], a[2]
            gain_us = sum(our_vals[r] * (get[r] - give[r]) for r in range(5))
            best_p, best_j = 0.0, -1
            for j in range(state.num_players):
                if j == me or not self._safe_partner(state, me, j, give):
                    continue
                pj = self.predict_accept(state, j, give, get, proposer=me, belief=belief)
                if pj > best_p:
                    best_p, best_j = pj, j
            out.append({"action": a, "p_accept": best_p, "partner": best_j, "gain": gain_us,
                        "score": best_p * max(0.0, gain_us)})
        out.sort(key=lambda d: -d["score"])
        return out

    def arbitrage_opportunities(self, state: GameState, me: int, needed: Optional[Sequence[int]] = None,
                                belief: Optional[HandBelief] = None, min_prob: float = 0.3) -> List[dict]:
        """Deals where the counterpart's implied valuation disagrees with ours.

        Direct: buy X with Y from j when j values X < Y and we value X > Y.
        Intermediary: buy X with Y from j, sell X for Z to k when j values
        X < Y, k values Z < X and we value Z > Y (we net Y -> Z).
        Returns dicts with ``steps`` (list of PROPOSE_TRADE actions),
        ``p``, ``gain``, ``reason`` sorted by expected gain.
        """
        p = state.players[me]
        our_vals = our_resource_values(state, me, needed)
        stage = trade_stage_factor(state)
        out = []
        n = state.num_players
        for j in range(n):
            if j == me:
                continue
            pj = self.profile_of(state, j)
            for x in range(5):          # what we buy
                for y in range(5):      # what we pay
                    if x == y or p.resources[y] <= 0:
                        continue
                    give_vec = [1 if r == y else 0 for r in range(5)]
                    if not self._safe_partner(state, me, j, give_vec):
                        continue
                    their_edge = pj.value[y] - pj.value[x]      # >0: they prefer y (they give x cheaply)
                    our_edge = our_vals[x] - our_vals[y]        # >0: we prefer x
                    give = tuple(1 if r == y else 0 for r in range(5))
                    get = tuple(1 if r == x else 0 for r in range(5))
                    prob = self.predict_accept(state, j, give, get, proposer=me, belief=belief)
                    if prob < min_prob:
                        continue
                    if our_edge > 0.05 and their_edge > -0.05:
                        gain = prob * our_edge * stage
                        out.append({"steps": [(A.PROPOSE_TRADE, give, get)], "p": prob, "gain": gain,
                                    "partner": j,
                                    "reason": f"{_pname(state, j)} values {B.RESOURCE_NAMES[x]} "
                                              f"{'below' if their_edge > 0 else 'about like'} {B.RESOURCE_NAMES[y]}; "
                                              f"we need {B.RESOURCE_NAMES[x]} more (edge {our_edge:+.2f}, P(accept) {prob:.0%})"})
                    # intermediary: sell x to k for z
                    for k in range(n):
                        if k in (me, j) or not self._safe_partner(state, me, k, [1 if r == x else 0 for r in range(5)]):
                            continue
                        pk = self.profile_of(state, k)
                        for z in range(5):
                            if z in (x, y):
                                continue
                            k_edge = pk.value[x] - pk.value[z]    # >0: k gives z for x happily
                            net_edge = our_vals[z] - our_vals[y]  # we end up converting y -> z
                            if k_edge <= 0.05 or net_edge <= 0.05:
                                continue
                            give2 = tuple(1 if r == x else 0 for r in range(5))
                            get2 = tuple(1 if r == z else 0 for r in range(5))
                            s2 = state.copy()
                            s2.players[me].resources[y] -= 1
                            s2.players[me].resources[x] += 1
                            prob2 = self.predict_accept(s2, k, give2, get2, proposer=me, belief=belief)
                            if prob2 < min_prob:
                                continue
                            gain = prob * prob2 * net_edge * stage
                            out.append({"steps": [(A.PROPOSE_TRADE, give, get), (A.PROPOSE_TRADE, give2, get2)],
                                        "p": prob * prob2, "gain": gain, "partner": j,
                                        "reason": f"intermediary: buy {B.RESOURCE_NAMES[x]} from {_pname(state, j)} "
                                                  f"with {B.RESOURCE_NAMES[y]}, sell it to {_pname(state, k)} for "
                                                  f"{B.RESOURCE_NAMES[z]} (P {prob:.0%} x {prob2:.0%})"})
        out.sort(key=lambda d: -d["gain"])
        return out[:8]

    @staticmethod
    def _safe_partner(state: GameState, me: int, j: int, give: Sequence[int]) -> bool:
        """Never feed a player about to win or whom the cards would hand a build."""
        from .trading import offer_is_feeding_leader
        if state.public_vp(j) + expected_hidden_vp(state, j) >= B.VP_TO_WIN - 1:
            return False
        return not offer_is_feeding_leader(state, me, j, give)[0]

    def summary(self, state: GameState, me: Optional[int] = None) -> List[str]:
        lines = []
        for i in range(state.num_players):
            if i == me:
                continue
            prof = self.profile_of(state, i)
            lines.append(f"{_pname(state, i)}: {prof.style_summary()}")
        lines.append(f"Game stage {game_stage(state):.0%}: trade willingness x{trade_stage_factor(state):.2f}")
        return lines

    # --- persistence -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": 1, "profiles": {k: v.to_dict() for k, v in self.profiles.items()}}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def from_dict(d: dict) -> "OpponentModel":
        m = OpponentModel()
        for k, v in d.get("profiles", {}).items():
            m.profiles[k] = OpponentProfile.from_dict(v)
        return m

    @staticmethod
    def load(path: str) -> "OpponentModel":
        with open(path) as f:
            return OpponentModel.from_dict(json.load(f))
