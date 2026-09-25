"""Coalition detection from revealed preferences.

Two players are in a coalition when they keep favouring each other *at a
cost*: A trades with B although the bank or another player offered a better
deal, B accepts A's offers that lose value, A's robber spares B when B was
the better target, C refuses A's fair offers while accepting B's.  Every
such event is scored by the **EV sacrifice** - how much worse the choice
was for the actor than the best alternative, in card-value units - and
accumulated **quadratically**: a blatant sacrifice is a far bigger signal
than ten subtle ones, and random noise (small deviations) barely counts.
Signals decay every turn like political capital.

``favour[a][b]`` is how much a has demonstrably favoured b; the symmetric
``strength(a, b)`` is the coalition score; ``blocs()`` groups players whose
pairwise strength exceeds a threshold.  Consumers:

* ``politics.PoliticalState`` - target weights (hit the bloc that includes
  the leader, do not expect allies to accept our offers), favour slack;
* the CLI advice ("blue and orange act as a bloc: 3 blatant deals").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import board as B
from .placement import RESOURCE_DEMAND, player_production, resource_scarcity
from .state import GameState, TradeOffer

SCALE = 0.5          # sacrifice (card-value units) that counts as one full signal
CAP = 4.0            # max signal from a single event (8x SCALE, squared)
DECAY = 0.97         # per turn
BLOC_THRESHOLD = 1.0 # symmetric strength above which two players are treated as a bloc


def _pname(state: GameState, i: int) -> str:
    p = state.players[i]
    return p.name or p.color


def resource_values(state: GameState, i: int) -> List[float]:
    """Marginal card values for any player (mean 1); mirrors opponent_model.our_resource_values
    but only needs public information (production + rough needs)."""
    p = state.players[i]
    prod = player_production(state, i, ignore_robber=True)
    scarcity = resource_scarcity(state)
    hand = p.resources if p.hand_known else [0] * 5
    vals = []
    for r in range(5):
        v = RESOURCE_DEMAND[r] * (scarcity[r] ** 0.5) / (1.0 + 4.0 * prod[r] + 0.4 * max(0, hand[r] - 2))
        vals.append(v)
    m = sum(vals) / 5.0
    return [v / m for v in vals]


def deal_value(vals: Sequence[float], receive: Sequence[int], pay: Sequence[int]) -> float:
    return sum(vals[r] * (receive[r] - pay[r]) for r in range(5))


def bank_alternative_value(state: GameState, i: int, vals: Sequence[float], get: Sequence[int]) -> float:
    """Best value of obtaining ``get`` from the bank / ports with the least valuable cards."""
    p = state.players[i]
    hand = list(p.resources) if p.hand_known else [3] * 5
    total = 0.0
    for r in range(5):
        for _ in range(get[r]):
            if state.bank[r] <= 0:
                return -1e9
            best = None
            for g in range(5):
                if g == r:
                    continue
                ratio = state.port_ratio(i, g)
                if hand[g] >= ratio:
                    cost = ratio * vals[g]
                    if best is None or cost < best[0]:
                        best = (cost, g, ratio)
            if best is None:
                return -1e9
            total += vals[r] - best[0]
            hand[best[1]] -= best[2]
    return total


def signal(sacrifice: float) -> float:
    """Quadratic signal from an EV sacrifice (card-value units), capped."""
    if sacrifice <= 0:
        return 0.0
    return min(CAP, (sacrifice / SCALE) ** 2)


class CoalitionDetector:
    def __init__(self, n: int):
        self.n = n
        self.favour: List[List[float]] = [[0.0] * n for _ in range(n)]
        self.events: List[str] = []
        self.blatant: Dict[Tuple[int, int], int] = {}

    def ensure(self, n: int) -> None:
        if n != self.n:
            self.__init__(n)

    def _add(self, a: int, b: int, sig: float, why: str) -> None:
        if a == b or sig <= 0:
            return
        self.favour[a][b] += sig
        if sig >= 1.0:
            key = (min(a, b), max(a, b))
            self.blatant[key] = self.blatant.get(key, 0) + 1
        self.events.append(f"{why} (signal {sig:.2f})")
        if len(self.events) > 40:
            self.events = self.events[-40:]

    def decay(self, factor: float = DECAY) -> None:
        for i in range(self.n):
            for j in range(self.n):
                self.favour[i][j] *= factor

    # --- events -------------------------------------------------------------
    def observe_trade(self, state: GameState, proposer: int, partner: int, offer: TradeOffer,
                      acceptance: Optional[Dict[int, bool]] = None) -> None:
        """An executed trade: proposer gave ``offer.give`` and received ``offer.get``."""
        self.ensure(state.num_players)
        # Proposer: executed deal vs the best alternative (bank / another accepter).
        vp = resource_values(state, proposer)
        executed = deal_value(vp, offer.get, offer.give)
        alt = bank_alternative_value(state, proposer, vp, offer.get)
        # The same deal with another player who also accepted would have the same value,
        # so the only "better elsewhere" for the proposer is the bank - unless the partner
        # was chosen over a less dangerous accepter (handled by politics, not here).
        sacrifice_p = alt - executed
        if sacrifice_p > 0.05:
            self._add(proposer, partner, signal(sacrifice_p),
                      f"{_pname(state, proposer)} traded with {_pname(state, partner)} although the bank was "
                      f"{sacrifice_p:.2f} better")
        # Partner: accepted a deal that loses value for them.
        vq = resource_values(state, partner)
        gain_q = deal_value(vq, offer.give, offer.get)
        if gain_q < -0.05:
            self._add(partner, proposer, signal(-gain_q),
                      f"{_pname(state, partner)} accepted a losing deal from {_pname(state, proposer)} "
                      f"({gain_q:+.2f})")
        # Third parties who accepted but were not chosen: the proposer preferred the partner.
        if acceptance:
            others = [j for j, ok in acceptance.items() if ok and j != partner]
            if others:
                self._add(proposer, partner, 0.15,
                          f"{_pname(state, proposer)} chose {_pname(state, partner)} over "
                          + ", ".join(_pname(state, j) for j in others))

    def observe_rejection(self, state: GameState, responder: int, offer: TradeOffer) -> None:
        """Responder refused a deal that was good for them: a signal against the proposer."""
        self.ensure(state.num_players)
        vq = resource_values(state, responder)
        gain = deal_value(vq, offer.give, offer.get)
        if gain > 0.15:
            # Refusing a favourable deal favours "everyone but the proposer"; spread over the others.
            others = [k for k in range(self.n) if k not in (responder, offer.proposer)]
            for k in others:
                self._add(responder, k, signal(gain) / max(1, len(others)),
                          f"{_pname(state, responder)} refused a good deal from {_pname(state, offer.proposer)}")

    def observe_robber(self, state: GameState, actor: int, chosen_hex: int, chosen_victim: int) -> None:
        """Actor spared a better target: a signal in favour of the spared player."""
        from .robber import hex_damage, choose_victim
        self.ensure(state.num_players)
        best_score, best_victim = -1e9, -1
        chosen_score = None
        for h in range(B.NUM_HEXES):
            if h == state.robber:
                continue
            opp, own = hex_damage(state, h, actor)
            victim = choose_victim(state, h, actor)
            sc = opp - 1.6 * own + (1.0 if victim >= 0 else 0.0)
            if h == chosen_hex:
                chosen_score = sc
            if sc > best_score:
                best_score, best_victim = sc, victim
        if chosen_score is None or best_victim < 0 or best_victim == chosen_victim:
            return
        sacrifice = (best_score - chosen_score) / 6.0   # pips-equivalent -> card-value scale
        if sacrifice > 0.1:
            self._add(actor, best_victim, signal(sacrifice),
                      f"{_pname(state, actor)} spared {_pname(state, best_victim)} with the robber")

    # --- queries ------------------------------------------------------------
    def strength(self, a: int, b: int) -> float:
        if a == b or a < 0 or b < 0:
            return 0.0
        return self.favour[a][b] + self.favour[b][a]

    def allies(self, i: int, threshold: float = BLOC_THRESHOLD) -> List[int]:
        return [j for j in range(self.n) if j != i and self.strength(i, j) >= threshold]

    def blocs(self, threshold: float = BLOC_THRESHOLD) -> List[List[int]]:
        seen = set()
        out = []
        for i in range(self.n):
            if i in seen:
                continue
            group = [i]
            stack = [i]
            seen.add(i)
            while stack:
                a = stack.pop()
                for b in range(self.n):
                    if b not in seen and self.strength(a, b) >= threshold:
                        seen.add(b)
                        group.append(b)
                        stack.append(b)
            if len(group) > 1:
                out.append(sorted(group))
        return out

    def against(self, me: int, threshold: float = BLOC_THRESHOLD) -> float:
        """Strength of the strongest bloc that excludes ``me`` (how ganged-up we are)."""
        best = 0.0
        for g in self.blocs(threshold):
            if me not in g:
                s = sum(self.strength(a, b) for a in g for b in g if a < b)
                best = max(best, s)
        return best

    def summary(self, state: GameState, me: int) -> List[str]:
        lines = []
        for g in self.blocs():
            names = " + ".join(_pname(state, i) for i in g)
            pairs = [(a, b) for a in g for b in g if a < b]
            blat = sum(self.blatant.get((a, b), 0) for a, b in pairs)
            s = sum(self.strength(a, b) for a, b in pairs)
            tag = " (includes you)" if me in g else ""
            lines.append(f"Bloc: {names}{tag} - strength {s:.1f}, {blat} blatant favour(s); "
                         + ("they will favour each other in trades and spare each other with the robber"
                            if me not in g else "keep it useful to them"))
        weak = [(self.strength(a, b), a, b) for a in range(self.n) for b in range(a + 1, self.n)
                if 0.3 <= self.strength(a, b) < BLOC_THRESHOLD]
        for s, a, b in sorted(weak, reverse=True)[:2]:
            lines.append(f"Possible alignment: {_pname(state, a)} & {_pname(state, b)} (strength {s:.1f})")
        if self.events:
            lines.append("Coalition signals: " + "; ".join(self.events[-3:]))
        return lines

    # --- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"n": self.n, "favour": self.favour, "blatant": {f"{a},{b}": v for (a, b), v in self.blatant.items()},
                "events": self.events[-40:]}

    @staticmethod
    def from_dict(d: dict) -> "CoalitionDetector":
        c = CoalitionDetector(int(d.get("n", 4)))
        fav = d.get("favour")
        if fav and len(fav) == c.n:
            c.favour = [[float(x) for x in row] for row in fav]
        for k, v in d.get("blatant", {}).items():
            a, b = k.split(",")
            c.blatant[(int(a), int(b))] = int(v)
        c.events = list(d.get("events", []))
        return c
