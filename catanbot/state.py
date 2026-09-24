"""Game state data model.

The engine (``catanbot.engine``) is a *perfect information* simulator: every
card is known.  Hidden information (opponent hands, dev cards, the dev deck
order) is handled outside the engine by ``catanbot.inference`` which produces
*determinizations*: fully specified states sampled from the belief.  This
keeps the rules code simple and fast.

States are plain Python objects with list fields; ``GameState.copy()`` is a
cheap structural copy used by the search.  The JSON format produced by
``to_dict``/``from_dict`` is the interchange format used by the CLI, the
screenshot parsers and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import board as B

# Turn phases
PHASE_SETUP_SETTLEMENT = "setup_settlement"
PHASE_SETUP_ROAD = "setup_road"
PHASE_ROLL = "roll"                    # current player must roll (may play a knight first)
PHASE_DISCARD = "discard"              # players in ``discard_queue`` must discard
PHASE_ROBBER = "robber"                # current player must move the robber (and pick a victim)
PHASE_MAIN = "main"                    # build / trade / play dev / end turn
PHASE_TRADE_RESPONSE = "trade_response"  # responders accept/reject ``pending_trade``
PHASE_TRADE_SELECT = "trade_select"    # proposer picks one of the accepting partners
PHASE_GAME_OVER = "game_over"

PLAYER_COLORS = ["red", "blue", "orange", "green", "white", "purple", "brown", "pink"]


@dataclass
class Player:
    color: str = "red"
    name: str = ""
    resources: List[int] = field(default_factory=lambda: [0] * 5)
    # Dev cards that may be played (bought on an earlier turn), by type.
    dev_cards: List[int] = field(default_factory=lambda: [0] * 5)
    # Dev cards bought this turn (cannot be played until next turn), by type.
    dev_cards_new: List[int] = field(default_factory=lambda: [0] * 5)
    played_knights: int = 0
    settlements: List[int] = field(default_factory=list)   # vertex ids
    cities: List[int] = field(default_factory=list)        # vertex ids
    roads: List[int] = field(default_factory=list)         # edge ids
    # --- hidden-information bookkeeping (used when the state came from a
    # screenshot).  The engine ignores these; ``inference`` uses them. ---
    hand_known: bool = True    # True: ``resources`` is exact.  False: only ``hand_size`` is exact.
    hand_size: int = 0         # exact number of resource cards (kept in sync by the engine)
    dev_known: bool = True     # True: dev_cards are exact.  False: only ``dev_count`` exact.
    dev_count: int = 0         # exact number of unplayed dev cards held
    # Colonist shows these publicly: knights played, and the public VP.

    def copy(self) -> "Player":
        p = Player(self.color, self.name, list(self.resources), list(self.dev_cards),
                   list(self.dev_cards_new), self.played_knights, list(self.settlements),
                   list(self.cities), list(self.roads), self.hand_known, self.hand_size,
                   self.dev_known, self.dev_count)
        return p

    # --- convenience -----------------------------------------------------
    @property
    def total_resources(self) -> int:
        return sum(self.resources)

    @property
    def total_dev(self) -> int:
        return sum(self.dev_cards) + sum(self.dev_cards_new)

    @property
    def vp_cards(self) -> int:
        return self.dev_cards[B.DEV_VP] + self.dev_cards_new[B.DEV_VP]

    def can_afford(self, cost) -> bool:
        r = self.resources
        return all(r[i] >= cost[i] for i in range(5))

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "name": self.name,
            "resources": {B.RESOURCE_NAMES[i]: self.resources[i] for i in range(5)},
            "dev_cards": {B.DEV_NAMES[i]: self.dev_cards[i] for i in range(5)},
            "dev_cards_new": {B.DEV_NAMES[i]: self.dev_cards_new[i] for i in range(5)},
            "played_knights": self.played_knights,
            "settlements": list(self.settlements),
            "cities": list(self.cities),
            "roads": list(self.roads),
            "hand_known": self.hand_known,
            "hand_size": self.hand_size if not self.hand_known else sum(self.resources),
            "dev_known": self.dev_known,
            "dev_count": self.dev_count if not self.dev_known else self.total_dev,
        }

    @staticmethod
    def from_dict(d: dict) -> "Player":
        p = Player()
        p.color = d.get("color", "red")
        p.name = d.get("name", "")
        res = d.get("resources", {})
        if isinstance(res, dict):
            p.resources = [int(res.get(B.RESOURCE_NAMES[i], 0)) for i in range(5)]
        else:
            p.resources = [int(x) for x in res]
        for attr in ("dev_cards", "dev_cards_new"):
            dv = d.get(attr, {})
            if isinstance(dv, dict):
                setattr(p, attr, [int(dv.get(B.DEV_NAMES[i], 0)) for i in range(5)])
            else:
                setattr(p, attr, [int(x) for x in dv] if dv else [0] * 5)
        p.played_knights = int(d.get("played_knights", 0))
        p.settlements = [int(v) for v in d.get("settlements", [])]
        p.cities = [int(v) for v in d.get("cities", [])]
        p.roads = [int(e) for e in d.get("roads", [])]
        p.hand_known = bool(d.get("hand_known", True))
        p.hand_size = int(d.get("hand_size", sum(p.resources)))
        if p.hand_known:
            p.hand_size = sum(p.resources)
        p.dev_known = bool(d.get("dev_known", True))
        p.dev_count = int(d.get("dev_count", p.total_dev))
        if p.dev_known:
            p.dev_count = p.total_dev
        return p


@dataclass
class TradeOffer:
    proposer: int
    give: List[int]        # resources the proposer gives (5 counts)
    get: List[int]         # resources the proposer wants (5 counts)
    responses: Dict[int, bool] = field(default_factory=dict)  # responder idx -> accepted?

    def copy(self) -> "TradeOffer":
        return TradeOffer(self.proposer, list(self.give), list(self.get), dict(self.responses))

    def to_dict(self) -> dict:
        return {"proposer": self.proposer, "give": list(self.give), "get": list(self.get),
                "responses": {str(k): v for k, v in self.responses.items()}}

    @staticmethod
    def from_dict(d: dict) -> "TradeOffer":
        return TradeOffer(int(d["proposer"]), [int(x) for x in d["give"]], [int(x) for x in d["get"]],
                          {int(k): bool(v) for k, v in d.get("responses", {}).items()})


@dataclass
class GameState:
    # Board
    hexes: List[Tuple[int, int]] = field(default_factory=lambda: list(B.STANDARD_HEXES))  # (resource, number)
    robber: int = 9
    ports: Dict[int, int] = field(default_factory=lambda: dict(B.STANDARD_PORTS))  # vertex -> port type
    # Players and supplies
    players: List[Player] = field(default_factory=list)
    bank: List[int] = field(default_factory=lambda: [B.BANK_PER_RESOURCE] * 5)
    dev_deck: List[int] = field(default_factory=lambda: list(B.DEV_DECK_COUNTS))  # remaining by type
    # Turn structure
    current: int = 0
    phase: str = PHASE_SETUP_SETTLEMENT
    turn: int = 0                 # number of completed turns (setup turns included)
    setup_round: int = 0          # 0 = first (forward), 1 = second (reverse)
    setup_last_settlement: int = -1
    dice: int = 0                 # last roll (0 = not rolled yet this turn)
    dev_played_this_turn: bool = False
    free_roads: int = 0           # roads granted by Road Building still to place
    discard_queue: List[int] = field(default_factory=list)
    pending_trade: Optional[TradeOffer] = None
    trade_responder: int = -1     # whose response is awaited in PHASE_TRADE_RESPONSE
    trades_this_turn: int = 0
    # Awards
    longest_road_owner: int = -1
    longest_road_len: int = 0
    largest_army_owner: int = -1
    winner: int = -1
    # Bookkeeping useful for training / explanations
    rolls_history_len: int = 0
    max_turns: int = 400          # safety cap for simulations

    # ------------------------------------------------------------------
    def copy(self) -> "GameState":
        s = GameState.__new__(GameState)
        s.hexes = self.hexes            # immutable during a game
        s.robber = self.robber
        s.ports = self.ports            # immutable during a game
        s.players = [p.copy() for p in self.players]
        s.bank = list(self.bank)
        s.dev_deck = list(self.dev_deck)
        s.current = self.current
        s.phase = self.phase
        s.turn = self.turn
        s.setup_round = self.setup_round
        s.setup_last_settlement = self.setup_last_settlement
        s.dice = self.dice
        s.dev_played_this_turn = self.dev_played_this_turn
        s.free_roads = self.free_roads
        s.discard_queue = list(self.discard_queue)
        s.pending_trade = self.pending_trade.copy() if self.pending_trade else None
        s.trade_responder = self.trade_responder
        s.trades_this_turn = self.trades_this_turn
        s.longest_road_owner = self.longest_road_owner
        s.longest_road_len = self.longest_road_len
        s.largest_army_owner = self.largest_army_owner
        s.winner = self.winner
        s.rolls_history_len = self.rolls_history_len
        s.max_turns = self.max_turns
        return s

    @property
    def num_players(self) -> int:
        return len(self.players)

    def player_index(self, color: str) -> int:
        for i, p in enumerate(self.players):
            if p.color == color:
                return i
        raise KeyError(color)

    # --- victory points ---------------------------------------------------
    def public_vp(self, i: int) -> int:
        p = self.players[i]
        vp = len(p.settlements) + 2 * len(p.cities)
        if self.longest_road_owner == i:
            vp += 2
        if self.largest_army_owner == i:
            vp += 2
        return vp

    def total_vp(self, i: int) -> int:
        """Public VP + VP dev cards (assumes perfect information)."""
        return self.public_vp(i) + self.players[i].vp_cards

    # --- occupancy helpers ------------------------------------------------
    def vertex_owner(self, v: int) -> int:
        for i, p in enumerate(self.players):
            if v in p.settlements or v in p.cities:
                return i
        return -1

    def edge_owner(self, e: int) -> int:
        for i, p in enumerate(self.players):
            if e in p.roads:
                return i
        return -1

    def occupied_vertices(self) -> Dict[int, int]:
        occ: Dict[int, int] = {}
        for i, p in enumerate(self.players):
            for v in p.settlements:
                occ[v] = i
            for v in p.cities:
                occ[v] = i
        return occ

    def occupied_edges(self) -> Dict[int, int]:
        occ: Dict[int, int] = {}
        for i, p in enumerate(self.players):
            for e in p.roads:
                occ[e] = i
        return occ

    def port_ratio(self, i: int, res: int) -> int:
        """Best bank-trade ratio player i has for giving ``res`` (4, 3 or 2)."""
        p = self.players[i]
        ratio = 4
        for v in p.settlements + p.cities:
            t = self.ports.get(v)
            if t is None:
                continue
            if t == B.PORT_GENERIC:
                ratio = min(ratio, 3)
            elif t == res:
                return 2
        return ratio

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "hexes": [{"resource": B.RESOURCE_NAMES[r], "number": n} for r, n in self.hexes],
            "robber": self.robber,
            "ports": {str(v): B.PORT_NAMES[t] for v, t in sorted(self.ports.items())},
            "players": [p.to_dict() for p in self.players],
            "bank": {B.RESOURCE_NAMES[i]: self.bank[i] for i in range(5)},
            "dev_deck": {B.DEV_NAMES[i]: self.dev_deck[i] for i in range(5)},
            "current": self.current,
            "phase": self.phase,
            "turn": self.turn,
            "setup_round": self.setup_round,
            "setup_last_settlement": self.setup_last_settlement,
            "dice": self.dice,
            "dev_played_this_turn": self.dev_played_this_turn,
            "free_roads": self.free_roads,
            "discard_queue": list(self.discard_queue),
            "pending_trade": self.pending_trade.to_dict() if self.pending_trade else None,
            "trade_responder": self.trade_responder,
            "trades_this_turn": self.trades_this_turn,
            "longest_road_owner": self.longest_road_owner,
            "longest_road_len": self.longest_road_len,
            "largest_army_owner": self.largest_army_owner,
            "winner": self.winner,
        }

    @staticmethod
    def from_dict(d: dict) -> "GameState":
        s = GameState()
        hexes = []
        for h in d["hexes"]:
            if isinstance(h, dict):
                res = h["resource"]
                num = int(h.get("number") or 0)
            else:
                res, num = h[0], int(h[1] or 0)
            if isinstance(res, str):
                res = B.RESOURCE_ALIASES[res.lower()]
            hexes.append((int(res), 0 if int(res) == B.DESERT else num))
        s.hexes = hexes
        s.robber = int(d.get("robber", 9))
        ports = d.get("ports")
        if ports is None:
            s.ports = dict(B.STANDARD_PORTS)
        else:
            s.ports = {}
            for v, t in ports.items():
                if isinstance(t, str):
                    t = B.PORT_GENERIC if t in ("3:1", "generic", "any") else B.RESOURCE_ALIASES[t.lower()]
                s.ports[int(v)] = int(t)
        s.players = [Player.from_dict(p) for p in d.get("players", [])]
        bank = d.get("bank")
        if bank is None:
            s.bank = [B.BANK_PER_RESOURCE] * 5
        elif isinstance(bank, dict):
            s.bank = [int(bank.get(B.RESOURCE_NAMES[i], B.BANK_PER_RESOURCE)) for i in range(5)]
        else:
            s.bank = [int(x) for x in bank]
        dd = d.get("dev_deck")
        if dd is None:
            s.dev_deck = list(B.DEV_DECK_COUNTS)
        elif isinstance(dd, dict):
            s.dev_deck = [int(dd.get(B.DEV_NAMES[i], 0)) for i in range(5)]
        else:
            s.dev_deck = [int(x) for x in dd]
        s.current = int(d.get("current", 0))
        s.phase = d.get("phase", PHASE_MAIN)
        s.turn = int(d.get("turn", 0))
        s.setup_round = int(d.get("setup_round", 0))
        s.setup_last_settlement = int(d.get("setup_last_settlement", -1))
        s.dice = int(d.get("dice", 0))
        s.dev_played_this_turn = bool(d.get("dev_played_this_turn", False))
        s.free_roads = int(d.get("free_roads", 0))
        s.discard_queue = [int(x) for x in d.get("discard_queue", [])]
        pt = d.get("pending_trade")
        s.pending_trade = TradeOffer.from_dict(pt) if pt else None
        s.trade_responder = int(d.get("trade_responder", -1))
        s.trades_this_turn = int(d.get("trades_this_turn", 0))
        s.longest_road_owner = int(d.get("longest_road_owner", -1))
        s.longest_road_len = int(d.get("longest_road_len", 0))
        s.largest_army_owner = int(d.get("largest_army_owner", -1))
        s.winner = int(d.get("winner", -1))
        return s


def new_game(num_players: int = 4, rng=None, hexes=None, ports=None, colors=None) -> GameState:
    """Fresh game in the setup phase.  ``rng`` is a ``random.Random`` (random board if given)."""
    s = GameState()
    if hexes is not None:
        s.hexes = list(hexes)
    elif rng is not None:
        s.hexes = B.random_hexes(rng)
    s.robber = next(i for i, (r, _) in enumerate(s.hexes) if r == B.DESERT)
    if ports is not None:
        s.ports = dict(ports)
    colors = colors or PLAYER_COLORS[:num_players]
    s.players = [Player(color=c, name=c) for c in colors[:num_players]]
    s.phase = PHASE_SETUP_SETTLEMENT
    s.current = 0
    return s
