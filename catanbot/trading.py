"""Trading logic: bank / port versus player trades, offers and responses.

Rules of thumb encoded (from the user's brief):

* Use the bank / a port when we can pay the ratio and no cheaper deal is
  realistically available - a 2:1 port is almost always taken, a 3:1 port is
  taken unless a 1:1 player trade is likely, a 4:1 bank trade is a last
  resort (try players first).
* Trade with players when the bank has run out of the resource, when we
  cannot make the bank ratio, or when a 1:1 offer is cheaper.
* Never feed the leader: don't give a card that completes a leader's build,
  and reject offers from a player about to win.
* Evaluate accept / reject with the same value function as the search when
  one is provided; otherwise use "does it complete a build for me and not
  for them".
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from . import actions as A
from . import board as B
from .actions import Action
from .counting import HandBelief, expected_opponent_hands
from .placement import RESOURCE_DEMAND, buildable_settlements, player_production, resource_scarcity
from .robber import estimated_vp
from .state import GameState, TradeOffer
from .opponent_model import OpponentModel, trade_stage_factor, game_stage


def missing_for(hand: Sequence[int], cost: Sequence[int]) -> List[int]:
    return [max(0, cost[r] - hand[r]) for r in range(5)]


def surplus_over(hand: Sequence[int], cost: Sequence[int]) -> List[int]:
    return [max(0, hand[r] - cost[r]) for r in range(5)]


def _can_afford(hand: Sequence[int], cost: Sequence[int]) -> bool:
    return all(hand[r] >= cost[r] for r in range(5))


def leader_index(state: GameState, exclude: int) -> int:
    best, best_vp = -1, -1.0
    for i in range(state.num_players):
        if i == exclude:
            continue
        vp = estimated_vp(state, i)
        if vp > best_vp:
            best, best_vp = i, vp
    return best


def available_builds(state: GameState, player: int) -> List[Tuple[Sequence[int], str]]:
    """Builds the player could physically place (ignoring resources), most valuable first."""
    p = state.players[player]
    out: List[Tuple[Sequence[int], str]] = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        out.append((B.COST_CITY, "city"))
    if len(p.settlements) < B.MAX_SETTLEMENTS and buildable_settlements(state, player):
        out.append((B.COST_SETTLEMENT, "settlement"))
    if sum(state.dev_deck) > 0:
        out.append((B.COST_DEV, "dev card"))
    if len(p.roads) < B.MAX_ROADS:
        out.append((B.COST_ROAD, "road"))
    return out


def completes_build(hand_before: Sequence[int], hand_after: Sequence[int],
                    builds: Optional[List[Tuple[Sequence[int], str]]] = None) -> Optional[str]:
    """Name of the most valuable build affordable after but not before."""
    if builds is None:
        builds = [(B.COST_CITY, "city"), (B.COST_SETTLEMENT, "settlement"),
                  (B.COST_DEV, "dev card"), (B.COST_ROAD, "road")]
    for cost, name in builds:
        if _can_afford(hand_after, cost) and not _can_afford(hand_before, cost):
            return name
    return None


def offer_is_feeding_leader(state: GameState, giver: int, receiver: int, give: Sequence[int]) -> Tuple[bool, str]:
    """True when giving ``give`` to ``receiver`` likely completes a build for a dangerous player."""
    rvp = estimated_vp(state, receiver)
    my_vp = estimated_vp(state, giver)
    hands = expected_opponent_hands(state, me=giver)
    before = hands[receiver]
    after = [before[r] + give[r] for r in range(5)]
    # Expected hands are floats; use "likely affords" = expected >= cost - 0.5
    def likely(hand, cost):
        return all(hand[r] + 0.5 >= cost[r] for r in range(5))
    completes = None
    for cost, name in ((B.COST_CITY, "city"), (B.COST_SETTLEMENT, "settlement")):
        if likely(after, cost) and not likely(before, cost):
            completes = name
            break
    if rvp >= B.VP_TO_WIN - 2 and completes:
        return True, f"would give {state.players[receiver].name or state.players[receiver].color} ({rvp:.0f} VP) a {completes}"
    if rvp >= B.VP_TO_WIN - 1:
        return True, f"{state.players[receiver].name or state.players[receiver].color} is at {rvp:.0f} VP"
    if rvp >= my_vp + 2 and completes:
        return True, f"would hand the leader a {completes}"
    return False, ""


def give_preference(state: GameState, player: int, needed: Sequence[int]) -> List[int]:
    """Resources ordered from most to least disposable for ``player``."""
    p = state.players[player]
    prod = player_production(state, player, ignore_robber=True)
    scarcity = resource_scarcity(state)
    surplus = [p.resources[r] - needed[r] for r in range(5)]
    # Cards the *next* builds will want are less disposable.
    future = [0.0] * 5
    for cost, name in available_builds(state, player):
        w = {"city": 1.0, "settlement": 0.8, "dev card": 0.5, "road": 0.3}[name]
        for r in range(5):
            future[r] += w * cost[r]
    order = sorted(range(5), key=lambda r: -(surplus[r] * 3.0 + prod[r] * 36.0 * 0.25
                                             - RESOURCE_DEMAND[r] * scarcity[r] - 0.8 * future[r]))
    return order


def partner_likelihood(state: GameState, player: int, partner: int, res: int,
                       belief: Optional[HandBelief] = None) -> float:
    """P(partner holds at least one ``res``)."""
    if belief is not None:
        return belief.probability_has(partner, res)
    p = state.players[partner]
    if p.hand_known:
        return 1.0 if p.resources[res] > 0 else 0.0
    hands = expected_opponent_hands(state, me=player)
    exp = hands[partner][res]
    n = p.hand_size
    if n <= 0:
        return 0.0
    q = min(1.0, exp / n)
    return 1.0 - (1.0 - q) ** n


def plan_trades(state: GameState, player: int, target_cost: Sequence[int],
                belief: Optional[HandBelief] = None, max_steps: int = 4,
                model: Optional[OpponentModel] = None) -> List[dict]:
    """Ordered plan to turn the hand into ``target_cost``.

    Returns steps ``{"action": Action, "kind": "bank"|"player", "reason": str,
    "fallback": Action | None}``; ``fallback`` is the bank action to use if
    a proposed player trade gets rejected.
    """
    p = state.players[player]
    hand = list(p.resources)
    needed = list(target_cost)
    missing = missing_for(hand, needed)
    if sum(missing) == 0:
        return []
    steps: List[dict] = []
    proposals_left = max(0, 4 - state.trades_this_turn)
    leader = leader_index(state, player)
    for _ in range(max_steps):
        missing = missing_for(hand, needed)
        if sum(missing) == 0:
            break
        # Most valuable missing resource first (scarce ones are harder to get later).
        scarcity = resource_scarcity(state)
        get = max((r for r in range(5) if missing[r] > 0), key=lambda r: RESOURCE_DEMAND[r] * scarcity[r])
        order = give_preference(state, player, needed)
        # --- bank / port option ------------------------------------------------
        bank_action = None
        bank_ratio = 99
        bank_give = -1
        if state.bank[get] > 0:
            for give in order:
                if give == get:
                    continue
                ratio = state.port_ratio(player, give)
                if hand[give] - needed[give] >= ratio and ratio < bank_ratio:
                    bank_ratio, bank_give = ratio, give
            if bank_give >= 0:
                bank_action = (A.BANK_TRADE, bank_give, get)
        # --- player option (1-for-1, best partner) -------------------------------
        player_action = None
        player_reason = ""
        best_prob = 0.0
        if proposals_left > 0:
            for give in order:
                if give == get or hand[give] - needed[give] <= 0:
                    continue
                for j in range(state.num_players):
                    if j == player:
                        continue
                    prob = partner_likelihood(state, player, j, get, belief)
                    if prob < 0.35:
                        continue
                    feeding, why = offer_is_feeding_leader(state, player, j, [1 if r == give else 0 for r in range(5)])
                    if feeding:
                        continue
                    if model is not None:
                        # Exploitative estimate: their profile + implied valuations + game stage.
                        pa = model.predict_accept(state, j,
                                                  [1 if r == give else 0 for r in range(5)],
                                                  [1 if r == get else 0 for r in range(5)],
                                                  proposer=player, belief=belief)
                        score = pa
                        prob = pa
                    else:
                        # Would the partner plausibly want ``give``?  Prefer partners who produce little of it.
                        want = 1.0 / (1.0 + 8.0 * player_production(state, j, ignore_robber=True)[give])
                        score = prob * (0.5 + want)
                    if score > best_prob:
                        best_prob = score
                        give_c = tuple(1 if r == give else 0 for r in range(5))
                        get_c = tuple(1 if r == get else 0 for r in range(5))
                        player_action = (A.PROPOSE_TRADE, give_c, get_c)
                        player_reason = (f"1:1 with {state.players[j].name or state.players[j].color} "
                                         + (f"({prob:.0%} likely to accept)" if model is not None
                                            else f"({prob:.0%} likely to hold {B.RESOURCE_NAMES[get]})"))
        # --- decide --------------------------------------------------------------
        if bank_action is None and player_action is None:
            break
        stage_f = trade_stage_factor(state)
        if bank_action is not None and (bank_ratio == 2 or (bank_ratio == 3 and best_prob * stage_f < 0.6)
                                        or (bank_ratio == 4 and best_prob * stage_f < 0.25)):
            reason = f"{bank_ratio}:1 {'port' if bank_ratio < 4 else 'bank'} {B.RESOURCE_NAMES[bank_give]} -> {B.RESOURCE_NAMES[get]} (cheapest available)"
            steps.append({"action": bank_action, "kind": "bank", "reason": reason, "fallback": None})
            hand[bank_give] -= bank_ratio
            hand[get] += 1
            continue
        if player_action is not None:
            why = player_reason
            if bank_action is None:
                why += "; bank cannot help" + (" (bank is out of that resource)" if state.bank[get] <= 0 else " (cannot pay the ratio)")
            else:
                why += f"; cheaper than {bank_ratio}:1"
            steps.append({"action": player_action, "kind": "player", "reason": why, "fallback": bank_action})
            proposals_left -= 1
            hand[[r for r in range(5) if player_action[1][r]][0]] -= 1
            hand[get] += 1
            continue
        # only the bank remains (4:1 or 3:1 with no partner)
        reason = f"{bank_ratio}:1 {B.RESOURCE_NAMES[bank_give]} -> {B.RESOURCE_NAMES[get]} (no plausible player trade)"
        steps.append({"action": bank_action, "kind": "bank", "reason": reason, "fallback": None})
        hand[bank_give] -= bank_ratio
        hand[get] += 1
    return steps


def candidate_offers(state: GameState, player: int, needed: Sequence[int],
                     belief: Optional[HandBelief] = None, max_offers: int = 6,
                     model: Optional[OpponentModel] = None) -> List[Action]:
    """Bounded list of promising PROPOSE_TRADE actions (1:1 and 2:1) for the search."""
    p = state.players[player]
    hand = p.resources
    missing = missing_for(hand, needed)
    if sum(missing) == 0:
        return []
    order = give_preference(state, player, needed)
    offers: List[Tuple[float, Action]] = []
    for get in range(5):
        if missing[get] <= 0:
            continue
        for give in order:
            if give == get or hand[give] - needed[give] <= 0:
                continue
            prob = max((partner_likelihood(state, player, j, get, belief)
                        for j in range(state.num_players) if j != player), default=0.0)
            if prob < 0.2:
                continue
            give1 = tuple(1 if r == give else 0 for r in range(5))
            get1 = tuple(1 if r == get else 0 for r in range(5))
            offers.append((prob, (A.PROPOSE_TRADE, give1, get1)))
            if hand[give] - needed[give] >= 2:
                give2 = tuple(2 if r == give else 0 for r in range(5))
                offers.append((prob * 0.9, (A.PROPOSE_TRADE, give2, get1)))
    offers.sort(key=lambda t: -t[0])
    out = []
    seen = set()
    for _, a in offers:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    if model is not None and out:
        ranked = model.rank_offers(state, player, out, needed, belief)
        out = [d["action"] for d in ranked if d["p_accept"] > 0.05] or out
    return out[:max_offers]


def should_accept(state: GameState, responder: int, offer: TradeOffer, evaluator=None,
                  margin: float = 0.002, model: Optional[OpponentModel] = None,
                  politics=None) -> Tuple[bool, str]:
    """Accept/reject an incoming offer (``offer.give`` is what we would receive).

    The required gain rises with the game stage: early trades grow both
    economies, late trades mostly help whoever is closer to 10 VP.
    """
    p = state.players[responder]
    proposer = offer.proposer
    if not _can_afford(p.resources, offer.get):
        return False, "cannot pay"
    stage = game_stage(state)
    margin = margin + 0.03 * stage * stage
    slack = politics.favor_slack(state, responder, proposer) if politics is not None else 0.0
    margin -= 0.01 * slack          # friends get a little slack, the leader pays a premium
    pvp = estimated_vp(state, proposer)
    if stage >= 0.6 and pvp >= estimated_vp(state, responder) and pvp >= 7:
        return False, f"late game: no trades with a player ahead of us ({pvp:.0f} VP)"
    feeding, why = offer_is_feeding_leader(state, responder, proposer, offer.get)
    my_before = list(p.resources)
    my_after = [my_before[r] + offer.give[r] - offer.get[r] for r in range(5)]
    mine = completes_build(my_before, my_after, available_builds(state, responder))
    if feeding and not (mine in ("city", "settlement")):
        return False, "don't feed the leader: " + why
    # Card-count: never accept a deal that makes us give more cards than we get
    # unless it completes a build for us.
    net = sum(offer.give) - sum(offer.get)
    if evaluator is not None:
        s2 = state.copy()
        q = s2.players[responder]
        pr = s2.players[proposer]
        for r in range(5):
            q.resources[r] += offer.give[r] - offer.get[r]
            pr.resources[r] += offer.get[r] - offer.give[r]
        vals = evaluator.evaluate([state, s2, state, s2], [responder, responder, proposer, proposer])
        gain_me = float(vals[1] - vals[0])
        gain_them = float(vals[3] - vals[2])
        # Late in the game we also require the deal not to help them more than us.
        share = 0.6 if stage < 0.5 else 1.0
        if gain_me > margin and (gain_me >= share * gain_them or mine is not None or gain_them <= 0.02):
            return True, f"value +{gain_me:.3f} for us (+{gain_them:.3f} for them)"
        if gain_me > margin:
            return False, f"helps them more than us (+{gain_them:.3f} vs +{gain_me:.3f})"
        return False, f"no gain for us ({gain_me:+.3f})"
    if mine is not None:
        return True, f"completes a {mine} for us"
    if net > 0 and estimated_vp(state, proposer) < estimated_vp(state, responder) + 2:
        return True, "we receive more cards than we give"
    # Same-count trade: accept if it gives us something we lack and produce little of.
    prod = player_production(state, responder, ignore_robber=True)
    gain_scarce = sum(offer.give[r] * (1.0 / (1.0 + 8 * prod[r])) for r in range(5))
    lose_scarce = sum(offer.get[r] * (1.0 / (1.0 + 8 * prod[r])) for r in range(5))
    if net == 0 and gain_scarce > lose_scarce + 0.15 + 0.3 * stage - 0.5 * slack:
        return True, "swaps a surplus card for one we need" + (" (floating a friend a little)" if slack > 0.05 else "")
    return False, "no clear benefit" + (" (late game: trades must pay off immediately)" if stage >= 0.6 else "")


def trade_advice(state: GameState, player: int, belief: Optional[HandBelief] = None,
                 model: Optional[OpponentModel] = None) -> List[str]:
    """Human readable trade plan for the most sensible target build."""
    p = state.players[player]
    lines: List[str] = []
    stage = game_stage(state)
    lines.append(f"Game stage {stage:.0%}: " + ("trades are cheap growth now." if stage < 0.35 else
                 "be selective, only trades that pay off soon." if stage < 0.7 else
                 "trades mostly help whoever is closer to 10 VP; avoid unless it completes a build."))
    targets = []
    if p.settlements and len(p.cities) < B.MAX_CITIES:
        targets.append((B.COST_CITY, "city"))
    if len(p.settlements) < B.MAX_SETTLEMENTS and buildable_settlements(state, player):
        targets.append((B.COST_SETTLEMENT, "settlement"))
    if sum(state.dev_deck) > 0:
        targets.append((B.COST_DEV, "dev card"))
    targets.sort(key=lambda t: sum(missing_for(p.resources, t[0])))
    for cost, name in targets[:2]:
        missing = missing_for(p.resources, cost)
        if sum(missing) == 0:
            lines.append(f"You can already afford a {name}.")
            continue
        steps = plan_trades(state, player, cost, belief, model=model)
        if not steps:
            lines.append(f"No sensible trade reaches a {name} this turn (missing "
                         + ", ".join(f"{missing[r]} {B.RESOURCE_NAMES[r]}" for r in range(5) if missing[r]) + ").")
            continue
        lines.append(f"To afford a {name}:")
        for s in steps:
            lines.append(f"  - {A.describe(s['action'], state)}  [{s['reason']}]")
            if s["fallback"] is not None:
                lines.append(f"      if rejected: {A.describe(s['fallback'], state)}")
    short = [B.RESOURCE_NAMES[r] for r in range(5) if state.bank[r] <= 2]
    if short:
        lines.append("Bank is nearly out of: " + ", ".join(short) + " (player trades are the only source).")
    if model is not None:
        arbs = model.arbitrage_opportunities(state, player, belief=belief)
        if arbs:
            lines.append("Exploitable trades (by opponents' revealed valuations):")
            for d in arbs[:3]:
                steps_txt = " then ".join(A.describe(a, state) for a in d["steps"])
                lines.append(f"  - {steps_txt}  [{d['reason']}]")
        lines.extend("Opponent " + l for l in model.summary(state, me=player)[:-1])
    return lines
