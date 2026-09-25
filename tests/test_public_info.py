"""The counted information mode: exact card counting from public events, and a bot that never
reads a hidden card (catanbot.counting.CardCounter, catanbot.bench.public_info, CatanbotPlayer(info=...)).

Runs on catanatron 3.2.1 (PyPI wheel) and on the 3.3 engine (GitHub checkout).
"""
from __future__ import annotations

import importlib.util
import os
import random

import pytest

pytest.importorskip("catanatron")

from catanatron.models.enums import ActionPrompt, ActionType  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402

from catanbot import board as B  # noqa: E402
from catanbot import engine as E  # noqa: E402
from catanbot.bench import catanatron_adapter as AD  # noqa: E402
from catanbot.bench import public_info as PI  # noqa: E402
from catanbot.counting import CardCounter, HandBelief  # noqa: E402
from catanbot.selfplay import make_bot  # noqa: E402

SMALL = "search:depth=1,beam=2,expand=4,actions=3,evaluator=heuristic"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def true_hands(st):
    ps = st.player_state
    return [[int(ps[f"P{i}_{r}_IN_HAND"]) for r in AD.CB_TO_RESOURCE] for i in range(len(st.colors))]


def true_devs(st):
    ps = st.player_state
    return [[int(ps[f"P{i}_{d}_IN_HAND"]) for d in AD.CB_TO_DEV] for i in range(len(st.colors))]


def deck_counts(st):
    out = [0] * 5
    for card in st.development_listdeck:
        out[AD.DEV_TO_CB[card]] += 1
    return out


def weighted_game(seed: int):
    """A finished game of four WeightedRandomPlayers (fast; steals, discards and dev cards galore)."""
    g = AD.make_game([WeightedRandomPlayer(c) for c in AD.COLORS], seed=seed)
    g.play()
    return g


def hidden_to(me: int, entry, st_colors, discards_public=False):
    """``(kind, parties)`` when ``entry`` hides resource information from seat ``me``, else None."""
    a = AD.log_action(entry)
    idx = {c: i for i, c in enumerate(st_colors)}
    actor = idx[a.color]
    if a.action_type == ActionType.MOVE_ROBBER and a.value[1] is not None:
        victim = idx[a.value[1]]
        if me not in (actor, victim):
            return "steal", (victim, actor)
    if a.action_type in AD.DISCARD_TYPES and actor != me and not discards_public:
        return "discard", (actor,)
    return None


# ---------------------------------------------------------------------------
# CardCounter / HandBelief units
# ---------------------------------------------------------------------------
def test_counter_is_exact_while_every_event_is_public():
    c = CardCounter([[0] * 5] * 4)
    c.observe_gain(1, B.WOOD, 3)
    c.observe_delta(2, [0, 2, 1, 1, 0])
    c.observe_spend(1, [1, 0, 0, 0, 0])
    c.observe_bank_trade(2, B.BRICK, 2, B.ORE)
    c.observe_player_trade(1, [1, 0, 0, 0, 0], 2, [0, 0, 0, 0, 1])
    c.observe_steal(2, 3, B.SHEEP)                 # seen by the thief / victim
    c.observe_monopoly(3, B.WOOD, taken_by={0: 0, 1: 1, 2: 1})
    truth = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [2, 0, 1, 0, 0]]
    assert c.num_hypotheses == 1 and c.weight_of(truth) == 1.0
    assert c.expected == [[float(x) for x in h] for h in truth]
    assert c.size == [0, 1, 1, 3] and all(c.is_exact(i) for i in range(4))
    assert c.probability_has(3, B.WOOD, 2) == 1.0 and c.probability_has(3, B.WOOD, 3) == 0.0
    assert c.observe_bank([19 - 2, 19, 19 - 1, 19 - 1, 19 - 1])


def test_hidden_steal_is_a_mixture_resolved_by_later_public_events():
    c = CardCounter([[0] * 5, [2, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0] * 5])
    c.observe_steal(1, 2)                          # we (seat 0) did not see the card
    assert c.num_hypotheses == 2 and c.size == [0, 2, 2, 0]
    assert c.marginal(2) == pytest.approx({(1, 0, 1, 0, 0): 2 / 3, (0, 1, 1, 0, 0): 1 / 3})
    assert c.expected[1] == pytest.approx([4 / 3, 2 / 3, 0, 0, 0])
    assert c.observe_bank([17, 18, 18, 19, 19])     # both hypotheses fit the bank: still two
    assert c.num_hypotheses == 2 and not c.is_exact(1) and not c.is_exact(2)
    c.observe_spend(2, [0, 1, 0, 0, 0])            # the thief pays a brick: it stole the brick
    assert c.num_hypotheses == 1 and c.is_exact(1) and c.is_exact(2)
    assert c.weight_of([[0] * 5, [2, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0] * 5]) == 1.0


def test_seen_steal_is_weighted_by_its_likelihood():
    c = CardCounter([[0] * 5, [1, 1, 0, 0, 0], [0] * 5, [0] * 5])
    c.observe_steal(1, 2)                          # hidden: wood or brick, 1/2 each
    c.observe_steal(2, 0, B.WOOD)                  # we steal from the thief and see a wood
    # thief held {stolen, nothing else}: a wood was stolen from it, so it had stolen the wood
    assert c.num_hypotheses == 1 and c.expected[1] == [0.0, 1.0, 0.0, 0.0, 0.0]
    d = CardCounter([[0] * 5, [3, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0] * 5])
    d.observe_steal(1, 2)
    d.observe_steal(1, 0, B.WOOD)                  # likelihood 2/3 vs 3/3 (Bayes, not just feasibility)
    assert d.marginal(2) == pytest.approx({(1, 0, 0, 0, 0): (3 / 4 * 2 / 3) / (3 / 4 * 2 / 3 + 1 / 4),
                                           (0, 1, 0, 0, 0): (1 / 4) / (3 / 4 * 2 / 3 + 1 / 4)})


def test_hidden_discards_single_one_pinned_by_the_bank_split_of_two_stays_open():
    c = CardCounter([[0] * 5, [4, 2, 2, 0, 0], [0] * 5, [0] * 5])
    c.observe_discard(1, n=4)                      # count public, cards hidden
    assert c.num_hypotheses > 5 and sum(c.expected[1]) == pytest.approx(4)
    c.observe_bank([15, 19, 19, 19, 19])           # 2 brick + 2 sheep went back to the bank
    assert c.num_hypotheses == 1 and c.expected[1] == [4.0, 0.0, 0.0, 0.0, 0.0]
    two = CardCounter([[0] * 5, [2, 2, 0, 0, 0], [2, 2, 0, 0, 0], [0] * 5])
    two.observe_discard(1, n=2)
    two.observe_discard(2, n=2)
    two.observe_bank([17, 17, 19, 19, 19])         # the batch's total is known, not its split
    assert two.num_hypotheses == 3
    for joint in two.hyps:
        assert [joint[1][r] + joint[2][r] for r in range(5)] == [2, 2, 0, 0, 0]
        assert sum(joint[1]) == 2 and sum(joint[2]) == 2
    # hypergeometric discards: each keeps one of each w.p. 4/6, two of a kind w.p. 1/6 (then the bank)
    assert two.weight_of([[0] * 5, [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0] * 5]) == pytest.approx(16 / 18)


def test_simultaneous_hidden_discards_are_resolved_jointly_with_the_bank():
    """A 7's hidden discards are applied together once the bank shows their total: only
    combinations matching it are enumerated (no per-card blow-up)."""
    hands = [[0] * 5, [3, 3, 2, 0, 0], [2, 2, 2, 2, 0], [1, 1, 1, 1, 4]]
    c = CardCounter(hands)
    bank_before = [19 - sum(h[r] for h in hands) for r in range(5)]
    c.observe_discards({1: 4, 2: 4, 3: 4}, [bank_before[0] + 3, bank_before[1] + 3, bank_before[2] + 2,
                                            bank_before[3] + 2, bank_before[4] + 2])
    assert c.size == [0, 4, 4, 4] and c.stats["resets"] == 0
    for joint in c.hyps:
        assert [sum(p[r] for p in joint) for r in range(5)] == [3, 3, 3, 1, 2]
        assert all(joint[i][r] <= hands[i][r] for i in range(4) for r in range(5))
    assert 1 < c.num_hypotheses < 400
    lone = CardCounter(hands)
    lone.observe_discards({3: 4}, [bank_before[0], bank_before[1], bank_before[2], bank_before[3], bank_before[4] + 4])
    assert lone.num_hypotheses == 1 and lone.expected[3] == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_monopoly_conditions_every_victim_on_its_public_take():
    c = CardCounter([[0] * 5, [1, 1, 0, 0, 0], [0, 0, 0, 2, 0], [0] * 5])
    c.observe_steal(1, 2)                          # 2 holds wood or brick + 2 wheat
    c.observe_monopoly(3, B.WOOD, taken_by={0: 0, 1: 0, 2: 1})
    assert c.num_hypotheses == 1 and c.size == [0, 1, 2, 1]
    assert c.expected[1] == [0.0, 1.0, 0.0, 0.0, 0.0] and c.expected[3] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_contradiction_restarts_from_a_consistent_assignment():
    c = CardCounter([[0] * 5, [1, 0, 0, 0, 0], [0] * 5, [0] * 5])
    c.observe_spend(1, [0, 0, 0, 0, 1])            # impossible: nobody could hold an ore
    assert c.stats["resets"] == 1 and c.stats["contradictions"] >= 1
    assert c.num_hypotheses == 1 and c.size[1] == 0 and all(min(h) >= 0 for h in next(iter(c.hyps)))


def test_hand_belief_exactness_bank_fit_and_hidden_events():
    from catanbot.state import GameState, Player
    s = GameState()
    s.players = [Player(color=c) for c in ("red", "blue", "orange", "white")]
    s.players[1].resources = [2, 1, 0, 0, 0]
    s.players[2].resources = [0, 0, 3, 0, 0]
    s.bank = [17, 18, 16, 19, 19]
    b = HandBelief(s)
    assert all(b.is_exact(i) for i in range(4))
    b.observe_spend(1, [1, 1, 0, 0, 0])            # exact bookkeeping, no renormalisation
    assert b.expected[1] == [1.0, 0, 0, 0, 0] and b.is_exact(1)
    b.observe_steal(2, 1)                          # single-resource victim: the card is known
    assert b.expected[1] == [1.0, 0, 1.0, 0, 0] and b.is_exact(1) and b.is_exact(2)
    b.observe_steal(1, 3)                          # a real hidden steal
    assert not b.is_exact(1) and not b.is_exact(3) and b.expected[3] == pytest.approx([0.5, 0, 0.5, 0, 0])
    assert b.observe_bank([18, 19, 16, 19, 19])     # fit to the (true) bank: consistent totals
    assert not b.observe_bank([18, 19, 17, 19, 19])  # an inconsistent bank is refused
    assert [b.expected[1][r] + b.expected[3][r] for r in range(5)] == pytest.approx([1, 0, 1, 0, 0])
    b.observe_monopoly(0, B.SHEEP, taken_by={1: 0, 2: 2, 3: 1})   # 3 had the sheep, so 1 the wood
    assert b.size == [3, 1, 0, 0] and b.expected[3] == [0.0] * 5
    assert b.expected[1] == pytest.approx([1, 0, 0, 0, 0]) and b.expected[0] == [0, 0, 3.0, 0, 0]
    b.observe_discard(0, n=1)                      # hidden discard of a 1-type hand: exact
    assert b.expected[0] == [0.0, 0.0, 2.0, 0.0, 0.0]
    assert b.probability_has(0, B.SHEEP, 2) == 1.0 and b.probability_has(0, B.SHEEP, 3) == 0.0


# ---------------------------------------------------------------------------
# (1) exact at every step when nothing is hidden; exact for untouched opponents otherwise
# ---------------------------------------------------------------------------
GAME_SEEDS = (21, 22, 23, 24, 25)


def test_belief_equals_truth_at_every_step_when_hidden_events_are_revealed():
    """``reveal_hidden`` treats steals, discards and dev draws as public: the counter must then be
    the true hands (one hypothesis) and the dev bookkeeping the true cards, after every entry of 5 games."""
    steps = 0
    for seed in GAME_SEEDS:
        g = weighted_game(seed)
        me = AD.COLORS[seed % 4]
        tr = PI.PublicInfoTracker(me, reveal_hidden=True)
        for entry, shadow in tr.replay_log(g.state):
            hands = true_hands(shadow)
            assert tr.counter.num_hypotheses == 1 and tr.counter.weight_of(hands) == 1.0, (seed, entry)
            assert tr.counter.expected == [[float(x) for x in h] for h in hands]
            devs = true_devs(shadow)
            for j in range(4):
                if j != tr.me:
                    assert tr.known_dev[j] == devs[j]
            assert tr.dev_pool() == deck_counts(shadow)     # everything else is known: the pool is the deck
            steps += 1
        assert tr.counter.stats["resets"] == 0
    assert steps > 1000


def test_belief_is_exact_for_every_opponent_no_hidden_event_touched():
    """Default model: hands untouched by a hidden steal / discard are exact, the truth is always in the
    support, every hypothesis has the public sizes and (outside a 7's discards) matches the bank."""
    checked = uncertain = 0
    for seed in GAME_SEEDS:
        g = weighted_game(seed)
        tr = PI.PublicInfoTracker(AD.COLORS[seed % 4])
        tainted = set()
        for entry, shadow in tr.replay_log(g.state):
            h = hidden_to(tr.me, entry, shadow.colors)
            if h is not None:
                tainted.update(h[1])
            hands = true_hands(shadow)
            c = tr.counter
            assert tr.support_weight(hands) > 0, (seed, entry)
            for j in range(4):
                if j not in tainted:
                    assert tr.is_exact(j) and tr.expected_hand(j) == [float(x) for x in hands[j]]
                    checked += 1
            if AD.log_action(entry).action_type not in AD.DISCARD_TYPES:
                bank = [int(x) for x in shadow.resource_freqdeck]
                for joint in c.hyps:
                    assert [sum(p[r] for p in joint) + bank[r] for r in range(5)] == [19] * 5
            pend = tr.pending_discards   # a 7's hidden discards so far (resolved with the bank at its end)
            assert all(sum(p) - pend.get(i, 0) == sum(hands[i]) for joint in c.hyps for i, p in enumerate(joint))
            assert tr.pool_is_consistent()
            uncertain += c.num_hypotheses > 1
        assert tr.stats["hidden_steals"] > 0
        assert c.stats["resets"] == 0 and c.stats["pruned_mass"] == 0
    assert checked > 2000 and uncertain > 0


def test_counted_player_belief_is_exact_at_every_decision_without_hidden_events(monkeypatch):
    """The CatanbotPlayer path itself (tracker started by the player, fed through ``decide``): with
    the hidden events revealed, the belief equals every opponent's true hand at each decision."""
    real_init = PI.PublicInfoTracker.__init__

    def revealing(self, *a, **kw):
        kw["reveal_hidden"] = True
        real_init(self, *a, **kw)

    monkeypatch.setattr(PI.PublicInfoTracker, "__init__", revealing)
    decisions = 0
    for seed in GAME_SEEDS:
        seat = seed % 4
        me = AD.CatanbotPlayer(AD.COLORS[seat], spec="heuristic", info="counted", strict=True, seed=seed)
        inner = me.decide

        def checked(game, playable, me=me, inner=inner):
            nonlocal decisions
            out = inner(game, playable)
            c = me.tracker.counter
            assert c.num_hypotheses == 1 and me.tracker.support_weight(true_hands(game.state)) == 1.0
            decisions += 1
            return out

        me.decide = checked
        AD.play_game([WeightedRandomPlayer(c) if i != seat else me for i, c in enumerate(AD.COLORS)], seed=seed)
        assert me.stats["errors"] == 0 and me.stats["info_errors"] == 0 and me.stats["fallback"] == 0
    assert decisions > 200


# ---------------------------------------------------------------------------
# (2) a hidden steal leaves exactly one card uncertain
# ---------------------------------------------------------------------------
def test_hidden_steal_leaves_only_that_card_uncertain():
    seen = 0
    for seed in GAME_SEEDS + (26, 27, 28):
        g = weighted_game(seed)
        tr = PI.PublicInfoTracker(AD.COLORS[seed % 4])
        exact_before = True
        victim_hand = None
        for entry, shadow in tr.replay_log(g.state):
            h = hidden_to(tr.me, entry, shadow.colors)
            c = tr.counter
            if h is not None and h[0] == "steal" and exact_before and victim_hand is not None:
                victim, thief = h[1]
                vh = victim_hand[victim]
                if sum(vh) > 0:
                    hands = true_hands(shadow)
                    types = [r for r in range(5) if vh[r] > 0]
                    assert c.num_hypotheses == len(types)
                    for joint, w in c.hyps.items():
                        for j in range(4):
                            if j not in (victim, thief):
                                assert list(joint[j]) == hands[j]
                        dv = [joint[victim][r] - hands[victim][r] for r in range(5)]
                        dt = [joint[thief][r] - hands[thief][r] for r in range(5)]
                        assert dv == [-x for x in dt] and sum(abs(x) for x in dv) in (0, 2)   # one card apart
                        moved = [r for r in range(5) if vh[r] - joint[victim][r] == 1]
                        assert len(moved) == 1 and w == pytest.approx(vh[moved[0]] / sum(vh))
                        assert sum(joint[victim]) == sum(hands[victim]) and sum(joint[thief]) == sum(hands[thief])
                    bank = [int(x) for x in shadow.resource_freqdeck]
                    for joint in c.hyps:
                        assert [sum(p[r] for p in joint) + bank[r] for r in range(5)] == [19] * 5
                    assert c.weight_of(hands) > 0
                    seen += 1
            exact_before = c.num_hypotheses == 1 and not tr.pending_discards
            victim_hand = true_hands(shadow)
    assert seen >= 5


# ---------------------------------------------------------------------------
# (3) the counted player never reads a hidden quantity
# ---------------------------------------------------------------------------
NON_VP = [d for d in AD.CB_TO_DEV if d != "VICTORY_POINT"]


def perturb_hidden(game, me_seat: int, rng: random.Random):
    """A copy of ``game`` whose change is invisible to the information model: one card swapped
    between two opponents (sizes and the bank unchanged) and, when possible, an opponent's unplayed
    non-VP development card exchanged with a card of another non-VP type from the deck."""
    g2 = game.copy()
    st = g2.state
    ps = st.player_state
    opps = [j for j in range(4) if j != me_seat]
    hands = true_hands(st)
    swaps = [(a, b, r, s) for a in opps for b in opps if a < b for r in range(5) for s in range(5)
             if r != s and hands[a][r] > 0 and hands[b][s] > 0]
    if not swaps:
        return None
    a, b, r, s = rng.choice(swaps)
    R, S = AD.CB_TO_RESOURCE[r], AD.CB_TO_RESOURCE[s]
    ps[f"P{a}_{R}_IN_HAND"] -= 1
    ps[f"P{a}_{S}_IN_HAND"] += 1
    ps[f"P{b}_{S}_IN_HAND"] -= 1
    ps[f"P{b}_{R}_IN_HAND"] += 1
    deck = st.development_listdeck
    for j in opps:
        for t in NON_VP:
            if ps[f"P{j}_{t}_IN_HAND"] <= 0:
                continue
            for k, u in enumerate(deck):
                if u in NON_VP and u != t:
                    ps[f"P{j}_{t}_IN_HAND"] -= 1
                    ps[f"P{j}_{u}_IN_HAND"] += 1
                    deck[k] = t
                    return g2
    return g2


def _assert_public(state, me: int):
    for j, p in enumerate(state.players):
        if j == me:
            continue
        assert not p.hand_known and p.resources == [0] * 5
        assert not p.dev_known and p.dev_cards == [0] * 5 and p.dev_cards_new == [0] * 5


def test_counted_player_never_reads_hidden_information():
    seat = 1
    me = AD.CatanbotPlayer(AD.COLORS[seat], spec=SMALL, info="counted", strict=True, seed=5)
    rng = random.Random(9)
    public_states = []
    det_checks = [0]

    real_observe = me.bot.observe

    def observe(state, action, player):
        _assert_public(state, seat)
        public_states.append(state)
        return real_observe(state, action, player)

    me.bot.observe = observe
    real_bot_decide = me.bot.decide

    def bot_decide(state, legal, r):
        # a determinization: every opponent hand is a hypothesis of the public counter, every
        # opponent dev card and the deck come from the public pool
        tr = me.tracker
        assert tr.support_weight([p.resources for p in state.players]) > 0
        pool = [sum(p.dev_cards[t] + p.dev_cards_new[t] for j, p in enumerate(state.players) if j != seat)
                + state.dev_deck[t] for t in range(5)]
        assert pool == tr.dev_pool()
        assert set(legal) <= set(E.legal_actions(state))   # our own actions are legal in every sample
        det_checks[0] += 1
        return real_bot_decide(state, legal, r)

    me.bot.decide = bot_decide
    real_views = me._views

    def views(game):
        pub, view = real_views(game)
        _assert_public(pub, seat)
        assert E.legal_actions(view) == E.legal_actions(me.tracker.determinize(pub, random.Random(1))) or \
            view.phase not in ("main", "roll", "robber")
        return pub, view

    me._views = views
    full_bot = make_bot(SMALL)
    positions = []
    visible = 0
    real_decide = me.decide

    def decide(game, playable):
        nonlocal visible
        playable = list(playable)
        before = (me.rng.getstate(), me._pending_robber, list(me._pending_discard))
        searched = me.stats["searched"]
        out = real_decide(game, playable)
        if me.stats["searched"] == searched or len(positions) >= 20 or game.state.current_prompt in AD.TRADE_PROMPTS:
            return out
        g2 = perturb_hidden(game, seat, rng)
        if g2 is None:
            return out
        after = (me.rng.getstate(), me._pending_robber, list(me._pending_discard))
        ranking = [(r.action, r.value) for r in me.bot.last_results]
        me.rng.setstate(before[0])
        me._pending_robber, me._pending_discard = before[1], list(before[2])
        out2 = real_decide(g2, list(AD.playable_actions_of(g2)))
        assert AD.playable_key(out2) == AD.playable_key(out) and out2.color == out.color
        assert [(r.action, r.value) for r in me.bot.last_results] == ranking
        # the perturbation is real: the full-information state (and search) sees it
        full1 = AD.to_catanbot_state(game)
        full2 = AD.to_catanbot_state(g2)
        assert full1.to_dict() != full2.to_dict()
        legal = E.legal_actions(full1)
        if len(legal) > 1:
            full_bot.decide(full1, legal, random.Random(0))
            v1 = [(r.action, round(r.value, 9)) for r in full_bot.last_results]
            full_bot.decide(full2, legal, random.Random(0))
            v2 = [(r.action, round(r.value, 9)) for r in full_bot.last_results]
            visible += v1 != v2
        positions.append(game.state.current_prompt)
        me.rng.setstate(after[0])
        me._pending_robber, me._pending_discard = after[1], list(after[2])
        return out

    me.decide = decide
    for seed in (31, 32, 33, 34):
        AD.play_game([WeightedRandomPlayer(c) if i != seat else me for i, c in enumerate(AD.COLORS)], seed=seed)
        assert me.stats["errors"] == 0 and me.stats["fallback"] == 0 and me.stats["info_errors"] == 0
        if len(positions) >= 20:
            break
    assert len(positions) == 20
    assert public_states and det_checks[0] >= 20 * me.info_samples
    assert visible >= 10     # a full-information search's values move with the swapped cards (18-20 of 20 measured)


# ---------------------------------------------------------------------------
# (4) the default ("full") mode is the pre-information-mode behaviour
# ---------------------------------------------------------------------------
def reference_full_decision(me, game, playable, rng):
    """The pre-information-mode decision procedure, spelled out: catanatron's true state, every
    hand known, the bot's first ranked action that has a catanatron equivalent."""
    st = game.state
    cb = AD.to_catanbot_state(game, me.color, me._mapping, me.suppress_trades)
    index = AD.index_playable(playable)
    lookup = {}
    cb_legal = []
    for a in E.legal_actions(cb):
        key = AD.catanbot_action_to_key(a, cb, me._mapping, st.colors)
        ca = index.get(key) if key is not None else None
        if ca is not None:
            lookup[a] = ca
            cb_legal.append(a)
    decision = me.bot.decide(cb, cb_legal, rng)
    for r in me.bot.last_results:
        if r.action in lookup:
            return lookup[r.action], cb
    return lookup[decision], cb


def test_default_full_mode_decisions_are_unchanged():
    seat = 2
    me = AD.CatanbotPlayer(AD.COLORS[seat], spec=SMALL, strict=True, seed=11)
    assert me.info == "full" and me.tracker is None
    assert not any(k.startswith(("info_", "hidden_")) for k in me.stats)
    handed = []
    real_bot_decide = me.bot.decide

    def bot_decide(state, legal, r):
        handed.append(state)
        return real_bot_decide(state, legal, r)

    me.bot.decide = bot_decide
    compared = []
    real_decide = me.decide

    def decide(game, playable):
        playable = list(playable)
        before = me.rng.getstate()
        searched = me.stats["searched"]
        n_handed = len(handed)
        out = real_decide(game, playable)
        if (me.stats["searched"] == searched or len(compared) >= 10
                or game.state.current_prompt == ActionPrompt.DISCARD or game.state.current_prompt in AD.TRADE_PROMPTS):
            return out
        after = me.rng.getstate()
        assert len(handed) == n_handed + 1
        state = handed[-1]
        assert all(p.hand_known and p.dev_known for p in state.players)
        me.rng.setstate(before)
        ref, cb = reference_full_decision(me, game, playable, me.rng)
        assert state.to_dict() == cb.to_dict()            # the bot saw catanatron's true state
        assert AD.playable_key(ref) == AD.playable_key(out)
        me.rng.setstate(after)
        compared.append(AD.playable_key(out))
        return out

    me.decide = decide
    for seed in (41, 42, 43):
        AD.play_game([WeightedRandomPlayer(c) if i != seat else me for i, c in enumerate(AD.COLORS)], seed=seed)
        if len(compared) >= 10:
            break
    assert len(compared) == 10
    assert me.tracker is None and me.stats["errors"] == 0


def test_bad_information_options_are_rejected():
    with pytest.raises(ValueError):
        AD.CatanbotPlayer(AD.COLORS[0], spec="heuristic", info="partial")
    with pytest.raises(ValueError):
        AD.CatanbotPlayer(AD.COLORS[0], spec="heuristic", info="counted", info_samples=0)


# ---------------------------------------------------------------------------
# (5) smoke: counted mode through the benchmark script
# ---------------------------------------------------------------------------
def _bench():
    spec = importlib.util.spec_from_file_location("bench_catanatron_pi", os.path.join(ROOT, "scripts", "bench_catanatron.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_counted_smoke_through_the_bench(tmp_path, capsys):
    import json
    bench = _bench()
    opponent = "value" if AD.API_33 else "vf"
    out = tmp_path / "counted.json"
    rc = bench.main(["--games", "4", "--opponent", opponent, "--spec", SMALL, "--info", "counted",
                     "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "information : counted" in text
    s = json.loads(out.read_text())
    assert s["info"] == {"mode": "counted", "samples": 4, "discards_public": False}
    st = s["stats"]
    assert st["errors"] == 0 and st["fallback"] == 0 and st["observe_errors"] == 0 and st["searched"] > 0
    ist = s["info_stats"]
    assert ist["info_errors"] == 0 and ist["info_resets"] == 0
    assert ist["info_samples"] == 4 * st["searched"]
    for r in s["results"]:
        assert r["info_stats"]["info_errors"] == 0


@pytest.mark.skipif(not AD.DOMESTIC_TRADING, reason=f"catanatron {AD.CATANATRON_VERSION} has no domestic trades")
def test_counted_mode_with_domestic_trades():
    seat = 0
    me = AD.CatanbotPlayer(AD.COLORS[seat], spec=SMALL, info="counted", strict=True, seed=2, suppress_trades=False)
    from catanatron.players.value import ValueFunctionPlayer
    players = [AD.BenchOpponent(ValueFunctionPlayer(c), trade_rule="value") if i != seat else me
               for i, c in enumerate(AD.COLORS)]
    real_decide = me.decide
    checks = [0]

    def decide(game, playable):
        out = real_decide(game, playable)
        assert me.tracker.support_weight(true_hands(game.state)) > 0
        checks[0] += 1
        return out

    me.decide = decide
    for seed in (51, 52):
        AD.play_game(players, seed=seed)
        assert me.stats["errors"] == 0 and me.stats["info_errors"] == 0 and me.stats["info_resets"] == 0
    assert me.stats["trades_confirmed"] + me.stats["offers_accepted_by_us"] >= 1 and checks[0] > 50
