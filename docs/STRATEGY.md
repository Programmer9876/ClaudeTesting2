# How catanbot decides

The bot optimises **win probability**, not victory points.  A value network
trained by self-play estimates each player's chance of winning from any
position; an expectimax search uses it to look ahead through our own turn
(with exact dice / draw / steal / trade-acceptance expectations) and through
the opponents' turns.  The explicit rules below are *priors* and
explanations: they order the moves the search tries first, they drive the
opponents' simulated behaviour, and they produce the advice text.  When
the value net disagrees with a rule, the net wins.

## Trading (`trading.py`, `opponent_model.py`, `politics.py`)

1. Work out the **target build** (closest affordable of city, settlement on a
   real spot, dev card) and the cards missing for it.
2. For every missing card compare:
   * the **bank / port** option: cheapest ratio we can pay from *surplus*
     (cards not needed by the target or the next build);
   * the **player** option: a 1:1 offer to the opponent most likely to hold
     the card and accept it.
3. Decision: 2:1 port -> always bank.  3:1 port -> bank unless a player
   trade is very likely.  4:1 -> try players first.  If the bank is out of
   the resource, or we cannot pay the ratio, only players remain.
4. Never feed the leader: no offer that plausibly completes a build for a
   player two or more VP ahead, none at all to someone at 9 VP.
5. **Acceptance prediction** uses the opponent profile: overall generosity,
   what they accept / refuse per resource, their *implied valuations*
   (updated by every accept, reject, proposal and bank trade they make),
   the stage of the game, and the political slack (below).
6. **Arbitrage**: if opponent A's implied valuation says they give X for Y
   cheaply and we value X more than Y, buy.  If B pays Z for X, run the
   chain Y -> X -> Z as an intermediary.  The search sees these deals: the
   first step of the three best exploitable deals is promoted above the
   ordinary proposals in the candidate list (its explanation names the
   counterpart's revealed valuation), and once the first leg of a chain has
   executed the second leg is tried first at the next level.
7. **Game stage**: early trades grow both economies; as the leader nears
   10 VP every trade mostly helps whoever is closer, so the required gain
   rises and late trades with anyone ahead of us are refused.  Inside the
   search the priors of `PROPOSE_TRADE` candidates are multiplied by the
   trade stage factor and proposals are capped per turn (4 early, 2 late;
   `SearchConfig.trade_cap_early / trade_cap_late`).
8. Incoming offers are judged with the same value function (accept if our
   win probability rises and theirs does not rise more), otherwise by
   "does it complete a build for us and not for them".
9. The simulated opponents trade too: in the lookahead each opponent may
   make one profile-ranked proposal per turn (`plan_trades` with the
   opponent model, `SearchConfig.opponent_proposals`), answered with the
   same acceptance rules (politics included), so the value net is trained
   on positions where deals are offered to us.
10. **Self-play trading styles** (`train.py`, bot specs): besides epsilon
    and the temperature over search values, every training bot may carry a
    per-game acceptance bias (`accept_bias=0.3` draws b ~ U(-0.3, 0.3)
    once per game and shifts the acceptance threshold: generous or stingy
    games), a temperature over the offer ranking (`offer_temp`) and a
    probability of a random response / proposal (`trade_eps`).  The bias of
    the bot that produced each sample is stored with the replay buffer
    (`bias`) for later analysis.

## 7-protection (`discard.py`)

* Risk = 1 - (5/6)^k where k is the number of opponent rolls before our
  next roll; with a hand above 7 the expected loss is shown in the advice.
* Before ending a turn with more than 7 cards the search sees the cheap
  dumps (build, dev card, bank-trade surplus into a needed card): the top
  `surplus_dump_actions` are ranked at least like the trade plan in the
  candidate list, so road / offer spam never crowds them out of the beam.
  (The static hand penalty itself is mirrored in the C++ evaluator port and
  is left evaluator-side; the value net learns the 7-risk from the
  `discard_exposure` / phase features.)
* When a 7 hits, the discard keeps the cards for the top target build (and
  half of the second), then throws away what we produce most easily and
  value least.

## Robber and knights (`robber.py`, `politics.py`)

* Who is dangerous (`danger.py`): not the VP rank but the **distance to a
  win**.  For every player the cheapest path to 10 VP given their hand is
  worked out (city upgrades, settlements on spots up to two roads away,
  Longest Road, Largest Army, VP cards), then the cards still missing, their
  production and port substitutes per resource, the rolls that feed the
  path and the turns until it is affordable.  Target weight = VP threat x
  (0.4 + 1.6 x danger): the leader is the default target, but a loaded
  runner-up (cards in hand, a spot, the right numbers) outranks an
  overextended leader (all cities built, no spot, empty hand).
* Robber target = the hex that removes the most demand-weighted pips from
  the most dangerous players, never on our own production unless nothing
  else exists.  Blocking is **need-aware**: a hex counts for the share of the
  target's supply of a resource they still need; a resource they hold,
  produce elsewhere or can buy through a port from a surplus is discounted,
  while the surplus resource that feeds a 2:1 / 3:1 port becomes worth
  blocking.  Victim = most dangerous player, weighted by what their hand
  likely holds (what they need, what we need) and by a large bonus when one
  stolen card can break a can-win-now hand; then the fattest hand.
* Out of turn we are a target too (`robber.steal_exposure`): every opponent
  rolling before us may hit us with a 7 or a knight, and whether they aim at
  us follows from their own best robber move.  The evaluator discounts a
  valuable hand by the expected loss, so the search spends it or keeps cheap
  cards when we are the obvious victim; the advice says so.
* Play a knight (before or after rolling) when: it takes Largest Army
  (especially if that wins), the robber blocks a critical tile of ours
  (>= 3 pips-equivalent), a player at 8+ VP can be slowed, an opponent is
  about to steal Largest Army from us, or late in the game we hold enough
  knights to build the army.  Otherwise hold it.
* Opponents in the simulation pick victims by danger × grudge × their
  observed habits (whom they keep robbing, whether they go for the leader,
  from the opponent model), so the search "knows" who gets robbed - the
  loaded player, not necessarily the visible leader - and who is spared.
  Our own robber ordering in the search uses the same grudge-weighted
  targets as the advice text, so the two never name different victims.

## Development cards (`devcards.py`, `counting.py`)

* Buy when: at 8-9 VP (hidden VP cards win unseen), pushing for Largest
  Army with knights left in the deck, more than 7 cards in hand, or
  ore/sheep/wheat surplus with no better build.  Never instead of an
  affordable city or a settlement on a real spot.
* The deck is counted: 25 cards minus knights played and cards held, so the
  expected value of the next card (knight / VP / road building / year of
  plenty / monopoly) is known.
* Monopoly when the expected take (from the hand beliefs) is >= 3 cards or
  completes a build; Year of Plenty takes the pair that completes the most
  valuable build, limited by bank stock.
* Cards bought this turn cannot be played until next turn (VP cards count
  immediately); the engine enforces this and the advice says so.

## Card counting (`counting.py`, `inference.py`)

Bank stock per resource, dev deck composition, and an expected-count belief
over every opponent's hand updated by production, builds, bank trades,
discards, steals and monopolies.  From a screenshot only hand sizes are
known, so hands are sampled from a production-weighted prior
(determinization) and the search is averaged over several samples.

## Placement (`placement.py`)

Settlement spots score pips weighted by resource demand and board scarcity,
with diminishing returns on what we already produce, diversity, new
resource types, port synergy, expansion room and a blocking bonus when an
opponent wanted the spot, minus a *blockability* penalty (below).  Cities go
on the best producers (ore/wheat weighted), minus the same penalty.  Roads
head for the best reachable spot, discounted by distance and contest, and
count towards Longest Road; a road that cuts an opponent off from their best
reachable spot (`road_block_values`: the drop of their best two-road spot
score when the edge is taken) gets the same bonus as a contested spot, and a
little more when it also extends our own Longest Road candidate.

### Blockability

Income is not only expected value.  The robber sits on one hex, so two of our
buildings on the same hex - or a city next to our settlement on it - let a
single robber placement switch all of that income off; the best-EV hex is not
the best spot when it concentrates what we already have.
`robber_exposure(state, player, extra_settlement=None, extra_city=None)`
measures that concentration in the pips-equivalent units of the spot scorers:

* `W_b(h)`: demand-weighted pips of our building `b` on hex `h` (settlement =
  pips, city = 2 x pips, times `RESOURCE_DEMAND[res] x scarcity[res] ** 0.5`).
  The robber's current position is ignored: the term is about where it *can*
  go.  `W(h) = sum_b W_b(h)`.
* `P_block(h) = PLACEMENT_ROBBER_Q x W(h)^2 / sum_h' W(h')^2 x (1 + 0.5 x shared_strong(h))`:
  opponents aim the robber at our juiciest hex (the square makes concentration
  costly), and they park it on a strong player's hexes anyway -
  `shared_strong(h)` is 1 when an opponent with `robber.threat >=
  PLACEMENT_STRONG_THREAT` (5+ estimated VP, hidden VP cards included) has a
  building on `h`.  The threshold is absolute rather than "the strongest at
  the table", so early on, when nobody is a robber magnet yet, only
  concentration is charged.
* `stack(h) = 1 - sum_b W_b(h)^3 / W(h)^3`: the share of the hex's weight that
  is there because buildings share it (0 for one building, 0.75 for two equal
  settlements, 0.67 for a city next to a settlement, 0.89 for three).
* `exposure = sum_h P_block(h) x W(h) x stack(h)` and a candidate building is
  charged `PLACEMENT_BLOCK_WEIGHT x (exposure_after - exposure_before)`.

Measuring exposure against the same buildings on separate hexes (`stack`) is
what keeps a first building - and any layout without shared hexes - free: the
scorers never trade raw pips against blockability, only stacking (a spread-out
building can even earn a small bonus for diluting an existing stack).  The
constants are module-level and tunable: `PLACEMENT_ROBBER_Q = 0.35` (share of
the time the robber sits on one of our hexes when we are an ordinary target),
`PLACEMENT_BLOCK_WEIGHT = 1.0` (0 switches the term off and restores the old
scores exactly), `PLACEMENT_STRONG_THREAT = 1.3`.  `BlockContext` caches the
per-player part for callers that score many candidates, and
`cpp/heuristic.cpp` (`score_spot` / `BlockContext`) is the bit-exact port that
`static_value`'s expansion term uses.

Worked numbers on the standard board (weight 1.0):

* First settlement on ore 10 + sheep 2 + brick 6.  A second settlement on the
  brick 6 (brick 6 + sheep 4, 8 pips) is charged 1.93 points, a brick 6 +
  wood 11 corner 2.01: about 1.9 pips of brick (one brick pip is worth 1.02
  in the production term at that point).  The same 8 pips on hexes we do not
  work cost 0, so wheat 12 + brick 6 + wood 11 drops from 15.1 to 13.1 and the
  spread spots (wheat 9 + wood 11 + wood 8: 20.5) stay ahead.
* Two settlements on the ore 8 (brick 10 + ore 8 and a coast corner of the
  8): the second costs 3.01 points because most of that income sits on one
  hex; when a 7-VP leader (threat 2.47) also works the 8 it costs 4.52 (the
  1.5x factor).  The city upgrade there costs 0.86 / 1.28 with the leader; a
  city on a settlement that shares no hex with our other buildings costs 0.
* Setup: round-1 picks are unchanged (no penalty for a first building); on 10
  random boards 2 of the 40 round-2 picks change, exactly where the old top
  spot shared a hex with the player's first settlement (seed 1, player 3:
  ore 11 + wood 6 + sheep 4 next to its own ore 11 gives way to ore 5 +
  brick 10; seed 3, player 1: wood 4 + wheat 8 next to its wheat 8 gives way
  to sheep 10 + brick 4 + ore 12).

## Opponent modelling (`opponent_model.py`)

Nash play is intractable with 3-4 players, so the bot plays
**exploitatively**: each opponent's deviations from expected play are
tracked with exponential decay - offer acceptance (overall, per resource
received, per resource paid), implied resource valuations, who they rob,
whether they hit the leader, build preferences, risk of holding more than 7
cards, and how often they deviate from our heuristic's prediction.  The
statistics are used, not just displayed: acceptance and valuations drive
P(accept) and the arbitrage deals, build preferences shade which resources
an opponent is assumed to want (a city builder wants ore / wheat), the
robber habits drive the simulated opponents' robber moves, and a high
surprise rate makes the profile's statistics count sooner than the
heuristic prior (`confidence`).  Hand sizes are public, so the
"holds > 7 cards" statistic only feeds the style summary.
Profiles persist across games / screenshots (`--profiles`) and can be fed
manually with `--event "blue accepted give ore get wood"`.

## Politics (`politics.py`)

* **Political capital** `capital[i][j]` = how favourably j views i, baseline
  low-to-moderate, decaying back to baseline every turn.  Robbing,
  monopolising, blocking spots / roads, taking an award and refusing fair
  offers cost capital; executed trades and accepted offers build it.  The
  size of each update scales with how far the action was from the actor's
  selfish best alternative ("business" vs "personal") and with the game
  stage (a late favour counts more than an early one).
* **Target pressure** = visible relative position (VP lead, Longest Road /
  Largest Army, production lead) plus grudges: the more ahead you look, the
  more you get robbed and the fewer trades you get.  The advice warns the
  visible leader to prefer hidden progress (dev cards) and small hands.
* **Favour slack**: capital is a bounded modifier on the acceptance
  threshold.  A friend will accept a deal slightly unfavourable for them
  (never a clearly unfair one); the visible leader must pay a premium.
* **Kingmaker-lite**: the search also considers trades that give a trailing
  player the cards to take Longest Road / Largest Army from the leader,
  when our own win probability does not drop and the leader's does - buying
  runway.

## Coalition detection (`coalitions.py`)

Nash-style play cannot see alliances; revealed preferences can.  Every
executed trade and robber choice is scored by the **EV sacrifice** of the
actor - how much worse the choice was than their best alternative (the
bank / port rate for the same cards, a losing deal accepted, a better
robber target spared, a fair offer refused).  Sacrifices are accumulated
**quadratically** per ordered pair with per-turn decay, so one blatant
favour outweighs ten subtle ones and noise barely registers.  Pairs above a
threshold form blocs (connected components).  Consequences:

* opponents are expected to spare their allies with the robber and to
  float them in trades, and to demand a premium from outsiders;
* a bloc that contains the leader is treated as a bigger threat than its
  members look individually when choosing robber targets;
* the advice reports blocs, blatant-favour counts, and whether you are
  being ganged up on (with the counter-play: hidden strength, small hand,
  peel off the weaker member).

Typed events feed it too: `--event "blue traded orange give 2 ore get 1 wood"`.

## Search (`search.py`)

Beam-pruned expectimax over our turn (actions ordered by the priors above,
END_TURN always available), exact expectations over dice, dev-card draws,
steals and trade acceptance, then opponents' turns simulated greedily with
the same value function (max^n) under common random dice samples, and
optionally our next turn again (depth 3).  Leaves are evaluated in batches
by the value net.  Hidden information (screenshots) is handled by averaging
over sampled determinizations.

The lookahead is a *correction* on top of the static value, and its noise is
larger than the margins it has to resolve: under common random numbers the
future values of two end-of-turn candidates still differ by ~0.028
win-probability per dice sample (the greedy opponents diverge), while the
best two root actions are typically 0.002-0.008 apart.  So (DESIGN section 4)
every end-of-turn node gets the lookahead - never only the top few, which
valued identical candidates differently by membership - the mean
`future - static` is added to the leaves of pruned branches without clamping,
and a node's own deviation from that mean is weighted by
`n / (n + lookahead_shrink)` (0.5 at the default 12 samples).  With the
heuristic evaluator a de-noised depth 2 plays at depth-1 strength against the
1-ply stand-in (23.9 % vs 24.9 % over 720 games; the pre-fix depth 2 scored
21.2 %) and still 6-8 points below depth 1 against the alpha-beta stand-in,
whatever the lookahead's sampling (DESIGN section 4): the lookahead's residual
signal is small (rollouts rate its choices +0.001 +/- 0.014 against depth 1).
A gain from lookahead needs an evaluator error the simulation can correct (a
value net trained on end-of-opponent-round targets, or exact one-round threat
terms), not more sampled max^n.  Depth 3 scores the lookahead leaves with a
reduced sub-search for every leaf or for none (500 nodes per leaf must be
left in `max_nodes` after the opponents' turns, DESIGN section 4 rule 4):
at the SearchBot budget of 20000 nodes it is therefore exactly depth 2, and
a real depth 3 costs ~250k nodes and seconds per decision.
