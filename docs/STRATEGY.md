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
   chain Y -> X -> Z as an intermediary.
7. **Game stage**: early trades grow both economies; as the leader nears
   10 VP every trade mostly helps whoever is closer, so the required gain
   rises and late trades with anyone ahead of us are refused.
8. Incoming offers are judged with the same value function (accept if our
   win probability rises and theirs does not rise more), otherwise by
   "does it complete a build for us and not for them".

## 7-protection (`discard.py`)

* Risk = 1 - (5/6)^k where k is the number of opponent rolls before our
  next roll; with a hand above 7 the expected loss is shown in the advice.
* Before ending a turn with more than 7 cards the search sees the cheap
  dumps (build, dev card, bank-trade surplus into a needed card).
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
* Opponents in the simulation pick victims by danger × grudge, so the search
  "knows" who gets robbed - the loaded player, not necessarily the visible
  leader.

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
opponent wanted the spot.  Cities go on the best producers (ore/wheat
weighted).  Roads head for the best reachable spot, discounted by distance
and contest, and count towards Longest Road.

## Opponent modelling (`opponent_model.py`)

Nash play is intractable with 3-4 players, so the bot plays
**exploitatively**: each opponent's deviations from expected play are
tracked with exponential decay - offer acceptance (overall, per resource
received, per resource paid), implied resource valuations, who they rob,
whether they hit the leader, build preferences, risk of holding more than 7
cards, and how often they deviate from our heuristic's prediction.
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
