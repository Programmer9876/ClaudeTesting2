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

## Win-path races (`winpaths.py`; off by default)

Top players treat Longest Road and Largest Army as *zero-sum races*: a race is only
worth entering if you can win it, a crowded race wastes the cards of everyone in it,
and the uncrowded path gets cheaper.  `static_value` does not know this: it credits
an award holder the full 2 VP as if the award were permanent, and it gives every
challenger a flat progress credit (0.15 / 0.35 per road length, 0.45 per played
knight, 0.45 of the 0.7 per held knight) whether anybody else races or not.  Measured
in self-play, a mid-game Longest Road holder keeps the award only 40 / 61 / 70 % of
the time with a lead of 0 / 1 / 2, a Largest Army holder 65 / 94 % with a lead of 1 / 2.

`catanbot/winpaths.py` replaces those award credits, for every seat, with a
race-aware expected value.  It is an *evaluator wrapper* (`PathsEvaluator`) plus a
move-ordering nudge: `static_value` and its C++ port are untouched.  It only runs
with `search.paths = 1` (bot spec `...,paths=1`); the default bot never imports it.

**Units and horizon.**  Static points (10 per VP); time in rounds.  The horizon is
`H = clamp(0.5 + 2.0 (10 - vmax), 1, 16)` rounds, `vmax` the leader's VP (hidden VP
cards as `counting.expected_hidden_vp`), fitted on 7.6k turn-start states
(remaining rounds = 0.13 + 2.42 (10 - vmax), MAE 3.2).  Root quantities (H, the dev
pool, the bank factor, liveness, the opponents' road room) are fixed per decision so
sibling leaves are compared on one scale.

**Supply** (cards per round, per seat): production with the robber's block counted
for `min(1, 2 / H)` of the horizon, times a bank factor `min(1, bank / (table income +
1))`.  Road rate and dev rate are *bundle rates* (`bundle_rate`): the most roads (wood +
brick) or dev cards (sheep + wheat + ore) per round such that the missing cards are
covered by 0.35 of the surplus left after the bundle's own cards, converted at the
seat's port ratios - one conversion budget shared by every missing resource (capped at
total / 2 and total / 3).  Your production decides which paths are cheap for you: an
ore + wheat seat without sheep buys dev cards ~3x faster than a wood + brick seat.
(The per-resource supply `income + 0.35 x converted income of the other resources`
is kept for the contested-spot timing.)

**Race levels.**  Longest Road: official trail length + half the roads the hand (and
held Road Building cards) could pay for + growth `min(room left, 0.6 x road rate x H)`
(0.6 = the share of roads that extend the trail), + 1.5 for the holder (a tie keeps
it).  Largest Army: played + held knights + half the dev cards the hand could buy x
P(knight) + growth `min(1, P(knight) x dev rate) x H`, all seats' growth scaled to
the knights left in the shared pool, + 1.0 for the holder.

**Solver** (`solve_race`).  `P_i = softmax(beta F_i)` over the projected levels
`F = b + G` plus a "nobody" outcome at the minimum level (5 roads / 3 knights) while
unheld; `beta = 1.2 / sqrt(0.8^2 + max growth)` (more growth ahead = more
uncertainty).  The **crowded factor** `N_close_i` is the soft count of rivals ahead
of or within one unit of seat i.  The units a seat still has to spend to beat the
strongest rival, `U = min(G, gap)`, escalate by 50 % per extra close rival, and cost
`c_LR = 1.2 x crowd` points per road (0.6 points per future card, 2 cards per road,
0.6 of the roads extend the trail, 0.4 of their value remains because race roads also
reach spots) or `c_LA ~ 1.0-1.3 x crowd` per knight (a dev card's 3 cards minus the
value of the VP and progress cards it may be instead).  The prize is
`20 (1 - 0.5 (1 - P))` for the holder and `20 x 0.5 P` for a challenger (one KAPPA, so
with no cost the race is exactly zero-sum).  **credit = max(V(P) - cost U, V(P0))**:
`P0` is the chance if the seat stops growing now, the passive floor.  This is "only
enter a race you can win": in a crowded race the credit falls to the floor and one
more road / knight adds almost nothing, in a race nobody else is in the gap is 0 and
the credit pulls hard.

**Ledger.**  The correction `C_i = credit_i - S_i` cancels static's implicit award
credit `S_i` exactly (20 for a holder, the 0.15 / 0.35 road-length terms, 0.45 per
played knight, 0.45 per held knight - static's 0.25 knight-play utility, the progress
cards' and the VP cards' credit stay).  A race is *live* once somebody holds it or has
4 roads / 2 knights; a race that is not live is skipped and static applies unchanged.
The leaf value is `softmax((static + g C) / 16)` (for a blended net: the heuristic
half is corrected; for a plain net the difference of the two softmaxes is added).

**Contested spots** (`paths_spots=1`, off by default).  For every spot within two
roads that a rival also reaches, `P_me = r_me / sum r` with `r = 1 / (T + o)`: `T` =
rounds until the seat can pay settlement + roads from hand and supply, `o` = its turn
offset (every rival moves before our next turn).  Static's settlement-reach terms
(0.12 x best spot, 0.6 per buildable spot) are rescaled by `P_me`.

**Move ordering** (`paths_priors=1`, on with `paths=1`).  BUY_DEV gets
`+4 P(knight) x DeltaLA` while Largest Army is live (and at least prior 45, i.e. it is
expanded even when `should_buy_dev` says save, when `P(knight) x DeltaLA >= 1` point);
a paid road gets `4 x DeltaLR` (x 0.3 unless it extends a dead end of our network, cap
12) *instead of* action_priors' flat +8 Longest Road bonus: no bonus for roads in a
lost race.  `DeltaR` = our credit change for one more unit, from one re-solve.

**Deliberately not in the value.**  Cities are not zero-sum and static already values
ore / wheat production (their competition enters only the race supply and the advice
text); VP cards are exchangeable draws from the shared pool that static and the dev
chance node already count exactly; "the race leader gets targeted" is in
`robber.target_weight` and politics; there is no plan memory (decisions stay a pure
function of the state, which keeps paired ablations deterministic - late switching is
already expensive because sunk progress stays in `b`, H shrinks and the waste grows).

**Limits (each is a test in tests/test_winpaths.py).**  A won path (holder, lead 3,
thin pool): P ~ 0.98, C ~ 0, one more knight ~0.01 VP.  A race nobody else is in: a
strong pull (two knights played and one held, rivals on at most one: +1.9 points for
the next knight, credit 6.2 vs static's 1.35).  A crowded race (three rivals on 9, 9
and 10 roads, us on 4): P 0.007 with 3.0 close rivals, the next road is worth 0.006 VP
and the road priors lose action_priors' +8 (net -7.7 to -7.9).  An opponent holding
with a big lead: our credit ~0 and C cancels static's progress credit.  Endgame (vmax
9, H = 2.5): current levels decide, a challenger two roads behind has P 0.008.

**Advisor.**  The CLI's "Win paths" section (always shown, text only) prints the
horizon, "Your best path: ...; crowded: ...", one line per award (holder, our level,
rate, P, close rivals and a verdict: safe / defend / OPEN - worth racing / CROWDED -
don't race / passive), the best contested spot, the VP cards left, our ore + wheat
share and a portfolio (points per card: city, settlement, dev card, road-for-length).

**Knobs.**  Spec keys `paths=1` (on), `paths_w` (value weight, default 1), `paths_crowd`
(crowding strength, default 1; 0 = probability share only), `paths_priors` (default
1), `paths_spots` (default 0); `recommend --paths W` in the CLI.  Tunables
`search.paths`, `search.paths_w`, `search.paths_crowd`, `search.paths_priors`,
`search.paths_spots` and the model constants `winpaths.KAPPA`, `BETA`, `H_B`,
`HAND_W`, `ESC`, `TIE_LR`, `LIVE_LR`, `LIVE_LA`, `PRIOR_SCALE` and `PLACEBO` (the
seat-rotated control); the constants only matter with `paths=1` in the base spec.
**Cost** (depth 1, beam 4, expand 8, 30 mid-game positions, loaded machine):
1.25-1.26x the default's decision time (p95 1.3-1.5x), 1.56-1.64x with `paths_spots=1`; ~26 us per
leaf on top of the batched C++ evaluation.  Experiments and decision rules:
docs/ABLATIONS_WINPATHS.md.

## Conversion cost: diversification weighted by ports (`conversion.py`; off by default)

Area: diversification / expansion.  The bot opens ore/wheat and builds cities first on every board (docs/RESULTS.md
2026-09-26 05:00: 3.85 resource types vs 4.67 for Catanatron's bots, 75-78 % of its bank trades at 4:1).  At depth
1 the search never sees that every card of a missing resource is bought later at 4:1; static pays only 0.4 per
type produced.  The user's rule: resource diversity matters a lot early without a port, a little less with a 3:1
port, much less with a 2:1 port on a resource we produce plenty of.

**The term** (our seat only; feature `ports.conversion_cost`, spec `conv=1`).  Robber-free production `p_r` per
roll, income `I`; need shares `n_r` = `placement.RESOURCE_DEMAND` normalised (the build-cost mix: 0.187 / 0.187 /
0.168 / 0.234 / 0.224); shortfall `s_r = max(0, n_r I - p_r)`, surplus `u_r` the other way.  Each acquired card
costs `rho - 1` extra cards on its route: a 2:1 port on `s` carries up to `u_s / 2` cards per roll, the rest goes
3:1 (generic port) or 4:1 (bank) - so 3 / 2 / 1 extra cards per missing card.  Over `R = n x H` rolls left
(`winpaths.horizon`, frozen at the root; 64 rolls early, 18 at 8 VP) the correction is
`-KAPPA_CONV x R x (c(leaf) - c(root))` static points, `KAPPA_CONV = 0.12` (static's price of a held card), and
static's own port credit is cancelled for our seat (`PORT_LEDGER`), so a port settlement gains exactly its trade
savings.  Calibration against the proof logs, for our typical opening: 12.5 extra cards a game at 4:1 (measured
~15), a generic port saves 4.2 (measured ~5), a 2:1 wheat port 3.2 (measured 3.6-3.8); no factor was fitted.
Setup: the `conversion` opening policy (`openings.policy=conversion`) adds `CONV_WEIGHT` (1.25 spot units per
point) x the same cost change of a candidate spot to `setup_pick`, minus the spot score's port bonus.

**What moves.**  A settlement on a missing type gains, a city on surplus ore/wheat loses a little (about 2 of its
~18 static points early, ~10 %), both by two thirds as much with a 3:1 port; a port settlement gains its saving;
everything fades with the horizon.  A road changes nothing unless `REACH_W > 0` (0 by default): then a leaf also
gets `REACH_W x` the best conversion saving among the spots within two roads, discounted `1 / (1 + 0.9 d)` like
static's reach term.

**Decision shadow** (60 proof T1 games vs ValueFunction, our seat, C++ evaluator; 1,997 shadowed decisions, every
2nd main decision; A/A 0 changed):

| arm | changed | main | city-legal | settlement-legal | chosen road share (main) |
|---|---|---|---|---|---|
| default | - | - | - | - | 26.7 % (END_TURN 16.2 %) |
| `conv=1` | 1.1 % | 22 / 1,248 | 16 / 126 | 6 / 104 | 26.7 % |
| `conv=1`, `REACH_W=0.3` | 2.4 % | 48 / 1,248 | 16 / 126 | 6 / 104 | 26.9 % (3 END_TURN -> road) |
| `paths=1` | 11.1 % | 178 / 1,248 | 3 / 126 | 2 / 104 | 30.1 % (END_TURN 11.4 %) |
| `conv=1,paths=1` | 12.0 % | 195 / 1,248 | 17 / 126 | 6 / 104 | 30.1 % |

`conv=1` changes *which* city or settlement (15 + 6 of 22) and almost never the kind of action (one city -> bank
trade): at depth 1 a leaf term cannot make the bot save cards for a settlement next turn instead of building a
city now.  The setup policy is
the larger lever: it changes 56 % of our setup settlements (setup_pick forced: 38 %).  Second settlements of those
positions (union with the logged first): types 3.78 -> 4.28, wheat+ore 11.6 -> 10.2 pips, wood+brick 7.4 -> 7.6,
sheep 1.9 -> 2.4 (sheep is our largest shortfall under the build-cost mix), total pips 20.9 -> 20.2, port 7 ->
13 %.  On 200 random boards (seat vs three default search bots): types 4.07 -> 4.62, 5-type openings 26 -> 62 %,
pips 20.2 -> 18.9.

**With the Longest Road race** (`paths=1,conv=1`).  The two are hub providers, summed per leaf, and value
different things bought by the same roads: this term is *resource access* (cards not lost to the bank, from our
buildings and ports); winpaths' credit is the *prize* (P(hold the 2-VP card at game end) from trail lengths, hand
and road rate).  Their ledgers cancel disjoint parts of static (award progress vs the port credit), and neither
reads the other's output (winpaths' road rate does count port-converted surplus: the race's feasibility, not the
saving).  In the shadow the pair is additive - it differs from `paths=1` alone in 23 decisions, `conv=1` from the
default in 22, and only one decision is changed by both - and it does not over-build roads (road share 30.1 %,
the same as `paths=1`; with `REACH_W=0.3` 30.5 %).  Known overlap: `paths_spots=1` rescales static's reach credit
by our chance at contested spots, the `REACH_W` saving is not rescaled.

**Cost**: 1.01-1.04x the default's CPU per decision (1.06-1.09x with the reach component); the setup policy
replaces the setup search and costs ~0.12x of it.

## Acquisition: get what we need from the cheapest source (`acquisition.py`; off by default)

Area: trading (docs/PRIORITY_PLAN.md step 3; design `docs/designs/priority_areas_2026-09-26.json`, "trades").
Against Catanatron the bank / port side is already close to its per-turn optimum (0 impossible 4:1 trades, 2 %
with a cheaper give, ~90 % of bought cards spent the same turn); the 4:1 gap is structural (ports area).  Every
switch below is off by default and needs neither the Python evaluator nor a C++ change.

* **acq.progress** (`search.acq` 1 / 2, `acq_w`, `acq_self`): a hub provider that replaces static's hand-progress
  term for our seat by an *effective* progress - bank / port conversions of the surplus (capped by the bank; half
  credit for a started conversion) and, with 2, `E[min(Y_r, missing_r)]` from the exact distribution of what we roll
  in the `n` rolls up to our next turn (the same horizon at every node of our turn and after END_TURN).  No
  correction above 7 cards; no pending-port credit.  It also applies when we answer an offer: the reject leaf keeps
  the surplus our own port converts next turn.  **Stage 0 failed its pre-registered rule** (`scripts/acq_audit.py`,
  300 trade-legal T1 positions, A/A 0): 8.0 % of decisions change, 0 END_TURN flips above 7 cards, 1.12-1.15x ms,
  but only 1 of the 13 bank trade -> END_TURN delays at 7 or fewer cards is waitable (P(roll the bought card before
  our next turn) >= 0.5; 50 % required; T2: 0 of 6).  The conversion credit makes a held surplus worth as much as
  the converted card, so the search keeps options instead of trading; production does not turn those delays back
  into trades.  So no games (the feature stops); `acquisition.best_target_left` stays for advice text.
* **Composition with `conv=1`**: both are hub providers, summed per leaf (tested).  Conversion cost is a *flow*
  (cards our buildings lose to the bank over the rest of the game; moves with buildings and ports), acq.progress a
  *stock* (this hand's distance to the next build; moves with the hand).  They cancel disjoint parts of static (the
  port credit / the hand-progress term) and share only the port ratios.  In the audit `conv=1,acq=2` changes 25
  decisions, `acq=2` 24, `conv=1` 1, none by both, and the pair always picks acq's or conv's action.
* **acq.breadth** (`search.trade_proposals` 5 = (A); `search.acq_shapes` = (B): mixed-give 2-for-1 offers - one
  card each of two surplus types for one missing card - injected as real proposal candidates of our main-phase nodes,
  ranked by `p_any` exactly as `_trade_outcomes` computes it, never where a 2:1 port already pays the card, never to
  a likely accepter at 9 VP or fed a build; `search.acq_breadth` = both as one switch).  Self-play smoke: injected
  offers accepted 4 of 20 (3 executed); shadow: the bundle changes 9.1 % of main / trade decisions (mostly which
  offer), 1.17x ms.
* **Player-trade premium** (`search.acq_floor`, `acquisition.W_PREMIUM`; the user's direction): a player trade is
  riskier than the same exchange with the bank because the partner also gains.  Every trade we would propose or
  accept is compared with our best bank / port plan for the same cards (this turn, or our next turn when we answer)
  in win probability of the search's own multi-seat evaluator; it must win by `W_PREMIUM x partner's gain x
  danger_multiplier(partner)` (danger.py, 0.4 far from a win .. 2.0 can win now); ties go to the bank.  The
  feed-the-leader and late-game rules are unchanged.  `candidate_offers` drops offers our own port rate matches.  The
  reason is the explanation ("your port gives you the same 1 brick for 2 wheat on your next turn without helping
  blue (~3 turns from winning)"); the advisor has it as `catanbot recommend --trade-floor W`.  Shadow: 1.6 % of
  decisions (25 accept -> reject), 1.21x ms; in self-play the default fails the rule in 0.6 % of its proposals and
  6.7 % of its accepts.
* **acq.flow** (`acquisition.port_flow_value`): a port's value = the bank trades still to come x the ratio it saves,
  `sum_r R(t) w_r (rho_r - rho'_r)`, `R(t)` the trades our bot still makes after `t` turns (6.7 at the start, T1
  table) and `w_r` a logit on our production share.  Fitted on T1, validated on T2 (`scripts/acq_flow_fit.py`:
  mean absolute error per resource per game 0.69 for the give split vs 1.27 for flat shares).  A pure function for
  the ports area's provider (step 4); the bot never calls it.
* **acq.calib** (`opponent_model.CALIB_RATE` / `CALIB_PRIOR`; fallback `REJECT_STREAK`; politics rule): the
  acceptance model is ~2.5x overconfident even against copies of our bot (0.51-0.56 predicted vs 0.18-0.22
  realised in the self-play smokes).  An
  online per-opponent recalibration `sigmoid(a0 + a_j + beta l)` of its raw logit `l` (updated at every observed
  answer, before the profile learns from it) halves the Brier score in self-play (0.153 vs 0.310).  The streak rule
  gives a seat P(accept) = 0 after 3 rejections in a row until it accepts.

## Counter-offers and out-of-turn trade analysis (`counteroffers.py`; off by default)

**Rules (Colonist.io).**  A rules variant, `GameState.allow_counters` (off: the base game is unchanged).
When the current player P proposes a trade, every other player may accept, reject or *counter*
with a modified deal aimed back at P (trades only ever happen with the current player); P may
accept a counter.  Engine protocol (docstring of `catanbot/engine.py`): responders answer in seat
order with `ACCEPT_TRADE`, `REJECT_TRADE` or `(COUNTER_TRADE, give, get)` (from the counterer's
side); a counter declines the offer as proposed.  Once everyone has answered, P sees the counters
one at a time in seat order, each as an ordinary pending offer from the counterer (`PHASE_TRADE_RESPONSE`,
responder P, the original offer suspended in `counter.origin`): accept = the counter executes and
the round closes (Colonist closes the offer on a completed counter); reject = the next counter,
then the original offer's partner selection among the plain accepters.  One answer (so at most
one counter) per responder per offer, no counter to a counter, counters count toward no limit.
Mapping to Colonist: Colonist's answers are simultaneous and P may take a counter at any moment;
here P decides after seeing every answer (the order that gives P the most information, and one an
old bot can play: a counter is just an offer to answer).

**Responder.**  `SearchConfig.counters = 1` (spec `counter=1`).  Candidates are small edits of the
offer (`engine.counter_candidates`: ask for one more card, give one card fewer, swap one card we
give for another resource).  `rank_counters` drops counters we cannot pay, counters that would hand
the leader a build (`offer_is_feeding_leader`), and every counter where `should_accept` would refuse
to trade with P at all (P about to win; late game with P at >= 7 VP and not behind us).  The rest
are ranked by `P(P accepts)^(1/counter_aggr) x gain` (the gain is the search evaluator's value of
the executed counter minus the value now), and the best `counter_candidates` (2) enter the search
as chance nodes: the other responders answer the original offer, then P takes the counter with the
model's probability (1.2 logits less when someone accepted the original as proposed) or rejects it
and the original offer continues.  Under the model a counter weakly dominates rejecting (a rejected
counter is a rejection), so a bot without friction counters almost every offer it would refuse:
without a margin the smoke games had 100-140 counters per game from two bots, ~10 % taken.
`counter_margin` (0.002 win probability, like `should_accept`'s base margin) makes a counter
beat the plain answers by that much; it stands for what the model does not price (P's patience,
what the counter reveals about our needs).  With it: see the smoke numbers below.

**P(P takes the counter)** (`OpponentModel.predict_counter_accept`): P proposed the original deal,
so the logit starts at +1.0 for P's own deal and moves by 2.5 x the change of the deal's worth for
P - P's implied valuation x needs like `predict_accept`, what P asked for weighted 1.8x (a swap for
another card is a real loss), what P offered 0.7x.  Calibrated on self-play against search-bot
proposers: a first version (prior 1.4, slope 1.8, no need weights) predicted swaps at 43 % and
"one more card" at 20 % while 17 % / 14 % were taken; the current one predicts 9-17 % against 9-16 %
taken.  P's record on counters (`counter_accept`), the stage, favour slack and the leader penalty
shift it.

**Out-of-turn analysis** (`SearchConfig.respond_lookahead = 1`, spec `resp_la=1`; works under
the default rules too).  Answering an offer (accept / reject / each counter), every outcome gets the
rest of P's current turn played out with the lookahead's greedy opponent policy (`_greedy_turn`,
stopping after P's END_TURN; P has rolled, so only dev draws and steals are random) before it is
evaluated, and the outcomes are end-of-decision nodes.  Without it a depth-1 search values the trade
at the moment the cards change hands - it cannot see the settlement the card lets P build on our
spot.  Example (tests): blue offers 2 ore for our brick, the brick completes blue's settlement on the
8-9-11 spot we are one road from; statically accept 0.277 vs reject 0.268, after blue's turn accept
0.181 vs reject 0.268.

**Proposer.**  When our offer is countered we are the current player: the counter is an ordinary
`ACCEPT` / `REJECT` decision and the search simply continues our turn after either (a rejected
counter leads to the next counter or to our own partner choice, which the search then decides).

**Opponent model and card counting.**  A counter is strong evidence: the counterer's implied
valuation moves 1.5x an accept's step towards what they ask for, their per-resource accept
tendencies count an acceptance for every card on both sides, their overall acceptance counts half
(they declined the deal as proposed but want a deal), `counters` counts it.  What they asked for is a
shortage hint (`short`, 0.6 per counter, x0.6 per END_TURN) that lowers `predict_accept`'s
"can they pay it" for hidden hands.  P's answers to counters feed `counter_accept`; an accepted
counter counts as a trade (`traded_with`, politics goodwill both ways - there is no EXECUTE_TRADE).
Card counting (`HandBelief` / `CardCounter.observe_offer`, alias `observe_counter`): every public
offer - a counter or a plain proposal - proves the offerer holds what they offer (a `CardCounter`
drops the hypotheses without it, a `HandBelief` raises those expectations) and hints they lack what
they ask for (soft, `strength` 0.5).  `SearchBot.observe` feeds both kinds when it has a belief.

**Advisor.**  `recommend --offer ...` adds a section "Offer response (accept / reject / counter,
after the proposer's turn)": the best answer after P's turn (and what the at-the-trade view would
have said when it differs), then accept / reject / up to three counters, each with its value after
P's turn and at the trade, P(they take it) for counters, and the reason: what the trade lets P build
this turn ("accepting lets blue build a settlement on vertex 28 (the 8-9-11 spot you are heading
for)"; builds P makes anyway after our reject are not blamed on the trade) and who would take the
deal if we refuse ("rejecting: orange will likely accept instead (~N%)", the model's P(accept)).
Existing lines are unchanged; `--json` has `offer_response`.

**Knobs.**  Rules: `play_game(..., allow_counters=True)`, `scripts/ablate.py --counters`.  Bot:
`counter=1`, `counter_n=2`, `counter_aggr=1.0`, `counter_margin=0.002`, `resp_la=1`; tunables
`search.counters`, `search.respond_lookahead`, `search.counter_aggr`, `search.counter_margin`.  The
league referee plays the standard rules, so only `resp_la=1` can be gated there today.

**Smoke** (20 games per setting under the counters rule, 2 x `counter=1` vs 2 default search bots at
depth 1 with the heuristic evaluator, same seeds and seat patterns, one process on a loaded
machine - far too few games to say anything about strength): with `resp_la=0` the two counter bots
made 2.1 counters per game, 0.4 were taken; mean decision 10.8 ms vs 9.8 ms for the default seats,
an answer to an offer 0.98 ms (p95 2.1) vs 0.43 ms.  With `resp_la=1`: 4.8 counters per game, 0.8
taken; mean decision 11.5 ms vs 9.8 ms, an answer to an offer 2.9 ms (p95 10.1) vs 0.43 ms.
Answering a counter (a normal search of our turn) took 17-20 ms.  Wins 16-4 and 12-8 for the
counter side: noise at 20 games; the paired ablation (>= 1000 games) decides.
