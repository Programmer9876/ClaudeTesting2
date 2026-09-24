"""Bot interface shared by all agents."""
from __future__ import annotations

from typing import List, Optional

from ..actions import Action
from ..state import GameState


class Bot:
    """A decision maker for one seat.

    ``decide`` receives the current state, the complete list of legal actions
    for the acting player (which is always this bot's seat) and a
    ``random.Random``.  It must return one of ``legal_actions``.
    """

    name: str = "bot"

    def reset(self) -> None:
        """Called at the start of every game."""

    def decide(self, state: GameState, legal_actions: List[Action], rng) -> Action:
        raise NotImplementedError

    def observe(self, state: GameState, action: Action, player: int) -> None:
        """Hook called after *any* player's action (for card-counting trackers)."""

    def explain(self, state: GameState) -> Optional[str]:
        """Optional human readable rationale for the last decision."""
        return None
