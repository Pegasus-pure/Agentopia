"""World-state tracking for the ACTIVITY stage micro-execution layer.

Tracks NPC positions, phase, and active actions.  Provides encounter
detection (public-location-only grouping with configurable limits).
"""

from __future__ import annotations

import random


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import get_config
from src.utils import get_logger
from src.world.clock import DayPhase
from src.world.action import Action, ActionStatus

config = get_config()


@dataclass
class EncounterGroup:
    """A group of NPCs co-located during the same day phase at a public location."""

    location: str
    phase: DayPhase
    agent_names: List[str] = field(default_factory=list)


class WorldState:
    """Tracks per-phase positions and actions for encounter detection.

    Each ACTIVITY phase the world updates NPC positions, then
    ``detect_encounters()`` scans for groups of ≥2 NPCs at the same
    public location.  Encounters are capped at
    ``max_encounters_per_phase`` from the day_phases config, sorted by
    group size descending.
    """

    def __init__(self, agent_names: List[str]) -> None:
        """Initialise world state for the given list of agent names.

        Args:
            agent_names: All NPC names participating in the simulation.
        """
        self._agent_names: List[str] = list(agent_names)

        # location → list of agent names currently at that location
        self._positions: Dict[str, List[str]] = {}

        # agent_name → Action (current action for this phase)
        self._actions: Dict[str, Action] = {}

        # Current day phase (set by the world loop)
        self._phase: Optional[DayPhase] = None

        # Agent name → location (reverse mapping for fast lookup)
        self._agent_location: Dict[str, str] = {}

        self._logger = get_logger("world", quiet=False)

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def update_position(self, agent_name: str, location: str) -> None:
        """Move *agent_name* to *location*, updating both indices."""
        # Remove from old location if already positioned
        old_location = self._agent_location.get(agent_name)
        if old_location is not None and old_location in self._positions:
            try:
                self._positions[old_location].remove(agent_name)
            except ValueError:
                pass

        # Add to new location
        self._positions.setdefault(location, []).append(agent_name)
        self._agent_location[agent_name] = location

    def get_position(self, agent_name: str) -> Optional[str]:
        """Return the location *agent_name* is currently at, or None."""
        return self._agent_location.get(agent_name)

    def get_agents_at(self, location: str) -> List[str]:
        """Return all NPC names at *location* (may be empty)."""
        return list(self._positions.get(location, []))

    # ------------------------------------------------------------------
    # Action CRUD
    # ------------------------------------------------------------------

    def set_action(self, agent_name: str, action: Action) -> None:
        """Record an Action for *agent_name*."""
        self._actions[agent_name] = action

    def get_action(self, agent_name: str) -> Optional[Action]:
        """Return the Action for *agent_name*, or None."""
        return self._actions.get(agent_name)

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------

    def set_phase(self, phase: DayPhase) -> None:
        """Set the current day phase."""
        self._phase = phase

    # ------------------------------------------------------------------
    # Phase-position snapshot
    # ------------------------------------------------------------------

    def get_phase_positions(self) -> Dict[str, List[str]]:
        """Return a copy of {location: [agent_names]} for the current phase."""
        return {loc: list(names) for loc, names in self._positions.items()}

    # ------------------------------------------------------------------
    # Encounter detection
    # ------------------------------------------------------------------

    def detect_encounters(self) -> List[EncounterGroup]:
        """Scan current positions and return encounter groups.

        Rules:
        1. Only locations with ≥2 NPCs are candidates.
        2. Private homes (keys starting with ``home/``) are excluded.
        3. Groups are capped at ``max_encounters_per_phase`` from config,
           with larger groups taking priority.
        4. Result is sorted by group size descending.

        Returns:
            List of EncounterGroup, sorted largest-first.
        """
        if self._phase is None:
            return []

        day_phases_cfg = config["world"]["time"].get("day_phases", {})
        max_encounters = day_phases_cfg.get("max_encounters_per_phase", 3)

        groups: List[EncounterGroup] = []

        for location, names in self._positions.items():
            # Skip private homes
            if location.startswith("home/"):
                continue
            if len(names) < 2:
                continue
            # Randomly split into groups of 1-4; only ≥2 trigger encounters
            pool = list(names)
            random.shuffle(pool)
            while pool:
                max_size = min(4, len(pool))
                possible_sizes = list(range(1, max_size + 1))
                # Weight: favours pairs (2) and trios (3) over solo/quad
                weights = [
                    1 if s == 1 else (5 if s == 2 else (3 if s == 3 else 1))
                    for s in possible_sizes
                ]
                size = random.choices(possible_sizes, weights=weights, k=1)[0]
                chunk = sorted(pool[:size])
                pool = pool[size:]
                if len(chunk) >= 2:
                    groups.append(
                        EncounterGroup(
                            location=location,
                            phase=self._phase,
                            agent_names=chunk,
                        )
                    )

        # Sort by group size descending, then by location for determinism
        groups.sort(key=lambda g: (-len(g.agent_names), g.location))

        # Cap at max_encounters_per_phase
        if len(groups) > max_encounters:
            self._logger.debug(
                f"[WorldState] detect_encounters: capped from {len(groups)} "
                f"to {max_encounters} (max_encounters_per_phase={max_encounters})"
            )
            groups = groups[:max_encounters]

        return groups

    # ------------------------------------------------------------------
    # Action advancement & encounter results
    # ------------------------------------------------------------------

    def advance_actions(self, amount: float) -> None:
        """Advance progress on every RUNNING action by *amount*.

        Actions that are PLANNED, PAUSED, COMPLETED, or INTERRUPTED are
        skipped for advancement, but PAUSED actions have their
        paused-phase counter incremented so that :meth:`resume` can
        correctly shift ``end_phase`` later.
        """
        for action in self._actions.values():
            if action.status == ActionStatus.RUNNING:
                action.advance(amount)
            elif action.status == ActionStatus.PAUSED:
                action.tick_paused()

    def apply_encounter_results(self, group: EncounterGroup) -> None:
        """Pause every action belonging to NPCs in *group*.

        Records the current phase on each paused action so that
        :meth:`Action.resume` can correctly shift ``end_phase``.

        Caller should subsequently invoke
        :meth:`apply_delta_to_agent` for each participant to persist
        encounter deltas to agent DM.
        """
        for agent_name in group.agent_names:
            action = self._actions.get(agent_name)
            if action is not None:
                action.pause(phase=self._phase)

    def resume_actions_after_encounter(self) -> None:
        """Resume all PAUSED actions after an encounter completes.

        Each resumed action has its ``end_phase`` shifted forward by
        the number of phases spent paused.
        """
        for action in self._actions.values():
            if action.status == ActionStatus.PAUSED:
                action.resume()

    def apply_delta_to_agent(
        self,
        agent_name: str,
        delta: Dict,
        agent_dm_writer,
    ) -> None:
        """Persist encounter outcome delta to an agent's data manager.

        This is an interface hook — actual persistence is delegated to
        the caller-provided ``agent_dm_writer`` callable which receives
        ``(agent_name, delta)``.

        Args:
            agent_name: Name of the NPC.
            delta: Delta dict produced by the encounter dialogue
                   (e.g. mood changes, item transfers).
            agent_dm_writer: A callable ``f(agent_name, delta)`` that
                             writes to the agent's DataManager.
        """
        try:
            agent_dm_writer(agent_name, delta)
        except Exception:
            self._logger.warning(
                f"[WorldState] apply_delta_to_agent failed for {agent_name}",
                exc_info=True,
            )
