"""Action data model for the ACTIVITY stage micro-execution layer.

An Action represents a planned activity that an NPC intends to perform
during a specific set of day phases.  Actions support a full lifecycle:
PLANNED → RUNNING → (PAUSED ⇄ RUNNING) → COMPLETED / INTERRUPTED.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.world.clock import DayPhase


class ActionStatus(Enum):
    """Lifecycle states of an Action."""

    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass
class Action:
    """A planned activity performed by an agent during specific day phases.

    Actions progress from PLANNED through RUNNING (possibly PAUSED by
    encounters) to COMPLETED or INTERRUPTED.  The ``advance()`` method
    drives progress toward 1.0 (completion).
    """

    agent_name: str
    name: str
    location: str
    day: int
    start_phase: DayPhase
    end_phase: DayPhase
    status: ActionStatus = ActionStatus.PLANNED
    progress: float = 0.0
    sub_location: Optional[str] = None
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8], init=False)

    # Encounter-related runtime state (not constructor parameters)
    _paused_at_phase: Optional[DayPhase] = field(default=None, init=False, repr=False)
    _paused_phase_count: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_schedule(cls, schedule, phase: "DayPhase") -> "Action":
        """Build an Action from a Schedule object for a single phase.

        Args:
            schedule: A ``Schedule`` with ``.participants``, ``.activity_id``,
                      ``.location``, and ``.activity_time`` attributes.
            phase: The ``DayPhase`` this action starts and ends within.

        Returns:
            A new ``Action`` ready to be tracked by ``WorldState``.
        """
        agent_name = schedule.participants[0] if schedule.participants else ""
        return cls(
            agent_name=agent_name,
            name=schedule.activity_id or "unknown",
            location=schedule.location or "unknown",
            day=schedule.activity_time.day if schedule.activity_time else 1,
            start_phase=phase,
            end_phase=phase,
        )

    def advance(self, amount: float) -> None:
        """Advance progress by *amount*.

        Once progress reaches or exceeds 1.0, the action is automatically
        marked COMPLETED.

        Args:
            amount: A float in (0.0, 1.0] representing the fraction of
                    the total action duration that elapsed.
        """
        if self.status == ActionStatus.PAUSED:
            # Cannot advance a paused action; silently ignore
            return
        if self.status in (ActionStatus.COMPLETED, ActionStatus.INTERRUPTED):
            # Terminal states – no further advancement
            return
        if self.status == ActionStatus.PLANNED:
            self.status = ActionStatus.RUNNING

        self.progress = min(1.0, self.progress + amount)
        if self.progress >= 1.0:
            self.complete()

    def pause(self, phase: Optional[DayPhase] = None) -> None:
        """Pause a running action (e.g. interrupted by an encounter).

        Records the phase at which the pause occurred so that
        :meth:`resume` can correctly adjust ``end_phase``.

        Args:
            phase: The current ``DayPhase`` when the pause occurs.
                   If not provided, ``_paused_at_phase`` stays ``None``
                   and no end-phase adjustment will happen on resume.
        """
        if self.status == ActionStatus.RUNNING:
            self.status = ActionStatus.PAUSED
            self._paused_at_phase = phase
            self._paused_phase_count = 1

    def resume(self) -> None:
        """Resume a previously paused action.

        Shifts ``end_phase`` forward by the number of phases spent
        paused so the action still has enough time to complete.
        Resets the paused-phase counter afterwards.

        When the action was paused without a known phase (e.g. via the
        legacy ``pause()`` call), no end-phase adjustment is applied.
        """
        if self.status == ActionStatus.PAUSED:
            if self._paused_at_phase is not None and self._paused_phase_count > 0:
                new_end = min(self.end_phase.value + self._paused_phase_count, DayPhase.NIGHT.value)
                self.end_phase = DayPhase(new_end)
            self._paused_at_phase = None
            self._paused_phase_count = 0
            self.status = ActionStatus.RUNNING

    def paused_phases(self) -> int:
        """Return the number of phases this action has been paused.

        Resets to 0 after :meth:`resume` is called.
        """
        return self._paused_phase_count

    def tick_paused(self) -> None:
        """Increment the paused-phase counter by one.

        Called by ``WorldState.advance_actions`` for each phase that
        passes while the action remains PAUSED.
        """
        if self.status == ActionStatus.PAUSED:
            self._paused_phase_count += 1

    def interrupt(self) -> None:
        """Permanently interrupt this action (terminal state)."""
        self.status = ActionStatus.INTERRUPTED

    def complete(self) -> None:
        """Mark the action as successfully completed (terminal state)."""
        self.status = ActionStatus.COMPLETED
        self.progress = 1.0

    def is_active(self) -> bool:
        """Return True if the action is currently active (RUNNING or PAUSED)."""
        return self.status in (ActionStatus.RUNNING, ActionStatus.PAUSED)
