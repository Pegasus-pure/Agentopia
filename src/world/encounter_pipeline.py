"""EncounterPipeline – Encapsulates encounter detection and execution.

Replaces the inline encounter logic in world.py's ACTIVITY stage with a
single ``pipeline.run()`` call: pause → build activity → run dialogue →
resume.
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, TYPE_CHECKING

from src.utils import get_logger

if TYPE_CHECKING:
    from src.world.state import EncounterGroup, WorldState
    from src.agents.role_agent import RoleAgent
    from src.agents.player_agent import PlayerAgent


class EncounterPipeline:
    """Runs a single encounter group: pause, execute dialogue, resume.

    Encapsulates the three-step encounter flow:
    1. Pause all NPC actions in the encounter group via WorldState
    2. Build a temporary Schedule + JointActivity and run the dialogue
    3. Resume paused actions (with end_phase adjustment)

    If the encounter group includes the Player, delegates to the World's
    player-encounter handler for input()-based interaction.
    """

    def __init__(self) -> None:
        self._logger = get_logger("encounter_pipeline", quiet=False)

    def run(
        self,
        enc_group: "EncounterGroup",
        name2agent: Dict[str, "RoleAgent"],
        world_state: "WorldState",
        location_store,
        player_agent: Optional["PlayerAgent"] = None,
    ) -> None:
        """Execute a full encounter pipeline for a single encounter group.

        Args:
            enc_group: The encounter group (location + agent names).
            name2agent: Mapping from agent name to agent instance.
            world_state: The current WorldState for pause/resume.
            location_store: LocationStore for surroundings text.
            player_agent: The PlayerAgent instance (if Player is in
                          the encounter); used for input()-based dialogue.
        """
        if len(enc_group.agent_names) < 2:
            return

        # ── Step 1: Pause actions of all NPCs in the encounter group ──
        world_state.apply_encounter_results(enc_group)

        # ── Step 2: Build temporary Schedule + JointActivity and execute ──
        participants = [
            name2agent[n]
            for n in enc_group.agent_names
            if n in name2agent
        ]
        if len(participants) >= 2:
            has_player = "Player" in enc_group.agent_names
            if has_player and player_agent is not None:
                # Player encounter: custom input()-based dialogue
                self._run_player_encounter(
                    enc_group, participants, name2agent,
                    location_store, player_agent, world_state,
                )
            else:
                # Normal NPC-only encounter
                self._run_npc_encounter(
                    enc_group, participants, name2agent, location_store,
                )

        # ── Step 3: Resume paused actions ──
        world_state.resume_actions_after_encounter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_npc_encounter(
        self,
        enc_group: "EncounterGroup",
        participants: List["RoleAgent"],
        name2agent: Dict[str, "RoleAgent"],
        location_store,
    ) -> None:
        """Run a normal NPC-only encounter dialogue."""
        from src.world.scheduling import Schedule
        from src.world.activity import JointActivity

        try:
            p1, p2 = participants[0].name, participants[1].name
            temp_schd = Schedule(
                activity_name=f"encounter_{p1}_{p2}",
                activity_time=participants[0].clock.get_time(),
                location=enc_group.location,
                type="encounter",
                participants=list(enc_group.agent_names),
            )
            joint_act = JointActivity.from_schedule(
                temp_schd, participants, location_store,
            )
            joint_act.run()
        except Exception:
            self._logger.warning(
                f"[EncounterPipeline] encounter execution failed for "
                f"{enc_group.agent_names} at {enc_group.location}",
                exc_info=True,
            )

    def _run_player_encounter(
        self,
        enc_group: "EncounterGroup",
        participants: List["RoleAgent"],
        name2agent: Dict[str, "RoleAgent"],
        location_store,
        player_agent: "PlayerAgent",
        world_state: "WorldState",
    ) -> None:
        """Run a Player-involved encounter using input()-based dialogue.

        Delegates the actual dialogue loop to the World's existing
        ``_run_player_encounter`` method, then adds the post-encounter
        choice prompt.
        """
        # Build temporary schedule for NPC context setup
        from src.world.scheduling import Schedule

        temp_schd = Schedule(
            activity_name=f"encounter_{enc_group.agent_names[0]}_{enc_group.agent_names[1]}",
            activity_time=player_agent.clock.get_time(),
            location=enc_group.location,
            type="encounter",
            participants=list(enc_group.agent_names),
        )

        # We reuse the existing _run_player_encounter pattern but
        # encapsulated here.  The World instance method is still used
        # because it has access to config and logger we'd otherwise
        # need to duplicate.  This is intentional — the pipeline only
        # wraps the pause/resume control flow, not the dialogue itself.
        #
        # However, since EncounterPipeline is called FROM World, we
        # need access to the World's config.  The simplest approach:
        # import config here.
        from src.config import get_world_config

        cfg = get_world_config()
        max_turns = int(cfg["activity"]["joint_activity_max_turns"])
        min_turns = int(cfg["activity"]["joint_activity_min_turns"])

        npc_agents = [a for a in participants if a.name != "Player"]

        location_desc = (
            location_store.get_surroundings_text(enc_group.location)
            if enc_group.location
            else ""
        )

        activity_background = (
            f"Encounter between {', '.join(enc_group.agent_names)} "
            f"at {enc_group.location or 'an unknown location'}."
        )

        agent_names = list(enc_group.agent_names)

        # Initialize NPC contexts
        for npc in npc_agents:
            npc.enter_joint_activity(
                activity_background=activity_background,
                activity_type="joint",
                participants=agent_names,
                location_desc=location_desc,
            )

        # Initialize Player context
        player_agent.enter_joint_activity(
            activity_background=activity_background,
            activity_type="joint",
            participants=agent_names,
            location_desc=location_desc,
        )

        print(f"\n{'='*60}")
        print(f"[偶遇] 你在 {enc_group.location or '某个地方'} 遇到了:")
        for npc in npc_agents:
            print(f"  - {npc.name}")
        print(f"(输入 再见/bye 可提前结束对话)")
        print(f"{'='*60}")

        # Natural conversation end markers (LLM output contains these)
        FAREWELL_KEYWORDS = ["再见", "拜拜", "bye", "see you", "回头见", "下次聊"]

        min_turns = int(cfg["activity"]["joint_activity_min_turns"])
        turn_count = 0

        for turn in range(max_turns):
            # --- NPC speaks ---
            for npc in npc_agents:
                try:
                    resp = npc.act_in_activity(
                        activity_type="joint", i_turn=turn + 1
                    )
                    last_npc_line = resp
                    last_speaker = npc.name
                    for other in participants:
                        if other.name != npc.name:
                            other.receive_in_activity(
                                f"[{npc.name}]: {resp}"
                            )
                    turn_count += 1
                except Exception:
                    self._logger.warning(
                        f"[EncounterPipeline] NPC {npc.name} generation failed",
                        exc_info=True,
                    )
                    continue

            # --- Player speaks ---
            if not last_npc_line:
                continue
            reply = player_agent.player_dialogue(last_speaker, last_npc_line)

            # Early exit: empty input or farewell keyword
            if not reply or reply.strip() in FAREWELL_KEYWORDS or (
                turn_count >= min_turns and any(kw in reply for kw in FAREWELL_KEYWORDS)
            ):
                print("[偶遇] 对话自然结束。")
                break

            for npc in npc_agents:
                npc.receive_in_activity(f"[{player_agent.name}]: {reply}")
            turn_count += 1

        # Exit dialogue for NPCs
        for npc in npc_agents:
            try:
                npc.exit_activity("joint")
            except Exception:
                self._logger.warning(
                    f"[EncounterPipeline] exit_activity failed for {npc.name}",
                    exc_info=True,
                )

        # Clean up Player context
        player_agent.activity_context = None

        # ── Post-encounter: prompt Player for choice ──
        self._post_encounter_player_choice(
            enc_group, name2agent, world_state, player_agent,
        )

    def _post_encounter_player_choice(
        self,
        enc_group: "EncounterGroup",
        name2agent: Dict[str, "RoleAgent"],
        world_state: "WorldState",
        player_agent: "PlayerAgent",
    ) -> None:
        """Prompt the Player: follow the NPC or continue original plan.

        After an encounter ends, the Player is asked whether to follow
        one of the NPCs to their next position or continue with their
        own schedule.
        """
        npc_names = [n for n in enc_group.agent_names if n != "Player"]
        if not npc_names:
            return

        # For each NPC in the encounter, get their next position
        npc_positions: Dict[str, str] = {}
        for npc_name in npc_names:
            pos = world_state.get_position(npc_name)
            if pos:
                npc_positions[npc_name] = pos

        if not npc_positions:
            return

        print(f"\n{'='*60}")
        print("[偶遇后] 对话结束了。")
        for npc_name, pos in npc_positions.items():
            print(f"  1) 跟着 {npc_name} 一起去 {pos}")
        print(f"  2) 继续原计划")
        print(f"{'='*60}")

        choice = input("你的选择 (1/2，默认2): ").strip()
        if choice == "1":
            # Pick the first NPC (or prompt for which one if multiple)
            if len(npc_positions) == 1:
                target_name, target_pos = next(iter(npc_positions.items()))
            else:
                # Multiple NPCs: ask which one
                print("选哪个NPC？")
                npc_list = list(npc_positions.items())
                for idx, (npc_name, pos) in enumerate(npc_list, 1):
                    print(f"  {idx}) {npc_name} → {pos}")
                sub_choice = input("选择: ").strip()
                try:
                    idx = int(sub_choice) - 1
                    if 0 <= idx < len(npc_list):
                        target_name, target_pos = npc_list[idx]
                    else:
                        target_name, target_pos = npc_list[0]
                except (ValueError, IndexError):
                    target_name, target_pos = npc_list[0]

            world_state.update_position("Player", target_pos)
            print(f"[OK] 你决定跟着 {target_name} 去 {target_pos}")
            self._logger.info(
                f"[EncounterPipeline] Player follows {target_name} to {target_pos}"
            )
        else:
            print("[OK] 继续原计划")
