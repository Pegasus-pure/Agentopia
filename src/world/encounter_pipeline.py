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

        # ── F2+F3: Post-Encounter Effects (统一 rumor + 感知) ──
        self._apply_post_encounter_effects(
            enc_group, name2agent, participants, world_state,
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

            # ── #031: NPC 看到自己本天后续的安排 ──
            # 注入到 NPC 自己的上下文而非对话中。
            # NPC 的 LLM 读到后续安排后，可能会自然在对话里提到
            # "我下个 phase 要去图书馆"——此时 Player/其他 NPC 才能从对话文本得知。
            npc_own_next = self._get_agent_next_in_day(npc, player_agent.clock.get_time())
            if npc_own_next:
                npc.receive_in_activity(
                    f"[Context: You have {npc_own_next} after this encounter]"
                )

        # Initialize Player context
        player_agent.enter_joint_activity(
            activity_background=activity_background,
            activity_type="joint",
            participants=agent_names,
            location_desc=location_desc,
        )

        # 显示 Player 后续安排
        player_next = EncounterPipeline._get_agent_next_in_day(
            player_agent, player_agent.clock.get_time()
        )
        if player_next:
            print(f"[提示] 你今晚还有: {player_next}")

        print(f"\n{'='*60}")
        print(f"[偶遇] 你在 {enc_group.location or '某个地方'} 遇到了:")
        for npc in npc_agents:
            print(f"  - {npc.name}")
        print(f"(输入 再见/bye 可提前结束对话)")
        print(f"{'='*60}")

        # Natural conversation end markers (LLM output contains these)
        FAREWELL_KEYWORDS = ["再见", "拜拜", "bye", "see you", "回头见", "下次聊"]

        # ── #032: NPC 主动离开检测 ──
        # 当 NPC 的 LLM 输出包含这些词时，表示 NPC 要走了，系统自动结束对话
        NPC_DEPARTURE_KEYWORDS = [
            "该走了", "我得走了", "我该走了", "先走了", "告辞",
            "要走了", "得走了", "必须走了", "该告辞了",
            "我下个", "我接下来", "我还有事", "还有事",
            "不能多聊", "下次再聊", "没时间了", "时间不早了",
            "have to go", "gotta go", "must go", "got to go",
            "i should go", "i need to go", "i'm off",
        ]

        min_turns = int(cfg["activity"]["joint_activity_min_turns"])
        turn_count = 0
        extension_limit = max_turns  # total max = 2× original
        join_count = 0
        MAX_JOINERS = 2

        while turn_count < max_turns + extension_limit:
            # --- NPC speaks ---
            npc_departure_detected = False
            for npc in npc_agents:
                try:
                    resp = npc.act_in_activity(
                        activity_type="joint", i_turn=turn_count + 1
                    )
                    last_npc_line = resp
                    last_speaker = npc.name
                    for other in participants:
                        if other.name != npc.name:
                            other.receive_in_activity(
                                f"[{npc.name}]: {resp}"
                            )
                    turn_count += 1

                    # ── #032: NPC 主动说"该走了" → 触发结束 ──
                    resp_lower = resp.lower()
                    if any(kw in resp_lower for kw in NPC_DEPARTURE_KEYWORDS):
                        npc_departure_detected = True
                        self._logger.info(
                            f"[#032] NPC {npc.name} signaled departure "
                            f"at turn {turn_count}: {resp[:80]}"
                        )
                except Exception:
                    self._logger.warning(
                        f"[EncounterPipeline] NPC {npc.name} generation failed",
                        exc_info=True,
                    )
                    continue

            if npc_departure_detected:
                print("[偶遇] NPC 表示该走了，对话自然结束。")
                break

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

            # ── Q2: 检测同位置 solo NPC 加入（≥2 轮后）──
            if (
                turn_count >= 2
                and join_count < MAX_JOINERS
                and world_state is not None
                and enc_group.location
            ):
                all_at_location = world_state.get_agents_at(
                    enc_group.location
                )
                current_participants = set(agent_names)
                candidates = [
                    n for n in all_at_location
                    if n not in current_participants
                ]

                for candidate_name in candidates:
                    if join_count >= MAX_JOINERS:
                        break
                    candidate_agent = name2agent.get(candidate_name)
                    if not candidate_agent:
                        continue

                    candidate_agent.enter_joint_activity(
                        activity_background=activity_background,
                        activity_type="joint",
                        participants=agent_names,
                        location_desc=location_desc,
                    )
                    candidate_agent.receive_in_activity(
                        "[System: You overhear a conversation nearby. "
                        "You may join by speaking up, or stay silent "
                        "by responding with '...']"
                    )
                    try:
                        resp = candidate_agent.act_in_activity(
                            activity_type="joint", i_turn=1
                        )
                    except Exception:
                        self._logger.warning(
                            f"[EncounterPipeline] Candidate "
                            f"{candidate_name} join check failed",
                            exc_info=True,
                        )
                        candidate_agent.exit_activity("joint")
                        continue

                    if resp and resp.strip() not in ("", "..."):
                        # Joins!
                        enc_group.agent_names.append(candidate_name)
                        agent_names.append(candidate_name)
                        participants.append(candidate_agent)
                        npc_agents.append(candidate_agent)
                        join_count += 1
                        print(
                            f"[偶遇] {candidate_name} 加入了对话！"
                        )
                        for p in participants:
                            p.receive_in_activity(
                                f"[System: {candidate_name} joined "
                                f"the conversation]"
                            )
                        turn_count = 0  # reset for fresh energy
                        break  # 同一轮只允许 1 人加入
                    else:
                        candidate_agent.exit_activity("joint")

            # ── Q3: 动态轮次扩展 ──
            if turn_count >= max_turns and turn_count < max_turns + extension_limit:
                all_done = True
                for npc in npc_agents:
                    try:
                        resp = npc.act_in_activity(
                            activity_type="joint",
                            i_turn=turn_count + 1,
                        )
                        last_npc_line = resp
                        last_speaker = npc.name
                        for other in participants:
                            if other.name != npc.name:
                                other.receive_in_activity(
                                    f"[{npc.name}]: {resp}"
                                )
                        turn_count += 1
                        if not any(
                            kw in resp for kw in FAREWELL_KEYWORDS + NPC_DEPARTURE_KEYWORDS
                        ):
                            all_done = False
                    except Exception:
                        self._logger.warning(
                            f"[EncounterPipeline] NPC {npc.name} "
                            f"generation failed during extension",
                            exc_info=True,
                        )
                        continue

                if all_done:
                    print("[偶遇] 对话自然收尾。")
                    break

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

    # ── P0-3 helper ─────────────────────────────────────────────────
    @staticmethod
    def _get_agent_next_in_day(
        agent: "RoleAgent",
        current_time,
    ) -> str:
        """Read agent's upcoming scheduled activities for the rest of the
        current day, skipping encounter-type schedules.

        Returns a human-readable summary of the next activity, or empty string.
        """
        try:
            future = agent.dm.get_future_schedules()
            if not future:
                return ""

            now = current_time
            # Filter: same day, after current time, not encounter type
            upcoming = [
                s for s in future
                if s.activity_time is not None
                and s.activity_time.year == now.year
                and s.activity_time.week == now.week
                and s.activity_time.day >= now.day
                and s.type != "encounter"
                and agent.name in (s.participants or [])
            ]
            upcoming.sort(key=lambda s: (s.activity_time.day, s.activity_time.phase.value if s.activity_time.phase else 0))

            if not upcoming:
                return ""

            next_s = upcoming[0]
            day_str = f"D{next_s.activity_time.day}" if next_s.activity_time.day else ""
            loc_str = f" @ {next_s.location}" if next_s.location else ""

            if next_s.type == "joint":
                others = [p for p in (next_s.participants or []) if p != agent.name]
                other_str = f" with {others[0]}" if others else ""
                return f"{day_str}: {next_s.activity_name}{loc_str}{other_str}"
            else:
                return f"{day_str}: {next_s.activity_name}{loc_str}"
        except Exception:
            return ""

    def _apply_post_encounter_follow(
        self,
        enc_group: "EncounterGroup",
        name2agent: Dict[str, "RoleAgent"],
        world_state: "WorldState",
        all_agents: List["RoleAgent"],
        dialogue_text: str,
    ) -> None:
        """Post-Encounter Follow: 用一次LLM调用替代菜单/随机，统一处理 Player + NPC 跟随决策。

        Args:
            enc_group: 当前偶遇组。
            name2agent: 名字→Agent 映射。
            world_state: WorldState 用于位置更新。
            all_agents: 需要决策的 Agent 列表。
            dialogue_text: 偶遇对话全文。
        """
        from src.world.scheduling import Schedule, make_activity_id
        from src.world.clock import Stage, TimeState
        from src.utils import get_response_json, get_config
        from src.agents.prompts import build_post_encounter_follow_prompt

        # 1. 收集所有参与者的后续计划
        schedule_lines = []
        for agent in all_agents:
            try:
                future = agent.dm.get_future_schedules()
                if future:
                    # 只取下一个 plan
                    next_s = future[0]
                    at = next_s.activity_time
                    loc = next_s.location or "unknown"
                    stype = next_s.type
                    parts = ", ".join(next_s.participants or [agent.name])
                    schedule_lines.append(
                        f"- {agent.name}: {next_s.activity_name} @ {loc} "
                        f"({stype}, participants: [{parts}])"
                    )
                else:
                    pos = world_state.get_position(agent.name)
                    schedule_lines.append(
                        f"- {agent.name}: (no schedule, currently at {pos or 'unknown'})"
                    )
            except Exception:
                schedule_lines.append(f"- {agent.name}: (schedule unavailable)")

        # 2. 调用 LLM
        config = get_config()
        model = config.get("role_model", "gpt-5-mini")
        if isinstance(model, list):
            model = model[0]

        prompt = build_post_encounter_follow_prompt(dialogue_text, "\n".join(schedule_lines))
        result = get_response_json(model=model, messages=prompt)

        if not result or not isinstance(result, dict):
            self._logger.info("[FOLLOW] LLM returned no decision — continuing original plans")
            return

        needs_change = result.get("needs_change", False)
        if not needs_change:
            self._logger.info("[FOLLOW] LLM decided no schedule changes needed")
            return

        # 3. 执行 schedule 更新
        updates = result.get("schedule_updates", [])
        for update in updates:
            agent_name = update.get("agent", "")
            agent = name2agent.get(agent_name)
            if not agent:
                self._logger.warning(f"[FOLLOW] Unknown agent: {agent_name}")
                continue

            stype = update.get("type", "solo")
            location = update.get("location", "")
            participants = update.get("participants", [agent_name])
            activity_name = update.get("activity_name", stype)

            if location:
                world_state.update_position(agent_name, location)

            if participants and len(participants) >= 1:
                try:
                    now = agent.clock.get_time()
                    new_schedule = Schedule(
                        activity_id=make_activity_id(stype, now, f"post_encounter_{agent_name}"),
                        activity_name=f"{activity_name} @ {location}",
                        activity_time=TimeState(now.year, now.week, Stage.ACTIVITY, day=now.day, phase=now.phase),
                        location=location,
                        type=stype,
                        status="created",
                        participants=participants,
                    )
                    agent.dm.add_schedule(new_schedule)
                    self._logger.info(
                        f"[FOLLOW] {agent_name} schedule updated: {stype} @ {location} "
                        f"with {participants}"
                    )
                except Exception as e:
                    self._logger.warning(f"[FOLLOW] Failed to update {agent_name} schedule: {e}")

        # 4. 发送通知
        notifications = result.get("notifications", [])
        for note in notifications:
            frm = note.get("from", "")
            to = note.get("to", "")
            msg = note.get("message", "")
            if frm and to and msg:
                sender = name2agent.get(frm)
                if sender:
                    try:
                        sender.dm.send_message(to=to, content=msg)
                        self._logger.info(
                            f"[FOLLOW] Notification: {frm} → {to}: {msg[:60]}"
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"[FOLLOW] Failed to notify {to}: {e}"
                        )

        # 5. 打印描述
        desc = result.get("description", "")
        if desc:
            print(f"\n[偶遇后] {desc}")

    def _post_encounter_player_choice(
        self, enc_group, name2agent, world_state, player_agent,
    ) -> None:
        """Post-encounter: prompt Player for follow choice, NPCs use LLM."""
        # 1. NPCs: LLM 跟随决策（排除 Player）
        npc_agents = [
            n for n in enc_group.agent_names
            if n in name2agent and n != "Player"
        ]
        if npc_agents:
            from src.world.state import EncounterGroup
            npc_only_group = EncounterGroup(
                location=enc_group.location,
                phase=enc_group.phase,
                agent_names=npc_agents,
            )
            self._apply_post_encounter_follow(
                npc_only_group, name2agent, world_state,
                [name2agent[n] for n in npc_agents],
                "(encounter dialogue)",
            )

        # 2. Player: 终端询问
        if "Player" in enc_group.agent_names:
            print(f"\n--- 偶遇后 ---")
            # 收集同位置其他 NPC 的后续计划
            other_npcs = [n for n in enc_group.agent_names if n != "Player"]
            if other_npcs:
                print(f"同行者:")
                for i, npc_name in enumerate(other_npcs):
                    agent = name2agent.get(npc_name)
                    if agent:
                        future = agent.dm.get_future_schedules()
                        if future:
                            next_s = future[0]
                            print(f"  [{i+1}] {npc_name} — 接下来: {next_s.activity_name} @ {next_s.location or '?'}")
                        else:
                            print(f"  [{i+1}] {npc_name}")
                    else:
                        print(f"  [{i+1}] {npc_name}")
                choice = input("跟随谁？(编号, 回车=不跟): ").strip()
                if choice:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(other_npcs):
                            choice = other_npcs[idx]
                        else:
                            print(f"  [错误] 编号超出范围 (1-{len(other_npcs)})")
                            choice = ""
                    except ValueError:
                        print(f"  [错误] 请输入编号")
                        choice = ""
                if choice and choice in other_npcs:
                    from src.world.scheduling import Schedule, make_activity_id
                    from src.world.clock import TimeState, Stage
                    now = player_agent.clock.get_time()
                    target_agent = name2agent[choice]
                    target_future = target_agent.dm.get_future_schedules()
                    target_loc = target_future[0].location if target_future else None
                    if target_loc:
                        new_schd = Schedule(
                            activity_id=make_activity_id("solo", now, "Player_follow"),
                            activity_name=f"Following {choice}",
                            activity_time=TimeState(
                                now.year, now.week, Stage.ACTIVITY,
                                day=now.day, phase=now.phase,
                            ),
                            location=target_loc,
                            type="solo",
                            status="created",
                            participants=["Player"],
                        )
                        player_agent.dm.add_schedule(new_schd)
                        print(f"[OK] 你决定跟随 {choice} 去 {target_loc}")
                    else:
                        print(f"[?] {choice} 没有后续计划，不跟随")

    # ── F2+F3: Post-Encounter Effects (统一 rumor + 感知) ──────────

    def _apply_post_encounter_effects(
        self,
        enc_group: "EncounterGroup",
        name2agent: Dict[str, "RoleAgent"],
        participants: list,
        world_state: "WorldState",
    ) -> None:
        """F2+F3 unified: 1 LLM call -> extract rumors + bystander perception -> distribute.

        1. Collect dialogue text from participants
        2. LLM 1-shot: extract {rumors: [...], perception: "..."}
        3. Distribute rumors to ALL agents (participants + bystanders)
           - Participants: fidelity=1.0
           - Known bystanders: fidelity=0.3-0.8 based on relationship
           - Stranger bystanders: skip irrelevant rumors
        4. Inject perception text into bystanders' activity_context
        """
        if len(enc_group.agent_names) < 2:
            return

        location = enc_group.location or "unknown"

        # 1. Collect dialogue text
        lines: list[str] = []
        for agent in participants:
            if not agent.activity_context:
                continue
            for msg in agent.activity_context:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and content:
                    lines.append(f"[{agent.name}]: {content}")
        if len(lines) < 2:
            return
        dialogue_text = "\n".join(lines)

        # 2. LLM extraction (1 call)
        from src.utils import get_response_with_retry, num_tokens_from_string
        from src.agents.prompts import POST_ENCOUNTER_EFFECTS_PROMPT
        from src.config import get_config

        config = get_config()
        model = config.get("god_model", "") or config.get("role_model", "")
        if isinstance(model, list):
            model = model[0]
        if not model:
            self._logger.warning("[F2+F3] No model available")
            return

        # Truncate long dialogues
        max_tokens = 2000
        while num_tokens_from_string(dialogue_text) > max_tokens:
            lines.pop(0)
            dialogue_text = "\n".join(lines)

        try:
            prompt = (
                f"A conversation just took place:\n\n{dialogue_text}\n\n"
                f"{POST_ENCOUNTER_EFFECTS_PROMPT}"
            )
            raw = get_response_with_retry(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=384,
                temperature=0.3,
            )
            import json
            try:
                result = json.loads(raw)
            except (json.JSONDecodeError, AttributeError):
                self._logger.warning(f"[F2+F3] JSON parse failed: {raw[:100]}")
                return

            rumors = result.get("rumors", [])
            perception = result.get("perception", "")

            if not rumors and not perception:
                return

            # 3. Identify bystanders
            pos_snapshot = world_state.get_phase_positions()
            nearby = set(pos_snapshot.get(location, []))
            participants_set = set(enc_group.agent_names)
            bystander_names = sorted(nearby - participants_set)

            # 4. Distribute rumors
            for rumor in rumors:
                content = rumor.get("content", "")
                topic = rumor.get("topic", "general")
                tags = rumor.get("tags", [])
                base_fidelity = rumor.get("fidelity", 1.0)

                # Participants: LLM-assigned fidelity
                for agent in participants:
                    try:
                        agent.dm.append_rumor(
                            content=content,
                            topic=topic,
                            source_type="encounter_dialogue",
                            source_location=location,
                            fidelity=base_fidelity,
                            tags=tags,
                        )
                    except Exception:
                        pass

                # Bystanders: fidelity decays further based on relationship
                for name in bystander_names:
                    agent = name2agent.get(name)
                    if not agent:
                        continue
                    rel = self._estimate_bystander_relation(agent, participants_set)
                    if rel < 0.2:
                        continue
                    fidelity = min(1.0, base_fidelity * (0.3 + rel * 0.5))
                    try:
                        agent.dm.append_rumor(
                            content=content,
                            topic=topic,
                            source_type="overheard_encounter",
                            source_location=location,
                            fidelity=fidelity,
                            tags=tags,
                        )
                    except Exception:
                        pass

            # 5. Inject perception into bystanders
            if perception and len(perception) >= 10:
                whisper = f"[You overhear nearby conversation]: {perception}"
                for name in bystander_names:
                    agent = name2agent.get(name)
                    if agent is None:
                        continue
                    try:
                        agent.receive_in_activity(whisper)
                    except Exception:
                        pass

            # 6. P203: Record affection + respect deltas to scratchpad
            for delta_type, delta_list in [("affection_delta", result.get("affection_deltas", [])),
                                           ("respect_delta", result.get("respect_deltas", []))]:
                for delta_entry in delta_list:
                    from_name = delta_entry.get("from", "")
                    to_name = delta_entry.get("to", "")
                    delta = delta_entry.get("delta", 0)
                    reason = delta_entry.get("reason", "interaction")
                    if not from_name or not to_name or from_name == to_name:
                        continue
                    from_agent = name2agent.get(from_name)
                    if not from_agent:
                        continue
                    try:
                        sp_path = from_agent.dm.character_scratchpads / f"{to_name}.jsonl"
                        sp_path.parent.mkdir(parents=True, exist_ok=True)
                        from_agent.dm._append_jsonl(sp_path, {
                            "content": reason,
                            delta_type: delta,
                        })
                    except Exception:
                        pass

            # 7. Record encounter summary to each participant's scratchpad
            #    Zero LLM: splice first 2 lines + total turn count
            time_str = str(self._clock.get_time()) if hasattr(self, '_clock') else ""
            summary_prefix = f"[Encounter at {location}"
            summary_prefix += f" {time_str}" if time_str else ""
            summary_prefix += "]"
            snippet = lines[:2] if len(lines) >= 2 else lines[:1]
            summary_body = "; ".join(snippet)
            if len(lines) > 3:
                summary_body += f" ... ({len(lines)} turns total)"
            encounter_summary = f"{summary_prefix}: {summary_body}"

            for participant in participants:
                for other in participants:
                    if other.name == participant.name:
                        continue
                    try:
                        sp_path = participant.dm.character_scratchpads / f"{other.name}.jsonl"
                        sp_path.parent.mkdir(parents=True, exist_ok=True)
                        participant.dm._append_jsonl(sp_path, {
                            "content": encounter_summary,
                            "encounter_event": True,
                        })
                    except Exception:
                        pass

            self._logger.info(
                f"[F2+F3] Post-encounter: {len(rumors)} rumors, "
                f"{len(bystander_names)} bystanders at {location} "
                f"overheard {', '.join(participants_set)}"
            )
        except Exception as e:
            self._logger.warning(
                f"[F2+F3] Post-encounter effects failed: {e}",
                exc_info=True,
            )

    def _estimate_bystander_relation(
        self, bystander, participants_set: set,
    ) -> float:
        """Estimate how familiar a bystander is with encounter participants.

        Checks scratchpad for mention of any participant name.
        Returns 0.0 (stranger) to 1.0 (close relation).
        """
        try:
            score = 0.0
            checked = 0
            for p_name in participants_set:
                sp = bystander.dm.character_scratchpads / f"{p_name}.jsonl"
                if sp.exists():
                    entries = bystander.dm._read_jsonl(sp, max_lines=1)
                    if entries:
                        score += 1.0
                checked += 1
            if checked == 0:
                return 0.0
            return score / checked
        except Exception:
            return 0.0

