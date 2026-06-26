"""
PlayerAgent - Human player as an agent in the Agentopia simulation.

Replaces LLM calls with terminal input() for plan, contact,
and encounter dialogue phases.
"""

from __future__ import annotations

import difflib
from typing import List, Optional, Dict, Any

from src.agents.role_agent import RoleAgent
from src.world.clock import Clock, Stage, DayPhase
from src.world.scheduling import MessageCenter, Schedule


class PlayerAgent(RoleAgent):
    """A human-controlled agent that uses terminal input() for all decisions.

    Inherits from RoleAgent to maintain interface compatibility with the
    World simulation loop. All LLM generation methods are overridden to
    use user input instead.
    """

    def __init__(
        self,
        name: str,
        clock: Clock,
        msg_center: MessageCenter,
        model: str = "",
        *,
        world_name: str = "school",
        no_context_engineering: bool = False,
        no_history: bool = False,
    ) -> None:
        # Call parent init but with a dummy model since we don't use LLM
        super().__init__(
            name=name,
            clock=clock,
            msg_center=msg_center,
            model=model or "gpt-5-mini",
            world_name=world_name,
            no_context_engineering=no_context_engineering,
            no_history=no_history,
        )
        # T02/T03: Pending invites from Player to NPCs: (day, phase) -> invite_info
        # For n_phases=1, key is int (day); for n_phases>1, key is tuple[int, DayPhase]
        self._pending_invites: Dict = {}
        # T03: Public activities for display in contact()
        self._public_activities = None

    # ------------------------------------------------------------------
    # PLAN stage: player inputs a weekly plan
    # ------------------------------------------------------------------
    def plan(self) -> None:
        """Let the player input their weekly schedule.

        Supports two modes based on config n_phases:
        - n_phases > 1: multi-phase input with `=` separator (one line per day)
        - n_phases = 1: legacy single-phase per day (no separator)
        """
        t = self.clock.get_time()
        self.logger.info(f"[PLAYER PLAN][year={t.year} week={t.week}] start planning")

        # Read config for days count and phases
        from src.config import get_world_config

        world_cfg = get_world_config()
        n_days = int(world_cfg["time"]["n_day"])
        day_phases_cfg = world_cfg["time"].get("day_phases", {})
        n_phases = day_phases_cfg.get("n_phases", 5)
        phases = DayPhase.from_config(day_phases_cfg)

        # Show available locations
        public_locs, private_locs = self.dm.location_store.list_all()
        all_locs = sorted(public_locs + private_locs)

        # Read current state
        try:
            state = self.dm.read_state()
            deposit = state["assets"]["deposit"]
        except Exception:
            deposit = "unknown"

        print("\n" + "=" * 60)
        print(f"===== 周计划 | Year {t.year} Week {t.week:02d} =====")
        print(f"余额: ${deposit}")
        print(f"本周共 {n_days} 天")
        print(f"可用地点 ({len(all_locs)}):")
        for i, loc in enumerate(all_locs):
            if i > 0 and i % 4 == 0:
                print()
            print(f"  [{i+1:2d}] {loc}", end="")
        print()

        if n_phases > 1:
            # ── Multi-phase mode ──────────────────────────────────────
            phase_labels = [DayPhase.label(p) for p in phases]
            phase_label_str = " = ".join(phase_labels)
            print("\n输入每天的计划（用 = 分隔各时段）:")
            print("  每个时段: 地点 活动名       → solo（自己去）")
            print("           约 人名 地点 活动   → 邀请对方")
            print("  跳过某时段留空（= =）")
            print(f"  示例: {phase_label_str}")
            print()

            import re
            from src.world.clock import TimeState, Stage

            scheduled = 0
            for day in range(1, n_days + 1):
                raw = input(f"第{day}天> ").strip()
                if not raw:
                    continue

                # Split by "=" into per-phase texts
                phase_texts = [p.strip() for p in raw.split("=")]

                for i, phase_text in enumerate(phase_texts):
                    if i >= len(phases):
                        break  # more segments than configured phases — ignore extras
                    day_phase = phases[i]
                    if not phase_text:
                        continue  # empty phase → skip
                    ok = self._process_phase_input(
                        day=day,
                        phase=day_phase,
                        phase_text=phase_text,
                        all_locs=all_locs,
                        t=t,
                    )
                    if ok:
                        scheduled += 1

            print(f"本周共安排了 {scheduled}/{n_days} 天 × {n_phases} 时段")

        else:
            # ── n_phases = 1: legacy single-phase per day ─────────────
            print("\n输入每天的计划（用空格分隔）:")
            print("  地点 活动名        → solo（自己去）")
            print("  约 人名 地点 活动   → 邀请对方")
            print("  空行              → 跳过当天")
            print()

            import re
            from src.world.clock import TimeState, Stage

            scheduled = 0
            for day in range(1, n_days + 1):
                raw = input(f"第{day}天> ").strip()
                if not raw:
                    continue

                # T02: Detect invite prefix
                is_invite = False
                invite_target = None
                for prefix in ["约", "邀请", "invite", "Invite"]:
                    if raw.startswith(prefix):
                        is_invite = True
                        invite_raw = raw[len(prefix):].strip()
                        # Parse: first word is the target name
                        parts = invite_raw.split(None, 1)
                        if len(parts) >= 1:
                            invite_target = parts[0]
                            invite_activity = parts[1] if len(parts) > 1 else "一起活动"
                        else:
                            invite_target = None
                            invite_activity = invite_raw
                        break

                if is_invite and invite_target:
                    # Validate: target must be a known NPC name
                    known_names = self._get_known_npc_names()
                    if invite_target not in known_names:
                        print(f"  [错误] 未找到 NPC: '{invite_target}'，请用空格分隔，例如: 邀请 amber cinema 电影")
                        continue
                    # Create a joint-type schedule for the invite
                    location = self._match_location(raw, all_locs)
                    if not location:
                        location = self._fuzzy_match_location(raw, all_locs)

                    activity_time = TimeState(
                        year=t.year,
                        week=t.week,
                        stage=Stage.ACTIVITY,
                        day=day,
                    )

                    schedule = Schedule(
                        activity_id=None,
                        activity_name=invite_activity,
                        activity_time=activity_time,
                        location=location or "",
                        type="joint",
                        status="created",
                        participants=["Player", invite_target],
                        proposer="Player",
                    )
                    self.dm.add_schedule(schedule)
                    # Store pending invite for CONTACT injection
                    self._pending_invites[day] = {
                        "target": invite_target,
                        "activity_name": invite_activity,
                        "location": location or "",
                    }
                    loc_str = f" @ {location}" if location else ""
                    print(f"  [INVITE] 约 {invite_target}: {invite_activity}{loc_str}")
                    scheduled += 1
                    continue

                scheduled += 1

                # Parse location from input
                location = self._match_location(raw, all_locs)
                activity_name = raw

                if location:
                    activity_name = raw.replace(location, "", 1).strip()
                    if not activity_name:
                        activity_name = "Hanging out"

                if not location:
                    location = self._fuzzy_match_location(raw, all_locs)
                    if location:
                        print(f"  [匹配] 地点: {location}")
                        activity_name = raw
                    else:
                        print(f"  [警告] 未找到匹配地点，请输入列表中的地点名（可用编号，如 [1]）")
                        continue

                activity_time = TimeState(
                    year=t.year,
                    week=t.week,
                    stage=Stage.ACTIVITY,
                    day=day,
                )

                schedule = Schedule(
                    activity_id=f"player_plan_{t}_d{day}",
                    activity_name=activity_name,
                    activity_time=activity_time,
                    location=location,
                    type="solo",
                    status="created",
                    participants=["Player"],
                )
                self.dm.add_schedule(schedule)
                print(f"  [OK] {activity_name} @ {location}")

            print(f"本周共安排了 {scheduled}/{n_days} 天")

    # ------------------------------------------------------------------
    # T03: Process a single phase_text into a Schedule (multi-phase mode)
    # ------------------------------------------------------------------
    def _process_phase_input(
        self,
        day: int,
        phase: DayPhase,
        phase_text: str,
        all_locs: list,
        t,  # TimeState for year/week
    ) -> bool:
        """Parse a single phase's text and create a solo or joint Schedule.

        Args:
            day: The day number (1-based).
            phase: The DayPhase for this activity.
            phase_text: The user input for this phase (e.g. "图书馆 看书" or "约 amber cinema 电影").
            all_locs: List of all known location names.
            t: Current TimeState (for year/week).

        Returns:
            True if a Schedule was successfully created, False otherwise.
        """
        from src.world.clock import TimeState, Stage

        # Detect invite prefix
        is_invite = False
        invite_target = None
        for prefix in ["约", "邀请", "invite", "Invite"]:
            if phase_text.startswith(prefix):
                is_invite = True
                invite_raw = phase_text[len(prefix):].strip()
                parts = invite_raw.split(None, 1)
                if len(parts) >= 1:
                    invite_target = parts[0]
                    invite_activity = parts[1] if len(parts) > 1 else "一起活动"
                else:
                    invite_target = None
                    invite_activity = invite_raw
                break

        if is_invite and invite_target:
            known_names = self._get_known_npc_names()
            if invite_target not in known_names:
                print(f"  [错误] 未找到 NPC: '{invite_target}'，请用空格分隔，例如: 邀请 amber cinema 电影")
                return False

            location = self._match_location(phase_text, all_locs)
            if not location:
                location = self._fuzzy_match_location(phase_text, all_locs)

            activity_time = TimeState(
                year=t.year,
                week=t.week,
                stage=Stage.ACTIVITY,
                day=day,
                phase=phase,
            )

            schedule = Schedule(
                activity_id=None,
                activity_name=invite_activity,
                activity_time=activity_time,
                location=location or "",
                type="joint",
                status="created",
                participants=["Player", invite_target],
                proposer="Player",
            )
            self.dm.add_schedule(schedule)
            # T03: store with (day, phase) key
            self._pending_invites[(day, phase)] = {
                "target": invite_target,
                "activity_name": invite_activity,
                "location": location or "",
            }
            phase_label = DayPhase.label(phase)
            loc_str = f" @ {location}" if location else ""
            print(f"  [{phase_label}] [INVITE] 约 {invite_target}: {invite_activity}{loc_str}")
            return True

        # Solo activity path
        location = self._match_location(phase_text, all_locs)
        activity_name = phase_text

        if location:
            activity_name = phase_text.replace(location, "", 1).strip()
            if not activity_name:
                activity_name = "Hanging out"

        if not location:
            location = self._fuzzy_match_location(phase_text, all_locs)
            if location:
                print(f"  [匹配] 地点: {location}")
                activity_name = phase_text
            else:
                print(f"  [警告] 未找到匹配地点，请输入列表中的地点名（可用编号，如 [1]）")
                return False

        activity_time = TimeState(
            year=t.year,
            week=t.week,
            stage=Stage.ACTIVITY,
            day=day,
            phase=phase,
        )

        schedule = Schedule(
            activity_id=f"player_plan_{t}_d{day}_p{phase.value}",
            activity_name=activity_name,
            activity_time=activity_time,
            location=location,
            type="solo",
            status="created",
            participants=["Player"],
        )
        self.dm.add_schedule(schedule)
        phase_label = DayPhase.label(phase)
        print(f"  [{phase_label}] [OK] {activity_name} @ {location}")
        return True

    def _get_known_npc_names(self) -> set[str]:
        """Return the set of known NPC names from the persona directory."""
        try:
            persona_root = self.dm.root.parent  # e.g. data/{world}/persona/{Player} → persona/
            return {d.name for d in persona_root.iterdir() if d.is_dir() and d.name != "Player"}
        except Exception:
            return set()

    def _match_location(self, raw: str, all_locs: List[str]) -> Optional[str]:
        """Try to find a location name as a substring of raw input.

        Returns the matched location string, or None.
        Longer location names are tried first to avoid partial matches.
        """
        # Sort by length descending so "Coffee Shop Corner" matches before "Coffee"
        for loc in sorted(all_locs, key=len, reverse=True):
            if loc.lower() in raw.lower():
                return loc
        return None

    def _fuzzy_match_location(
        self, raw: str, all_locs: List[str], cutoff: float = 0.6
    ) -> Optional[str]:
        """Fuzzy match a location name from raw input using difflib."""
        # Try matching each word in raw against location names
        words = raw.split()
        for word in words:
            matches = difflib.get_close_matches(word, all_locs, n=1, cutoff=cutoff)
            if matches:
                return matches[0]
        # Try matching the whole raw string
        matches = difflib.get_close_matches(raw, all_locs, n=1, cutoff=0.4)
        return matches[0] if matches else None

    def set_public_activities(self, events) -> None:
        """T03: Store public activities for display in contact()."""
        self._public_activities = events

    # ------------------------------------------------------------------
    # CONTACT stage: display incoming messages, let player reply
    # ------------------------------------------------------------------
    def contact(self) -> None:
        """Contact stage: show public activities, received messages, and let player respond."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER CONTACT][year={t.year} week={t.week} slot={t.slot}]"
        )

        # T03: Display this week's public activities (only in first slot)
        if t.slot == 1 and self._public_activities is not None:
            events = self._public_activities
            if events:
                print(f"\n{'='*60}")
                print(f"===== 本周公共活动 | Year {t.year} Week {t.week:02d} =====")
                for evt in events:
                    print(f"  [{evt.start_day}] {evt.event_name}: {evt.description}")
                print(f"{'='*60}")
            else:
                print(f"\n[本周无公共活动]")

        # Check for messages from other agents via DataManager
        msgs = self._get_player_inbox()
        if not msgs:
            return

        print(f"\n--- 联络阶段 | Slot {t.slot} ---")
        for msg in msgs:
            raw_action = msg.get('raw_action', msg.get('message', '(无内容)'))
            msg_from = msg['from']

            # T03: Detect invitation-type messages
            is_invitation = (
                "[邀请]" in raw_action
                or "想约你" in raw_action
                or "invite" in raw_action.lower()
            )

            if is_invitation:
                print(f"\n[来自 {msg_from} 的邀约]")
                print(f"  {raw_action}")
                print(f"  [{msg_from}想约你...接受/拒绝？]")
                reply = input("回复 (accept/yes=接受, reject/no=拒绝, 回车跳过): ").strip()
                if reply:
                    if reply.lower() in ("accept", "yes", "接受", "y"):
                        # Accept the invitation
                        structured_reply = f"respond_invitation: yes, accepted invitation from {msg_from}"
                        self.dm.send_message(to=msg_from, content=structured_reply)
                        self.msg_center.add({
                            "time": str(t),
                            "from": self.name,
                            "to": msg_from,
                            "type": "respond_invitation",
                            "activity_name": raw_action,
                            "decision": "yes",
                            "raw_action": structured_reply,
                            "message": structured_reply,
                        })
                        print(f"[OK] 已接受 {msg_from} 的邀约")
                    elif reply.lower() in ("reject", "no", "拒绝", "n"):
                        structured_reply = f"respond_invitation: no, declined invitation from {msg_from}"
                        self.dm.send_message(to=msg_from, content=structured_reply)
                        self.msg_center.add({
                            "time": str(t),
                            "from": self.name,
                            "to": msg_from,
                            "type": "respond_invitation",
                            "activity_name": raw_action,
                            "decision": "no",
                            "raw_action": structured_reply,
                            "message": structured_reply,
                        })
                        print(f"[OK] 已拒绝 {msg_from} 的邀约")
                    else:
                        # Free-form reply
                        self.dm.send_message(to=msg_from, content=reply)
                        self.msg_center.add({
                            "time": str(t),
                            "from": self.name,
                            "to": msg_from,
                            "type": "contact",
                            "raw_action": reply,
                            "message": reply,
                        })
                        print(f"[OK] 已回复 {msg_from}")
            else:
                print(f"\n[来自 {msg_from} 的消息]")
                print(f"  {raw_action}")

                reply = input("回复 (回车跳过): ").strip()
                if reply:
                    self.dm.send_message(to=msg_from, content=reply)
                    self.msg_center.add(
                        {
                            "time": str(t),
                            "from": self.name,
                            "to": msg_from,
                            "type": "contact",
                            "raw_action": reply,
                            "message": reply,
                        }
                    )
                    print(f"[OK] 已回复 {msg_from}")

    def _get_player_inbox(self) -> List[dict]:
        """Read recent contact messages sent to Player from other agents.

        Returns a list of message dicts sorted by time.
        """
        import json
        from pathlib import Path

        msgs: List[dict] = []
        contact_dir = self.dm.root / "contact"
        if not contact_dir.exists():
            return msgs

        for contact_file in sorted(contact_dir.glob("*.jsonl")):
            try:
                with open(contact_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        content = record.get("content", {})
                        entry = content.get("entry", "")
                        # Messages sent TO Player (from NPCs)
                        if isinstance(entry, str) and "发送给" in entry:
                            msgs.append(
                                {
                                    "from": contact_file.stem,
                                    "raw_action": entry,
                                    "message": entry,
                                }
                            )
            except Exception:
                continue

        return msgs

    def finalize_contact(self) -> None:
        """Override: skip LLM-based finalize, just persist joint activity results."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER FINALIZE CONTACT][year={t.year} week={t.week}] finalize contact"
        )

        # Pull joint activities from MessageCenter and persist
        sched_res = self.msg_center.get_scheduling_result(self.name)
        for sch in sched_res:
            if sch.status == "created" and self.name in sch.participants:
                self.dm.add_schedule(sch)

        # Read notifications
        notifications = self.msg_center.get_notifications(self.name)
        if notifications:
            print(f"\n--- 系统通知 ---")
            for note in notifications:
                print(f"  {note}")

    # ------------------------------------------------------------------
    # ACTIVITY stage: encounter dialogue
    # ------------------------------------------------------------------
    def player_dialogue(self, npc_name: str, npc_line: str) -> str:
        """Prompt the player for a dialogue response during an encounter.

        Args:
            npc_name: Name of the NPC who just spoke.
            npc_line: What the NPC said.

        Returns:
            The player's typed response.
        """
        print(f"\n[{npc_name}]: {npc_line}")
        reply = input("你说: ").strip()
        return reply

    def act_in_activity(
        self, activity_type: str = "joint", i_turn: Optional[int] = None
    ) -> str:
        """Override: in encounter activity, use player input.

        For solo/public activities where Player is alone, just return a
        default reflection since there's no one to talk to.
        """
        if activity_type == "joint":
            # When called as part of normal JointActivity flow (with NPCs),
            # prompt the player. The context is in self.activity_context.
            # Show the last received content as context
            context = ""
            if self.activity_context:
                for msg in reversed(self.activity_context):
                    if msg["role"] == "user" and msg.get("content"):
                        context = msg["content"]
                        break

            if context:
                print(f"\n--- 当前场景 ---")
                # Show last ~500 chars of context
                preview = context[-500:] if len(context) > 500 else context
                print(preview)
                print("--- --- ---")

            reply = input("你说 (回车跳过): ").strip()
            if not reply:
                reply = f"{self.name} nods silently."
            return reply

        elif activity_type == "solo":
            # Player is doing a solo activity - no LLM needed
            return f"{self.name} completed their solo activity."

        else:
            # Public activity or encounter - minimal response
            return f"{self.name} participates in the activity."

    def receive_in_activity(self, content: str) -> None:
        """Override: append observation to activity context (same as parent)."""
        if self.activity_context and self.activity_context[-1]["role"] == "user":
            self.activity_context[-1]["content"] += "\n\n" + content
        else:
            self.activity_context.append({"role": "user", "content": content})

    def enter_joint_activity(
        self,
        activity_background: str,
        activity_type: str,
        participants: Optional[List[str]] = None,
        location_desc: Optional[str] = None,
    ) -> None:
        """Override: minimal context setup, no LLM analysis."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER ACTIVITY][year={t.year} week={t.week} day={t.day}] "
            f"enter {activity_type}: {activity_background[:80]}..."
        )

        # Build minimal activity context
        self.activity_context = [
            {
                "role": "system",
                "content": (
                    f"Activity: {activity_background}\n"
                    f"Participants: {participants}\n"
                    f"Location: {location_desc or 'Unknown'}"
                ),
            }
        ]

    def enter_solo_activity(self) -> None:
        """Override: minimal solo activity entry."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER SOLO][year={t.year} week={t.week} day={t.day}] enter solo"
        )
        self.activity_context = [
            {"role": "system", "content": "Solo activity."}
        ]

    def enter_public_activity(
        self,
        activity_name: str,
        event_description: str,
        participants: List[str],
        group_info: str = "",
    ) -> None:
        """Override: minimal public activity entry."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER PUBLIC][year={t.year} week={t.week} day={t.day}] "
            f"enter: {activity_name}{group_info}"
        )
        self.activity_context = [
            {
                "role": "system",
                "content": (
                    f"Public activity: {activity_name}\n"
                    f"Description: {event_description}\n"
                    f"Participants: {participants}"
                ),
            }
        ]

    # ------------------------------------------------------------------
    # No-op / simplified overrides for stages Player doesn't interact with
    # ------------------------------------------------------------------
    def _generate_with_functions(self, *args, **kwargs):
        """Override: PlayerAgent never calls LLM. Return empty placeholder."""
        return [{"role": "assistant", "content": ""}]

    def review(self) -> None:
        """Weekly review: print summary of the week."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER REVIEW][year={t.year} week={t.week}]"
        )

        # Gather schedule data
        schedules = []
        try:
            sched_path = self.dm.root / "schedule.jsonl"
            if sched_path.exists():
                import json
                with open(sched_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        content = entry.get("content", {})
                        if content.get("activity_time", {}).get("week") == t.week:
                            schedules.append(content)
        except Exception:
            pass

        print("\n" + "=" * 60)
        print(f"===== 本周回顾 | Year {t.year} Week {t.week:02d} =====")

        if schedules:
            # T03: detect n_phases for per-phase display
            from src.config import get_world_config
            world_cfg = get_world_config()
            day_phases_cfg = world_cfg["time"].get("day_phases", {})
            n_phases_review = day_phases_cfg.get("n_phases", 5)

            print(f"本周安排了 {len(schedules)} 天:")
            for s in sorted(schedules, key=lambda x: x.get("activity_time", {}).get("day", 0)):
                d = s.get("activity_time", {}).get("day", "?")
                stype = s.get("type", "?")
                name = s.get("activity_name", "?")
                loc = s.get("location", "?")
                participants = s.get("participants", [])
                # T03: extract phase info
                phase_val = s.get("activity_time", {}).get("phase")
                if n_phases_review > 1 and phase_val is not None:
                    try:
                        p_label = DayPhase.label(DayPhase(phase_val))
                    except (ValueError, KeyError):
                        p_label = f"P{phase_val}"
                    d_str = f"D{d}({p_label})"
                else:
                    d_str = f"D{d}"

                if stype == "joint" and len(participants) >= 2:
                    other = [p for p in participants if p != "Player"][0]
                    print(f"  {d_str}: 约 {other} — {name} @ {loc}")
                elif stype == "solo" or stype == "joint":
                    print(f"  {d_str}: {name} @ {loc}")
                else:
                    print(f"  {d_str}: [{stype}] {name} @ {loc}")
        else:
            print("本周无安排")

        # Read contact history
        try:
            contact_path = self.dm.root / "contact.jsonl"
            msgs = []
            if contact_path.exists():
                import json
                with open(contact_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        content = entry.get("content", {})
                        msgs.append(content)
            if msgs:
                print(f"\n联络 ({len(msgs)} 条):")
                for m in msgs[-5:]:  # last 5
                    sender = m.get("from", "?")
                    text = m.get("message", m.get("raw_action", ""))
                    if text:
                        print(f"  [{sender}] {text[:60]}")
        except Exception:
            pass

        print("=" * 60)

    def settle_week(self) -> None:
        """Weekly settlement: skip LLM-based discard, just log."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER SETTLE][year={t.year} week={t.week}] skipped (no LLM)"
        )

    def express_position_application_wishes(
        self, positions: List, forced_out: bool = False
    ) -> List[str]:
        """Override: skip position application for Player."""
        return []

    def judge_others(self):
        """Override: skip social ranking for Player."""
        from src.world.reward import SocialRanking

        return SocialRanking(
            agent_name=self.name,
            time=str(self.clock.get_time()),
            affection_scores={},
            respect_scores={},
        )
