"""
PlayerAgent - Human player as an agent in the Agentopia simulation.

Replaces LLM calls with terminal input() for plan, contact,
and encounter dialogue phases.
"""

from __future__ import annotations

import difflib
from typing import List, Optional, Dict, Any

from src.agents.role_agent import RoleAgent
from src.world.clock import Clock, Stage, DayPhase, TimeState
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

        if n_phases > 1:
            # ── Multi-phase mode: per-phase input ────────────────────
            phase_labels = [DayPhase.label(p) for p in phases]
            print("\n地点列表:")
            for i, loc in enumerate(all_locs):
                if i > 0 and i % 4 == 0:
                    print()
                print(f"  [{i+1:2d}] {loc}", end="")
            print()
            print("\n输入: 地点编号,活动名（编号必填，活动可选；空行跳过）")
            print()

            from src.world.clock import TimeState, Stage

            scheduled = 0
            for day in range(1, n_days + 1):
                day_entries = []
                for i, phase in enumerate(phases):
                    phase_label = phase_labels[i]
                    while True:
                        raw = input(f"第{day}天[{phase_label}]> ").strip()
                        if not raw:
                            if phase_label in ("Dawn", "Night"):
                                self._add_sleeping_schedule(day, phase, t)
                                day_entries.append((phase_label, "Sleeping @ home/Player", True))
                                scheduled += 1
                            else:
                                day_entries.append((phase_label, "(skip)", False))
                            break
                        # Parse: "number" or "number, activity"
                        if "," in raw:
                            num_str, _, activity_name = raw.partition(",")
                            activity_name = activity_name.strip() or "Hanging out"
                        else:
                            num_str = raw
                            activity_name = "Hanging out"
                        try:
                            idx = int(num_str.strip()) - 1
                            if not (0 <= idx < len(all_locs)):
                                print(f"  [错误] 地点编号超出范围 (1-{len(all_locs)})")
                                continue
                        except ValueError:
                            print(f"  [错误] 请输入地点编号，如 1 或 1,看书")
                            continue
                        location = all_locs[idx]
                        ok = self._add_phase_schedule(day, phase, location, activity_name, t)
                        if ok:
                            scheduled += 1
                            day_entries.append((phase_label, f"{activity_name} @ {location}", True))
                        else:
                            day_entries.append((phase_label, raw + " (failed)", False))
                        break

                print(f"\n第{day}天计划:")
                for phase_label, text, ok in day_entries:
                    print(f"  [{phase_label:8}] {text}")
                print()

            # ── 生活标准选择 ──
            print("\n选择本周生活标准:")
            print("  [1] frugal    — 节俭 (消费$100, 材料-5)")
            print("  [2] moderate  — 适中 (消费$200, 材料+0)")
            print("  [3] comfortable — 舒适 (消费$300, 材料+5)")
            print("  [4] luxurious — 奢华 (消费$500, 材料+10)")
            std_choice = input("选择 (1-4, 回车=moderate): ").strip()
            standard_map = {"1": "frugal", "2": "moderate", "3": "comfortable", "4": "luxurious"}
            living_std = standard_map.get(std_choice, "moderate")
            fake_output = [{"content": f"<living_standard>{living_std}</living_standard>"}]
            self._apply_living_standard(fake_output)

            print(f"本周共安排了 {scheduled}/{n_days} 天 × {n_phases} 时段")

            # ── F2: 显示最近 rumor ──
            try:
                rumors = self.dm.read_rumors_retrieved(query="", limit=3)
                if rumors:
                    print(f"\n📢 你最近听到的消息:")
                    for r in rumors:
                        content = r.get("content", "")
                        fidelity = r.get("fidelity", 0)
                        if content:
                            print(f"  - {content} (可信度: {fidelity:.0%})")
            except Exception:
                pass

        else:
            # ── n_phases = 1: legacy single-phase per day ─────────────
            print("\n输入每天的计划: 地点 活动名（空行跳过当天）")
            print()

            import re
            from src.world.clock import TimeState, Stage

            scheduled = 0
            for day in range(1, n_days + 1):
                while True:
                    raw = input(f"第{day}天> ").strip()
                    if not raw:
                        break

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
                            print(f"  [错误] 未找到匹配的地点，请重新输入（输入空行跳过当天）:")
                            print(f"         可用地点: {', '.join(all_locs[:8])}{'...' if len(all_locs) > 8 else ''}")
                            continue

                    activity_time = TimeState(
                        year=t.year, week=t.week,
                        stage=Stage.ACTIVITY, day=day,
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
                    scheduled += 1
                    break

            # ── 生活标准选择 ──
            print("\n选择本周生活标准:")
            print("  [1] frugal    — 节俭 (消费$100, 材料-5)")
            print("  [2] moderate  — 适中 (消费$200, 材料+0)")
            print("  [3] comfortable — 舒适 (消费$300, 材料+5)")
            print("  [4] luxurious — 奢华 (消费$500, 材料+10)")
            std_choice = input("选择 (1-4, 回车=moderate): ").strip()
            standard_map = {"1": "frugal", "2": "moderate", "3": "comfortable", "4": "luxurious"}
            living_std = standard_map.get(std_choice, "moderate")
            fake_output = [{"content": f"<living_standard>{living_std}</living_standard>"}]
            self._apply_living_standard(fake_output)

            print(f"本周共安排了 {scheduled}/{n_days} 天")
    def _add_sleeping_schedule(self, day: int, phase: "DayPhase", t) -> None:
        """Dawn / Night 空输入时，自动创建回家睡觉的 solo 日程。"""
        from src.world.clock import TimeState, Stage
        from src.world.scheduling import Schedule

        activity_time = TimeState(
            year=t.year, week=t.week,
            stage=Stage.ACTIVITY, day=day, phase=phase,
        )
        schedule = Schedule(
            activity_id=f"player_sleep_{t}_d{day}_p{phase.value}",
            activity_name="Sleeping",
            activity_time=activity_time,
            location="home/Player",
            type="solo",
            status="created",
            participants=["Player"],
        )
        self.dm.add_schedule(schedule)

    def _add_phase_schedule(self, day: int, phase, location: str, activity_name: str, t) -> bool:
        """Create a solo schedule for the given phase (location already resolved)."""
        from src.world.scheduling import Schedule, make_activity_id
        from src.world.clock import TimeState, Stage
        activity_time = TimeState(year=t.year, week=t.week, stage=Stage.ACTIVITY, day=day, phase=phase)
        schedule = Schedule(
            activity_id=make_activity_id("solo", activity_time, self.name),
            activity_name=activity_name or "Hanging out",
            activity_time=activity_time,
            location=location,
            type="solo",
            status="created",
            participants=["Player"],
        )
        self.dm.add_schedule(schedule)
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_npc_names: list[str] = []

    def set_active_npc_names(self, names: list[str]) -> None:
        self._active_npc_names = names

    def _get_known_npc_names(self) -> set[str]:
        """Return active NPC names from the agent list."""
        return set(self._active_npc_names)

    def _read_player_affection_scores(self) -> Dict[str, int]:
        """从 Player 的 reward 数据中读取对各 NPC 的好感度。"""
        import json
        from src.utils import FileReadBackwards
        try:
            reward_path = self.dm.root / "reward.jsonl"
            if not reward_path.exists():
                return {}
            with FileReadBackwards(reward_path, encoding="utf-8") as frb:
                last_line = next(frb, "")
            if not last_line:
                return {}
            data = json.loads(last_line.strip())
            ranking = data.get("ranking", {})
            return ranking.get("affection_scores", {})
        except Exception:
            return {}

    def _read_npc_to_player_deltas(self, npc_name: str) -> tuple:
        """Read NPC→Player affection/respect deltas from the NPC's scratchpad.

        Returns (aff_total, resp_total), or (0, 0) if no data.
        """
        try:
            # NPC scratchpad lives in persona/{npc_name}/memory/scratchpad/
            from pathlib import Path
            from src.world.god import _god_data_dir
            data_dir = Path("data") / (_god_data_dir or "school")
            persona_root = data_dir / "persona"
            sp_path = persona_root / npc_name / "memory" / "scratchpad" / "characters" / "Player.jsonl"
            if not sp_path.exists():
                return 0, 0
            import json
            aff = 0
            resp = 0
            for line in sp_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    aff += e.get("affection_delta", 0)
                    resp += e.get("respect_delta", 0)
                except Exception:
                    continue
            return aff, resp
        except Exception:
            return 0, 0

    def _build_player_affection_display(self, known_npcs: list) -> str:
        """Build a formatted display string showing NPC→Player relationship labels.

        Used by CONTACT and PLAN terminal output.
        """
        lines = []
        for name in sorted(known_npcs):
            aff, resp = self._read_npc_to_player_deltas(name)
            from src.utils import affection_label
            label = affection_label(aff, resp)
            lines.append(
                f"  {name}: {label}"
                f" (aff={'+' if aff >= 0 else ''}{aff} resp={'+' if resp >= 0 else ''}{resp})"
            )
        return "\n".join(lines) if lines else ""

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

    def signup_public_events(self, events) -> list:
        """Override: show public events and let Player choose which to attend.

        Provides a terminal menu for public event signup, since PlayerAgent
        cannot call LLM for this decision.
        """
        from src.world.clock import Stage, TimeState
        from src.world.scheduling import Schedule
        from src.config import get_world_config

        # Filter eligible
        eligible = [e for e in events if e.is_eligible(self.name)]
        if not eligible:
            return []

        # Filter days not already busy
        busy_days = self.dm.get_busy_days_this_week()
        available = [e for e in eligible if e.start_day not in busy_days]
        if not available:
            return []

        print(f"\n{'='*60}")
        print("===== 本周公共活动 =====")
        for i, evt in enumerate(available, 1):
            day_label = f"D{evt.start_day}"
            repeat_label = "每周" if evt.repeat_weeks > 1 else "一次"
            print(f"  [{i}] {day_label} | {evt.event_name} ({repeat_label})")
            print(f"      {evt.description[:100]}")
        print(f"  [0] 都不参加")
        print(f"{'='*60}")

        raw = input("选择参加的活动 (编号, 逗号分隔, 0=跳过): ").strip()
        if not raw or raw == "0":
            return []

        selected_indices = []
        for part in raw.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(available):
                    selected_indices.append(idx)
            except ValueError:
                continue

        signups = []
        day_phases = self.clock.get_phases()
        default_phase = day_phases[0] if day_phases else None

        for idx in selected_indices:
            evt = available[idx]
            at_ts = evt.start_t
            if at_ts.stage == Stage.ACTIVITY and at_ts.phase is None and default_phase:
                at_ts = TimeState(
                    year=at_ts.year, week=at_ts.week,
                    stage=at_ts.stage, day=at_ts.day,
                    slot=at_ts.slot, phase=default_phase,
                )
            schedule = Schedule(
                activity_name=evt.event_name,
                activity_time=at_ts,
                participants=[self.name],
                type="public",
                status="created",
                event_description=evt.description,
            )
            self.dm.add_schedule(schedule)
            signups.append(schedule.activity_id)
            print(f"  [PUBLIC] 已报名: {evt.event_name} (D{evt.start_day})")

        return signups

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

        if msgs:
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

        # ── 收件箱处理完毕后：主动操作菜单 ──
        # ── Contact 速率限制 ──
        from src.config import get_config
        contact_limit = int(get_config()["world"]["contact"]["n_action_per_slot"])
        contact_actions_this_slot = 0

        known_npcs = sorted(self._get_known_npc_names())
        print(f"\n--- 主动操作 ---")
        print(f"[1] 发送消息  格式: NPC编号,消息内容")
        print(f"[2] 提议活动  格式: NPC编号,活动名,地点编号,第几天,附言")
        print(f"[3] 赠送礼物")
        print(f"\n## NPC 列表 ({len(known_npcs)} 人):")
        for i, name in enumerate(known_npcs):
            aff, resp = self._read_npc_to_player_deltas(name)
            from src.utils import affection_label
            label = affection_label(aff, resp)
            if i > 0 and i % 3 == 0:
                print()
            print(f"  [{i+1:3d}] {name} ({label})", end="")
        print()
        choice = input("\n选择 (1/2/3, 回车跳过): ").strip()

        if choice == "1":
            while True:
                outgoing = input("发送消息 (NPC编号,消息内容): ").strip()
                if not outgoing:
                    break
                if "," not in outgoing:
                    print("[提示] 格式错误，请使用「NPC编号, 消息内容」（用逗号分隔）")
                    continue
                num_str, _, msg_text = outgoing.partition(",")
                msg_text = msg_text.strip()
                if not msg_text:
                    print("[提示] 消息内容不能为空")
                    continue
                try:
                    idx = int(num_str.strip()) - 1
                    if not (0 <= idx < len(known_npcs)):
                        print(f"[错误] NPC编号超出范围 (1-{len(known_npcs)})")
                        continue
                except ValueError:
                    print(f"[错误] 请输入NPC编号，如 1,你好")
                    continue
                to_name = known_npcs[idx]
                self.dm.send_message(to=to_name, content=msg_text)
                self.msg_center.add({
                    "time": str(t), "from": self.name, "to": to_name,
                    "type": "contact", "raw_action": msg_text, "message": msg_text,
                })
                print(f"[OK] 已向 {to_name} 发送消息")
                contact_actions_this_slot += 1
                if contact_actions_this_slot >= contact_limit:
                    print("已用尽本回合联络次数")
                    return
                break

        elif choice == "2":
            # Show public locations only (exclude home/ private residences)
            public_locs, _ = self.dm.location_store.list_all()
            all_locs = sorted(loc for loc in public_locs if not loc.startswith("home/"))
            print(f"\n可用地点 ({len(all_locs)}):")
            for i, loc in enumerate(all_locs):
                if i > 0 and i % 4 == 0:
                    print()
                print(f"  [{i+1:2d}] {loc}", end="")
            print()

            while True:
                propose_raw = input("提议活动 (NPC编号,活动名,地点编号,第几天,附言): ").strip()
                if not propose_raw:
                    break
                parts = [p.strip() for p in propose_raw.split(",", 4)]
                if len(parts) < 4:
                    print("[提示] 格式错误，请使用「NPC编号, 活动名, 地点编号, 第几天, 附言」")
                    continue
                npc_str, activity_name, loc_str, day_str = parts[:4]
                proposal_msg = parts[4] if len(parts) > 4 else f"一起去{activity_name}"
                # ── NPC 编号 ──
                try:
                    npc_idx = int(npc_str) - 1
                    if not (0 <= npc_idx < len(known_npcs)):
                        print(f"[错误] NPC编号超出范围 (1-{len(known_npcs)})")
                        continue
                except ValueError:
                    print(f"[错误] 请输入NPC编号")
                    continue
                to_name = known_npcs[npc_idx]
                # ── 地点编号 ──
                try:
                    loc_idx = int(loc_str) - 1
                    if not (0 <= loc_idx < len(all_locs)):
                        print(f"[错误] 地点编号超出范围 (1-{len(all_locs)})")
                        continue
                except ValueError:
                    print(f"[错误] 请输入地点编号")
                    continue
                location = all_locs[loc_idx]
                # ── 天数有效性 ──
                try:
                    day = int(day_str)
                except ValueError:
                    print(f"[错误] 天数无效: '{day_str}'，请输入数字")
                    continue
                if day < t.day:
                    print(f"[错误] 第{day}天已过去（当前第{t.day}天），请选择未来日期")
                    continue
                if day > 5:
                    print(f"[错误] 天数超出范围（1-5）")
                    continue
                # ── 活动名唯一性 ──
                if activity_name in self.proposed_activities:
                    print(f"[错误] 活动名 '{activity_name}' 已在本周提议过，请使用不同的活动名")
                    continue
                # 构造 propose_joint_activity 消息
                activity_time = TimeState(
                    year=t.year, week=t.week,
                    stage=Stage.ACTIVITY, day=day,
                )
                invite_msg = (
                    f"[邀请] {self.name} 想约你第{day}天"
                    f"去{location}: {activity_name}"
                )
                self.msg_center.add({
                    "time": str(t), "from": self.name, "to": to_name,
                    "type": "propose_joint_activity",
                    "activity_name": activity_name,
                    "activity_time": str(activity_time),
                    "invited_persons": [to_name],
                    "required_participants": [to_name, self.name],
                    "raw_action": invite_msg,
                    "message": proposal_msg,
                    "location": location,
                    "proposal": proposal_msg,
                })
                self.proposed_activities[activity_name] = {
                    "invited_persons": [to_name],
                    "activity_time": activity_time,
                }
                print(f"[OK] 已向 {to_name} 提议 {activity_name} @ {location} (第{day}天)")
                contact_actions_this_slot += 1
                if contact_actions_this_slot >= contact_limit:
                    print("已用尽本回合联络次数")
                    return
                break

        elif choice == "3":
            self._send_gift_via_terminal()

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
        """Override: skip LLM-based finalize, persist results and show scheduling outcome."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER FINALIZE CONTACT][year={t.year} week={t.week}] finalize contact"
        )

        # Pull joint activities from MessageCenter and persist
        sched_res = self.msg_center.get_scheduling_result(self.name)
        accepted = []
        rejected = []

        for sch in sched_res:
            if self.name not in sch.participants:
                continue

            if sch.status == "created":
                self.dm.add_schedule(sch)
                if sch.proposer == self.name:
                    # Player invited NPC → NPC accepted
                    others = [p for p in sch.participants if p != self.name]
                    other_str = ", ".join(others)
                    day_str = f"D{sch.activity_time.day}" if sch.activity_time.day else ""
                    accepted.append(f"  约 {other_str} 已接受: {sch.activity_name}")
                else:
                    # NPC invited Player → Player accepted
                    day_str = f"D{sch.activity_time.day}" if sch.activity_time.day else ""
                    accepted.append(f"  {sch.proposer} 的邀约已确认: {sch.activity_name}")

            elif sch.status in ("failed", "canceled"):
                if sch.proposer == self.name:
                    # Player invited NPC → NPC rejected
                    others = [p for p in sch.participants if p != self.name]
                    other_str = ", ".join(others)
                    reason = f" ({sch.cancel_reason})" if sch.cancel_reason else ""
                    rejected.append(f"  约 {other_str} 被拒{reason}")
                else:
                    # NPC invited Player → Player rejected
                    reason = f" ({sch.cancel_reason})" if sch.cancel_reason else ""
                    rejected.append(f"  {sch.proposer} 的邀约已拒绝{reason}")

        # Print results
        if accepted or rejected:
            print(f"\n{'='*60}")
            print("===== 联络结果 =====")
            for line in accepted:
                print(line)
            for line in rejected:
                print(line)

        # Read system notifications
        notifications = self.msg_center.get_notifications(self.name)
        if notifications:
            if not accepted and not rejected:
                print(f"\n--- 系统通知 ---")
            for note in notifications:
                print(f"  {note}")

    # ------------------------------------------------------------------
    # ACTIVITY stage: encounter dialogue
    # ------------------------------------------------------------------
    def _show_activity_menu(self, phase_label: str, location: str, scheduled: str) -> str:
        """P202: Show per-phase Player activity menu (before encounter detection).

        Args:
            phase_label: Human-readable phase name (e.g. 'Morning').
            location: Player's current location.
            scheduled: What the Player's schedule says for this phase.

        Returns:
            One of: 'plan', 'search', 'social', 'skip'
        """
        print(f"\n{'─'*50}")
        print(f"  D{self.clock.get_time().day} {phase_label} | 你在 {location}")
        if scheduled:
            print(f"  计划: {scheduled}")
        print(f"{'─'*50}")
        print("  [1] 照计划行动")
        print("  [2] 四处搜索")
        print("  [3] 找人聊天")
        print("  [4] 跳过本时段")
        while True:
            choice = input("> ").strip()
            if choice == "1":
                return "plan"
            elif choice == "2":
                return "search"
            elif choice == "3":
                return "social"
            elif choice == "4":
                return "skip"
            print("  输入 1-4")

    def _do_search(self, location: str, all_locs: list | None = None) -> str | None:
        """P202-A: Search the current location for items.

        DC based on location type + Player stats. On success, add an item
        to the Player's possessions.

        Args:
            location: Where the Player is searching.
            all_locs: Unused, kept for future expansion.

        Returns:
            Item name if found, None otherwise.
        """
        import random

        # Read Player stats for DC modifier
        try:
            state = self.dm.read_state()
            profile = self.dm._profile_cache if hasattr(self.dm, '_profile_cache') else None
            if profile is None:
                try:
                    t = self.clock.get_time()
                    profile = self.dm._read_profile(t.year)
                except Exception:
                    profile = {}
            talents = profile.get("talents", {}).get("quantitative", {})
            intel = talents.get("intelligence", 80)
            creativity = talents.get("creativity", 80)
        except Exception:
            intel = 80
            creativity = 80

        # Location DC table (higher = harder to find things)
        loc_dc = {
            "library": 10, "bookstore": 10, "study_room": 11,
            "gym": 12, "sport": 12, "sports_center": 12,
            "dormitory": 14, "dorm": 14, "home": 16,
            "classroom": 13, "lab": 12, "laboratory": 12,
            "cafeteria": 14, "canteen": 14, "dining": 14,
            "shop": 11, "store": 11, "market": 11, "mall": 11,
            "park": 14, "garden": 14, "outdoor": 14,
            "office": 15, "clinic": 15, "hospital": 15,
            "corridor": 16, "hallway": 16, "street": 16,
        }
        base_dc = 14  # default
        for key, dc in loc_dc.items():
            if key in location.lower():
                base_dc = dc
                break

        # Player stat modifier: higher stats = easier search
        avg_stat = (intel + creativity) / 2
        dc_mod = -2 if avg_stat >= 90 else (-1 if avg_stat >= 80 else (0 if avg_stat >= 60 else 1))
        effective_dc = max(6, min(18, base_dc + dc_mod))

        # Roll
        roll = random.randint(1, 20)
        success = roll >= effective_dc

        print(f"\n  你开始四处搜寻...")
        if success:
            # Pick a location-appropriate item
            loc_items = {
                "library": [{"name": "旧书", "value": 5}, {"name": "笔记残页", "value": 3},
                           {"name": "古籍抄本", "value": 15}, {"name": "书签", "value": 2}],
                "bookstore": [{"name": "二手书", "value": 8}, {"name": "明信片", "value": 2}],
                "gym": [{"name": "运动手环", "value": 20}, {"name": "水壶", "value": 5},
                       {"name": "发带", "value": 3}],
                "lab": [{"name": "试剂瓶", "value": 10}, {"name": "实验笔记", "value": 8}],
                "classroom": [{"name": "粉笔头", "value": 1}, {"name": "遗忘的课本", "value": 5}],
                "cafeteria": [{"name": "零食", "value": 3}, {"name": "优惠券", "value": 2}],
                "shop": [{"name": "小饰品", "value": 8}, {"name": "钥匙扣", "value": 4}],
                "park": [{"name": "四叶草", "value": 5}, {"name": "光滑的石头", "value": 2}],
                "default": [{"name": "零钱", "value": 3}, {"name": "小物件", "value": 3}],
            }
            pool = None
            for key, items in loc_items.items():
                if key in location.lower():
                    pool = items
                    break
            if pool is None:
                pool = loc_items["default"]
            item = random.choice(pool)

            # Add to possessions
            state = self.dm.read_state()
            possessions = state.get("assets", {}).get("possessions", [])
            possessions.append(item)
            state["assets"]["possessions"] = possessions
            self.dm.write_state(state)

            print(f"  ✓ 发现: {item['name']} (价值 ${item['value']})")
            return item["name"]
        else:
            print(f"  什么也没找到... (掷 {roll} < DC {effective_dc})")
            return None

    def _do_social(self, positions: dict, player_location: str) -> str | None:
        """P202-B: List NPCs at same location, let Player choose one to approach.

        NPC→Player attitude labels are shown. Strongly disliked → auto-rejected.
        Moderately negative → confirmation required.

        Args:
            positions: dict mapping location -> list of agent names (from WorldState).
            player_location: Player's current location.

        Returns:
            Selected NPC name, or None if Player skips / NPC rejects.
        """
        nearby = [n for n in positions.get(player_location, []) if n != "Player"]
        if not nearby:
            print("  附近没有其他人。")
            return None

        # Build NPC→Player attitude labels
        npc_attitudes = {}
        for nm in nearby:
            aff, resp = self._read_npc_to_player_deltas(nm)
            from src.utils import affection_label
            npc_attitudes[nm] = (aff, resp, affection_label(aff, resp))

        print(f"\n  在{player_location}的人:")
        for i, name in enumerate(nearby):
            _, _, label = npc_attitudes[name]
            print(f"  [{i+1}] {name} — {label}")
        print("  [0] 算了，不找了")
        while True:
            choice = input("找谁聊天？> ").strip()
            if choice == "0":
                return None
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(nearby):
                    target = nearby[idx]
                    aff, resp, label = npc_attitudes[target]

                    # ① 严重反感 → NPC 拒绝
                    if aff <= -20:
                        print(f"  {target} 看见你走过来，转身离开了... ({label})")
                        return None

                    # ② 中度负面 → 确认
                    if aff <= -5:
                        print(f"  ⚠️ {target} 似乎不太想见到你 ({label})")
                        confirm = input("  还要继续吗？[y/N] ").strip().lower()
                        if confirm != "y":
                            return None

                    return target
            print(f"  输入 1-{len(nearby)} 或 0")

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

            # ── 送礼检测（Encounter 对话中）──
            if reply in ("/gift", "送", "给", "送礼"):
                if self._send_gift_via_terminal():
                    reply = f"{self.name} gave a gift."
                else:
                    reply = f"{self.name} nods silently."

            return reply

        elif activity_type == "solo":
            # Player solo: 从日程中获取活动内容，让 God Model 评估
            schd = self.get_schedule()
            loc = schd.location if schd and schd.location else "unknown"
            act_name = schd.activity_name if schd and schd.activity_name else "solo activity"
            return f"Activity: I'm at {loc} to {act_name}."

        elif activity_type == "public":
            # 从 activity_context 读取活动信息
            act_info = ""
            if self.activity_context:
                for msg in self.activity_context:
                    if msg["role"] == "system":
                        act_info = msg.get("content", "")
                        break
            print(f"\n--- 公共活动 ---")
            if act_info:
                print(act_info[:500])
            action = input("你在活动中做什么? (回车跳过): ").strip()
            return action if action else f"{self.name} participated in the public activity."

        return f"{self.name} acted in the activity."

    def _send_gift_via_terminal(self) -> bool:
        from src.world.activity import JointActivity

        state = self.dm.read_state()
        possessions = state.get("assets", {}).get("possessions", [])
        if not possessions:
            print("  [提示] 你没有任何物品可以赠送")
            return False

        known = sorted(self._get_known_npc_names())
        print(f"\n  🎁 赠送对象 ({len(known)} 人):")
        for i, name in enumerate(known[:10]):
            print(f"    [{i+1}] {name}")
        if len(known) > 10:
            print(f"    ...还有 {len(known)-10} 人")
        target = input("  选择对象序号 (0=取消): ").strip()
        if not target.isdigit() or int(target) < 1 or int(target) > len(known):
            return False
        target_npc = known[int(target) - 1]

        print(f"\n  📦 你的背包 ({len(possessions)} 件):")
        for i, item in enumerate(possessions):
            name = item.get("name", item.get("item_name", f"物品{i+1}"))
            print(f"    [{i+1}] {name}")
        choice = input("  选择物品序号 (0=取消): ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(possessions):
            return False
        item = possessions[int(choice) - 1]
        item_name = item.get("name", item.get("item_name", "?"))

        JointActivity._exec_gift(JointActivity, self.name, target_npc, item_name, self.dm)
        print(f"  [OK] 你送了 {target_npc} {item_name}")
        return True

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
        """Override: build solo context from planned schedule."""
        t = self.clock.get_time()
        self.logger.info(
            f"[PLAYER SOLO][year={t.year} week={t.week} day={t.day}] enter solo"
        )
        # 读取当前日程，构建活动上下文
        schd = self.get_schedule()
        loc = schd.location if schd and schd.location else "unknown"
        act_name = schd.activity_name if schd and schd.activity_name else "unknown"
        self.activity_context = [
            {"role": "system", "content": f"Solo activity: {act_name} @ {loc}"}
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

        # ── 显示当前技能和背包 ──
        try:
            state = self.dm.read_state()
            skills = state.get("skills", {})
            if skills:
                print(f"\n📊 当前技能:")
                for skill, level in sorted(skills.items()):
                    print(f"  {skill}: {level}")
            possessions = state.get("assets", {}).get("possessions", [])
            if possessions:
                deposit = state.get("assets", {}).get("deposit", "?")
                print(f"\n📦 背包 ({len(possessions)} 件, 余额 ${deposit}):")
                for item in possessions:
                    name = item.get("name", item.get("item_name", "?"))
                    print(f"  - {name}")
        except Exception:
            pass

    def settle_week(self) -> None:
        """检查 possessions 数量，超上限时让玩家选择丢弃，否则随机丢弃。"""
        from src.config import get_config
        from src.utils import get_logger

        logger = get_logger(f"agent_{self.name}")
        config = get_config()
        max_possessions = int(config["world"]["solo_activity"]["max_possessions"])

        state = self.dm.read_state()
        possessions = state.get("assets", {}).get("possessions", [])

        if len(possessions) <= max_possessions:
            return  # 未超上限，不需要丢弃

        excess = len(possessions) - max_possessions
        print(f"\n--- 物品清理 ---")
        print(f"背包物品 ({len(possessions)}) 已超过上限 ({max_possessions})，需要丢弃 {excess} 件")

        discard_indices = []
        for i, item in enumerate(possessions[:excess + 5]):
            item_name = item.get("name", item.get("item_name", f"物品{i+1}"))
            print(f"  [{i+1}] {item_name}")

        if excess > 0:
            choice = input(f"选择要丢弃的物品序号 (逗号分隔, 如 1,3,5; 回车则随机丢弃): ").strip()
            if choice:
                for s in choice.split(","):
                    s = s.strip()
                    if s.isdigit() and 1 <= int(s) <= len(possessions):
                        discard_indices.append(int(s) - 1)

        # 如果玩家没选够，随机补足
        import random
        remaining = list(range(len(possessions)))
        random.shuffle(remaining)
        for idx in remaining:
            if len(discard_indices) >= excess:
                break
            if idx not in discard_indices:
                discard_indices.append(idx)

        # 执行丢弃
        new_possessions = []
        for i, item in enumerate(possessions):
            if i not in discard_indices:
                new_possessions.append(item)

        # 写回 state
        self.dm.update_possessions(new_possessions)

        logger.info(f"[SETTLE] {self.name} discarded {len(discard_indices)} items")
        print(f"[OK] 已丢弃 {len(discard_indices)} 件物品，剩余 {len(new_possessions)} 件")

    def express_position_application_wishes(
        self, positions: List = None, forced_out: bool = False
    ) -> List[str]:
        """Player terminal input for 3 position preferences."""
        from src.world.position_application import PositionManager
        from src.config import get_world_config

        cfg = get_world_config()
        pm = PositionManager(self.name, self.dm, cfg)
        available = pm.get_available_positions()

        if not available:
            return []

        print(f"\n--- 职位申请 ---")
        print(f"可选职位:")
        for i, pos in enumerate(available[:15]):
            org = pos.get("organization", "")
            role = pos.get("role", "")
            income = pos.get("weekly_income", 0)
            print(f"  [{i+1}] {org}/{role} (${income}/周)")
        print(f"  [0] 跳过")
        selected = input("选择 3 个偏好 (用逗号分隔序号, 如 1,3,5): ").strip()
        if not selected or selected == "0":
            return []

        wishes = []
        for s in selected.split(","):
            s = s.strip()
            if s.isdigit() and 1 <= int(s) <= len(available):
                pos = available[int(s) - 1]
                wishes.append(f"{pos.get('organization')}/{pos.get('role')}")
                if len(wishes) >= 3:
                    break

        return wishes[:3]

    def judge_others(self) -> "SocialRanking":
        """Player terminal input for social ranking."""
        from src.world.reward import SocialRanking

        # 获取认识的 NPC
        known_names = self.dm.get_top_related_names(limit=10)
        if not known_names:
            return SocialRanking(
                agent_name=self.name,
                time=str(self.clock.get_time()),
                affection_scores={},
                respect_scores={},
            )

        print(f"\n--- 社会关系评分 ---")
        print(f"对以下 NPC 评分 (0-100):")
        affection_scores = {}
        respect_scores = {}
        for name in sorted(known_names):
            aff = input(f"  {name} 好感度 (0-100, 回车=50): ").strip()
            try:
                affection_scores[name] = max(0, min(100, int(aff))) if aff else 50
            except ValueError:
                affection_scores[name] = 50

            res = input(f"  {name} 尊重度 (0-100, 回车=50): ").strip()
            try:
                respect_scores[name] = max(0, min(100, int(res))) if res else 50
            except ValueError:
                respect_scores[name] = 50

        return SocialRanking(
            agent_name=self.name,
            time=str(self.clock.get_time()),
            affection_scores=affection_scores,
            respect_scores=respect_scores,
        )
