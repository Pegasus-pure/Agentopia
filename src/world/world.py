from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from src.world.clock import Clock, DayPhase, Stage, TimeState
from src.agents.role_agent import RoleAgent
from src.agents.player_agent import PlayerAgent
from src.world.scheduling import MessageCenter, Schedule, PublicEvent, make_activity_id
from src.world.cleanup import clean_append_only_jsonl_before
from src.world.god import init_god_module
from src.utils import get_logger, pool_size, set_log_run_id
from src.config import get_world_config, get_config
from src.world.activity import EncounterActivity, JointActivity, SoloActivity
from src.world.state import WorldState
from src.world.action import Action


class World:
    """Minimal world runner following the pseudo/world.py stages."""

    def __init__(
        self,
        *,
        no_context_engineering: bool = False,
        parallel: bool = False,
        config_path: str | None = None,
        no_history: bool = False,
        max_agents: int | None = None,
        resume_from: Optional[Tuple[int, int]] = None,
        player_enabled: Optional[bool] = None,
    ) -> None:
        self.config = get_world_config()
        # CLI override for player mode
        if player_enabled is not None:
            self.config["enable_player"] = player_enabled
        self.clock = Clock(start_year=self.config["time"]["start_year"], start_week=0)

        # data_dir: file path (includes run_id); name: logical identifier (used for prompts)
        self.data_dir = self.config["data_dir"]

        # Set up the run-specific log directory: logs/{run_id}/
        run_id = Path(self.data_dir).name
        set_log_run_id(run_id)

        # world logger prints to console; log file: logs/{run_id}/world.log
        self.logger = get_logger("world", quiet=False)

        # Initialize God module for SFT data collection
        init_god_module(clock=self.clock, data_dir=self.data_dir)

        # Create cache logger with run_id for cache hit/miss logs
        # File: logs/world_{worldname}_{runid}.log (e.g., logs/world_schooldays_01151703.log)
        # This logger only writes to file, not console
        cache_logger_name = f"world_{Path(self.data_dir).name}"
        get_logger(cache_logger_name, quiet=True)

        # Determine resume point (for cleanup and run loop)
        self._resume_year, self._resume_week = self._resolve_resume_point(resume_from)
        start_time = TimeState(self._resume_year, self._resume_week, Stage.BEGIN)
        self.logger.info(
            f"Resume point: Y{self._resume_year}-W{self._resume_week:02d} "
            f"(cleanup from {start_time})"
        )
        clean_append_only_jsonl_before(world_name=self.data_dir, start_time=start_time)

        self.no_context_engineering = no_context_engineering
        self.parallel = parallel
        # Concurrency cap from root config; fail fast if missing.
        root_cfg = get_config()

        # role_model: str | list[str] from config
        role_model_cfg = root_cfg["role_model"]
        if isinstance(role_model_cfg, str):
            self._role_models = [role_model_cfg]
        else:
            self._role_models = list(role_model_cfg)
        if not self._role_models:
            raise ValueError("role_model config must not be empty")
        self.max_concurrency = int(root_cfg["max_concurrency"])
        self.no_history = no_history
        self.max_agents = max_agents  # None means no limit
        # Message center: single instance shared by world and all agents
        self.msg_center = MessageCenter(world_name=self.data_dir, clock=self.clock)

        # WorldState: per-phase position & action tracking (initialised later
        # when day-loop begins; import deferred to avoid circular reference).
        # TODO: from src.world.state import WorldState
        self.world_state = None  # type: Optional[WorldState]

        # Bootstrap agents from existing dataset directories to avoid inventing personas here.
        # Initialize agents from data directory with a configurable cap
        self.agents: List[RoleAgent] = self._init_agents_from_data(
            max_agents=self.max_agents
        )
        # Cache name -> agent mapping (agents don't change after init)
        self._name2agent: Dict[str, RoleAgent] = {a.name: a for a in self.agents}

    # Initialization ---------------------------------------------------------
    def _persona_root(self) -> Path:
        return Path("data") / self.data_dir / "persona"

    def _init_agents_from_data(
        self, *, max_agents: int | None = None
    ) -> List[RoleAgent]:
        root = self._persona_root()
        if not root.exists():
            raise FileNotFoundError(f"persona root not found: {root}")
        # Every persona should have a profile; for simplicity we no longer filter here. If one is missing, an error will be raised later at read time to surface the problem early.
        all_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        # Exclude Player from the normal selection pool
        persona_dirs = [d for d in all_dirs if d.name != "Player"]
        names = (
            [p.name for p in persona_dirs[:max_agents]]
            if max_agents
            else [p.name for p in persona_dirs]
        )
        # Add Player only if enabled in config
        player_enabled = self.config.get("enable_player", False)
        player_dir = next((d for d in all_dirs if d.name == "Player"), None)
        if player_dir and player_enabled:
            names.append("Player")

        # Ensure locations file and private homes exist for current run world
        from src.world.locations import get_location_store

        self.location_store = get_location_store(self.data_dir)

        # Assign role_model to each agent (uniform distribution, persisted)
        model_assignment = self._load_or_assign_models(names)

        # Create agents first (needed for agents_summary in location generation)
        agents = []
        for n in names:
            if n == "Player":
                agent = PlayerAgent(
                    n,
                    clock=self.clock,
                    msg_center=self.msg_center,
                    world_name=self.data_dir,
                    no_context_engineering=self.no_context_engineering,
                    no_history=self.no_history,
                )
            else:
                agent = RoleAgent(
                    n,
                    clock=self.clock,
                    msg_center=self.msg_center,
                    model=model_assignment[n],
                    world_name=self.data_dir,
                    no_context_engineering=self.no_context_engineering,
                    no_history=self.no_history,
                )
            agents.append(agent)

        # Build agents summary for location generation
        agents_summary = "\n\n".join(
            f"## {a.name}\n{a.dm.get_brief_intro()}" for a in agents
        )
        self.location_store.ensure(persona_names=names, agents_summary=agents_summary)

        # Ensure positions exist (generate via God Model if needed)
        from src.world.position_application import get_position_store
        from src.agents.prompts import get_world_setting

        self.position_store = get_position_store(self.data_dir)
        world_setting = get_world_setting(self.data_dir)
        self.position_store.ensure(agents=agents, world_setting=world_setting)

        # Store initial position count for yearly growth calculation
        self._initial_position_count = self.position_store.count()

        # Ensure initial state exists (writes W00-begin entry).
        # Only on fresh start — on resume, state.jsonl already has data.
        # Writing at W00 during resume would append a time-misordered entry
        # after existing W01+ data, corrupting backward reads.
        is_fresh_start = (
            self._resume_year == self.config["time"]["start_year"]
            and self._resume_week <= 1
        )
        if is_fresh_start:
            for agent in agents:
                agent.dm.read_state()

        # Give PlayerAgent the list of active agent names for contact display
        active_names = [a.name for a in agents if a.name != "Player"]
        for agent in agents:
            if agent.name == "Player":
                agent.set_active_npc_names(active_names)

        return agents

    # Model Assignment ---------------------------------------------------------
    def _model_assignment_path(self) -> Path:
        return Path("data") / self.data_dir / "model_assignment.json"

    def _load_or_assign_models(self, names: List[str]) -> Dict[str, str]:
        """Load or create model assignment for each agent.

        If model_assignment.json exists, load it (resume scenario).
        Otherwise, uniformly distribute role_models across agents and persist.
        """
        path = self._model_assignment_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                assignment = json.load(f)
            self.logger.info(f"Loaded model assignment from {path}")
            # Assign models for any missing agents (resume with changed persona set)
            missing = [n for n in names if n not in assignment and n != "Player"]
            if missing:
                self.logger.warning(
                    f"model_assignment.json missing {len(missing)} agents, "
                    f"assigning models for: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                )
                models = self._role_models
                # Deterministic assignment: hash the agent name + data_dir as seed
                for name in missing:
                    seed = abs(hash((name, self.data_dir))) % len(models)
                    assignment[name] = models[seed]
                # Persist updated assignment
                with path.open("w", encoding="utf-8") as f:
                    json.dump(dict(sorted(assignment.items())), f, indent=2, ensure_ascii=False)
                self.logger.info(f"Updated model_assignment.json with {len(missing)} new agents")
            # Warn if config models differ from persisted assignment
            assigned_models = sorted(set(assignment.values()))
            config_models = sorted(self._role_models)
            if assigned_models != config_models:
                self.logger.warning(
                    f"Model assignment locked from previous run: {assigned_models}. "
                    f"Current config role_model={config_models} is ignored."
                )
            return assignment

        # New run: assign models uniformly with deterministic seed
        rng = random.Random(self.data_dir)
        models = self._role_models
        # Shuffle to avoid alphabetical bias (e.g., first N agents all get model-a)
        shuffled_names = list(names)
        rng.shuffle(shuffled_names)
        assignment = {
            name: models[i % len(models)] for i, name in enumerate(shuffled_names)
        }

        # Persist
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            # Sort by name for readability
            json.dump(dict(sorted(assignment.items())), f, indent=2, ensure_ascii=False)
        self.logger.info(
            f"Assigned {len(names)} agents to {len(models)} model(s): "
            + ", ".join(
                f"{m}={sum(1 for v in assignment.values() if v == m)}" for m in models
            )
        )
        return assignment

    # Checkpoint & Resume -----------------------------------------------------
    def _checkpoint_path(self) -> Path:
        return Path("data") / self.data_dir / "checkpoint.json"

    def _read_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Read checkpoint.json. Returns None if not found or corrupted."""
        p = self._checkpoint_path()
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                cp = json.load(f)
            # Validate required fields
            _ = cp["year"], cp["week"]
            return cp
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Corrupted checkpoint.json, ignoring: {e}")
            return None

    def _write_checkpoint(self, year: int, week: int) -> None:
        """Write checkpoint.json atomically (tmp + rename).

        Progression:
        - After year-start: {"year": Y, "week": 0}
        - After week W: {"year": Y, "week": W}
        - After year-end: {"year": Y+1, "week": 0} (advance to next year)
        """
        data = {"year": year, "week": week}
        p = self._checkpoint_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(p)

    def _resolve_resume_point(
        self, resume_from: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Determine (resume_year, resume_week) for the run loop.

        Priority:
        1. --resume-from override
        2. checkpoint.json auto-detection
        3. Fresh start from config start_year, week 1

        Checkpoint semantics: {"year": Y, "week": W}
        - week == 0: year-start done, no week completed → resume from (Y, 1)
        - week < n_week: week W done → resume from (Y, W+1)
        - week == n_week: all weeks done, year-end may not have completed
          → resume from (Y, n_week) to re-run last week + year-end
          (cheap: cache hit on the week, ensures year-end completes)
        """
        start_year = self.config["time"]["start_year"]

        if resume_from is not None:
            return resume_from

        cp = self._read_checkpoint()
        if cp is not None:
            n_week = self.config["time"]["n_week"]
            year = cp["year"]
            week = cp["week"]
            if week == 0:
                return (year, 1)
            elif week < n_week:
                return (year, week + 1)
            else:
                # week == n_week: year-end may not have completed
                return (year, week)

        return (start_year, 1)

    # Utilities --------------------------------------------------------------
    def by_name(self) -> Dict[str, RoleAgent]:
        """Return cached mapping from character name to agent."""
        return self._name2agent

    def _collect_existing_schedules(self) -> Dict[str, Dict[str, Schedule]]:
        """Collect existing schedules for conflict detection.

        Returns schedules created before current time that are scheduled for
        future (including current week). Used to detect conflicts when confirming
        new joint activities.

        Returns:
            Dict mapping person -> activity_time (str) -> Schedule
        """
        t = self.clock.get_time()
        result: Dict[str, Dict[str, Schedule]] = {}

        for agent in self.agents:
            # Reuse get_future_schedules which handles the scheduling window
            schedules = agent.dm.get_future_schedules()
            for schd in schedules:
                # Only include schedules created before current time
                # (schedules created this week are handled by confirm_schedule itself)
                if schd.time is None or schd.time >= t:
                    continue
                act_time = str(schd.activity_time)
                if agent.name not in result:
                    result[agent.name] = {}
                result[agent.name][act_time] = schd

        return result

    def build_all_agents_summary(self) -> str:
        """Build a summary of all agents for GuardModel context.

        Returns:
            A formatted string with each agent's profile summary.
        """
        lines = []
        for agent in self.agents:
            lines.append(f"### {agent.name}\n{agent.dm.get_brief_intro()}")
        return "\n\n".join(lines)

    # Public Events Persistence -----------------------------------------------
    def _public_events_path(self) -> Path:
        """Return path to public_events.jsonl."""
        return Path("data") / self.data_dir / "public_events.jsonl"

    def _save_public_events(self, events: List[PublicEvent]) -> None:
        """Append public events to file (with time field for cleanup support)."""
        path = self._public_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        time_str = str(self.clock.get_time())
        with open(path, "a", encoding="utf-8") as f:
            for evt in events:
                d = {"time": time_str, **evt.to_dict()}
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _load_public_events(self) -> Dict[str, PublicEvent]:
        """Load public events from file, filtering out expired ones.

        File is append-only (oldest first), so we read backwards.
        Once we hit an event older than max_repeat_weeks, we can stop.

        Returns:
            Dict of event_id -> PublicEvent for all non-expired events.
        """
        import json

        path = self._public_events_path()
        if not path.exists():
            return {}

        t = self.clock.get_time()
        n_weeks_per_year = self.config["time"]["n_week"]
        current_absolute_week = t.year * n_weeks_per_year + t.week
        max_repeat_weeks = self.config["public_activity"]["max_repeat_weeks"]

        # Read all lines, then iterate backwards
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        events: Dict[str, PublicEvent] = {}
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            evt = PublicEvent.from_dict(d)

            # Early exit: if event is older than max_repeat_weeks, all previous are older
            start_absolute_week = evt.start_year * n_weeks_per_year + evt.start_week
            if current_absolute_week - start_absolute_week >= max_repeat_weeks:
                break

            # Check if expired based on actual repeat_weeks
            end_absolute_week = start_absolute_week + evt.repeat_weeks
            if current_absolute_week < end_absolute_week:
                events[evt.event_id] = evt
        return events

    SKILL_DECAY_RATE = 0.05  # 5% per week for unused skills

    def _apply_fulfillment_decay(self) -> None:
        """Apply proportional fulfillment decay: value * (1 - ratio) per dimension."""
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="fulfillment")
        decay_ratio = self.config["fulfillment_decay_min_ratio"]

        for agent in self.agents:
            fulfillment = agent.dm.get_fulfillment()
            decays = {
                key: int(value * decay_ratio[key]) for key, value in fulfillment.items()
            }

            if verify_logger:
                for key in decays:
                    new_val = max(0, fulfillment[key] - decays[key])
                    verify_logger.info(
                        f"[VERIFY-FULFILLMENT] {agent.name}.{key}: "
                        f"{fulfillment[key]} → {new_val} (-{decays[key]})"
                    )
            agent.dm.apply_fulfillment_decay(decays)

    def _settle_weekly_income(self) -> None:
        """Distribute weekly_income to all agents at the start of each week.

        REQ-10: Income has two sources.
        total_income = position_income + extra_income
        """
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="economy")

        if verify_logger:
            verify_logger.info(
                "[VERIFY-ECONOMY] Distributing weekly_income to all agents"
            )

        for agent in self.agents:
            profile = agent.dm.read_profile()
            position = profile["position"]
            position_income = position["weekly_income"]
            extra_income = profile["extra_income"]
            total_income = position_income + extra_income

            if total_income > 0:
                current_deposit = agent.dm.get_deposit()
                new_deposit = current_deposit + total_income
                agent.dm.update_deposit(new_deposit)
                self.logger.info(
                    f"{agent.name} received weekly_income {total_income}, deposit: {current_deposit} -> {new_deposit}"
                )
                if verify_logger:
                    verify_logger.info(
                        f"[VERIFY-ECONOMY] {agent.name}: total_income={total_income} "
                        f"(position={position_income}, extra={extra_income}), "
                        f"deposit {current_deposit} → {new_deposit}"
                    )
            elif verify_logger:
                verify_logger.warning(f"[VERIFY-ECONOMY] {agent.name} has zero income")

    def _apply_skill_decay(self) -> None:
        """Apply weekly skill decay with frequency-based rates.

        Decay rates per usage tier:
        - 0 uses:    5% decay (full decay)
        - 5-10 uses: 2% decay (slight maintenance)
        - 10+ uses:  no decay (skill maintained)
        """
        for agent in self.agents:
            state = agent.dm.read_state()
            skills = state.get("skills", {})
            if not skills:
                continue

            used_counts = getattr(agent.dm, "_skills_used_this_week", {})

            decayed = {}
            for skill_name, value in skills.items():
                use_count = used_counts.get(skill_name, 0)
                if use_count < 5:
                    decay_rate = 0.05      # 0-4 uses → 5% decay (barely used)
                elif use_count < 10:
                    decay_rate = 0.02      # 5-9 uses → 2% decay (slight decay)
                else:
                    continue               # 10+ uses → no decay (well maintained)
                new_value = max(0, int(value * (1 - decay_rate)))
                decayed[skill_name] = new_value

            if decayed:
                from src.utils import get_verify_logger
                verify_logger = get_verify_logger(feature="skill_decay")
                if verify_logger:
                    verify_logger.info(
                        f"[VERIFY-SKILL-DECAY] {agent.name}: {len(decayed)} skills decayed"
                    )
                agent.dm.apply_skill_decay(decayed)

            # Reset tracking for next week
            agent.dm._skills_used_this_week = {}

    def _apply_familiarity_decay(self) -> None:
        """Apply weekly familiarity decay with frequency-based rates.

        Decay rates per contact tier:
        - 0 contacts:      5% decay (no interaction → rapid fade)
        - 5-9 contacts:    2% decay (occasional → slight fade)
        - 10+ contacts:    no decay (frequent contact → maintained)
        """
        for agent in self.agents:
            contacts_this_week = getattr(agent.dm, "_contacts_this_week", {})

            char_dir = agent.dm.character_scratchpads
            if not char_dir.exists():
                continue

            for sp_file in char_dir.iterdir():
                if not sp_file.suffix == ".jsonl":
                    continue
                target = sp_file.stem
                contact_count = contacts_this_week.get(target, 0)

                if contact_count >= 10:
                    continue  # 10+ contacts → no decay

                # Read current familiarity total
                entries = agent.dm._read_jsonl(sp_file, max_lines=50)
                fam_deltas = [e.get("familiarity_delta", 0) for e in entries if "familiarity_delta" in e]
                if not fam_deltas:
                    continue
                current_fam = sum(fam_deltas)
                if current_fam <= 0:
                    continue

                if contact_count < 5:
                    decay_rate = 0.05      # 0-4 contacts → 5% decay
                else:
                    decay_rate = 0.02      # 5-9 contacts → 2% decay

                decay = int(current_fam * decay_rate)
                if decay > 0:
                    sp_path = char_dir / f"{target}.jsonl"
                    agent.dm._append_jsonl(sp_path, {
                        "content": "familiarity_decay",
                        "familiarity_delta": -decay,
                    })

    def _before_week_start(self) -> None:
        """Execute all operations that should happen before each week starts."""
        # Record each agent's deposit at week start (for weekly economy reward)
        for agent in self.agents:
            agent._week_start_deposit = agent.dm.get_deposit()

        self._apply_fulfillment_decay()
        self._settle_weekly_income()
        self._apply_skill_decay()
        self._apply_familiarity_decay()
        # Reset contact tracking for next week
        for agent in self.agents:
            agent.dm._contacts_this_week = {}

    def _run_position_application_season(self) -> None:
        """Run position application season at year end.

        Calls run_position_application_season() from position_application module to handle
        the 1.5-round position application process.
        """
        from src.world.position_application import run_position_application_season
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="position_application")
        t = self.clock.get_time()

        # position_store is initialized in World.__init__, use it directly
        positions = self.position_store.get_all()

        self.logger.info(
            f"== POSITION APPLICATION SEASON == year={t.year} positions={len(positions)}"
        )

        # Run position application
        results = run_position_application_season(self, self.clock, parallel=self.parallel)

        # Log summary (single pass)
        accepted = sum(1 for r in results.values() if r.get("accepted"))
        unemployed = len(results) - accepted

        self.logger.info(
            f"[POSITION_APPLICATION] Complete: {accepted} accepted, {unemployed} unemployed"
        )

    def _merge_profile_positions(self) -> None:
        """Merge positions from agent profiles into PositionStore.

        Agents' initial positions (from profile templates) may not be in
        positions.json if god_design_positions() missed them. This method
        scans all agent profiles, identifies missing positions, and adds
        them with created_year=-1 (pre-simulation).
        """
        from src.world.position_application import Position

        # Scan profiles: group agents by their position name
        pos_agents: Dict[str, List[str]] = {}  # pos_name -> [agent_names]
        pos_data: Dict[str, Dict] = {}  # pos_name -> profile position data

        for agent in self.agents:
            profile = agent.dm.read_profile()
            pos = profile["position"]
            name = f"{pos['organization']}/{pos['role']}"
            pos_agents.setdefault(name, []).append(agent.name)
            if name not in pos_data:
                pos_data[name] = pos

        # Add missing positions with created_year=-1
        added = 0
        for pos_name in sorted(pos_agents.keys()):
            if self.position_store.get(pos_name) is not None:
                continue
            data = pos_data[pos_name]
            agents_holding = pos_agents[pos_name]
            org, role = Position.parse_name(pos_name)
            new_pos = Position(
                organization=org,
                role=role,
                type=data["type"],
                description=data.get("description", ""),
                weekly_income=data["weekly_income"],
                weekly_delta_skills=data["weekly_delta_skills"],
                capacity=len(agents_holding),
                occupied_by=sorted(agents_holding),
                created_year=-1,
            )
            self.position_store.add(new_pos)
            added += 1
            self.logger.info(
                f"[POSITIONS] Merged from profile: {pos_name} "
                f"(capacity={len(agents_holding)}, created_year=-1)"
            )

        if added:
            self.logger.info(
                f"[POSITIONS] Merged {added} positions from agent profiles"
            )

    def _grow_positions(self, current_year: int, is_first_year: bool) -> None:
        """Grow positions at the start of each year.

        - First year: Set created_year for all initial positions
        - Subsequent years: Generate new challenging positions

        Args:
            current_year: The current simulation year
            is_first_year: Whether this is the first year of simulation
        """
        from src.world.god import god_grow_positions
        from src.world.position_application import Position
        from src.agents.prompts import get_world_setting
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="position_application")

        if is_first_year:
            # First year: set created_year for all existing positions
            for pos in self.position_store.get_all():
                if pos.created_year is None:
                    pos.created_year = current_year

            # Merge initial positions from agent profiles that aren't in the store.
            # These are positions agents start with but god_design_positions() missed.
            self._merge_profile_positions()

            self.position_store.save()
            self.logger.info(
                f"[POSITIONS] Year {current_year}: initialized {self.position_store.count()} positions"
            )
            return

        # Subsequent years: generate new challenging positions
        # Remove any positions already created for this year (idempotent on resume)
        removed = self.position_store.remove_by_created_year(current_year)
        if removed:
            self.position_store.save()
            self.logger.info(
                f"[POSITIONS] Year {current_year}: removed {removed} stale positions (resume)"
            )

        # Formula: max(2, N/10) where N = initial position count (stable, not compounding)
        target_count = max(2, self._initial_position_count // 10)

        world_setting = get_world_setting(self.data_dir)
        existing_positions = self.position_store.get_all()

        self.logger.info(
            f"[POSITIONS] Year {current_year}: requesting {target_count} new challenging positions"
        )

        new_positions_data = god_grow_positions(
            agents=self.agents,
            world_setting=world_setting,
            existing_positions=existing_positions,
            count=target_count,
            created_year=current_year,
        )

        if new_positions_data:
            new_positions = [Position.from_dict(d) for d in new_positions_data]
            self.position_store.add_positions(new_positions)
            self.position_store.save()

            if verify_logger:
                for pos in new_positions:
                    verify_logger.info(
                        f"[POSITIONS] Added: {pos.name} (income={pos.weekly_income}, "
                        f"min_skills={pos.min_skills}, capacity={pos.capacity})"
                    )

        self.logger.info(
            f"[POSITIONS] Year {current_year}: added {len(new_positions_data) if new_positions_data else 0}, "
            f"total now {self.position_store.count()}"
        )

    # Execution --------------------------------------------------------------
    def run(self) -> None:
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="world")

        start_year = self.config["time"]["start_year"]
        total_years = self.config["time"]["n_year"]
        n_week = self.config["time"]["n_week"]
        resume_year = self._resume_year
        resume_week = self._resume_week

        for y in range(total_years):
            current_year = start_year + y

            if current_year < resume_year:
                continue

            self.clock.set_year(current_year)

            # Year-start: run _grow_positions unless resuming mid-year
            # resume_week > 1 means year-start already completed for resume_year
            need_year_start = (current_year > resume_year) or (resume_week <= 1)
            if need_year_start:
                self._grow_positions(
                    current_year, is_first_year=(current_year == start_year)
                )
                self._write_checkpoint(current_year, 0)

            for week in range(1, n_week + 1):
                if current_year == resume_year and week < resume_week:
                    continue

                self.clock.set_week(week)
                self.step()

                # Weekly reward: set day=6 to distinguish from daily D01-D05
                self.clock.set_day(6)
                self._calculate_weekly_rewards(week)

                self._write_checkpoint(current_year, week)

            # Year-end: set day=99 so yearly reward sorts after all daily/weekly
            self.clock.set_day(99)
            self._calculate_yearly_rewards()

            # Year-end (weekly rewards already calculated in the weekly loop)
            self._update_yearly_profiles()
            self._run_position_application_season()

            # ── P203: Yearly affection decay ──
            decay = 0
            for ag in self.agents:
                if ag.name == "Player":
                    continue
                try:
                    decay += ag.dm.apply_affection_decay(decay_factor=0.95)
                except Exception:
                    pass
            if decay:
                self.logger.info(f"[P203] Affection decay: {decay} scratchpad files")

            # Advance checkpoint to next year (year-end complete)
            self._write_checkpoint(current_year + 1, 0)

        if verify_logger:
            verify_logger.info(
                "[VERIFY-COMPLETE] World simulation completed successfully"
            )

    def step(self) -> None:
        self.clock.advance_step_cycle()
        # Reset stage to BEGIN before _before_week_start, so that save_state
        # calls inside it (fulfillment decay, weekly income) don't inherit
        # the stale SETTLE/REVIEW stage from the previous step() iteration.
        self.clock.set_stage(Stage.BEGIN)
        # Stage 0: before week start (fulfillment decay + weekly income)
        self._before_week_start()
        for agent in self.agents:
            agent.clear_on_week_start()

        # Stage 1: plan
        self.clock.set_stage(Stage.PLAN)
        t = self.clock.get_time()
        self.logger.info(f"== PLAN STAGE == year={t.year} week={t.week}")
        if self.parallel:
            # Player agent uses terminal input() — must run on main thread
            npc_agents = [a for a in self.agents if a.name != "Player"]
            player_agent = self._name2agent.get("Player")
            if player_agent:
                player_agent.plan()
            if npc_agents:
                with ThreadPoolExecutor(max_workers=pool_size(len(npc_agents))) as ex:
                    list(ex.map(lambda a: a.plan(), npc_agents))
        else:
            for agent in self.agents:
                agent.plan()

        # Stage 2: before_contact (God Model generates events → Agents respond)
        self.clock.set_stage(Stage.BEFORE_CONTACT)
        t = self.clock.get_time()
        self.logger.info(f"== BEFORE_CONTACT STAGE == year={t.year} week={t.week}")

        # God Model: generate public events for this week
        this_week_events = self._generate_public_events()
        # Store for PlayerAgent contact() display (T03)
        self.public_activities = this_week_events

        # Agents: sign up for public events (parallel)
        if this_week_events:
            if self.parallel:
                with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                    list(
                        ex.map(
                            lambda a: a.signup_public_events(this_week_events),
                            self.agents,
                        )
                    )
            else:
                for a in self.agents:
                    a.signup_public_events(this_week_events)
        else:
            self.logger.info("No public events available this week")

        # Stage 3: contact
        self.clock.set_stage(Stage.CONTACT)
        # start of contact phase: clear per-week msg queue
        self.msg_center.clear()

        # T02: Inject Player's pending invites into target NPCs' inbox
        player_agent = self._name2agent.get("Player")
        if player_agent is not None:
            # T03: Pass public activities to PlayerAgent for display
            if hasattr(player_agent, "set_public_activities"):
                player_agent.set_public_activities(
                    getattr(self, "public_activities", None)
                )
            # T03: Inject pending invites — removed (PlayerAgent no longer stores _pending_invites)

        for slot in range(1, self.config["time"]["n_contact_slot"] + 1):
            self.clock.set_slot(slot)
            t = self.clock.get_time()
            self.logger.info(
                f"== CONTACT STAGE == year={t.year} week={t.week} slot={t.slot}"
            )
            if self.parallel:
                # Player agent uses terminal input() — must run on main thread
                npc_agents = [a for a in self.agents if a.name != "Player"]
                player_agent = self._name2agent.get("Player")
                if player_agent:
                    player_agent.contact()
                if npc_agents:
                    with ThreadPoolExecutor(
                        max_workers=pool_size(len(npc_agents))
                    ) as ex:
                        list(ex.map(lambda a: a.contact(), npc_agents))
            else:
                for agent in self.agents:
                    agent.contact()

        # Stage 4: after_contact
        self.clock.set_stage(Stage.AFTER_CONTACT)
        self.logger.info(f"== AFTER CONTACT STAGE == year={t.year} week={t.week}")

        # Collect existing schedules for conflict detection
        # (schedules created in previous weeks for future days)
        existing_schedules = self._collect_existing_schedules()

        # Confirm joint schedules from contact messages
        self.msg_center.confirm_schedule(existing_schedules=existing_schedules)

        # Agents: finalize contact (writes joint schedules to agent.dm)
        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(lambda a: a.finalize_contact(), self.agents))
        else:
            for agent in self.agents:
                agent.finalize_contact()

        # Solo schedule fallback: for days without joint/public, write solo schedule
        t = self.clock.get_time()
        n_days = self.config["time"]["n_day"]
        public_locs, _ = self.location_store.list_all()
        for agent in self.agents:
            if agent.name == "Player":
                continue
            for day in range(1, n_days + 1):
                schd = agent.dm.get_schedule_for_day(t.year, t.week, day)
                if schd is not None:
                    continue  # already has a schedule for this day
                # Decide location: 50% home, 50% random public
                if random.random() < 0.5:
                    location = f"home/{agent.name}"
                else:
                    location = random.choice(sorted(public_locs)) if public_locs else f"home/{agent.name}"
                activity_time = TimeState(year=t.year, week=t.week, stage=Stage.ACTIVITY, day=day)
                solo_schd = Schedule(
                    activity_id=make_activity_id("solo", activity_time, agent.name),
                    activity_name="Solo Activity",
                    activity_time=activity_time,
                    location=location,
                    type="solo",
                    status="created",
                    participants=[agent.name],
                )
                agent.dm.add_schedule(solo_schd)

        # Stage 5: activity (day × phase double loop)
        self.clock.set_stage(Stage.ACTIVITY)

        # Initialise WorldState for the ACTIVITY stage lifecycle
        agent_names = [a.name for a in self.agents]
        self.world_state = WorldState(agent_names)
        n_phases = len(self.clock.get_phases())
        self.logger.info(f"Day phases: {n_phases} phase(s) per day")

        for day in range(1, self.config["time"]["n_day"] + 1):
            self.clock.set_day(day)

            for phase in self.clock.get_phases():
                self.clock.set_phase(phase)
                self.world_state.set_phase(phase)
                t = self.clock.get_time()

                # --- Phase: update positions from schedules ---
                for agent in self.agents:
                    schd = agent.get_schedule()
                    if schd is not None and schd.location:
                        self.world_state.update_position(agent.name, schd.location)
                    elif self.world_state.get_position(agent.name) is None:
                        # Idle agent: assign random public location
                        public_locs, _ = self.location_store.list_all()
                        if public_locs:
                            self.world_state.update_position(
                                agent.name, random.choice(sorted(public_locs))
                            )

                tracked = len(self.world_state._positions)

                # ── P202-B: Player priority encounter (before NPC detection) ──
                # Player gets to choose: plan/search/social. If Player chooses
                # social, we create a manual EncounterGroup. Player is excluded
                # from the NPC encounter pool regardless.
                name2agent_phase = self.by_name()
                player_agent_phase = name2agent_phase.get("Player")
                player_social_target = None
                player_used_menu = False  # True if Player chose search/social (skip normal schedule)
                if player_agent_phase:
                    player_pos = self.world_state.get_position("Player")
                    player_schd = player_agent_phase.get_schedule()
                    scheduled = ""
                    if player_schd and player_schd.activity_name:
                        scheduled = f"{player_schd.activity_name} ({player_schd.type})"

                    # Home: skip activity menu — treat as normal solo/joint
                    is_at_home = player_pos and player_pos.startswith("home/")
                    if not is_at_home:
                        from src.world.clock import DayPhase
                        phase_label = DayPhase.label(phase) if phase else "?"
                        choice = player_agent_phase._show_activity_menu(
                            phase_label, player_pos or "unknown", scheduled
                        )
                    else:
                        choice = "plan"  # at home: follow schedule like NPC

                    if choice == "social":
                        # Use positions snapshot BEFORE Player is removed
                        pos_snapshot = self.world_state.get_phase_positions()
                        player_social_target = player_agent_phase._do_social(
                            pos_snapshot, player_pos or "unknown"
                        )
                    elif choice == "search":
                        # P202-A: Search current location for items
                        player_agent_phase._do_search(player_pos or "unknown")
                        player_used_menu = True

                    # Exclude Player from NPC encounter pool
                    # (remove from all location lists in _positions)
                    ws = self.world_state
                    for loc in list(ws._positions.keys()):
                        if "Player" in ws._positions[loc]:
                            ws._positions[loc].remove("Player")
                            if not ws._positions[loc]:
                                del ws._positions[loc]

                # --- Phase: detect encounters (NPC-only) ---
                encounters = self.world_state.detect_encounters()

                # If Player chose social, inject manual EncounterGroup
                if player_social_target:
                    from src.world.state import EncounterGroup as EG, DayPhase
                    manual_group = EG(
                        location=player_pos or "unknown",
                        phase=phase,
                        agent_names=["Player", player_social_target],
                    )
                    encounters.insert(0, manual_group)
                    self.logger.info(
                        f"[P202] Player manually triggered encounter with {player_social_target} "
                        f"at {player_pos}"
                    )

                self.logger.info(
                    f"[ACTIVITY] day={t.day}/5 phase={t.phase.name.lower() if t.phase else '?'} | "
                    f"tracked={tracked} agents "
                    f"| encounters={len(encounters)}"
                )

                # --- Phase: execute regular activities ---
                # (per-phase activity building — runs each phase for
                #  phase-specific activity construction)
                public_acts, joint_acts, encounter_acts, solo_acts = (
                    self._build_today_activities_all_types()
                )

                # ── Optimization: Solo batch processing by location ──
                # Group solo agents at the same location → 1 LLM → individual outcomes
                try:
                    solo_by_loc: dict[str, list] = {}
                    solo_keep: list = []
                    for act in solo_acts:
                        agent = act.agents[0]
                        loc = self.world_state.get_position(agent.name)
                        if loc and loc in solo_by_loc:
                            solo_by_loc[loc].append(act)
                        elif loc:
                            solo_by_loc[loc] = [act]
                        else:
                            solo_keep.append(act)

                    batch_count = 0
                    for loc, acts in list(solo_by_loc.items()):
                        if len(acts) < 2:
                            solo_keep.extend(acts)
                            del solo_by_loc[loc]
                            continue
                        batch_count += 1
                        # Build shared context
                        agent_names = sorted(a.agents[0].name for a in acts)
                        sched_names = [a.agents[0].get_schedule() for a in acts]
                        sched_strs = [
                            f"{sn.agents[0].name}: {sn.activity_name}"
                            if sn and hasattr(sn, 'activity_name') else f"{sn.agents[0].name}: solo"
                            for sn in acts
                        ]

                        prompt = (
                            f"A group of people are at {loc}. Describe what happens "
                            f"as each person goes about their own activity:\n"
                            + "\n".join(f"  - {s}" for s in sched_strs)
                            + "\n\nOutput JSON: {\"scene\": \"brief scene description\", "
                            "\"individuals\": {\"name\": \"their activity detail\", ...}}"
                        )

                        from src.utils import get_response_with_retry, num_tokens_from_string
                        from src.config import get_config
                        cfg = get_config()
                        model = cfg.get("role_model", "")
                        if isinstance(model, list):
                            model = model[0]
                        if model:
                            raw = get_response_with_retry(
                                model=model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=256,
                                temperature=0.5,
                            )
                            import json
                            try:
                                results = json.loads(raw)
                                scene = results.get("scene", "")
                                individuals = results.get("individuals", {})
                                for act in acts:
                                    agent = act.agents[0]
                                    detail = individuals.get(agent.name, scene)
                                    if detail and not hasattr(agent, '_batched_solo_outcome'):
                                        agent._batched_solo_outcome = detail
                            except json.JSONDecodeError:
                                pass

                    self.logger.info(
                        f"[BATCH] Solo batch: {batch_count} locations merged, "
                        f"{len(solo_keep)} solo agents kept individual"
                    )
                    solo_acts = solo_keep + (
                        [a for acts in solo_by_loc.values() for a in acts]
                        if False else [a for acts in solo_by_loc.values() for a in acts]
                    )
                except Exception:
                    self.logger.warning("[BATCH] Solo batch optimization failed", exc_info=True)
                # Execute all activity types in parallel with
                # Semaphore-based concurrency control
                #
                # Design:
                # - Semaphore(max_concurrency) controls total concurrent tasks
                # - Joint/Solo: 1 slot each (one concurrent task)
                # - Public: N slots where N = min(participants, internal_parallelism)
                #
                # Submission order (priority):
                # 1. Joint (bottleneck, slow, gets slots first)
                # 2. Solo (fast, fills remaining slots)
                # 3. Public (sorted by size, small first to release slots quickly)
                if self.parallel:
                    from src.config import get_config
                    from threading import Semaphore

                    cfg = get_config()
                    max_concurrency = int(cfg["max_concurrency"])

                    n_joint = len(joint_acts) + len(encounter_acts)
                    n_solo = len(solo_acts)
                    n_public = len(public_acts)

                    # Public internal parallelism from config (must be > 0)
                    public_internal_parallelism = int(
                        self.config["public_activity"]["internal_parallelism"]
                    )
                    if public_internal_parallelism <= 0:
                        self.logger.error(
                            f"internal_parallelism must be > 0, "
                            f"got {public_internal_parallelism}, forcing to 5"
                        )
                        public_internal_parallelism = 5

                    self.logger.debug(
                        f"Activity parallel: joint={n_joint}, solo={n_solo}, "
                        f"public={n_public}, public_internal={public_internal_parallelism}, "
                        f"pool={max_concurrency}"
                    )

                    # Semaphore controls total concurrent tasks
                    capacity = Semaphore(max_concurrency)

                    def run_with_slots(fn, slots: int, *args, **kwargs):
                        """Run function while holding `slots` semaphore permits."""
                        for _ in range(slots):
                            capacity.acquire()
                        try:
                            return fn(*args, **kwargs)
                        finally:
                            for _ in range(slots):
                                capacity.release()

                    # Thread pool large enough for all tasks to be submitted
                    total_tasks = n_joint + n_solo + n_public

                    if total_tasks == 0:
                        self.logger.info(
                            f"[ACTIVITY] No activities to execute in this phase, skipping"
                        )
                    else:
                        with ThreadPoolExecutor(max_workers=total_tasks) as ex:
                            futures = []

                            # Phase 1: Joint (1 slot each, highest priority)
                            for act in joint_acts + encounter_acts:
                                futures.append(ex.submit(run_with_slots, act.run, 1))

                            # Phase 2: Solo (1 slot each, fast)
                            # ── P203: Solo 感知同位置 NPC ──
                            phase_positions = self.world_state.get_phase_positions()
                            for act in solo_acts:
                                ag_name = act.agents[0].name
                                pos = self.world_state.get_position(ag_name)
                                if pos:
                                    all_nearby = phase_positions.get(pos, [])
                                    act._nearby_names = [n for n in all_nearby if n != ag_name]
                                else:
                                    act._nearby_names = []

                            for act in solo_acts:
                                futures.append(ex.submit(run_with_slots, act.run, 1))

                            # Phase 3: Public (sorted by participant count, small first)
                            # Small Public completes faster, releases slots for others
                            sorted_public = sorted(public_acts, key=lambda a: len(a.agents))
                            for act in sorted_public:
                                n_participants = len(act.agents)
                                slots = min(n_participants, public_internal_parallelism)
                                futures.append(
                                    ex.submit(run_with_slots, act.run, slots, parallel=True)
                                )

                            # Wait for all
                            for f in futures:
                                f.result()
                else:
                    # Sequential fallback
                    # ── P203: Solo 感知同位置 NPC (sequential path) ──
                    phase_positions = self.world_state.get_phase_positions()
                    for act in solo_acts:
                        ag_name = act.agents[0].name
                        pos = self.world_state.get_position(ag_name)
                        if pos:
                            all_nearby = phase_positions.get(pos, [])
                            act._nearby_names = [n for n in all_nearby if n != ag_name]
                        else:
                            act._nearby_names = []

                    for act in joint_acts + encounter_acts:
                        act.run()
                    for act in public_acts:
                        act.run(parallel=False)
                    for act in solo_acts:
                        act.run()

                # --- Phase: execute encounter dialogues via EncounterPipeline ---
                # (detection already done above; now execute the actual LLM dialogs)

                # Filter encounter groups: only groups with at least 1 solo NPC participate.
                # Solo→solo: normal encounter. Solo→joint/public/encounter: post-activity.
                # Groups with zero solo NPCs (all busy) are discarded.
                from src.world.state import EncounterGroup as EG
                filtered_encounters = []
                for enc_group in encounters:
                    has_solo = False
                    for nm in enc_group.agent_names:
                        agent = self._name2agent.get(nm)
                        if not agent:
                            continue
                        schd = agent.get_schedule()
                        if schd and schd.type == "solo":
                            has_solo = True
                            break
                    if not has_solo:
                        continue
                    filtered_encounters.append(EG(
                        location=enc_group.location,
                        phase=enc_group.phase,
                        agent_names=list(enc_group.agent_names),
                    ))
                encounters = filtered_encounters

                from src.world.encounter_pipeline import EncounterPipeline

                pipeline = EncounterPipeline()
                name2agent = self.by_name()
                player_agent = name2agent.get("Player")
                for enc_group in encounters:
                    if len(enc_group.agent_names) < 2:
                        continue

                    pipeline.run(
                        enc_group,
                        name2agent,
                        self.world_state,
                        self.location_store,
                        player_agent=player_agent,
                    )

                    # ── Post-Encounter Follow: NPC-only 偶遇后跟随决策 ──
                    if "Player" not in enc_group.agent_names:
                        from src.world.encounter_pipeline import EncounterPipeline
                        # 收集 NPC 的对话文本
                        npc_agents_in_enc = [
                            self._name2agent[n] for n in enc_group.agent_names
                            if n in self._name2agent
                        ]
                        dialogue_text = ""
                        for npc in npc_agents_in_enc:
                            if npc.activity_context:
                                for msg in npc.activity_context:
                                    role = msg.get("role", "")
                                    content = msg.get("content", "")
                                    if role in ("user", "assistant") and content:
                                        speaker = npc.name if role == "assistant" else "System"
                                        dialogue_text += f"[{speaker}]: {content}\n"
                                break
                        if not dialogue_text:
                            dialogue_text = f"An encounter between {', '.join(enc_group.agent_names)} took place."

                        follow_pipeline = EncounterPipeline()
                        follow_pipeline._apply_post_encounter_follow(
                            enc_group, self._name2agent, self.world_state,
                            npc_agents_in_enc, dialogue_text,
                        )

                # --- Phase: advance actions ---
                self.world_state.advance_actions(1.0 / n_phases)

            # --- End of day: daily reward ---
            reward_cfg = self.config["reward"]
            if reward_cfg.get("granularity", "weekly") == "daily":
                self._calculate_daily_rewards(day)

        # Stage complete — discard WorldState
        self.world_state = None

        # Review phase
        self.clock.set_stage(Stage.REVIEW)
        t = self.clock.get_time()
        self.logger.info(f"== REVIEW STAGE == year={t.year} week={t.week}")
        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(lambda a: a.review(), self.agents))
        else:
            for agent in self.agents:
                agent.review()

        # Settle phase (weekly cleanup)
        self.clock.set_stage(Stage.SETTLE)
        t = self.clock.get_time()
        self.logger.info(f"== SETTLE STAGE == year={t.year} week={t.week}")
        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(lambda a: a.settle_week(), self.agents))
        else:
            for agent in self.agents:
                agent.settle_week()

        # ── F2: SETTLE stage 对所有 NPC 应用 fidelity 衰减 ──
        try:
            from src.config import get_world_config
            rumor_cfg = get_world_config().get("rumor", {})
            if rumor_cfg.get("fidelity_decay_per_day", 0.05) > 0:
                self.logger.info("[F2] Applying fidelity decay to all NPCs")
                for agent in self.agents:
                    try:
                        agent.dm._apply_fidelity_decay()
                    except Exception:
                        self.logger.warning(
                            f"[F2] Fidelity decay failed for {agent.name}",
                            exc_info=True,
                        )
        except Exception:
            self.logger.warning("[F2] Fidelity decay phase failed", exc_info=True)

        # ── Compress rumors: keep only most recent rumors ──
        try:
            self.logger.info("[F2] Compressing rumors for all NPCs")
            for agent in self.agents:
                try:
                    agent.dm.compress_rumors(keep_recent=100)
                except Exception:
                    self.logger.warning(
                        f"[F2] Rumor compression failed for {agent.name}",
                        exc_info=True,
                    )
        except Exception:
            self.logger.warning("[F2] Rumor compression phase failed", exc_info=True)

    # Internal --------------------------------------------------------------
    # def _build_today_activities(self) -> tuple[list[JointActivity], list[SoloActivity]]:
    #     """Build today's joint and solo activities based on current clock time."""
    #     t = self.clock.get_time()
    #     # 1) Collect today's joint activities (deduplicated by activity_id)
    #     aid_to_schd: Dict[str, Schedule] = {}
    #     aid_to_agents: Dict[str, set[str]] = {}
    #     for a in self.agents:
    #         schd = a.dm.get_today_schedule()
    #         if not schd:
    #             # agent has no joint activity on this day
    #             continue

    #         at = schd.activity_time
    #         aid = schd.activity_id
    #         assert (at == t) and (schd.type == "joint"), (
    #             f"Invalid schedule for agent {a.name}: {schd} "
    #         )

    #         aid_to_schd.setdefault(aid, schd)
    #         aid_to_agents.setdefault(aid, set()).add(a.name)

    #     # 2) Validate JointActivity (sorted by activity_id for determinism)
    #     name2agent = self.by_name()
    #     joint_acts = []
    #     for aid, schd in sorted(aid_to_schd.items()):
    #         participants = schd.participants

    #         # Validate: 1) at least 2 people; 2) all role names are valid and have agents; 3) participants == agents that have this activity
    #         assert len(participants) >= 2
    #         assert all(n in name2agent for n in participants)

    #         assert set(participants) == aid_to_agents[aid]

    #         p_agents = [name2agent[n] for n in participants]

    #         act = JointActivity.from_schedule(
    #             schd, p_agents, location_store=self.location_store
    #         )
    #         joint_acts.append(act)

    #     # 3) Determine SoloActivity
    #     engaged: set[str] = set()
    #     for aid, schd in sorted(aid_to_schd.items()):
    #         engaged.update(schd.participants)

    #     solo_agents = [ag for ag in self.agents if ag.name not in engaged]
    #     solo_acts = [
    #         SoloActivity(
    #             activity_id=f"{ag.name}-solo-{t}",
    #             activity_name="Solo",
    #             time=t,
    #             agents=[ag],
    #         )
    #         for ag in solo_agents
    #     ]

    #     return joint_acts, solo_acts

    def _build_today_activities_all_types(
        self,
    ) -> tuple[
        list[PublicActivity],
        list[JointActivity],
        list[EncounterActivity],
        list[SoloActivity],
    ]:
        """Build today's activities for all types.

        All activity types (joint, public, encounter, solo) are read from agent schedules.
        Solo activities are built from schedule data with location, not as leftover.

        Priority handling (Encounter > Joint > Public) is done in agent.get_schedule().

        Returns:
            Tuple of (public_acts, joint_acts, encounter_acts, solo_acts)
        """
        from src.world.activity import PublicActivity

        t = self.clock.get_time()
        name2agent = self.by_name()

        # Collect schedules grouped by (type, activity_id)
        # type -> activity_id -> (schd, set of agent names)
        schedules_by_type: dict[str, dict[str, tuple[Schedule, set[str]]]] = {
            "joint": {},
            "public": {},
            "encounter": {},
            "solo": {},
        }

        engaged: set[str] = set()

        for agent in self.agents:
            schd = agent.get_schedule()
            if not schd:
                continue

            at = schd.activity_time
            # Compare ignoring phase — schedules are stored at day granularity
            # while the clock may carry phase info during ACTIVITY stage.
            at_day = TimeState(at.year, at.week, at.stage, at.day, at.slot)
            t_day = TimeState(t.year, t.week, t.stage, t.day, t.slot)
            assert at_day == t_day, (
                f"Schedule time mismatch for {agent.name}: expected {t_day}, got {at_day}"
            )

            stype = schd.type
            assert stype in schedules_by_type, (
                f"Unknown schedule type '{stype}' for {agent.name}"
            )

            # Solo schedules are collected separately at the end —
            # they don't participate in joint/encounter/public building,
            # and are only built once per day (not per phase).
            if stype == "solo":
                continue

            aid = schd.activity_id
            if aid not in schedules_by_type[stype]:
                schedules_by_type[stype][aid] = (schd, set())
            schedules_by_type[stype][aid][1].add(agent.name)
            engaged.add(agent.name)

        # Build activities by type
        def build_joint_or_encounter(stype: str) -> list:
            """Build Joint or Encounter activities with strict validation."""
            cls = EncounterActivity if stype == "encounter" else JointActivity
            acts = []
            for aid in sorted(schedules_by_type[stype].keys()):
                schd, agent_names = schedules_by_type[stype][aid]
                participants = schd.participants

                # Validation: participants in schedule must match collected agents
                assert len(participants) >= 2, (
                    f"{stype} activity {aid} has < 2 participants: {participants}"
                )
                assert all(n in name2agent for n in participants), (
                    f"{stype} activity {aid} has unknown participants: {participants}"
                )
                assert set(participants) == agent_names, (
                    f"{stype} activity {aid} mismatch: "
                    f"schd.participants={participants}, collected={agent_names}"
                )

                p_agents = [name2agent[n] for n in participants]
                act = cls.from_schedule(
                    schd, p_agents, location_store=self.location_store
                )
                acts.append(act)
            return acts

        def build_public() -> list[PublicActivity]:
            """Build Public activities (min 1 participant).

            Unlike Joint/Encounter where participants are pre-determined,
            Public participants are independently signed up - each agent's
            schedule only contains themselves in participants field.
            We aggregate all sign-ups here.

            Note: If an agent signed up for Public but also has Joint activity,
            Joint takes priority (handled in get_schedule()), so that agent
            won't appear in agent_names here - this is correct behavior.
            """
            current_phase = self.clock.get_time().phase
            default_phase = self.clock.get_phases()[0] if self.clock.get_phases() else None
            acts = []
            for aid in sorted(schedules_by_type["public"].keys()):
                schd, agent_names = schedules_by_type["public"][aid]

                # Filter by phase: only include if schedule's phase matches current phase
                # Default to default_phase if phase is None (backward compatibility)
                schd_phase = schd.activity_time.phase if schd.activity_time.phase else default_phase
                if schd_phase != current_phase:
                    continue

                if len(agent_names) < 1:
                    continue

                # Aggregate all sign-ups (each agent's schedule has participants=[self])
                participant_names = sorted(agent_names)
                participant_agents = [name2agent[n] for n in participant_names]

                # Must update participants for downstream (original only has single agent)
                schd.participants = participant_names

                act = PublicActivity.from_schedule(
                    schd, participant_agents, event_description=schd.event_description
                )
                acts.append(act)
            return acts

        public_acts = build_public()
        joint_acts = build_joint_or_encounter("joint")
        encounter_acts = build_joint_or_encounter("encounter")

        # Public activities: NPC only participates in their chosen phase.
        # (No longer spans the entire day.)
        # Engagement is handled by the activity execution below.

        # Solo for remaining agents (one per agent per day, built from schedule for location)
        solo_agents = [ag for ag in self.agents if ag.name not in engaged]
        solo_acts = []
        for ag in solo_agents:
            schd = ag.dm.get_today_schedule()
            solo_acts.append(SoloActivity(
                activity_id=schd.activity_id if schd else None,
                activity_name=(schd.activity_name if schd else None) or "Solo Activity",
                time=t,
                agents=[ag],
                schedule=schd,
                location_store=self.location_store,
            ))

        return public_acts, joint_acts, encounter_acts, solo_acts

    # Player Encounter ---------------------------------------------------------
    def _run_player_encounter(
        self,
        enc_group,
        participants: list,
        name2agent: dict,
    ) -> None:
        """Run a Player-involved encounter using input()-based dialogue.

        Instead of the God Model orchestration used in JointActivity.run(),
        this method runs a simple turn-by-turn dialogue:
        - NPC agents use their LLM (act_in_activity) to respond
        - Player uses terminal input() to respond

        Args:
            enc_group: EncounterGroup with agent_names and location
            participants: List of agent objects in the encounter
            name2agent: Dict mapping name -> agent
        """
        player_agent = name2agent.get("Player")
        npc_agents = [a for a in participants if a.name != "Player"]

        if not player_agent or not npc_agents:
            return

        cfg = get_world_config()
        max_turns = int(cfg["activity"]["joint_activity_max_turns"])
        min_turns = int(cfg["activity"]["joint_activity_min_turns"])

        location_desc = (
            self.location_store.get_surroundings_text(enc_group.location)
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
        print(f"{'='*60}")

        # Simple turn-by-turn dialogue: NPC speaks first, then Player responds,
        # then next NPC, etc.
        # Build an interleaved order: NPC, Player, NPC, Player, ...
        dialogue_order: List[str] = []
        for turn in range(max_turns):
            for npc in npc_agents:
                dialogue_order.append(npc.name)
            dialogue_order.append("Player")

        last_npc_line = ""
        last_speaker = ""

        for i, speaker_name in enumerate(dialogue_order):
            if speaker_name == "Player":
                if not last_npc_line:
                    continue  # Skip if no NPC has spoken yet
                reply = player_agent.player_dialogue(last_speaker, last_npc_line)
                if reply:
                    # Broadcast Player's reply to all NPCs
                    for npc in npc_agents:
                        npc.receive_in_activity(f"[{player_agent.name}]: {reply}")
                last_speaker = player_agent.name
            else:
                npc = name2agent[speaker_name]
                try:
                    resp = npc.act_in_activity(activity_type="joint", i_turn=i + 1)
                    # Extract clean text from resp
                    last_npc_line = resp
                    last_speaker = speaker_name

                    # Broadcast to all other participants (including Player)
                    for other in participants:
                        if other.name != speaker_name:
                            other.receive_in_activity(
                                f"[{speaker_name}]: {resp}"
                            )
                except Exception:
                    self.logger.warning(
                        f"[PLAYER ENCOUNTER] NPC {speaker_name} generation failed",
                        exc_info=True,
                    )
                    last_npc_line = f"{speaker_name} looks at you expectantly."
                    last_speaker = speaker_name
                    for other in participants:
                        if other.name != speaker_name:
                            other.receive_in_activity(
                                f"[{speaker_name}]: {last_npc_line}"
                            )

            # Allow early exit
            if i >= min_turns * len(participants) and last_speaker == "Player":
                # After player speaks, check if they want to end
                pass  # Continue until max turns or explicit exit

        # Exit dialogue for NPCs
        for npc in npc_agents:
            try:
                npc.exit_activity("joint")
            except Exception:
                self.logger.warning(
                    f"[PLAYER ENCOUNTER] exit_activity failed for {npc.name}",
                    exc_info=True,
                )

        # Clean up Player context
        player_agent.activity_context = None

    # Public Stage Methods ----------------------------------------------------
    def _generate_public_events(self) -> List[PublicEvent]:
        """God Model generates public events for this week.

        1. Load existing events from file (auto-filters expired)
        2. Generate new events via God Model
        3. Persist new events to file
        4. Return list of events active this week (as this-week instances)
        """
        from src.world.god import generate_public_events
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="public_activity")
        t = self.clock.get_time()
        n_weeks_per_year = self.config["time"]["n_week"]
        n_days = self.config["time"]["n_day"]

        # 1. Load existing events (expired ones are filtered out on read)
        public_events = self._load_public_events()

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-PUBLIC] Loaded {len(public_events)} active events from file"
            )

        # 2. Generate new public events
        agent_summaries = self.build_all_agents_summary()
        previous_events = "\n".join(
            [
                f"- {evt.event_name}: {evt.description}"
                for evt in sorted(public_events.values(), key=lambda e: e.event_id)
            ]
        )
        valid_agent_names = [agent.name for agent in self.agents]
        existing_event_names = {
            evt.event_name.strip().lower()
            for evt in public_events.values()
            if evt.is_active_this_week(t.year, t.week, n_weeks_per_year)
        }

        new_events = generate_public_events(
            agent_summaries=agent_summaries,
            previous_events=previous_events,
            n_days=n_days,
            year=t.year,
            week=t.week,
            valid_agent_names=valid_agent_names,
            existing_event_names=existing_event_names,
        )

        # 3. Persist new events to file
        if new_events:
            self._save_public_events(new_events)
            for evt in new_events:
                public_events[evt.event_id] = evt

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-PUBLIC] Generated {len(new_events)} new events: "
                f"{[e.event_name for e in new_events]}"
            )

        self.logger.info(
            f"Public events this week: {len(public_events)} total, {len(new_events)} new"
        )

        # 4. Build this-week instances for active events
        this_week_events: List[PublicEvent] = []
        for evt in public_events.values():
            if not evt.is_active_this_week(t.year, t.week, n_weeks_per_year):
                continue
            this_week_evt = PublicEvent(
                event_id=evt.event_id,
                event_name=evt.event_name,
                start_year=t.year,
                start_week=t.week,
                start_day=evt.start_day,
                repeat_weeks=evt.repeat_weeks,
                description=evt.description,
                eligible_participants=evt.eligible_participants,
            )
            this_week_events.append(this_week_evt)
        this_week_events.sort(key=lambda e: e.event_id)

        return this_week_events

    # ---------- Year-end Profile Update ----------
    def _update_yearly_profiles(self) -> None:
        """Update profiles for all agents at year end.

        Called after all weeks of a year are complete, before moving to next year.
        GodModel generates new profile based on yearly experiences.
        """
        from src.world.god import update_yearly_profile
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="profile_update")
        current_year = self.clock.get_time().year
        next_year = current_year + 1

        self.logger.info(f"== PROFILE UPDATE == Y{current_year} → Y{next_year}")

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-PROFILE] === PROFILE UPDATE START === "
                f"Y{current_year} → Y{next_year}, {len(self.agents)} agents"
            )

        # Collect results for verification
        results = []

        def update_and_collect(agent):
            result = self._update_agent_profile(agent, current_year, next_year)
            return result

        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                results = list(ex.map(update_and_collect, [a for a in self.agents if a.name != "Player"]))
        else:
            for agent in self.agents:
                if agent.name != "Player":
                    results.append(update_and_collect(agent))

        if verify_logger:
            # Log all results
            verify_logger.info("[VERIFY-PROFILE] --- Per-Agent Profile Changes ---")
            for result in sorted(results, key=lambda x: x["agent_name"]):
                name = result["agent_name"]
                changes = result["changes"]
                if changes:
                    verify_logger.info(f"[VERIFY-PROFILE] {name}: {', '.join(changes)}")
                else:
                    verify_logger.info(
                        f"[VERIFY-PROFILE] {name}: no quantitative changes"
                    )

            verify_logger.info(
                f"[VERIFY-PROFILE] === PROFILE UPDATE COMPLETE === "
                f"{len(self.agents)} agents processed"
            )

    def _update_agent_profile(
        self,
        agent: "RoleAgent",
        current_year: int,
        next_year: int,
    ) -> Dict[str, Any]:
        """Update a single agent's profile for next year.

        Args:
            agent: RoleAgent instance
            current_year: Current year number
            next_year: Next year number

        Returns:
            Dict with agent_name and list of changes for verification logging
        """
        from src.world.god import update_yearly_profile
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="profile_update")

        # Read current profile for comparison
        current_profile = agent.dm.read_profile()

        new_profile = update_yearly_profile(agent, current_year, next_year)
        agent.dm.write_profile(new_profile, year=next_year)
        self.logger.info(f"Profile updated: {agent.name} for Y{next_year}")

        # Compute changes for verification
        changes = []

        # Personality trait changes
        cur_pq = current_profile["personality_traits"]["quantitative"]
        new_pq = new_profile["personality_traits"]["quantitative"]
        for key in cur_pq:
            if cur_pq[key] != new_pq.get(key):
                changes.append(f"personality.{key}: {cur_pq[key]}→{new_pq[key]}")

        # Talent changes
        cur_tq = current_profile["talents"]["quantitative"]
        new_tq = new_profile["talents"]["quantitative"]
        for key in cur_tq:
            if cur_tq[key] != new_tq.get(key):
                changes.append(f"talents.{key}: {cur_tq[key]}→{new_tq[key]}")

        # Log input summaries (weekly summaries count)
        n_week = self.config["time"]["n_week"]
        summaries = agent.dm.read_weekly_summaries(n_weeks=n_week)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-PROFILE] {agent.name} INPUT: {len(summaries)} weekly summaries"
            )

        return {"agent_name": agent.name, "changes": changes}

    # Reward Calculation -------------------------------------------------------
    def _calculate_daily_rewards(self, day: int) -> None:
        """Calculate subjective + economy rewards for today. No LLM calls.

        Called at the end of each day when granularity="daily".
        Results are saved to reward/{subdir}/year=Y/day=D.jsonl.

        LLM cost: ZERO (reads from state.jsonl and deposit data only).
        """
        from src.world.reward import (
            compute_subjective_rewards,
            calculate_total_rewards,
        )
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="reward")
        t = self.clock.get_time()
        reward_cfg = self.config["reward"]

        self.logger.info(f"== DAILY REWARD == year={t.year} week={t.week} day={day}")

        # =====================================================================
        # 1. Subjective Reward (no LLM — reads fulfillment from state.jsonl)
        # =====================================================================
        subj_rewards = compute_subjective_rewards(
            self.agents, time_str=str(t),
        )
        total_penalties = sum(r.n_penalties for r in subj_rewards.values())
        self.logger.info(
            f"Daily subjective: {len(subj_rewards)} agents, {total_penalties} penalties"
        )

        # =====================================================================
        # 2. Economy Reward (deposit delta since yesterday)
        # =====================================================================
        economy_scores: Dict[str, float] = {}
        for agent in self.agents:
            deposit_now = agent.dm.get_deposit()
            economy_scores[agent.name] = float(deposit_now)

        self.logger.info(f"Daily economy: {len(economy_scores)} agents")

        # =====================================================================
        # 3. Social Reward: SKIP (computed weekly with LLM)
        #    Store placeholder so daily reward.jsonl has consistent schema.
        # =====================================================================
        from src.world.reward import SocialReward
        social_rewards: Dict[str, SocialReward] = {
            a.name: SocialReward(
                agent_name=a.name, time=str(t),
                affection_score=0.0, respect_score=0.0, combined_score=0.0,
            )
            for a in self.agents
        }

        # =====================================================================
        # 4. Total Reward (subjective + economy only; social added at normalization)
        # =====================================================================
        temp_total = calculate_total_rewards(
            social_rewards=social_rewards,
            subjective_rewards=subj_rewards,
            economy_scores=economy_scores,
            time_str=str(t),
        )

        # =====================================================================
        # 5. Save per-agent daily reward
        # =====================================================================
        from src.world.reward import SocialRanking
        empty_rankings = {
            a.name: SocialRanking(
                agent_name=a.name, time=str(t),
                affection_scores={}, respect_scores={},
            )
            for a in self.agents
        }

        def save_agent_daily_reward(agent):
            ranking = empty_rankings[agent.name]
            social = social_rewards[agent.name]
            subjective = subj_rewards[agent.name]
            total = temp_total[agent.name]
            agent.dm.save_reward(ranking, social, subjective, total)

        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(save_agent_daily_reward, self.agents))
        else:
            for agent in self.agents:
                save_agent_daily_reward(agent)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-REWARD] Daily reward saved: Y{t.year}-W{t.week}-D{day}"
            )

    def _calculate_weekly_rewards(self, week: int) -> None:
        """Calculate weekly reward (NO LLM — subjective + economy only).

        Social reward is placeholder 0.0 — real social computed yearly.
        Called at the end of each week with clock.day=6.
        """
        from src.world.reward import (
            compute_subjective_rewards,
            calculate_total_rewards,
            SocialReward,
        )
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="reward")
        t = self.clock.get_time()
        reward_cfg = self.config["reward"]

        self.logger.info(f"== WEEKLY REWARD (no LLM) == year={t.year} week={week}")

        # =====================================================================
        # 1. Subjective Reward (from state.jsonl — no LLM)
        # =====================================================================
        subjective_rewards = compute_subjective_rewards(
            self.agents, time_str=str(t),
        )
        total_penalties = sum(r.n_penalties for r in subjective_rewards.values())
        self.logger.info(
            f"Weekly subjective: {len(subjective_rewards)} agents, "
            f"{total_penalties} misery penalties"
        )

        # =====================================================================
        # 2. Economy Reward (deposit delta during this week)
        # =====================================================================
        economy_scores: Dict[str, float] = {}
        for agent in self.agents:
            deposit_end = agent.dm.get_deposit()
            deposit_start = getattr(agent, "_week_start_deposit", deposit_end)
            economy_scores[agent.name] = float(deposit_end - deposit_start)

        self.logger.info(f"Weekly economy: {len(economy_scores)} agents")

        # =====================================================================
        # 3. Social Reward — placeholder (real social computed yearly with LLM)
        # =====================================================================
        social_rewards: Dict[str, SocialReward] = {
            a.name: SocialReward(
                agent_name=a.name, time=str(t),
                affection_score=0.0, respect_score=0.0, combined_score=0.0,
            )
            for a in self.agents
        }

        # =====================================================================
        # 4. Total Reward (Subjective + Economy; Social = 0 placeholder)
        # =====================================================================
        total_rewards = calculate_total_rewards(
            social_rewards=social_rewards,
            subjective_rewards=subjective_rewards,
            economy_scores=economy_scores,
            time_str=str(t),
        )
        self.logger.info(f"Weekly total: {len(total_rewards)} agents")

        # =====================================================================
        # 5. Save Per-Agent Reward Data (ranking=None → no LLM data)
        # =====================================================================
        def save_agent_weekly(agent):
            social = social_rewards[agent.name]
            subjective = subjective_rewards[agent.name]
            total = total_rewards[agent.name]
            agent.dm.save_reward(None, social, subjective, total)

        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(save_agent_weekly, self.agents))
        else:
            for agent in self.agents:
                save_agent_weekly(agent)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-REWARD] Weekly reward saved: Y{t.year}-W{t.week:02d}"
            )

        self.logger.info(f"== WEEKLY REWARD COMPLETE == year={t.year} week={week}")

    def _calculate_yearly_rewards(self) -> None:
        """Calculate yearly reward with LLM social evaluation.

        Called at year-end with clock.day=99.
        LLM cost: N calls to judge_others() (one per agent) — once per year.
        """
        from src.world.reward import (
            build_social_graphs,
            calculate_social_rewards,
            compute_social_metrics,
            save_rankings,
            save_social_metrics,
            compute_subjective_rewards,
            calculate_total_rewards,
        )
        from src.utils import get_verify_logger

        verify_logger = get_verify_logger(feature="reward")
        t = self.clock.get_time()
        reward_cfg = self.config["reward"]
        n_days = self.config["time"]["n_day"] * self.config["time"]["n_week"]

        self.logger.info(
            f"== YEARLY REWARD (LLM) == year={t.year} "
            f"n_days_for_subjective={n_days}"
        )

        # =====================================================================
        # 1. Social Reward — LLM call: judge_others() for each agent
        # =====================================================================
        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                rankings = list(ex.map(lambda a: a.judge_others(), self.agents))
        else:
            rankings = [a.judge_others() for a in self.agents]

        rankings_by_name = {r.agent_name: r for r in rankings}

        # Persist rankings centrally (yearly file)
        save_rankings(rankings, self.data_dir, t.year, week=0)

        # Compute and save absolute social metrics
        social_metrics = compute_social_metrics(rankings, str(t))
        save_social_metrics(social_metrics, self.data_dir, t.year, week=0)

        affection_graph, respect_graph = build_social_graphs(rankings)

        if not affection_graph and not respect_graph:
            self.logger.warning(
                f"[REWARD] All agents returned empty rankings - no social graph edges."
            )

        social_rewards = calculate_social_rewards(
            affection_graph=affection_graph,
            respect_graph=respect_graph,
            time_str=str(t),
            all_agent_names=[a.name for a in self.agents],
        )
        self.logger.info(f"Yearly social: {len(social_rewards)} agents")

        # =====================================================================
        # 2. Subjective Reward (full year, no LLM)
        # =====================================================================
        subjective_rewards = compute_subjective_rewards(
            self.agents, time_str=str(t), n_days=n_days,
        )
        total_penalties = sum(r.n_penalties for r in subjective_rewards.values())
        self.logger.info(
            f"Yearly subjective: {len(subjective_rewards)} agents, "
            f"{total_penalties} misery penalties"
        )

        # =====================================================================
        # 3. Economy Reward (deposit delta over full year)
        # =====================================================================
        economy_scores: Dict[str, float] = {}
        for agent in self.agents:
            deposit_end = agent.dm.get_deposit()
            deposit_start = getattr(agent, "_week_start_deposit", deposit_end)
            economy_scores[agent.name] = float(deposit_end - deposit_start)

        self.logger.info(f"Yearly economy: {len(economy_scores)} agents")

        # =====================================================================
        # 4. Total Reward (Social + Subjective + Economy)
        # =====================================================================
        total_rewards = calculate_total_rewards(
            social_rewards=social_rewards,
            subjective_rewards=subjective_rewards,
            economy_scores=economy_scores,
            time_str=str(t),
        )
        self.logger.info(f"Yearly total: {len(total_rewards)} agents")

        # =====================================================================
        # 5. Save Per-Agent Reward Data
        # =====================================================================
        def save_agent_yearly(agent):
            ranking = rankings_by_name[agent.name]
            social = social_rewards[agent.name]
            subjective = subjective_rewards[agent.name]
            total = total_rewards[agent.name]
            agent.dm.save_reward(ranking, social, subjective, total)

        if self.parallel:
            with ThreadPoolExecutor(max_workers=pool_size(len(self.agents))) as ex:
                list(ex.map(save_agent_yearly, self.agents))
        else:
            for agent in self.agents:
                save_agent_yearly(agent)

        if verify_logger:
            verify_logger.info(
                f"[VERIFY-REWARD] Yearly reward saved: Y{t.year}"
            )

        self.logger.info(f"== YEARLY REWARD COMPLETE == year={t.year}")