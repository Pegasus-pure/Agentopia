#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal world following pseudo/world.py"
    )
    parser.add_argument(
        "--years", type=int, default=None, help="Override number of years"
    )
    parser.add_argument(
        "--weeks", type=int, default=None, help="Override weeks per year"
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Specify run id (MMDDHHMM) to continue a previous run",
    )
    parser.add_argument(
        "--no-ce",
        dest="no_ce",
        action="store_true",
        help="Disable context engineering; agents only use working memory",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel LLM calls (parallel is enabled by default)",
    )
    parser.set_defaults(parallel=True)
    parser.add_argument(
        "--no-history",
        dest="no_history",
        action="store_true",
        help="Disable history usage (skip read/write history)",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Maximum number of agents to bootstrap from data (default: no limit)",
    )
    parser.add_argument(
        "--player",
        dest="player",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Player mode: 'on' or 'off' (overrides config.enable_player)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for world/utils/agents",
    )
    parser.add_argument(
        "--world",
        type=str,
        default=None,
        help="Override world name from config.json (e.g., school, apartment)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override world language from config.json (e.g., en, zh)",
    )
    parser.add_argument(
        "--role-model",
        dest="role_model",
        type=str,
        default=None,
        help="Override role_model from config.json. "
             "Comma-separated for multiple models (e.g., 'claude-4.5-sonnet,gemini-3-flash-preview')",
    )
    parser.add_argument(
        "--god-model",
        dest="god_model",
        type=str,
        default=None,
        help="Override god_model from config.json (e.g., claude-4.5-sonnet, Qwen3.5-397B-A17B)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a run directory path (e.g., data/schooldays_01201830 or schooldays_01201830). "
             "Loads config from that directory. Mutually exclusive with --run-id.",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        type=str,
        default=None,
        help="Resume from specific point: Y{year} or Y{year}-W{week} (requires --run-id or --resume)",
    )
    parser.add_argument(
        "--resume-year",
        dest="resume_year",
        type=int,
        default=None,
        help="Resume from year (requires --run-id or --resume; alternative to --resume-from)",
    )
    parser.add_argument(
        "--resume-week",
        dest="resume_week",
        type=int,
        default=None,
        help="Resume from week (used with --resume-year, defaults to 1)",
    )
    args = parser.parse_args()

    # --resume and --run-id are mutually exclusive
    if args.resume and args.run_id:
        parser.error("--resume and --run-id are mutually exclusive")

    # Validate resume-from / resume-year arguments
    resume_from_parsed: tuple[int, int] | None = None
    if args.resume_from and args.resume_year is not None:
        parser.error("--resume-from and --resume-year are mutually exclusive")
    if args.resume_week is not None and args.resume_year is None:
        parser.error("--resume-week requires --resume-year")
    if args.resume_from:
        if not args.run_id and not args.resume:
            parser.error("--resume-from requires --run-id or --resume")
        m = re.fullmatch(r"Y(\d+)(?:-W(\d+))?", args.resume_from)
        if not m:
            parser.error(
                f"Invalid --resume-from format: '{args.resume_from}'. "
                f"Expected Y{{year}} or Y{{year}}-W{{week}}"
            )
        r_year = int(m.group(1))
        r_week = int(m.group(2)) if m.group(2) else 1
        resume_from_parsed = (r_year, r_week)
    elif args.resume_year is not None:
        if not args.run_id and not args.resume:
            parser.error("--resume-year requires --run-id or --resume")
        resume_from_parsed = (args.resume_year, args.resume_week or 1)

    import json
    from src.world.run_manager import (
        generate_run_id,
        ensure_run_world_data,
        get_run_data_dir,
        save_run_config,
    )
    from src.config import load_config

    if args.resume:
        # ── Resume from directory path ──────────────────────────────────
        # Accept "data/schooldays_01201830" or just "schooldays_01201830"
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            # Strip leading "data/" if present, then normalize
            parts = resume_path.parts
            if parts[0] == "data":
                data_dir = str(Path(*parts[1:]))
            else:
                data_dir = str(resume_path)
        else:
            # Absolute path: extract relative to data/
            data_dir = str(resume_path.relative_to(Path("data").resolve()))

        run_dir = Path("data") / data_dir
        run_cfg_path = run_dir / "config.json"
        if not run_cfg_path.exists():
            parser.error(f"Config not found: {run_cfg_path}")

        load_config(run_cfg_path)
        print(f"[run_world] Resuming from {run_dir} (config loaded)")
    else:
        # ── Normal flow: new run or --run-id resume ─────────────────────
        # Read root config.json as raw input (not yet loaded into global _CONFIG)
        with (ROOT / "config.json").open("r", encoding="utf-8") as f:
            raw_config = json.load(f)

        # Apply world/language overrides early (before base_world_name)
        if args.world is not None:
            raw_config["world"]["name"] = args.world
            print(f"[run_world] Overriding world name: {args.world}")
        if args.language is not None:
            raw_config["world"]["language"] = args.language
            print(f"[run_world] Overriding language: {args.language}")

        base_world_name = str(raw_config["world"]["name"]).strip()
        run_id = args.run_id or generate_run_id()
        dst_path = ensure_run_world_data(base_world_name, run_id)
        data_dir = get_run_data_dir(base_world_name, run_id)
        run_cfg_path = Path("data") / data_dir / "config.json"

        if args.run_id and run_cfg_path.exists():
            # Resuming a previous run: load its config directly
            cli_overrides = [
                name for name, val in [
                    ("--world", args.world), ("--language", args.language),
                    ("--role-model", args.role_model), ("--god-model", args.god_model),
                    ("--years", args.years), ("--weeks", args.weeks),
                ] if val is not None
            ]
            if cli_overrides:
                print(f"[run_world] WARNING: Resuming run {args.run_id}, "
                      f"CLI overrides ignored: {', '.join(cli_overrides)}")
            load_config(run_cfg_path)
        else:
            # New run: apply CLI overrides, save to run directory, then load
            if args.role_model is not None:
                # Support comma-separated list: 'model-a,model-b' → ["model-a", "model-b"]
                models = [m.strip() for m in args.role_model.split(",") if m.strip()]
                if not models:
                    parser.error("--role-model must not be empty")
                raw_config["role_model"] = models[0] if len(models) == 1 else models
                print(f"[run_world] Overriding role_model: {raw_config['role_model']}")
            if args.god_model is not None:
                raw_config["god_model"] = args.god_model
                print(f"[run_world] Overriding god_model: {args.god_model}")
            if args.years is not None:
                raw_config["world"]["time"]["n_year"] = args.years
            if args.weeks is not None:
                raw_config["world"]["time"]["n_week"] = args.weeks
            raw_config["world"]["name"] = base_world_name
            raw_config["world"]["data_dir"] = data_dir
            save_run_config(data_dir, raw_config)
            load_config(run_cfg_path)

        print(
            f"[run_world] data_dir={data_dir}"
        )

    # Set up per-run cache directory (isolates cache between parallel runs)
    from src.utils import set_run_cache_dir

    set_run_cache_dir(data_dir)

    from src.config import get_config
    from src.world.world import World

    # Determine resume_from: CLI override or auto-detect from checkpoint
    resume_from = resume_from_parsed if (args.run_id or args.resume) else None

    # Resolve player mode: CLI --player on/off > config.enable_player
    if args.player == "on":
        player_enabled = True
    elif args.player == "off":
        player_enabled = False
    else:
        player_enabled = None  # use config default

    w = World(
        no_context_engineering=args.no_ce,
        parallel=args.parallel,
        no_history=args.no_history,
        max_agents=args.max_agents,
        player_enabled=player_enabled,
        resume_from=resume_from,
    )

    # If --debug is set, bump logger levels to DEBUG for world, utils and agents.
    if args.debug:
        import logging

        def _bump_logger(lg):
            if lg is None:
                return
            try:
                lg.setLevel(logging.DEBUG)
                for h in getattr(lg, "handlers", []) or []:
                    h.setLevel(logging.DEBUG)
            except Exception:
                pass

        # world logger (prints to console)
        _bump_logger(getattr(w, "logger", None))
        # utils logger (file-only by default)
        _bump_logger(logging.getLogger("utils"))
        # agent loggers (file-only by default)
        for a in getattr(w, "agents", []) or []:
            _bump_logger(getattr(a, "logger", None))

    # Flush caches on early termination:
    # - atexit: handles exceptions and normal exit
    # - signal: handles SIGINT (Ctrl+C) and SIGTERM (kill)
    import atexit
    from src.utils import flush_all_caches, merge_run_cache

    atexit.register(flush_all_caches)

    _flushing = False  # guard against double flush

    def _flush_and_exit(signum, frame):
        nonlocal _flushing
        sig_name = signal.Signals(signum).name
        if _flushing:
            print(f"\n[{sig_name}] Flush in progress. Press Ctrl+C again to force kill.")
            # Restore default handler so next Ctrl+C kills immediately
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return
        _flushing = True
        # Restore default handler so next Ctrl+C kills immediately during flush
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print(f"\n[{sig_name}] Flushing caches before exit...")
        try:
            flush_all_caches()
            merge_run_cache(data_dir)
            print(f"[{sig_name}] Cache flush done. Exiting.")
        except Exception as e:
            print(f"[{sig_name}] Error during flush: {e}")
        finally:
            os._exit(128 + signum)

    signal.signal(signal.SIGINT, _flush_and_exit)
    signal.signal(signal.SIGTERM, _flush_and_exit)

    w.run()

    # 1. Flush all thread deltas to disk (before merge)
    flush_all_caches()

    # 2. Merge run cache shards back into main cache files
    merge_run_cache(data_dir)


if __name__ == "__main__":
    main()

# python scripts/run_world.py --year 1 --week 5 --parallel
# python -m src.utils
