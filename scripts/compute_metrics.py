#!/usr/bin/env python3
"""Compute per-agent per-week and per-agent per-year metrics from simulation data.

Adapted from Agentopia github original, extended with encounter / follow / perception
awareness.  Data sources:

    persona/<agent>/state.jsonl         — vitality, fulfillment, skills, assets
    persona/<agent>/schedule.jsonl      — daily activity plans
    persona/<agent>/contact/sig.jsonl   — contact signals (from → to)
    persona/<agent>/generation/**/*.jsonl — LLM traces (token counts)
    persona/<agent>/profile/year=*.json   — static innate profile
    god/solo_activity/**/*.jsonl        — god model evaluations (growth deltas)

Usage:
    python scripts/compute_metrics.py --data-dir apartment_06271605
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── metric list ──────────────────────────────────────────────────────────────

ALL_METRICS = [
    # system
    "input_tokens", "output_tokens", "function_call_count",
    # contact
    "active_contacts", "passive_contacts",
    # activity
    "joint_proposed", "joint_participated", "public_participated", "solo_count",
    "total_spending_amount", "activity_consumption_count",
    # growth
    "extra_earning_count", "skill_improvement_count", "total_skills",
    "deposit", "deposit_diff",
    # fulfillment (per-week snapshot)
    "fulfillment_mood", "fulfillment_material", "fulfillment_social", "fulfillment_esteem",
    # encounter
    "encounter_participated", "encounter_followed",
]

# Yearly aggregation rules
_CUMULATIVE = {
    "input_tokens", "output_tokens", "function_call_count",
    "active_contacts", "passive_contacts",
    "joint_proposed", "joint_participated", "public_participated", "solo_count",
    "total_spending_amount", "activity_consumption_count",
    "extra_earning_count", "skill_improvement_count",
    "deposit_diff",
    "encounter_participated", "encounter_followed",
}
_SNAPSHOT = {"total_skills", "deposit"}
_SNAPSHOT_LAST_NONZERO = {
    "fulfillment_mood", "fulfillment_material", "fulfillment_social", "fulfillment_esteem",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def parse_time(time_str: str) -> Tuple[int, int]:
    """Extract (year, week) from time string like 'Y2020-W01-xxx'."""
    m = re.match(r"Y(\d+)-W(\d+)", time_str)
    if not m:
        raise ValueError(f"Cannot parse time: {time_str}")
    return int(m.group(1)), int(m.group(2))


def week_key(year: int, week: int) -> str:
    return f"Y{year}-W{week:02d}"


def extract_day(time_str: str) -> int | None:
    m = re.search(r"-D(\d+)", time_str)
    return int(m.group(1)) if m else None


_AGENT_NAME_RE = re.compile(r"^# (.+?)'s Context$", re.MULTILINE)


def extract_agent_from_god_prompt(system_content: str) -> str | None:
    m = _AGENT_NAME_RE.search(system_content)
    return m.group(1) if m else None


def iter_agent_dirs(data_dir: Path) -> List[Tuple[str, Path]]:
    persona_dir = data_dir / "persona"
    return [
        (d.name, d)
        for d in sorted(persona_dir.iterdir())
        if d.is_dir() and d.name != "Player"
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# State (read once, reused)
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_state(data_dir: Path) -> Dict[str, List[dict]]:
    result = {}
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        records = read_jsonl(agent_dir / "state.jsonl")
        if records:
            result[agent_name] = records
    return result


def get_week_snapshots(state_records: List[dict]) -> Dict[str, Tuple[dict, dict]]:
    """Group state records by week → (first_content, last_content)."""
    week_records: Dict[str, List[dict]] = defaultdict(list)
    for rec in state_records:
        yr, wk = parse_time(rec["time"])
        week_records[week_key(yr, wk)].append(rec)
    return {
        wk: (records[0]["content"], records[-1]["content"])
        for wk, records in week_records.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_system_metrics(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """input_tokens, output_tokens, function_call_count from generation/*.jsonl."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "function_call_count": 0})
    )
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        gen_dir = agent_dir / "generation"
        if not gen_dir.exists():
            continue
        for jsonl_path in sorted(gen_dir.rglob("*.jsonl"), key=lambda p: p.as_posix()):
            for record in read_jsonl(jsonl_path):
                if "time" not in record or "input_tokens" not in record:
                    continue
                yr, wk = parse_time(record["time"])
                wk = week_key(yr, wk)
                bucket = results[agent_name][wk]
                bucket["input_tokens"] += record.get("input_tokens", 0)
                bucket["output_tokens"] += record.get("output_tokens", 0)
                for out_msg in record.get("outputs", []):
                    tool_calls = out_msg.get("tool_calls")
                    if tool_calls:
                        bucket["function_call_count"] += len(tool_calls)
    return results


def compute_contact_metrics(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """active_contacts, passive_contacts from contact/sig.jsonl."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"active_contacts": 0, "passive_contacts": 0})
    )
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for record in read_jsonl(agent_dir / "contact" / "sig.jsonl"):
            yr, wk = parse_time(record["time"])
            wk = week_key(yr, wk)
            bucket = results[agent_name][wk]
            if record.get("from") == agent_name:
                bucket["active_contacts"] += 1
            if record.get("to") == agent_name:
                bucket["passive_contacts"] += 1
    return results


def compute_activity_metrics(
    data_dir: Path,
    all_state: Dict[str, List[dict]],
    n_day: int,
) -> Dict[str, Dict[str, dict]]:
    """Schedule + state → activity counts + spending + encounter."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "joint_proposed": 0, "joint_participated": 0, "public_participated": 0,
            "solo_count": 0, "total_spending_amount": 0, "activity_consumption_count": 0,
            "encounter_participated": 0, "encounter_followed": 0,
        })
    )
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        occupied_days: Dict[str, set] = defaultdict(set)

        for rec in read_jsonl(agent_dir / "schedule.jsonl"):
            activity_time = rec["activity_time"]
            yr, wk = parse_time(activity_time)
            wk_key_ = week_key(yr, wk)
            day = extract_day(activity_time)
            if day is not None:
                occupied_days[wk_key_].add(day)

            rec_type = rec.get("type", "")
            if rec_type == "joint":
                if rec.get("proposer") == agent_name:
                    results[agent_name][wk_key_]["joint_proposed"] += 1
                if agent_name in rec.get("participants", []):
                    results[agent_name][wk_key_]["joint_participated"] += 1
            elif rec_type == "public":
                if agent_name in rec.get("participants", []):
                    results[agent_name][wk_key_]["public_participated"] += 1
            elif rec_type == "encounter":
                if agent_name in rec.get("participants", []):
                    results[agent_name][wk_key_]["encounter_participated"] += 1

        # solo_count = days without occupancy
        state_records = all_state.get(agent_name)
        if state_records:
            snapshots = get_week_snapshots(state_records)
            for wk_key_ in snapshots:
                results[agent_name][wk_key_]["solo_count"] = (
                    n_day - len(occupied_days[wk_key_])
                )

        # spending: detect deposit drops between consecutive state records
        if not state_records:
            continue
        for i in range(1, len(state_records)):
            prev_deposit = state_records[i - 1]["content"]["assets"]["deposit"]
            cur_deposit = state_records[i]["content"]["assets"]["deposit"]
            if cur_deposit >= prev_deposit:
                continue

            time_str = state_records[i]["time"]
            is_plan = "-plan" in time_str
            is_activity = "-activity-" in time_str
            if not (is_plan or is_activity):
                continue

            yr, wk = parse_time(time_str)
            wk_key_ = week_key(yr, wk)
            drop = prev_deposit - cur_deposit
            results[agent_name][wk_key_]["total_spending_amount"] += drop
            if is_activity:
                results[agent_name][wk_key_]["activity_consumption_count"] += 1

        # encounter_followed: detect follow schedules (T7 system)
        for rec in read_jsonl(agent_dir / "schedule.jsonl"):
            activity_time = rec["activity_time"]
            yr, wk = parse_time(activity_time)
            wk_key_ = week_key(yr, wk)
            activity_name = rec.get("activity_name", "")
            if "Follow" in activity_name and agent_name in rec.get("participants", []):
                results[agent_name][wk_key_]["encounter_followed"] += 1

    return results


def compute_growth_metrics(
    data_dir: Path,
    all_state: Dict[str, List[dict]],
) -> Dict[str, Dict[str, dict]]:
    """extra_earning, skill_improvement, total_skills, deposit, deposit_diff."""
    results: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "extra_earning_count": 0, "skill_improvement_count": 0,
            "total_skills": 0, "deposit": 0, "deposit_diff": 0,
        })
    )
    # god solo_activity evaluations
    god_solo_dir = data_dir / "god" / "solo_activity"
    skipped = 0
    if god_solo_dir.exists():
        for jsonl_path in sorted(god_solo_dir.rglob("*.jsonl"), key=lambda p: p.as_posix()):
            for record in read_jsonl(jsonl_path):
                inputs = record.get("inputs", [])
                if not inputs:
                    continue
                sys_content = inputs[0].get("content", "")
                if not sys_content or "consumption activity" in sys_content:
                    continue

                yr, wk = parse_time(record["time"])
                wk_key_ = week_key(yr, wk)

                agent_name = extract_agent_from_god_prompt(sys_content)
                if not agent_name:
                    skipped += 1
                    continue

                outputs = record.get("outputs", [])
                if not outputs:
                    continue
                raw = outputs[0].get("content", "")
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    data = raw

                delta_money = data.get("delta_money", 0)
                if isinstance(delta_money, (int, float)) and delta_money > 0:
                    results[agent_name][wk_key_]["extra_earning_count"] += 1

                delta_skills = data.get("delta_skills") or {}
                if isinstance(delta_skills, dict) and any(
                    isinstance(v, (int, float)) and v > 0 for v in delta_skills.values()
                ):
                    results[agent_name][wk_key_]["skill_improvement_count"] += 1

    if skipped:
        print(f"    WARNING: {skipped} god records skipped (agent name extraction failed)")

    # state-derived: total_skills, deposit, deposit_diff
    for agent_name, state_records in sorted(all_state.items()):
        non_settle = [r for r in state_records if "settle" not in r["time"]]
        snapshots = get_week_snapshots(non_settle)
        sorted_weeks = sorted(snapshots.keys())
        prev_deposit = None
        prev_year = None
        for wk_key_ in sorted_weeks:
            first_content, last_content = snapshots[wk_key_]
            skills = last_content["skills"]
            deposit = last_content["assets"]["deposit"]
            yr, _ = parse_time(wk_key_)
            results[agent_name][wk_key_]["total_skills"] = sum(skills.values())
            results[agent_name][wk_key_]["deposit"] = deposit
            if prev_deposit is None or yr != prev_year:
                init_deposit = first_content["assets"]["deposit"]
                results[agent_name][wk_key_]["deposit_diff"] = deposit - init_deposit
            else:
                results[agent_name][wk_key_]["deposit_diff"] = deposit - prev_deposit
            prev_deposit = deposit
            prev_year = yr

    return results


def compute_fulfillment_metrics(
    all_state: Dict[str, List[dict]],
) -> Dict[str, Dict[str, dict]]:
    """fulfillment_{mood,material,social,esteem} from state (skip settle)."""
    results: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for agent_name, state_records in sorted(all_state.items()):
        non_settle = [r for r in state_records if "settle" not in r["time"]]
        snapshots = get_week_snapshots(non_settle)
        for wk_key_, (_, last_content) in snapshots.items():
            fulfillment = last_content["fulfillment"]
            results[agent_name][wk_key_] = {
                "fulfillment_mood": fulfillment["mood"],
                "fulfillment_material": fulfillment["material"],
                "fulfillment_social": fulfillment["social"],
                "fulfillment_esteem": fulfillment["esteem"],
            }
    return results


def compute_innate_metrics(
    data_dir: Path, start_year: int,
) -> Dict[str, dict]:
    """Static innate: weekly_income, init_deposit, init_relationship_count."""
    results: Dict[str, dict] = {}
    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        profile_path = agent_dir / "profile" / f"year={start_year}.json"
        if not profile_path.exists():
            continue
        with profile_path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        pos = profile.get("position", {})
        init = profile.get("init_assets", {})
        weekly_income = pos.get("weekly_income", 0) + profile.get("extra_income", 0)
        init_deposit = init.get("deposit", 0)
        chars_dir = agent_dir / "memory" / "scratchpad" / "characters"
        if chars_dir.exists():
            init_rel_count = len([p for p in chars_dir.iterdir() if p.suffix == ".jsonl"])
        else:
            init_rel_count = 0
        results[agent_name] = {
            "weekly_income": weekly_income,
            "init_deposit": init_deposit,
            "init_relationship_count": init_rel_count,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Merge & Aggregate
# ═══════════════════════════════════════════════════════════════════════════════

def merge_metrics(
    active_weeks: Dict[str, set],
    *metric_dicts: Dict[str, Dict[str, dict]],
) -> Dict[str, Dict[str, dict]]:
    merged: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for md in metric_dicts:
        for agent_name, weeks in md.items():
            allowed = active_weeks.get(agent_name, set())
            for wk_key_, metrics in weeks.items():
                if wk_key_ in allowed:
                    merged[agent_name][wk_key_].update(metrics)
    for agent_name in merged:
        for wk_key_ in merged[agent_name]:
            for metric in ALL_METRICS:
                merged[agent_name][wk_key_].setdefault(metric, 0)
    return dict(merged)


def find_last_complete_year(
    by_week: Dict[str, Dict[str, dict]],
    n_week: int,
) -> int | None:
    year_week_counts: Dict[int, List[int]] = defaultdict(list)
    for agent_name, weeks in by_week.items():
        agent_years: Dict[int, int] = defaultdict(int)
        for wk_key_ in weeks:
            yr, _ = parse_time(wk_key_)
            agent_years[yr] += 1
        for yr, count in agent_years.items():
            year_week_counts[yr].append(count)
    complete_years = [
        yr for yr in sorted(year_week_counts)
        if all(c == n_week for c in year_week_counts[yr])
    ]
    return complete_years[-1] if complete_years else None


def aggregate_yearly(
    by_week: Dict[str, Dict[str, dict]],
    last_complete_year: int | None = None,
) -> Dict[str, Dict[str, dict]]:
    by_year: Dict[str, Dict[str, dict]] = {}
    for agent_name, weeks in by_week.items():
        year_weeks: Dict[str, List[str]] = defaultdict(list)
        for wk_key_ in weeks:
            yr, _ = parse_time(wk_key_)
            if last_complete_year is not None and yr > last_complete_year:
                continue
            year_weeks[f"Y{yr}"].append(wk_key_)
        agent_yearly = {}
        for yr_key, wk_keys in sorted(year_weeks.items()):
            sorted_wks = sorted(wk_keys)
            yearly: Dict[str, Any] = {}
            for metric in _CUMULATIVE:
                yearly[metric] = sum(weeks[wk][metric] for wk in sorted_wks)
            for metric in _SNAPSHOT:
                yearly[metric] = weeks[sorted_wks[-1]][metric]
            for metric in _SNAPSHOT_LAST_NONZERO:
                val = 0
                for wk in reversed(sorted_wks):
                    v = weeks[wk][metric]
                    if v != 0:
                        val = v
                        break
                yearly[metric] = val
            agent_yearly[yr_key] = yearly
        by_year[agent_name] = agent_yearly
    return by_year


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(data_dir_name: str) -> None:
    data_dir = ROOT / "data" / data_dir_name
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return

    # prefer per-run config, fallback to global
    run_config_path = data_dir / "config.json"
    config_path = run_config_path if run_config_path.exists() else ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    n_day = config["world"]["time"]["n_day"]
    n_week = config["world"]["time"]["n_week"]
    start_year = config["world"]["time"]["start_year"]

    print(f"Computing metrics for {data_dir_name} (n_day={n_day}, n_week={n_week})")

    print("  Loading state data...")
    all_state = load_all_state(data_dir)
    active_weeks = {
        name: {week_key(*parse_time(r["time"])) for r in records}
        for name, records in all_state.items()
    }

    print("  Computing system metrics...")
    system_metrics = compute_system_metrics(data_dir)

    print("  Computing contact metrics...")
    contact_metrics = compute_contact_metrics(data_dir)

    print("  Computing activity metrics...")
    activity_metrics = compute_activity_metrics(data_dir, all_state, n_day)

    print("  Computing growth metrics...")
    growth_metrics = compute_growth_metrics(data_dir, all_state)

    print("  Computing fulfillment metrics...")
    fulfillment_metrics = compute_fulfillment_metrics(all_state)

    by_week = merge_metrics(
        active_weeks,
        system_metrics, contact_metrics, activity_metrics, growth_metrics,
        fulfillment_metrics,
    )

    last_complete = find_last_complete_year(by_week, n_week)
    if last_complete is not None:
        print(f"  Last complete year: Y{last_complete}")
    else:
        print("  WARNING: No complete year found, including all years in by_year")

    print("  Aggregating yearly...")
    by_year = aggregate_yearly(by_week, last_complete)

    print("  Computing innate metrics...")
    innate = compute_innate_metrics(data_dir, start_year)

    output_dir = ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_dir_name}_metrics.json"

    output = {
        "data_dir": data_dir_name,
        "config": {"n_day": n_day, "n_week": n_week, "start_year": start_year},
        "last_complete_year": last_complete,
        "innate": innate,
        "by_week": by_week,
        "by_year": by_year,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    n_agents = len(by_week)
    n_weeks_total = sum(len(w) for w in by_week.values())
    n_years = len(set(yr for a in by_year.values() for yr in a))
    print(f"Done. {n_agents} agents, {n_weeks_total} agent-weeks, {n_years} years in by_year.")
    print(f"Output: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics from simulation data")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data dir to process (e.g. apartment_06271605).",
    )
    args = parser.parse_args()
    run_one(args.data_dir)


if __name__ == "__main__":
    main()
