#!/usr/bin/env python3
"""Compute per-agent per-week social-emotional quality reward from simulation data.

Inspired by Sentipolis (arxiv:2601.18027) and Sullivan theories of interpersonal
reciprocity.  No extra LLM calls — pure computation from existing JSONL data.

Dimensions:
  1. social_engagement   — out-degree contact frequency (active outreach count)
  2. social_reach        — distinct people contacted this week (out + in, deduplicated)
  3. relationship_breadth — how many characters this agent knows (scratchpad count)
  4. emotional_richness  — standard deviation of fulfillment across 4 axes
  5. reciprocity         — proportion of relationships that are bidirectional

Each dimension is normalised 0..1, then weighted per config.json.

Usage:
    python scripts/emotion_analysis.py <run_id>

Example:
    python scripts/emotion_analysis.py apartment_06271605
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    m = re.match(r"Y(\d+)-W(\d+)", time_str)
    if not m:
        raise ValueError(f"Cannot parse time: {time_str}")
    return int(m.group(1)), int(m.group(2))


def week_key(year: int, week: int) -> str:
    return f"Y{year}-W{week:02d}"


def week_label(wk_key: str) -> str:
    yr, wk = parse_time(wk_key)
    return f"Y{yr}-W{wk:02d}"


def iter_agent_dirs(data_dir: Path) -> List[Tuple[str, Path]]:
    persona_dir = data_dir / "persona"
    return [
        (d.name, d)
        for d in sorted(persona_dir.iterdir())
        if d.is_dir() and d.name != "Player"
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension 1: Social Engagement (out-degree)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_engagement(
    data_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """active contact count per agent per week, normalised 0..1."""
    raw: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    max_any = 1

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for rec in read_jsonl(agent_dir / "contact" / "sig.jsonl"):
            yr, wk = parse_time(rec["time"])
            wk = week_key(yr, wk)
            if rec.get("from") == agent_name:
                raw[agent_name][wk] += 1
                if raw[agent_name][wk] > max_any:
                    max_any = raw[agent_name][wk]

    return {
        name: {wk: v / max_any for wk, v in weeks.items()}
        for name, weeks in raw.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension 2: Social Reach (distinct contacts, out + in, per week)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reach(
    data_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Distinct people interacted with per week (outgoing + incoming), 0..1."""
    raw: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    max_any = 1

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        for rec in read_jsonl(agent_dir / "contact" / "sig.jsonl"):
            yr, wk = parse_time(rec["time"])
            wk = week_key(yr, wk)
            frm = rec.get("from", "")
            to = rec.get("to", "")
            if frm == agent_name and to != agent_name:
                raw[agent_name][wk].add(to)
            elif to == agent_name and frm != agent_name:
                raw[agent_name][wk].add(frm)
            if len(raw[agent_name][wk]) > max_any:
                max_any = len(raw[agent_name][wk])

    return {
        name: {wk: len(contacts) / max_any for wk, contacts in weeks.items()}
        for name, weeks in raw.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension 3: Relationship Breadth (total scratchpad characters)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_breadth(
    data_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Total number of scratchpad character entries per week (0..1, 15+ = max)."""
    result: Dict[str, Dict[str, float]] = defaultdict(dict)

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        chars_dir = agent_dir / "memory" / "scratchpad" / "characters"
        if not chars_dir.exists():
            continue

        # Track when each character file first appears
        char_weeks: Dict[str, str] = {}
        for char_file in sorted(chars_dir.iterdir()):
            if char_file.suffix != ".jsonl":
                continue
            records = read_jsonl(char_file)
            for rec in records:
                t = rec.get("time", "")
                if t:
                    yr, wk = parse_time(t)
                    char_weeks[char_file.stem] = week_key(yr, wk)
                    break

        # Cumulative count: each week adds chars first seen that week
        seen: Set[str] = set()
        all_weeks = sorted(set(char_weeks.values()))
        for wk in all_weeks:
            for char, first_wk in char_weeks.items():
                if first_wk == wk:
                    seen.add(char)
            result[agent_name][wk] = min(len(seen) / 15.0, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension 4: Emotional Richness (fulfillment variance)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_richness(
    data_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Standard deviation of mood/material/social/esteem per week (0..1 normalised)."""
    result: Dict[str, Dict[str, float]] = defaultdict(dict)

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        records = read_jsonl(agent_dir / "state.jsonl")
        if not records:
            continue

        week_vals: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
        for rec in records:
            t = rec["time"]
            if "settle" in t:
                continue
            yr, wk = parse_time(t)
            wk_key_ = week_key(yr, wk)
            f = rec["content"]["fulfillment"]
            week_vals[wk_key_].append(
                (f["mood"], f["material"], f["social"], f["esteem"])
            )

        for wk_key_, vals in week_vals.items():
            if len(vals) < 2:
                continue
            # Per-axis std then average across axes
            n = len(vals)
            axis_means = [sum(a[i] for a in vals) / n for i in range(4)]
            axis_vars = [
                sum((a[i] - axis_means[i]) ** 2 for a in vals) / n
                for i in range(4)
            ]
            avg_std = math.sqrt(sum(axis_vars) / 4)
            # Normalise: std of 20+ = max richness
            result[agent_name][wk_key_] = min(avg_std / 20.0, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension 5: Reciprocity (bidirectional relationship ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reciprocity(
    data_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Proportion of contact relationships that are bidirectional (0..1)."""
    result: Dict[str, Dict[str, float]] = defaultdict(dict)

    for agent_name, agent_dir in iter_agent_dirs(data_dir):
        sigs = read_jsonl(agent_dir / "contact" / "sig.jsonl")
        if not sigs:
            continue

        # Track edges per week: (agent -> other)
        week_outgoing: Dict[str, Set[str]] = defaultdict(set)
        week_incoming: Dict[str, Set[str]] = defaultdict(set)

        for rec in sigs:
            yr, wk = parse_time(rec["time"])
            wk_key_ = week_key(yr, wk)
            frm = rec.get("from", "")
            to = rec.get("to", "")
            if frm == agent_name and to != agent_name:
                week_outgoing[wk_key_].add(to)
            elif to == agent_name and frm != agent_name:
                week_incoming[wk_key_].add(frm)

        for wk_key_ in week_outgoing:
            outgoing = week_outgoing[wk_key_]
            incoming = week_incoming.get(wk_key_, set())
            if not outgoing:
                continue
            mutual = outgoing & incoming
            result[agent_name][wk_key_] = len(mutual) / len(outgoing)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Merge & Write
# ═══════════════════════════════════════════════════════════════════════════════

def merge_dimensions(
    *dims: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Merge dimension dicts → {agent: {week: {dim: score}}}.  Fill missing with 0."""
    merged: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: dict.fromkeys(DIM_NAMES, 0.0))
    )
    for i, dim in enumerate(dims):
        name = DIM_NAMES[i]
        for agent, weeks in dim.items():
            for wk, val in weeks.items():
                merged[agent][wk][name] = val
    return {a: dict(w) for a, w in merged.items()}


DIM_NAMES = [
    "social_engagement",
    "social_reach",
    "relationship_breadth",
    "emotional_richness",
    "reciprocity",
]


def weighted_total(
    agent_weeks: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute weighted sum for one agent's week."""
    return {
        wk: sum(dim_vals.get(d, 0.0) * weights.get(d, 0.0) for d in DIM_NAMES)
        for wk, dim_vals in agent_weeks.items()
    }


def run_one(data_dir_name: str) -> None:
    data_dir = ROOT / "data" / data_dir_name
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return

    # Only include agents with actual simulation state (skip W00-only / empty persona dirs)
    active_agents = set()
    for _, agent_dir in iter_agent_dirs(data_dir):
        records = read_jsonl(agent_dir / "state.jsonl")
        for rec in records:
            t = rec.get("time", "")
            if "W00" not in t and len(t) > 0:
                active_agents.add(agent_dir.name)
                break
    active_count = len(active_agents)
    if active_agents:
        print(f"  Active agents (with state data): {active_count}")

    # Load weights from config
    run_config = data_dir / "config.json"
    config_path = run_config if run_config.exists() else ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    weights = config.get("world", {}).get("reward", {}).get("weights", {
        "social_engagement": 1.0,
        "social_reach": 1.5,
        "relationship_breadth": 2.0,
        "emotional_richness": 1.0,
        "reciprocity": 1.5,
    })
    # Fill missing with 0
    for d in DIM_NAMES:
        weights.setdefault(d, 0.0)

    # Compute
    print(f"Computing emotion metrics for {data_dir_name}")
    print("  Social engagement...")
    engagement = compute_engagement(data_dir)
    print("  Social reach...")
    reach = compute_reach(data_dir)
    print("  Relationship breadth...")
    breadth = compute_breadth(data_dir)
    print("  Emotional richness...")
    richness = compute_richness(data_dir)
    print("  Reciprocity...")
    reciprocity = compute_reciprocity(data_dir)

    # Merge
    merged = merge_dimensions(engagement, reach, breadth, richness, reciprocity)

    # Write aggregated result to analysis/results/
    output_dir = ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_dir_name}_emotion.json"

    # Build aggregated output: {agent_name: {week: {dims, total}}}
    aggregated = {}
    written = 0
    for agent_name in sorted(active_agents):
        weeks = merged.get(agent_name, {})
        if not weeks:
            continue
        totals = weighted_total(weeks, weights)
        agent_data = {}
        for wk_key_ in sorted(weeks.keys()):
            dims = weeks[wk_key_]
            agent_data[wk_key_] = {
                "dimensions": {d: round(dims.get(d, 0.0), 4) for d in DIM_NAMES},
                "weighted_total": round(totals[wk_key_], 4),
            }
            written += 1
        if agent_data:
            aggregated[agent_name] = agent_data

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "data_dir": data_dir_name,
            "weights": weights,
            "agents": aggregated,
            "computed_at": now,
        }, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(aggregated)} agents, {written} records written to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-agent weekly social-emotional reward"
    )
    parser.add_argument(
        "run_id",
        help="Run data directory name (e.g. apartment_06271605)",
    )
    args = parser.parse_args()
    run_one(args.run_id)


if __name__ == "__main__":
    main()
