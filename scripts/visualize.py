#!/usr/bin/env python3
"""Generate PNG trend charts from simulation data using matplotlib.

Usage:
    python scripts/visualize.py <run_id>

Outputs 4 PNG files to analysis/results/:
    1. emotion_trends.png           — weighted_total composite trend (4 agents, by volatility)
    2. emotion_dimensions.png       — 5 emotion dimension subplots (all 5 agents)
    3. metrics_fulfillment.png      — 4 fulfillment dimension subplots (2x2)
    4. metrics_deposit.png          — deposit & deposit_diff trend
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ── matplotlib setup (Agg backend, no GUI) ──────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Support Chinese characters in labels
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── constants ───────────────────────────────────────────────────────────

AGENT_COLORS = [
    "#4A90D9",  # blue
    "#E74C3C",  # red
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
]

# Color order for emotion_trends chart: 红(波动最大) → 橙 → 蓝 → 绿(波动最小)
TREND_COLORS = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]

DIM_NAMES = [
    "social_engagement",
    "social_reach",
    "relationship_breadth",
    "emotional_richness",
    "reciprocity",
]

DIM_LABELS = {
    "social_engagement": "社交活跃度",
    "social_reach": "社交覆盖面",
    "relationship_breadth": "关系广度",
    "emotional_richness": "情绪丰富度",
    "reciprocity": "关系对等度",
}

FULFILLMENT_DIMS = ["fulfillment_mood", "fulfillment_material", "fulfillment_social", "fulfillment_esteem"]
FULFILLMENT_LABELS = {
    "fulfillment_mood": "心情 (Mood)",
    "fulfillment_material": "物质 (Material)",
    "fulfillment_social": "社交 (Social)",
    "fulfillment_esteem": "自尊 (Esteem)",
}

OUTPUT_DIR = ROOT / "analysis" / "results"

# ── helpers (kept for backward compatibility) ───────────────────────────


def load_metrics(run_id: str) -> dict:
    """Load pre-computed metrics from analysis/results/."""
    results = ROOT / "analysis" / "results"
    metrics_path = results / f"{run_id}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return {"by_week": {}}


def read_daily_fulfillment(run_id: str, top_n: int = 15) -> dict:
    """Read daily fulfillment directly from state.jsonl.

    Returns:
        {
            "dims": ["mood", "material", "social", "esteem"],
            "agents": {
                "Aaron Whitfield": {
                    "mood":    {"D1": 65.0, "D2": 68.0, ...},
                    "material": {...},
                    ...
                },
                ...
            }
        }
    """
    data_dir = ROOT / "data" / run_id
    dims = ["mood", "material", "social", "esteem"]

    # Collect all agent names from persona dir
    persona_dir = data_dir / "persona"
    if not persona_dir.exists():
        return {"dims": dims, "agents": {}}

    agents_data = {}
    for agent_dir in sorted(persona_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        name = agent_dir.name
        state_path = agent_dir / "state.jsonl"
        if not state_path.exists():
            continue

        # Read state.jsonl, extract daily fulfillment with week+day
        day_vals = {d: defaultdict(list) for d in dims}
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                time_str = rec.get("time", "")
                # Only activity records (exclude settle)
                if "activity" not in time_str or "settle" in time_str:
                    continue
                # Extract week + day: "Y2020-W01-activity-D2-dawn" -> "W01-D2"
                mw = re.search(r"W(\d+)", time_str)
                md = re.search(r"D(\d+)", time_str)
                if not mw or not md:
                    continue
                day = f"W{mw.group(1)}-D{md.group(1)}"
                ful = rec.get("content", {}).get("fulfillment", {})
                for d in dims:
                    if d in ful:
                        day_vals[d][day].append(ful[d])

        # Average per day (sorted by week then day)
        agent_data = {}
        for d in dims:
            def sort_key(x):
                parts = x.split("-D")
                week = int(parts[0].replace("W", ""))
                day = int(parts[1])
                return (week, day)
            days_sorted = sorted(day_vals[d].keys(), key=sort_key)
            agent_data[d] = {
                day: sum(day_vals[d][day]) / len(day_vals[d][day])
                for day in days_sorted
                if day_vals[d][day]
            }
        agents_data[name] = agent_data

    return {"dims": dims, "agents": agents_data}


def read_weekly_from_state(run_id: str) -> dict:
    """Read weekly fulfillment from state.jsonl.

    Uses the LAST state entry of each week (settle or last activity day),
    NOT the W00-begin default values.

    Returns:
        {
            "dims": ["mood", "material", "social", "esteem"],
            "agents": {
                "Aaron Whitfield": {
                    "mood":    {"W0": 43, "W1": 95, ...},
                    ...
                },
                ...
            }
        }
    """
    data_dir = ROOT / "data" / run_id
    persona_dir = data_dir / "persona"
    dims = ["mood", "material", "social", "esteem"]

    # Collect all fulfillment values per (agent, week, dim), then average
    agent_week_vals: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for agent_dir in sorted(persona_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        state_path = agent_dir / "state.jsonl"
        if not state_path.exists():
            continue
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                time_str = rec.get("time", "")
                # Skip plan/settle entries -- only activity records
                if "activity" not in time_str:
                    continue
                # Extract week number
                m = re.search(r"W(\d+)", time_str)
                if not m:
                    continue
                week = f"W{m.group(1)}"
                ful = rec.get("content", {}).get("fulfillment", {})
                if not ful:
                    continue
                key = (agent_dir.name, week)
                for d in dims:
                    if d in ful:
                        agent_week_vals[key][d].append(ful[d])

    # Compute weekly average per dimension
    agents_data = {}
    for (name, week), dim_vals in agent_week_vals.items():
        if name not in agents_data:
            agents_data[name] = {d: {} for d in dims}
        for d in dims:
            vals = dim_vals.get(d, [])
            if vals:
                agents_data[name][d][week] = sum(vals) / len(vals)
            else:
                agents_data[name][d][week] = 0

    return {"dims": dims, "agents": agents_data}


def read_weekly_fulfillment(metrics: dict) -> dict:
    """Extract weekly fulfillment from metrics dict (legacy, use read_weekly_from_state instead).

    Returns same structure as read_daily_fulfillment but keyed by week.
    """
    dims = ["mood", "material", "social", "esteem"]
    dim_keys = {
        "mood": "fulfillment_mood",
        "material": "fulfillment_material",
        "social": "fulfillment_social",
        "esteem": "fulfillment_esteem",
    }
    by_week = metrics.get("by_week", {})
    agents_data = {}
    for name in by_week:
        agent_data = {}
        weeks_sorted = sorted(
            [w for w in by_week[name] if w != "Y2020-W00"],
            key=lambda w: int(w.split("-W")[1])
        )
        for d in dims:
            agent_data[d] = {
                w: by_week[name][w].get(dim_keys[d], 0)
                for w in weeks_sorted
            }
        agents_data[name] = agent_data
    return {"dims": dims, "agents": agents_data}


def compute_daily_emotion_dims(run_id: str) -> dict:
    """Compute all 5 emotion dimensions at daily granularity.

    Returns: {agent_name: {day_key: {dim: value, ...}}}
    """
    data_dir = ROOT / "data" / run_id
    persona_dir = data_dir / "persona"
    if not persona_dir.exists():
        return {}

    import math

    result = {}
    for agent_dir in sorted(persona_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        name = agent_dir.name
        # Skip agents with no state data
        state_path = agent_dir / "state.jsonl"
        if not state_path.exists():
            continue
        has_data = False
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    has_data = True
                    break
        if not has_data:
            continue

        result[name] = {}

        # 1. emotional_richness from state.jsonl
        day_ful = defaultdict(list)
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec.get("time", "")
                if "activity" not in t or "settle" in t:
                    continue
                mw = re.search(r"W(\d+)", t)
                md = re.search(r"D(\d+)", t)
                if not mw or not md:
                    continue
                dk = f"W{mw.group(1)}-D{md.group(1)}"
                ful = rec.get("content", {}).get("fulfillment", {})
                if ful:
                    day_ful[dk].append([ful.get(d, 0) for d in ["mood", "material", "social", "esteem"]])

        for dk, records in day_ful.items():
            if len(records) >= 2:
                n = len(records)
                means = [sum(r[i] for r in records) / n for i in range(4)]
                vars_ = [sum((r[i] - means[i]) ** 2 for r in records) / n for i in range(4)]
                avg_std = math.sqrt(sum(vars_) / 4)
                result[name].setdefault(dk, {})["emotional_richness"] = round(min(avg_std / 20.0, 1.0), 4)

        # 2-3-5. engagement, reach, reciprocity from contact/sig.jsonl
        sig_path = agent_dir / "contact" / "sig.jsonl"
        if sig_path.exists():
            week_contacts = defaultdict(lambda: {"out": set(), "in": set()})
            with open(sig_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    t = rec.get("time", "")
                    mw = re.search(r"W(\d+)", t)
                    if not mw:
                        continue
                    wk = f"W{mw.group(1)}"
                    frm = rec.get("from", "")
                    to = rec.get("to", "")
                    if frm == name and to != name:
                        week_contacts[wk]["out"].add(to)
                    elif to == name and frm != name:
                        week_contacts[wk]["in"].add(frm)

            # Distribute weekly contacts evenly to D1-D5
            for wk, c in week_contacts.items():
                out = c["out"]
                inc = c["in"]
                for day in range(1, 6):
                    dk = f"{wk}-D{day}"
                    out_list = sorted(out)
                    inc_list = sorted(inc)
                    daily_out = {out_list[i] for i in range(day - 1, len(out_list), 5)}
                    daily_inc = {inc_list[i] for i in range(day - 1, len(inc_list), 5)}
                    result[name].setdefault(dk, {})
                    result[name][dk]["social_engagement"] = round(len(daily_out) / max(len(out), 1), 4)
                    result[name][dk]["social_reach"] = round(len(daily_out | daily_inc) / max(len(out | inc), 1), 4)
                    result[name][dk]["reciprocity"] = round(
                        len(daily_out & daily_inc) / max(len(daily_out), 1), 4
                    ) if daily_out else 0

        # 4. relationship_breadth (cumulative -- replicate per day based on first seen)
        chars_dir = agent_dir / "memory" / "scratchpad" / "characters"
        if chars_dir.exists():
            char_first = {}
            for cf in sorted(chars_dir.iterdir()):
                if cf.suffix != ".jsonl":
                    continue
                with open(cf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        t = rec.get("time", "")
                        mw = re.search(r"W(\d+)", t)
                        md = re.search(r"D(\d+)", t)
                        if not mw or not md:
                            continue
                        dk = f"W{mw.group(1)}-D{md.group(1)}"
                        if cf.stem not in char_first or dk < char_first[cf.stem]:
                            char_first[cf.stem] = dk
                        break

            all_days = sorted(result[name].keys())
            seen_chars = set()
            for dk in all_days:
                for ch, first_dk in char_first.items():
                    if first_dk <= dk:
                        seen_chars.add(ch)
                result[name].setdefault(dk, {})
                result[name][dk]["relationship_breadth"] = round(min(len(seen_chars) / 15.0, 1.0), 4)

    return result


# ── PNG chart builders (matplotlib) ─────────────────────────────────────


def _extract_week_labels(weeks_data: dict) -> list:
    """Extract sorted week labels (e.g. ['W00', 'W01', ...]) from emotion week keys."""
    sorted_weeks = sorted(weeks_data.keys(), key=lambda w: int(w.split("-W")[1]))
    return [w.split("-W")[1] for w in sorted_weeks]


def _discover_target_agents(run_id: str, emotion: dict) -> List[Tuple[str, float]]:
    """Discover agents by weighted_total volatility (stdev).

    For each agent in the intersection of persona dir and emotion data,
    compute the stdev of its weighted_total across all weeks. Sort by
    stdev descending and take **top 2 + bottom 2 = 4** agents.

    Returns list of (agent_name, stdev) tuples, ordered: highest vol first,
    lowest vol last.
    """
    persona_dir = ROOT / "data" / run_id / "persona"
    persona_agents = set()
    if persona_dir.exists():
        for d in sorted(persona_dir.iterdir()):
            if d.is_dir():
                persona_agents.add(d.name)

    emotion_agents = set(emotion.get("agents", {}).keys())
    candidates = sorted(persona_agents & emotion_agents)

    if not candidates:
        return []

    try:
        scores = []
        for name in candidates:
            weeks_data = emotion.get("agents", {}).get(name, {})
            # Collect weighted_total sorted by week
            sorted_week_keys = sorted(weeks_data.keys(), key=lambda wk: int(wk.split("-W")[1]))
            totals = [weeks_data[k].get("weighted_total", 0) for k in sorted_week_keys]

            if len(totals) < 2:
                stdev = 0.0
            else:
                stdev = statistics.stdev(totals)
            scores.append((name, stdev))

        # Sort by stdev descending
        scores.sort(key=lambda x: -x[1])

        # Top 2 (highest vol) + Bottom 2 (lowest vol) = 4
        if len(scores) <= 4:
            result = scores[:]
        else:
            result = scores[:2] + scores[-2:]

        return result
    except Exception:
        # Fallback: alphabetical, take first 4
        fallback = sorted(candidates)[:4]
        return [(name, 0.0) for name in fallback]


def _extract_metrics_agents(metrics: dict, target_agents: list) -> dict:
    """Filter metrics by_week to only include target agents."""
    by_week = metrics.get("by_week", {})
    return {name: by_week[name] for name in target_agents if name in by_week}


def _plot_multiline(ax, x_labels, agent_values: dict, title: str,
                    y_label: str = "得分", y_range: tuple | None = None) -> None:
    """Plot multiple lines (one per agent) on a given Axes."""
    for i, (name, values) in enumerate(agent_values.items()):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(x_labels, values, color=color, linewidth=2, marker="o", markersize=4, label=name)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("周", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    if y_range:
        ax.set_ylim(y_range)


def make_emotion_trends_chart(emotion: dict, run_id: str,
                              target_agents: List[Tuple[str, float]]) -> str:
    """Generate emotion_trends.png — single chart, 4 lines ordered by volatility.

    Colors: red (highest σ) → orange → blue → green (lowest σ).
    Legend shows agent name + σ value.
    """
    agents = emotion.get("agents", {})
    if not agents or not target_agents:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (name, stdev) in enumerate(target_agents):
        weeks_data = agents.get(name, {})
        sorted_week_keys = sorted(weeks_data.keys(), key=lambda wk: int(wk.split("-W")[1]))
        labels = [w.split("-W")[1] for w in sorted_week_keys]
        totals = [weeks_data[k].get("weighted_total", 0) for k in sorted_week_keys]
        color = TREND_COLORS[i % len(TREND_COLORS)]
        label = f"{name} (σ={stdev:.3f})"
        ax.plot(labels, totals, color=color, linewidth=2.5, marker="o", markersize=5, label=label)

    ax.set_title("社交情感综合分波动趋势 (Weighted Total)", fontsize=13, pad=10)
    ax.set_xlabel("周", fontsize=10)
    ax.set_ylabel("综合分", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{run_id}_emotion_trends.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def make_emotion_dimensions_chart(emotion: dict, run_id: str,
                                  target_agents: List[Tuple[str, float]]) -> str:
    """Generate emotion_dimensions.png — 5 subplots (2x3) for each dimension.

    Each subplot shows the 4 target agents' scores for one dimension.
    Uses a 2x3 layout with the last cell empty (or removed).
    """
    agents = emotion.get("agents", {})
    if not agents or not target_agents:
        return ""

    sorted_names = [name for name, _ in target_agents]
    # Get week labels from first agent
    week_labels = _extract_week_labels(agents[sorted_names[0]])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("社交情感各维度变化趋势", fontsize=14, y=1.01)

    # Flatten axes for iteration
    axes_flat = axes.flatten()

    for idx, dim in enumerate(DIM_NAMES):
        ax = axes_flat[idx]
        ax.set_facecolor("white")

        dim_label = DIM_LABELS.get(dim, dim)

        agent_values = {}
        for name in sorted_names:
            weeks_data = agents[name]
            labels = _extract_week_labels(weeks_data)
            vals = [weeks_data[w]["dimensions"].get(dim, 0) for w in sorted(weeks_data.keys(),
                                                                             key=lambda x: int(x.split("-W")[1]))]
            agent_values[name] = vals

        _plot_multiline(ax, week_labels, agent_values, dim_label, y_label="得分 (0~1)", y_range=(0, 1.1))
        ax.tick_params(axis="x", labelsize=8)

    # Hide the last (6th) subplot if no dimension uses it (5 dims, 2x3=6 cells)
    axes_flat[5].set_visible(False)

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{run_id}_emotion_dimensions.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def make_metrics_fulfillment_chart(metrics: dict, run_id: str, target_agents: list) -> str:
    """Generate metrics_fulfillment.png — 4 fulfillment dimensions in 2x2 subplots."""
    by_week = _extract_metrics_agents(metrics, target_agents)
    if not by_week:
        return ""

    sorted_names = sorted(by_week.keys())
    # Get week labels from first agent (skip W00 which is start-of-sim defaults)
    first_weeks = sorted([w for w in by_week[sorted_names[0]].keys() if w != "Y2020-W00"],
                         key=lambda x: int(x.split("-W")[1]))
    week_labels = [w.split("-W")[1] for w in first_weeks]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("满足感各维度变化趋势", fontsize=14, y=1.01)

    for idx, dim_key in enumerate(FULFILLMENT_DIMS):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        ax.set_facecolor("white")

        dim_label = FULFILLMENT_LABELS.get(dim_key, dim_key)

        agent_values = {}
        for name in sorted_names:
            weeks = sorted([w for w in by_week[name].keys() if w != "Y2020-W00"],
                           key=lambda x: int(x.split("-W")[1]))
            labels_short = [w.split("-W")[1] for w in weeks]
            vals = [by_week[name][w].get(dim_key, 0) for w in weeks]
            agent_values[name] = vals

        _plot_multiline(ax, week_labels, agent_values, dim_label, y_label="得分", y_range=(30, 105))

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{run_id}_metrics_fulfillment.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def make_metrics_deposit_chart(metrics: dict, run_id: str, target_agents: list) -> str:
    """Generate metrics_deposit.png — deposit & deposit_diff trend subplots."""
    by_week = _extract_metrics_agents(metrics, target_agents)
    if not by_week:
        return ""

    sorted_names = sorted(by_week.keys())
    # Get week labels from first agent
    first_weeks = sorted(by_week[sorted_names[0]].keys(), key=lambda x: int(x.split("-W")[1]))
    week_labels = [w.split("-W")[1] for w in first_weeks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("存款趋势", fontsize=14, y=1.02)

    # Left: deposit
    ax1.set_facecolor("white")
    for i, name in enumerate(sorted_names):
        weeks = sorted(by_week[name].keys(), key=lambda x: int(x.split("-W")[1]))
        labels = [w.split("-W")[1] for w in weeks]
        vals = [by_week[name][w].get("deposit", 0) for w in weeks]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax1.plot(labels, vals, color=color, linewidth=2.5, marker="o", markersize=5, label=name)
    ax1.set_title("存款余额 (Deposit)", fontsize=11, pad=8)
    ax1.set_xlabel("周", fontsize=9)
    ax1.set_ylabel("金额", fontsize=9)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right: deposit_diff
    ax2.set_facecolor("white")
    for i, name in enumerate(sorted_names):
        weeks = sorted(by_week[name].keys(), key=lambda x: int(x.split("-W")[1]))
        labels = [w.split("-W")[1] for w in weeks]
        vals = [by_week[name][w].get("deposit_diff", 0) for w in weeks]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax2.plot(labels, vals, color=color, linewidth=2.5, marker="o", markersize=5, label=name)
    ax2.set_title("存款变动 (Deposit Diff)", fontsize=11, pad=8)
    ax2.set_xlabel("周", fontsize=9)
    ax2.set_ylabel("金额", fontsize=9)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{run_id}_metrics_deposit.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


# ── main ─────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    print(f"Loading data for {run_id}...")

    # Load metrics
    metrics = load_metrics(run_id)
    print(f"  metrics: {len(metrics.get('by_week', {}))} agents")

    # Load emotion
    emotion_path = OUTPUT_DIR / f"{run_id}_emotion.json"
    emotion = {}
    if emotion_path.exists():
        with open(emotion_path, encoding="utf-8") as f:
            emotion = json.load(f)
        print(f"  emotion: {len(emotion.get('agents', {}))} agents")
    else:
        print("  emotion: not found")

    # Discover target agents (sorted by weighted_total volatility)
    target_scores = _discover_target_agents(run_id, emotion)
    target_agents = [name for name, _ in target_scores]
    print(f"  target agents (intersection): {target_agents}")

    if not target_agents:
        print("No target agents found. Exiting.")
        sys.exit(1)

    # Print σ table
    print("\n=== 按 weighted_total 波动(stdev)排序 ===")
    if target_scores:
        flat_scores = sorted(target_scores, key=lambda x: -x[1])
        for idx, (name, stdev) in enumerate(flat_scores):
            tag = "波动最大" if idx < 2 else "波动最小"
            print(f"  {name:<25s} σ={stdev:.3f} ({tag})")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []

    # Chart 1: emotion_trends
    if emotion.get("agents"):
        print("\nGenerating emotion_trends.png ...")
        path = make_emotion_trends_chart(emotion, run_id, target_scores)
        if path:
            generated.append(path)
            print(f"  -> {path}")
    else:
        print("  Skipping emotion_trends: no emotion data")

    # Chart 2: emotion_dimensions
    if emotion.get("agents"):
        print("Generating emotion_dimensions.png ...")
        path = make_emotion_dimensions_chart(emotion, run_id, target_scores)
        if path:
            generated.append(path)
            print(f"  -> {path}")
    else:
        print("  Skipping emotion_dimensions: no emotion data")

    # Chart 3: metrics_fulfillment
    if metrics.get("by_week"):
        print("Generating metrics_fulfillment.png ...")
        path = make_metrics_fulfillment_chart(metrics, run_id, target_agents)
        if path:
            generated.append(path)
            print(f"  -> {path}")
    else:
        print("  Skipping metrics_fulfillment: no metrics data")

    # Chart 4: metrics_deposit
    if metrics.get("by_week"):
        print("Generating metrics_deposit.png ...")
        path = make_metrics_deposit_chart(metrics, run_id, target_agents)
        if path:
            generated.append(path)
            print(f"  -> {path}")
    else:
        print("  Skipping metrics_deposit: no metrics data")

    # Summary
    print(f"\nDone! Generated {len(generated)} PNG(s):")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
