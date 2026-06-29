# Agentopia — Living Society Engine

> **Emergent social simulation for LLM-powered NPCs.**
> Where autonomous agents build relationships, spread rumors, and encounter each other — spontaneously.

*This is an extended fork of [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia). For the original project's documentation in other languages, see the [upstream repo](https://github.com/Neph0s/Agentopia).*

---

Agentopia simulates a multi-agent society over years of simulated time. Each agent sets personal goals, develops skills, participates in economic life, manages long-term memory, and — most importantly — **encounters other agents spontaneously in shared spaces**.

This fork extends the original [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia) with a **real-time position system**, **collision-based encounter pipeline**, and **dynamic social-emotional relationship tracking** — turning a scheduled activity simulator into an emergent society.

---

## ✨ What Makes This Fork Different

| Dimension | Upstream | This Fork |
|-----------|----------|-----------|
| **Encounter** | Pre-scheduled via activity roster | **Real-time position collision** — any agents at the same location automatically trigger multi-turn dialogue |
| **Time granularity** | Day = 1 activity block | **5 phases per day** (dawn/morning/afternoon/dusk/night) — 25 phases/week, enabling intra-day scheduling dynamics |
| **After-encounter** | None | **Rumor propagation** + **affection tracking** + **follow behavior** |
| **Social-emotional analysis** | Weekly fulfillment metrics (mood/material/social/esteem) | **5-dimension social-emotional quality** (engagement, reach, breadth, richness, reciprocity) via offline analysis |
| **Fulfillment tracking** | Weekly snapshot at settle | **Daily + per-phase** fulfillment, plus emotional richness variance |
| **NPC memory** | Character scratchpad (read/write/list) | Adds **rumor injection**, fidelity decay over time, and propagation across agents |

---

## 🎯 Core Differentiators

### 🚶 Encounter System — Spontaneous Social Collisions

The heart of this fork. Instead of pre-assigning encounters like an event calendar, we track **every agent's real-time position** and detect collisions.

```
WorldState (per phase)
    │
    ├── _positions: location → [agent1, agent2, ...]
    └── _agent_location: agent → current location
    
    ▼
detect_encounters() ──→ same location, ≥2 agents?
    │                       │
    │                  NO: nothing
    │
    ▼ YES
EncounterGroup (2–4 agents)
    │
    ▼
Multi-turn LLM dialogue (dynamic termination)
    │
    ├── Rumor injection into scratchpad
    ├── Affection delta (liking / respect)
    └── Follow behavior (LLM decides to join or change plan)
```

Key features:
- **Dynamic grouping**: 2–4 agents per encounter, naturally formed by who's where
- **Multi-turn dialogue**: Configurable min/max turns with dynamic extension based on conversation quality
- **Natural termination**: Agents can leave mid-conversation when LLM judges they've had enough
- **Joiners welcome**: Latecomers to the same location can join ongoing encounters (up to MAX_JOINERS)

### 💞 Affection & Social Emotion System

Every social interaction leaves a trace. The system tracks two dimensions of interpersonal relationships:

```
                    ┌── affection (liking) ← encounter quality
    Relationship ───┤
                    └── respect (esteem)  ← social status / achievement
```

**Per-agent weekly social-emotional quality** (5 dimensions, computed offline from simulation data):

| Dimension | What it measures | Data source |
|-----------|-----------------|-------------|
| `social_engagement` | Active outreach frequency | `sig.jsonl` out-degree |
| `social_reach` | Distinct contacts per week | `sig.jsonl` out + in |
| `relationship_breadth` | Total known characters | scratchpad character count |
| `emotional_richness` | Fulfillment variance across 4 axes | `state.jsonl` std-dev |
| `reciprocity` | Bidirectional relationship ratio | mutual / total contacts |

These are aggregated into a **weighted_total** score per agent per week, enabling trend analysis over time.

### 🗣️ Rumor System — Information Spreads

When two agents encounter, what they discuss doesn't stay between them. The system injects key information into each agent's scratchpad as a **rumor entry**, which:
- Persists in the NPC's long-term memory
- Decays in fidelity over time (older entries become less reliable)
- Can be retrieved when prompted in future encounters or planning
- Spreads socially — A tells B, B tells C, and the rumor mutates

```
Encounter dialogue → extract key info
    → append_rumor() → JSONL entry (content, source, fidelity, timestamp)
    → future PLAN/REVIEW prompt reads active rumors
    → next encounter: agent may share the rumor → propagation
```

### ⏱️ Fine-Grained Time (5-Phase Days)

Each day is divided into 5 phases, giving NPCs intra-day scheduling flexibility:

```
Day:   Dawn → Morning → Afternoon → Dusk → Night
       W05     W05        W05        W05     W05       ← 5 phases/week
       D03     D03        D03        D03     D03       ← same day across all 5
```

Each phase has its own:
- **Plan** (what to do this phase)
- **Position** (where the agent is)
- **Activity execution** (solo or joint)
- **Contact slot** (reaching out to others)

This enables realistic scheduling: *"Meet Alice for coffee in the morning, go shopping alone in the afternoon, and attend a party in the evening."*

---

## 📊 Analysis & Visualization

Post-run analysis scripts generate actionable insights:

```
scripts/emotion_analysis.py <run_id>    → 5-dimension social-emotional scores
scripts/compute_metrics.py <run_id>     → per-agent weekly/yearly metrics
scripts/visualize.py <run_id>           → 4 PNG trend charts (matplotlib)
scripts/time_analysis.py <run_id>       → wall-clock time statistics
```

The **visualize.py** (now matplotlib-powered, no frontend) generates:
- **Emotion trends** — weighted_total composite over weeks, ranked by volatility
- **Emotion dimensions** — 5 subplots for each social-emotional dimension
- **Fulfillment metrics** — mood/material/social/esteem 2×2 chart
- **Economic trends** — deposit change over time

---

## 🗺️ Weekly Lifecycle

```
  ┌──────────────────────────────────────────────────────┐
  │                    WEEKLY CYCLE                      │
  │                                                      │
  │  PLAN → SIGNUP → CONTACT (×5 slots)                 │
  │                  │                                   │
  │                  ▼                                   │
  │         ┌──────────────────┐                         │
  │         │  ACTIVITY (×5 days × 5 phases)            │
  │         │  ┌─────────────────────────────────┐      │
  │         │  │ Phase loop: dawn→morning→...    │      │
  │         │  │   Plan → Update position        │      │
  │         │  │   Execute activity              │      │
  │         │  │   Detect encounters             │      │
  │         │  │   Run encounter dialogue        │      │
  │         │  │   Post-encounter effects        │      │
  │         │  └─────────────────────────────────┘      │
  │         └──────────────────┘                         │
  │                                                      │
  │  REVIEW → SETTLE                                     │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Configure

```bash
cp config.example.json config.json
# Edit config.json: set role_model, god_model, world.name, API keys
```

### Run

```bash
# Fresh simulation
python scripts/run_world.py

# With overrides: 5 agents, 10 weeks
python scripts/run_world.py --years 1 --weeks 10 --max-agents 5

# Resume from checkpoint
python scripts/run_world.py --run-id <run_id> --resume-from Y2020-W02
```

### Analyze

```bash
python scripts/emotion_analysis.py <run_id>
python scripts/compute_metrics.py <run_id>
python scripts/visualize.py <run_id>           # → 4 PNG trend charts
```

---

## 📁 Project Structure

```
├── src/
│   ├── agents/           # RoleAgent, PlayerAgent, DataManager, Prompts
│   ├── world/            # World, Clock, EncounterPipeline, WorldState
│   │   ├── encounter_pipeline.py   # Multi-turn LLM encounter dialogue
│   │   ├── state.py                # Real-time position tracking
│   │   └── world.py                # Main simulation loop & reward
├── scripts/              # run_world, compute_metrics, visualize, etc.
├── data/                 # Simulation run instances
├── analysis/results/     # Computed metrics & charts
└── docs/                 # Design documents & Mermaid diagrams
```

---

## 📜 License

Forked from [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia).
