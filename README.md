# Agentopia (Extended Fork)

This fork contains several extensions built on top of [Agentopia](https://github.com/Neph0s/Agentopia) by [@Pegasus-pure](https://github.com/Pegasus-pure). See the **What's New** section below for details.

> [!NOTE]
> For the original project description, documentation, and setup instructions, please see:
> - [Original README (English)](docs/README_original.md)
> - [简体中文](docs/README_zh.md) | [日本語](docs/README_ja.md) | [한국어](docs/README_ko.md)

---

## What's New

### 1. Player as the 101st Agent (Implemented)

The player joins as a normal NPC. The only difference: the input source is replaced from LLM API to terminal `stdin` via **dependency injection** — the core scheduler is not modified.

- Scheduler calls an abstract input interface (LLM for NPCs, stdin for player)
- Player reuses existing event types: `Solo`, `Joint`, `Public`
- Simulation runs asynchronously; only pauses when player input is required

**Value:** Introduces a "human variable" into the agent society without architectural changes.

---

### 2. Time Granularity: 5×5 Phase Plan (Implemented)

Motivation: The original `weekly_diary` produces only 5 records/week, too coarse for observing micro-emergent behaviors.

Changes:
- PLAN stage: Agents generate a **5 days × 5 phases** plan (25 items) in one shot
- ACTIVITY stage: Each day executes 5 phases, mapped to plan items
- REVIEW stage: 25 records/week instead of 5 (same logic, higher density)

Memory: Long-term preserved; short-term context reads latest 25 records (~1 week).

**Value:** Micro-emergence becomes observable and meaningful.

---

### 3. Encounter System Refactor (Implemented, Feedback Logic WIP)

The original system-driven `Encounter` event (Stage 4) is replaced with a **state-driven** approach:

- Trigger: Two agents both `Solo` + same location → auto `Contact`
- Emergence: Conversation outcome **feeds back into the original plan** — plans are mutable

The plan-feedback logic is implemented; tuning the degree of plan change via a Reward-style mechanism is planned.

**Value:** Enables "a conversation can change your trajectory" as a genuine emergent behavior, not a scripted event.

---

### 4. Dynamic Dialogue Termination (Implemented)

Problem: Fixed 20-turn dialogue is too expensive and unnatural under diary-mode density (25 events/week).

Solution:
1. Compress max turns to **8** (60% cost reduction)
2. LLM detects termination per turn: transition words ("but", "never mind") + topic drift → natural end; farewell words → early exit
3. Real-time confidence update on early termination → affects next planning cycle immediately

**Value:** Dialogue feels natural ("end when there's nothing left to say"), not mechanical.

---

### 5. Interest-Based Confidence Adjustment (Planned)

Inspired by the Reward mechanism:
- Topic-personality matchmaking factors into confidence adjustment
- Gradient decay prevents abrupt personality flips

Currently in design phase.

---

## Architectural Notes

All extensions are **non-intrusive**:
- No core scheduler rewrite
- Dependency injection pattern for player input
- Original event types reused wherever possible
- Backward compatible with the original 8-stage weekly design

---

Forked from [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia). Original README available in [`docs/README_original.md`](docs/README_original.md).
