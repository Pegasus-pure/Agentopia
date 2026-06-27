# Agentopia (Extended Fork)


> [!NOTE]
> For the original project description, documentation, and setup instructions, please see:
> - [English](docs/README_en.md) | [简体中文](docs/README_zh.md) | [日本語](docs/README_ja.md) | [한국어](docs/README_ko.md)


---

## 架构对比

| 维度 | 原项目 | 本项目 |
|------|--------|--------|
| 时间粒度 | 5天/周，无日内细分 | 5天 × 5时段，25 phase/周 |
| 偶遇系统 | God Model 全局分配 | WorldState 实时位置追踪，同地点碰撞触发 |
| 偶遇后处理 | 无 | 信息传播 + 情感追踪 + 跟随行为 |
| NPC 记忆 | 单次快照 | scratchpad 读写（目标/感知/记忆）+ 年度档案 |
| Player 交互 | 无 | 终端 UI 层（PLAN/CONTACT/Signup/Settle） |

---

## 核心系统

### 周循环

```
PLAN → Signup → CONTACT(×5槽) → ACTIVITY(×5天×5时段) → Review → Settle
```

CONTACT 阶段 NPC 主动联络、邀请活动。ACTIVITY 阶段按计划执行 solo/joint/public 活动，同地点 ≥2 人自动触发偶遇对话。（new）

### 偶遇 (WorldState + EncounterPipeline)

- WorldState 实时追踪每个 NPC 的位置
- 同地点碰撞 → 自动偶遇 → 多轮 LLM 对话（动态终结）
- 偶遇后自动：信息传播（rumour）+ 情感追踪（好感变化）+ 跟随（改计划去同一地点）

### 数据分析


可分析维度：
- 情感变化 — mood/material/social/esteem 四维满足感时序（new）
- 社交关系 — 主动/被动联络次数、自我中心网络
- 活动分布 — 各地点热度、活动类型占比
- 经济状态 — 存款变化、消费金额、额外收入
- Token 消耗 — 按周/按 agent 的 LLM 调用成本


---

## Player 状态

Player 系统已部分实现，当前**暂停开发**

---

Forked from [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia).
