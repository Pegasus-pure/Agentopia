# Agentopia (Extended Fork)

基于 [Agentopia](https://github.com/Neph0s/Agentopia) 深度扩展的 NPC 社会模拟引擎。原项目描述见 `docs/README_*.md`。

---

## 架构对比

| 维度 | 原项目 | 本项目 |
|------|--------|--------|
| 时间粒度 | 5天/周，无日内细分 | 5天 × 5时段，25 phase/周 |
| 偶遇系统 | God Model 全局分配 | WorldState 实时位置追踪，同地点碰撞触发 |
| 偶遇后处理 | 无 | 信息传播 + 情感追踪 + 跟随行为 |
| NPC 记忆 | 单次快照 | scratchpad 读写（目标/感知/记忆）+ 年度档案 |
| Player 交互 | 无 | 终端 UI 层（PLAN/CONTACT/Signup/Settle） |
| 数据分析 | 原始 28 指标 | 适配版 + encounter/follow 指标 + 排除 Player |

---

## 核心系统

### 周循环

```
PLAN → Signup → CONTACT(×5槽) → ACTIVITY(×5天×5时段) → Review → Settle
```

CONTACT 阶段 NPC 主动联络、邀请活动。ACTIVITY 阶段按计划执行 solo/joint/public 活动，同地点 ≥2 人自动触发偶遇对话。

### 偶遇 (WorldState + EncounterPipeline)

- WorldState 实时追踪每个 NPC 的位置
- 同地点碰撞 → 自动偶遇 → 多轮 LLM 对话（动态终结）
- 偶遇后自动：信息传播（rumour）+ 情感追踪（好感变化）+ 跟随（改计划去同一地点）

### 数据分析

```bash
python scripts/compute_metrics.py --data-dir <run_id>
```

可分析维度：
- 情感变化 — mood/material/social/esteem 四维满足感时序
- 社交关系 — 主动/被动联络次数、自我中心网络
- 活动分布 — 各地点热度、活动类型占比
- 经济状态 — 存款变化、消费金额、额外收入
- Token 消耗 — 按周/按 agent 的 LLM 调用成本

输出 `analysis/<run_id>/metrics.json`，按周+按年两层聚合。

---

## Player 状态

Player 系统已部分实现（终端交互、PLAN/CONTACT/Signup/Settle），当前 **暂停开发**，聚焦 NPC 社群完善后再回来做。

---

Forked from [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia).
