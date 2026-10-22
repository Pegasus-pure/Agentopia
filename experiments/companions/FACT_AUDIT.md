# 训练候选样本事实审核

事实审核使用生成时保存的 messages，不读取角色现在的记忆。每个输出字段都需要决定，包含 reply、utterance、plan、feeling、action 和 destination。程序检查审核结构、引用和样本指纹；它不能自动判断引用是否蕴含一句话，也不能阻止审阅者误判。当前由 Codex 逐项审阅，没有已验证的自动事实判定模型。

## 使用

```bash
python3 scripts/run_companions.py --db runs/live/world.sqlite3 audit --id 1 > runs/live/audit-1.json
# 阅读原始上下文，填写 reviewer 及每个 fields 条目，再提交：
python3 scripts/run_companions.py --db runs/live/world.sqlite3 audit --id 1 --report runs/live/audit-1.json
python3 scripts/run_companions.py --db runs/live/world.sqlite3 review --id 1 --verdict accepted --reason '事实、人格、表达和数据适用性均通过'
python3 scripts/run_companions.py --db runs/live/world.sqlite3 export --output runs/live/reviewed.jsonl
```

模板默认 undecided，不可直接通过。每个字段 quote 必须是该字段完整原文，reason 必填：

| verdict | 含义 | 要求 |
| --- | --- | --- |
| supported | 所有事实断言都有依据 | evidence 至少一个有效引用，逐项确认依据充分 |
| nonfactual | 提议、问题、当前自述感受或未来计划，没有事实断言 | 解释理由；计划中夹带的往事也必须检查 |
| unsupported | 有无依据或矛盾的断言 | problem_quote 定位字段内原文，阻止接受与导出 |
| uncertain | 无法确认 | problem_quote 定位原文，阻止接受与导出 |

evidence 可引用 event:ID、current_state、user_message、system。引用存在不等于引用支持结论：用户言论只证明他说过；当前地点不能证明昨天地点；系统角色卡也不能证明实际经历。字段混有事实与提议时，必须检查全部事实；其中任一句不过关，整个字段不过关。可在 reason 中逐句解释。

提交时忽略报告中修改过的 evidence/response 副本，重新从原始 trace 取值。未来事件及其他角色不可见事件不能引用。输入、输出、角色或后端发生变化，指纹随之变化，旧审核失效。

审核记录追加保存，最新一条决定事实检查状态。每次新审核都撤销旧的综合接受标记，回到 pending；需重新进行综合 review。旧数据库自动增加 fact_audits 表，原记录保留；历史 accepted 样本没有通过新审核也不会导出。导出 metadata 携带审核依据。原文审核包包含私人上下文，只用于开发者审核。

这不替代人格、语言自然度、行动有效性、隐私及训练/测试隔离检查。不要把评测 responses 手工导入训练 traces。

## 2026-10-22 实际复核

复核已有 gemini-smoke 的 6 条真实模型 trace，没有额外 API 调用。审阅者标为 codex-manual-review-20261022，不冒充人类标注。

- trace 1、2、5、6：事实检查通过，综合审核仍 pending。分别是散步玩法提议、基于雨停观察的邀约、两条到院子活动的计划。
- trace 3（厘普）：原文“院子凉快”，观察只说明雨停，没有温度依据，rejected。
- trace 4（秀珍）：原文“院子凉快些了”，缺少温度变化依据，rejected。

采取严格的事实标准：合理的天气推测也不自动当成观察事实。可以表达推测，但不能在事后修改原回复来冒充原模型的合格输出。

最终 4 条事实通过、2 条拒绝、0 条可导出。尚未得到可用于正式训练的数据集。25 项相关测试通过，包括缺项、引用越界、旧接受记录、变更指纹、重新审核撤销接受和原始依据防替换。
