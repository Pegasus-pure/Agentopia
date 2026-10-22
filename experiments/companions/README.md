# 多栋系列五角色实验

这是基于原作者 `Neph0s/Agentopia` 的独立实验入口，目标是让多栋、厘普、秀珍、小河、张田中在共同世界里行动，并分别记住自己的经历。当前交付是**可运行的事件与上下文实验，不是已训练的五角色模型**。

上游的 `src/`、时钟、经济体系、角色模板和 RFT 工具保持原样。实验复用上游模型调用接口，但不把动物角色强行放进人类职业和年度工资体系。没有移植旧 fork 的五时段或其他补丁。

## 离线运行

Python 3.10+，离线测试不需要安装第三方依赖。在仓库根目录执行：

```bash
python3 scripts/run_companions.py --db runs/demo/world.sqlite3 init
python3 scripts/run_companions.py --db runs/demo/world.sqlite3 observe --audience all --text '实验事件：院子下雨，今天的野餐取消。'
python3 scripts/run_companions.py --db runs/demo/world.sqlite3 chat --actor duodong --text '我给你准备了一份惊喜，先别告诉其他人。' --backend mock
python3 scripts/run_companions.py --db runs/demo/world.sqlite3 step --steps 3 --backend mock
python3 scripts/run_companions.py --db runs/demo/world.sqlite3 replay --actor lipu
python3 -m unittest discover -s tests -v
```

`mock` 只检查数据流，回复是明确标记的占位内容，不展示人格效果，不得用于训练。模拟与真实模型记录必须用不同数据库；运行器拒绝混用。

角色 ID：`duodong`、`lipu`、`xiuzhen`、`xiaohe`、`zhangtianzhong`。

## 连接真实模型

按上游 README 安装 `requirements.txt`，从 `config.example.json` 建立本地配置，把模型端点填入 `models`。不要把密钥或运行日志提交到 Git。

```bash
python3 scripts/run_companions.py --db runs/live/world.sqlite3 step --steps 1 --backend upstream --config config.json --model YOUR_CONFIGURED_MODEL
python3 scripts/run_companions.py --db runs/live/world.sqlite3 chat --actor duodong --text '今天过得怎么样？' --backend upstream --config config.json --model YOUR_CONFIGURED_MODEL
```

真实接口沿用上游的 `src.utils.get_response_json`，同样可对接本地模型服务。一次 `step` 请求五位角色，一次 `chat` 请求一位；上游可能调用配置中的 fallback。建议先单步验证，记录真实 token 与费用后再扩大。上游适配器尚未做服务商端到端验证；下述独立 Vertex 路径已通过真实请求。人格质量仍待评审。

## 复用现有 Gemini / Vertex AI

本机有 `gcloud auth login` 登录时，可以直接调用 Vertex Gemini，不需要启动模拟 VM，也不需要导出服务账号密钥。此路径使用 gcloud 登录身份；原服务器的 ADC 机器身份是另一种认证方式，不是把旧 fork 的补丁带回来。

```bash
python3 scripts/run_companions.py --db runs/gemini/world.sqlite3 chat --actor duodong --text '今天你想做什么？' --backend vertex --project YOUR_PROJECT_ID --model gemini-2.5-flash
python3 scripts/run_companions.py --db runs/gemini/world.sqlite3 step --steps 1 --backend vertex --project YOUR_PROJECT_ID --model gemini-2.5-flash
```

可通过 `--gcloud /absolute/path/to/gcloud` 指定 CLI；默认 `--location global`。访问令牌仅在内存中用于请求头，不写入记录。每次请求输出实际 token 使用量到 stderr，不自动估算账单。响应必须完整且为 JSON；截断、拦截、失败均停止，不进行隐藏的批量重试。支持模型取决于项目的实际权限与服务可用性。

[Google 生成接口文档](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)

## 行为与记忆语义

- 每一轮五位读取同一个轮前快照，选择移动、说话或休息；全部返回有效结果后才提交。如果任意调用失败，整轮不改变世界。已发出的模型请求仍可能产生费用。
- 移动按固定角色顺序结算，然后处理说话。同地点的角色能听到说话；离开的角色不会收到之后原地点的谈话。下一轮才会看到本轮的新事件。
- `plan` 和 `feeling` 是角色自述的私人状态，不是真实行动，也不向其他角色公开。行动结果由运行器记录。
- 用户私聊只给指定角色。`observe --audience ...` 用于明确指定谁知道一件事；`all` 是主动广播。
- 角色可以通过说话传播自己知道的信息，但没有自动“读心”接口。上下文隔离不保证模型不会猜测或主动泄露秘密，后者需要单独评估。
- SQLite 保存完整事件和当时的请求/结果，重启后继续。模型当前只能看到最近 40 条可见事件；尚未实现长程检索、自动记忆摘要或遗忘。
- `status`、`traces` 是开发者视图，包含私人计划或上下文，不应直接作为玩家页面。
- 数据库保存初始化时的角色卡快照。改卡后开新实验，不会静默改变旧实验的人设。
- 当前没有墙钟后台调度、完整关系状态机、语音、画面或自动训练器。

## 角色依据

`cast.json` 中五人的表演方向全部标记为 `provisional`。名单与身份用于建模；小院、地点和临时表演方向属于本实验设定。链接只支持名单，不能拿来证明全部性格。正式训练前需收集原作者公开角色内容，逐项登记事实、出处与允许的创作空间；旧机器人的聊天和旧草稿不会自动充当原作依据。

## 训练样本入口

每次模型调用记录原始上下文，避免用后来发生的事倒灌训练输入。接受前必须完成 [事实审核](FACT_AUDIT.md)，逐项填写依据和判断；旧 accepted 标记不能绕过新审核。

```bash
python3 scripts/run_companions.py --db runs/live/world.sqlite3 traces
python3 scripts/run_companions.py --db runs/live/world.sqlite3 review --id 1 --verdict accepted --reason '通过角色一致性、可见记忆与行动合理性检查'
python3 scripts/run_companions.py --db runs/live/world.sqlite3 export --output runs/live/reviewed.jsonl
```

默认所有样本待审，只导出通过事实审核且明确接受的真实模型样本；离线占位样本拒绝接受。导出文件使用排他创建，不覆盖旧数据。此入口是人工审核的 SFT 候选数据，不等同于上游 RFT 的优势计算，不代表已经完成蒸馏或强化学习。真实私聊也会进入 trace，导出前需确认其中内容适合训练。

下一步以整段场景划分训练/验证/测试集，避免同一事件的五人视角跨集合泄漏；平衡五位角色和行动/交流任务；保留拒绝样本，后续按需要构造偏好对。

## 设计与验证顺序

1. 整理五位的有来源角色卡，区分原作依据与实验创作。
2. 现成模型基线：测试私聊、共同活动、爽约、离开再回来等固定场景。
3. 分别评估人格区分、记忆准确、行动有效、社交连贯，不只看收入或社会地位奖励。
4. 生成和筛选少量真实轨迹；先人工评审，不把所有模拟输出当好数据。
5. 在适配 16 GB 部署目标的模型上做小规模微调，并与未训练基线使用相同场景比较。
6. 只有看到可重复改善，才扩大模拟量和训练规模。

这五位优先共用模型能力、独立保存状态与记忆。是否采用多教师、独立适配器或偏好训练，取决于基线缺陷和实验结果，尚未锁定。

## 固定场景评测

[来源登记与 v1/v2 对照方法](evaluation/README.md)。评测保留在训练入口之外，角色卡更新只影响新世界。

记忆提示进一步区分缺失记录、直接观察、转述和当前状态：无记录不能推断从未发生，当前地点不能证明昨日地点；明确给出的事件也不能随意否认。这是生成约束而非事实校验器，仍需真实评测和样本审核。

角色卡随数据库保存，但记忆规则属于运行时代码：升级后现有世界的后续调用也会使用新规则；历史 trace 中的原始提示不变。
