# Agentopia

[English](../README.md) | **简体中文** | [日本語](README.ja.md) | [한국어](README.ko.md)

**Agentopia** 是一个用于多智能体社会长期生活模拟的框架。Agentopia 以年为尺度模拟人类社会生活：在我们的实验中，100 个智能体在 10 个模拟年份里自主参与社会生活。
它们设定并追求自己的目标，发展并满足自身的需求，并与其他智能体互动，在社会中建立关系。

它围绕两个问题构建：我们能否构建一个让智能体有效模拟人类生活的 AI 智能体社会，以及来自这样一个社会的经验与奖励能否提升大语言模型的能力？为了回答后一个问题，我们定义了一种*生活奖励（life reward）*，它映射人类的幸福感——社会地位、主观满足感与经济状况——并用它来训练大语言模型，提升其拟人化与角色扮演能力。

---

## 概览

Agentopia 以年为尺度模拟人类社会生活。每个智能体：

- 设定并追求个人目标，发展技能，参与经济活动
- 在情绪、物质与社交维度上发展并满足自身需求
- 与其他智能体互动，在社会中建立关系
- 在此过程中管理自己的长期记忆
- 经历一个每周循环：**计划（Plan）→ 联络（Contact）→ 活动（Activity）→ 回顾（Review）**
- 在每个年末，更新其档案、申请新的职业，并获得一份反映社会地位、主观满足感与经济状况的*生活奖励*

一个**环境模型**（一个强大的 LLM）作为编排模拟的生成引擎——验证智能体的响应、提供反馈并调度事件——无需硬编码规则。

## 仓库结构

```
├── config.example.json     # 配置模板（复制为 config.json 并填写）
├── requirements.txt
├── data/
│   ├── apartment/          # 示例世界：现代公寓社区
│   ├── school/             # 示例世界：学校场景（中国高中）
│   └── persona_template/   # persona 数据格式模板
├── scripts/
│   ├── run_world.py        # 运行模拟的主入口
│   ├── build_rft_data.py   # 计算优势值 + 构建 RFT 训练数据
│   ├── compute_metrics.py  # 针对一次运行的每智能体 / 每年定量指标
│   ├── time_analysis.py    # 针对一次运行的每周墙钟时间统计
└── src/
    ├── agents/             # 角色扮演智能体：提示词、上下文、记忆
    └── world/             # 模拟引擎：调度、活动、奖励
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.json config.json
```

编辑 `config.json`：
- 将 `world.name` 设置为你想运行的世界（例如 `apartment`、`school`）
- 将 `role_model` 和 `god_model` 设置为 `models` 中定义的模型名称
- 填写你想使用的模型的 API 密钥和接入点
- 调整 `world.time.n_year` 以控制模拟的时长
- 设置 `fallback_model`，用于主调用失败时（例如响应无法被正确解析）的备用模型
- 调整 `max_concurrency` 以控制并行 LLM 请求的最大数量

### 3. 运行模拟

```bash
python scripts/run_world.py
```

在运行时覆盖世界设置：

```bash
python scripts/run_world.py --world apartment
```

## 模型配置

Agentopia 支持多种 LLM 后端。在 `config.json` 的 `models` 下进行配置：

| 后端 | 必填字段 |
|---|---|
| 兼容 OpenAI（vLLM、本地） | `url`、`api_key`、`vllm_model_name` |
| Anthropic（Claude） | `api_key`、`anthropic_model_name` |
| Google Gemini（Vertex AI） | `credentials_file`、`project`、`location` |
| Azure OpenAI | `url`、`api_key`、`api_version` |

对于通过 vLLM 提供服务的具备思考能力的模型，在模型配置中设置 `"enable_thinking": true`。

## 模拟数据布局

每次运行都会在 `data/` 下获得自己的目录，命名为 `worldname_MMDDHHMM`（例如
`school_06031205`）。启动时，它会从基础世界（例如 `data/school/`）复制而来，
随后所有模拟输出都会写入其中。除档案和配置文件外，数据均为追加写入（append-only）的 JSONL。

```
data/<world>_<MMDDHHMM>/      # 一次运行的目录（从基础世界 data/<world>/ 复制而来）
├── config.json               # 本次运行的生效配置（已应用 CLI 覆盖）
├── checkpoint.json           # 恢复检查点（最后完成的年/周/阶段）
├── worldview.json            # 世界设定 / 背景
├── positions.json            # 生成的可用职业岗位
├── locations.json            # 生成的地图
├── public_events.jsonl       # 世界级公共事件
├── persona/<name>/           # 每个智能体的数据
│   ├── profile/year=<YYYY>.json   # 年度档案快照
│   ├── state.jsonl                # 随时间变化的活力、满足感、技能、资产
│   ├── schedule.jsonl             # 每周日程
│   ├── activity.jsonl             # 活动结果
│   ├── reward.jsonl               # 每智能体奖励结果（社会/主观/经济/总计）
│   ├── generation/year=<YYYY>/week=<W>.jsonl   # 原始 LLM 生成轨迹
│   ├── memory/
│   │   ├── weekly_diary.jsonl     # 每周日记条目
│   │   ├── history.jsonl          # 长期生活史
│   │   └── scratchpad/            # 智能体在模拟过程中自主管理的记忆文件
│   │       ├── general.jsonl          # 核心笔记：长期目标、计划、进度、待办、反思等
│   │       ├── characters/<person>.jsonl   # 每人笔记：对该角色的了解以及智能体对彼此关系的看法（每个角色一个文件）
│   │       └── others/<thing>.jsonl        # 其他主题的笔记（每个主题一个文件）
│   └── contact/<person>.jsonl     # 智能体之间的消息记录
├── reward/                   # 世界级奖励数据
│   ├── rankings/year=<YYYY>/week=<W>.jsonl   # PageRank 输入（好感度/尊重度）
│   ├── metrics/year=<YYYY>/week=<W>.jsonl    # 每智能体的已计算奖励指标
│   └── advantages.jsonl                      # 轨迹回报 + 每期优势值
└── god/<feature>/year=<YYYY>/week=<W>.jsonl  # 环境模型的生成轨迹
```

## 生活奖励训练

Agentopia 的一个主要目标是通过社会模拟来提升 LLM 的拟人化角色扮演能力。
为此，`scripts/build_rft_data.py` 会从一次已完成的模拟中
选取高优势轨迹（参见论文第 4 节），并将其打包为训练数据。
它会衡量智能体的生活奖励、计算回报与优势值，
选取优势值最高的轨迹，并将其生成轨迹收集为一个训练集。

```bash
python scripts/build_rft_data.py --data-dir school_06031205 --top 0.25 
```

关键参数：

- `--data-dir`（必填）：`data/` 下的某次具体模拟运行目录，命名为
  `worldname_<runid>`（例如 `school_06031205`）——**而非**基础世界名称 `school`。
- `--top`：每期保留的顶部轨迹比例（默认为
  `config.json` 中的 `world.reward.rft_top_fraction`）。
- `--n-year`：将选择范围限制在前 N 个模拟年份内。

输出（位于 `rft_data/` 下）：

- `rft_data/<data-dir>_Y<year>W<week>.jsonl` — 训练样本
- `rft_data/<data-dir>_Y<year>W<week>.md` — 关于所选训练样本的统计报告
- `rft_data/god_<data-dir>_Y<year>W<week>.jsonl` — 采样的环境模型
  生成数据（仅当 `data/<data-dir>/god/` 存在时）

## 分析技能

本仓库在 `.claude/skills/` 下附带了一组 [Claude Code](https://claude.com/claude-code) 技能，
用于检视一次已完成的运行。在 Claude Code 中工作时，可按名称调用某个
技能（例如 `analyze run school_06031205`）；每个技能也在其 `SKILL.md` 前置元数据中列出了触发短语。

| 技能 | 作用 |
|---|---|
| `analyze-run` | 对一次运行进行定性深入分析——智能体的经历、内心历程、性格成长——在 `data/<run>/run_analysis/` 下生成系统级与每智能体报告。 |
| `run-metrics` | 一次运行的定量指标（token、联络、活动、消费、技能、满足感、社会评价）。封装 `scripts/compute_metrics.py`；写入 `analysis/results/<run>_metrics.json`。 |
| `analyze-activity` | 依据 `analyze-activity/PRINCIPLES.md` 中的标准，检查智能体在活动阶段的发言是否读起来像真人。使用 `scripts/extract_activity_dialogues.py`。 |
| `time-analysis` | 报告一次运行的每周墙钟耗时，从 `logs/<run>/world.log` 中解析。封装 `scripts/time_analysis.py`。 |

这些技能是可选的分析助手；它们并非运行模拟所必需。

## 许可证

本项目基于 MIT 许可证发布。
