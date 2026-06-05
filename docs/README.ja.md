# Agentopia

[English](../README.md) | [简体中文](README.zh.md) | **日本語** | [한국어](README.ko.md)

**Agentopia** は、マルチエージェント社会における長期的な生活シミュレーションのためのフレームワークです。Agentopia は人間の社会生活を年単位で何年にもわたってシミュレートします。私たちの実験では、100 体のエージェントが 10 シミュレーション年にわたって自律的に社会生活に参加しました。
エージェントは自ら目標を設定して追求し、自身のニーズを育てて満たし、他のエージェントと交流して社会の中で関係を築きます。

本フレームワークは 2 つの問いを軸に構築されています。すなわち、エージェントが人間の生活を効果的にシミュレートする AI エージェント社会を構築できるか、そしてそのような社会から得られる経験と報酬は LLM の能力を向上させられるか、という問いです。後者のために、私たちは人間のウェルビーイング（社会的地位、主観的充足感、経済状況）を反映する*生活報酬（life reward）*を定義し、それを用いて大規模言語モデルを訓練し、その擬人性とロールプレイ能力を向上させます。

---

## 概要

Agentopia は人間の社会生活を年単位でシミュレートします。各エージェントは次のことを行います。

- 個人的な目標を設定して追求し、スキルを伸ばし、経済活動に従事する
- 気分・物質・社会の各次元にわたってニーズを育て、満たす
- 他のエージェントと交流し、社会の中で関係を築く
- その過程で自身の長期記憶を管理する
- 週次サイクルを通じて生活する：**計画（Plan）→ 連絡（Contact）→ 活動（Activity）→ 振り返り（Review）**
- 各年末に、プロフィールを更新し、新しいキャリアに応募し、社会的地位・主観的充足感・経済状況を反映した*生活報酬*を受け取る

**環境モデル**（高性能な LLM）が、シミュレーションを統括する生成エンジンとして機能します。これは、ハードコードされたルールなしに、エージェントの応答を検証し、フィードバックを提供し、イベントをスケジューリングします。

## リポジトリ構成

```
├── config.example.json     # 設定テンプレート（config.json にコピーして記入）
├── requirements.txt
├── data/
│   ├── apartment/          # サンプルワールド：現代的なアパート団地
│   ├── school/             # サンプルワールド：学校設定（中国の高校）
│   └── persona_template/   # ペルソナデータ形式のテンプレート
├── scripts/
│   ├── run_world.py        # シミュレーション実行のメインエントリポイント
│   ├── build_rft_data.py   # アドバンテージの計算 + RFT 訓練データの構築
│   ├── compute_metrics.py  # 1 回の実行に対するエージェント別 / 年別の定量指標
│   ├── time_analysis.py    # 1 回の実行に対する週ごとの実時間計測
└── src/
    ├── agents/             # ロールプレイエージェント：プロンプト、コンテキスト、記憶
    └── world/             # シミュレーションエンジン：スケジューリング、活動、報酬
```

## はじめに

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 設定

```bash
cp config.example.json config.json
```

`config.json` を編集します。

- `world.name` を実行したいワールドに設定する（例：`apartment`、`school`）
- `role_model` と `god_model` を `models` で定義されたモデル名に設定する
- 使用したいモデルの API キーとエンドポイントを記入する
- `world.time.n_year` を調整してシミュレーションの長さを制御する
- `fallback_model` を設定する。これは主たる呼び出しが失敗した場合（例：応答が正しくパースできない場合）に使用されるモデルです
- `max_concurrency` を調整して並列 LLM リクエストの最大数を制御する

### 3. シミュレーションの実行

```bash
python scripts/run_world.py
```

実行時にワールドを上書きするには：

```bash
python scripts/run_world.py --world apartment
```

## モデル設定

Agentopia は複数の LLM バックエンドをサポートします。`config.json` の `models` の下で設定します。

| バックエンド | 必須フィールド |
|---|---|
| OpenAI 互換（vLLM、ローカル） | `url`、`api_key`、`vllm_model_name` |
| Anthropic（Claude） | `api_key`、`anthropic_model_name` |
| Google Gemini（Vertex AI） | `credentials_file`、`project`、`location` |
| Azure OpenAI | `url`、`api_key`、`api_version` |

vLLM 経由で提供される思考対応モデルの場合は、モデル設定で `"enable_thinking": true` を設定します。

## シミュレーションデータの配置

各実行は `data/` の下に独自のディレクトリを持ち、`worldname_MMDDHHMM`（例：
`school_06031205`）という名前が付けられます。開始時にベースワールド（例：`data/school/`）から
コピーされ、その後すべてのシミュレーション出力がその中に書き込まれます。プロフィールと
設定ファイルを除き、データは追記専用（append-only）の JSONL です。

```
data/<world>_<MMDDHHMM>/      # 1 回の実行ディレクトリ（ベースワールド data/<world>/ からコピー）
├── config.json               # この実行の有効な設定（CLI による上書きを適用済み）
├── checkpoint.json           # 再開用チェックポイント（最後に完了した年/週/段階）
├── worldview.json            # ワールド設定 / 背景
├── positions.json            # 生成された利用可能なキャリアポジション
├── locations.json            # 生成されたマップ
├── public_events.jsonl       # ワールドレベルの公開イベント
├── persona/<name>/           # エージェントごとのデータ
│   ├── profile/year=<YYYY>.json   # 年次プロフィールのスナップショット
│   ├── state.jsonl                # 時間経過に伴う活力、充足感、スキル、資産
│   ├── schedule.jsonl             # 週次スケジュール
│   ├── activity.jsonl             # 活動の結果
│   ├── reward.jsonl               # エージェントごとの報酬結果（社会/主観/経済/合計）
│   ├── generation/year=<YYYY>/week=<W>.jsonl   # 生の LLM 生成トレース
│   ├── memory/
│   │   ├── weekly_diary.jsonl     # 週次の日記エントリ
│   │   ├── history.jsonl          # 長期的な人生史
│   │   └── scratchpad/            # エージェントがシミュレーション中に自律的に管理する記憶ファイル
│   │       ├── general.jsonl          # 中核メモ：長期目標、計画、進捗、ToDo、振り返り など
│   │       ├── characters/<person>.jsonl   # 人物ごとのメモ：その人物に関する知識と、エージェントが捉える両者の関係（人物ごとに 1 ファイル）
│   │       └── others/<thing>.jsonl        # その他のトピックに関するメモ（トピックごとに 1 ファイル）
│   └── contact/<person>.jsonl     # エージェント間のメッセージログ
├── reward/                   # ワールドレベルの報酬データ
│   ├── rankings/year=<YYYY>/week=<W>.jsonl   # PageRank の入力（好意/尊敬）
│   ├── metrics/year=<YYYY>/week=<W>.jsonl    # エージェントごとの計算済み報酬指標
│   └── advantages.jsonl                      # 軌跡のリターン + 期間ごとのアドバンテージ
└── god/<feature>/year=<YYYY>/week=<W>.jsonl  # 環境モデルの生成トレース
```

## 生活報酬による訓練

Agentopia の主要な目標の 1 つは、社会シミュレーションを通じて LLM の擬人的な
ロールプレイ能力を向上させることです。そのために、`scripts/build_rft_data.py` は
完了したシミュレーションから高アドバンテージの軌跡（論文の第 4 節を参照）を選択し、
それらを訓練データにパッケージ化します。
これはエージェントの生活報酬を測定し、リターンとアドバンテージを計算し、
最も高いアドバンテージの軌跡を選択し、その生成トレースを訓練セットに収集します。

```bash
python scripts/build_rft_data.py --data-dir school_06031205 --top 0.25 
```

主要な引数：

- `--data-dir`（必須）：`data/` の下にある特定のシミュレーション実行ディレクトリ。
  `worldname_<runid>`（例：`school_06031205`）という名前で、**ベースワールド名 `school` ではありません**。
- `--top`：期間ごとに保持する上位軌跡の割合（既定値は
  `config.json` の `world.reward.rft_top_fraction`）。
- `--n-year`：選択範囲を最初の N シミュレーション年に制限します。

出力（`rft_data/` の下）：

- `rft_data/<data-dir>_Y<year>W<week>.jsonl` — 訓練サンプル
- `rft_data/<data-dir>_Y<year>W<week>.md` — 選択された訓練サンプルに関する統計レポート
- `rft_data/god_<data-dir>_Y<year>W<week>.jsonl` — サンプリングされた環境モデルの
  生成データ（`data/<data-dir>/god/` が存在する場合のみ）

## 分析スキル

本リポジトリは、完了した実行を検査するための [Claude Code](https://claude.com/claude-code) スキル一式を
`.claude/skills/` の下に同梱しています。Claude Code で作業する際は、名前で
スキルを呼び出します（例：`analyze run school_06031205`）。各スキルは `SKILL.md` のフロントマターにも
トリガーフレーズを列挙しています。

| スキル | 機能 |
|---|---|
| `analyze-run` | 実行に対する定性的な深掘り——エージェントの経験、内面の歩み、人格の成長——を行い、`data/<run>/run_analysis/` の下にシステムレベルおよびエージェント別のレポートを生成します。 |
| `run-metrics` | 実行に対する定量指標（トークン、連絡、活動、支出、スキル、充足感、社会的評価）。`scripts/compute_metrics.py` をラップし、`analysis/results/<run>_metrics.json` に書き込みます。 |
| `analyze-activity` | `analyze-activity/PRINCIPLES.md` の基準に照らして、エージェントの活動フェーズの発言が実在の人間のように読めるかをチェックします。`scripts/extract_activity_dialogues.py` を使用します。 |
| `time-analysis` | `logs/<run>/world.log` から解析した、実行の週ごとの実時間消費を報告します。`scripts/time_analysis.py` をラップします。 |

これらのスキルは任意の分析ヘルパーであり、シミュレーションの実行に必須ではありません。

## ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。
