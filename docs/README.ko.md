# Agentopia

[English](../README.md) | [简体中文](README.zh.md) | [日本語](README.ja.md) | **한국어**

**Agentopia**는 멀티 에이전트 사회에서의 장기 생활 시뮬레이션을 위한 프레임워크입니다. Agentopia는 인간의 사회생활을 여러 해에 걸쳐 시뮬레이션합니다. 우리의 실험에서는 100명의 에이전트가 10시뮬레이션 년 동안 자율적으로 사회생활에 참여했습니다.
에이전트들은 스스로 목표를 설정하고 추구하며, 자신의 욕구를 키우고 충족하고, 다른 에이전트와 상호작용하여 사회 안에서 관계를 맺습니다.

이 프레임워크는 두 가지 질문을 중심으로 구축되었습니다. 즉, 에이전트가 인간의 삶을 효과적으로 시뮬레이션하는 AI 에이전트 사회를 구축할 수 있는가, 그리고 그러한 사회에서 얻은 경험과 보상이 LLM의 능력을 향상시킬 수 있는가입니다. 후자를 위해, 우리는 인간의 웰빙(사회적 지위, 주관적 충족감, 경제적 상태)을 반영하는 *생활 보상(life reward)*을 정의하고, 이를 사용해 대규모 언어 모델을 훈련하여 그 의인화 능력과 롤플레잉 능력을 향상시킵니다.

---

## 개요

Agentopia는 인간의 사회생활을 연 단위로 시뮬레이션합니다. 각 에이전트는 다음을 수행합니다.

- 개인적인 목표를 설정하고 추구하며, 기술을 발전시키고, 경제 활동에 참여합니다
- 기분·물질·사회 차원에 걸쳐 욕구를 키우고 충족합니다
- 다른 에이전트와 상호작용하여 사회 안에서 관계를 맺습니다
- 그 과정에서 자신의 장기 기억을 관리합니다
- 주간 주기를 통해 살아갑니다: **계획(Plan) → 연락(Contact) → 활동(Activity) → 회고(Review)**
- 매년 말에 프로필을 갱신하고, 새로운 진로에 지원하며, 사회적 지위·주관적 충족감·경제적 상태를 반영한 *생활 보상*을 받습니다

**환경 모델**(강력한 LLM)이 시뮬레이션을 총괄하는 생성 엔진 역할을 합니다. 이 모델은 하드코딩된 규칙 없이 에이전트의 응답을 검증하고, 피드백을 제공하며, 이벤트를 스케줄링합니다.

## 저장소 구조

```
├── config.example.json     # 설정 템플릿 (config.json으로 복사한 후 작성)
├── requirements.txt
├── data/
│   ├── apartment/          # 예시 월드: 현대적인 아파트 단지
│   ├── school/             # 예시 월드: 학교 환경 (중국 고등학교)
│   └── persona_template/   # 페르소나 데이터 형식 템플릿
├── scripts/
│   ├── run_world.py        # 시뮬레이션 실행의 메인 진입점
│   ├── build_rft_data.py   # 어드밴티지 계산 + RFT 훈련 데이터 구축
│   ├── compute_metrics.py  # 실행에 대한 에이전트별 / 연도별 정량 지표
│   ├── time_analysis.py    # 실행에 대한 주별 실제 소요 시간 측정
└── src/
    ├── agents/             # 롤플레잉 에이전트: 프롬프트, 컨텍스트, 기억
    └── world/             # 시뮬레이션 엔진: 스케줄링, 활동, 보상
```

## 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 설정

```bash
cp config.example.json config.json
```

`config.json`을 편집합니다.

- `world.name`을 실행하려는 월드로 설정합니다 (예: `apartment`, `school`)
- `role_model`과 `god_model`을 `models`에 정의된 모델 이름으로 설정합니다
- 사용하려는 모델의 API 키와 엔드포인트를 입력합니다
- `world.time.n_year`를 조정하여 시뮬레이션의 길이를 제어합니다
- `fallback_model`을 설정합니다. 이는 기본 호출이 실패할 경우(예: 응답을 올바르게 파싱할 수 없는 경우) 사용되는 모델입니다
- `max_concurrency`를 조정하여 병렬 LLM 요청의 최대 개수를 제어합니다

### 3. 시뮬레이션 실행

```bash
python scripts/run_world.py
```

실행 시점에 월드를 재정의하려면:

```bash
python scripts/run_world.py --world apartment
```

## 모델 설정

Agentopia는 여러 LLM 백엔드를 지원합니다. `config.json`의 `models` 아래에서 설정합니다.

| 백엔드 | 필수 필드 |
|---|---|
| OpenAI 호환 (vLLM, 로컬) | `url`, `api_key`, `vllm_model_name` |
| Anthropic (Claude) | `api_key`, `anthropic_model_name` |
| Google Gemini (Vertex AI) | `credentials_file`, `project`, `location` |
| Azure OpenAI | `url`, `api_key`, `api_version` |

vLLM을 통해 제공되는 사고(thinking) 지원 모델의 경우, 모델 설정에서 `"enable_thinking": true`를 설정합니다.

## 시뮬레이션 데이터 레이아웃

각 실행은 `data/` 아래에 고유한 디렉터리를 가지며, `worldname_MMDDHHMM`(예:
`school_06031205`)으로 이름이 지정됩니다. 시작 시 베이스 월드(예: `data/school/`)에서
복사되며, 이후 모든 시뮬레이션 출력이 그 안에 기록됩니다. 프로필과 설정 파일을
제외하고 데이터는 추가 전용(append-only) JSONL입니다.

```
data/<world>_<MMDDHHMM>/      # 한 번의 실행 디렉터리 (베이스 월드 data/<world>/에서 복사)
├── config.json               # 이 실행의 유효 설정 (CLI 재정의 적용됨)
├── checkpoint.json           # 재개용 체크포인트 (마지막으로 완료된 연/주/단계)
├── worldview.json            # 월드 설정 / 배경
├── positions.json            # 생성된 가용 진로 포지션
├── locations.json            # 생성된 지도
├── public_events.jsonl       # 월드 수준의 공개 이벤트
├── persona/<name>/           # 에이전트별 데이터
│   ├── profile/year=<YYYY>.json   # 연간 프로필 스냅샷
│   ├── state.jsonl                # 시간에 따른 활력, 충족감, 기술, 자산
│   ├── schedule.jsonl             # 주간 일정
│   ├── activity.jsonl             # 활동 결과
│   ├── reward.jsonl               # 에이전트별 보상 결과 (사회/주관/경제/총합)
│   ├── generation/year=<YYYY>/week=<W>.jsonl   # 원본 LLM 생성 트레이스
│   ├── memory/
│   │   ├── weekly_diary.jsonl     # 주간 일기 항목
│   │   ├── history.jsonl          # 장기 생애 기록
│   │   └── scratchpad/            # 에이전트가 시뮬레이션 중 자율적으로 관리하는 기억 파일
│   │       ├── general.jsonl          # 핵심 메모: 장기 목표, 계획, 진행 상황, 할 일, 회고 등
│   │       ├── characters/<person>.jsonl   # 인물별 메모: 해당 인물에 대한 지식과 에이전트가 보는 둘의 관계 (인물당 한 파일)
│   │       └── others/<thing>.jsonl        # 기타 주제에 대한 메모 (주제당 한 파일)
│   └── contact/<person>.jsonl     # 에이전트 간 메시지 로그
├── reward/                   # 월드 수준의 보상 데이터
│   ├── rankings/year=<YYYY>/week=<W>.jsonl   # PageRank 입력 (호감도/존중도)
│   ├── metrics/year=<YYYY>/week=<W>.jsonl    # 에이전트별 계산된 보상 지표
│   └── advantages.jsonl                      # 궤적 리턴 + 기간별 어드밴티지
└── god/<feature>/year=<YYYY>/week=<W>.jsonl  # 환경 모델 생성 트레이스
```

## 생활 보상 훈련

Agentopia의 주요 목표 중 하나는 사회 시뮬레이션을 통해 LLM의 의인화 롤플레잉
능력을 향상시키는 것입니다. 이를 위해 `scripts/build_rft_data.py`는
완료된 시뮬레이션에서 고어드밴티지 궤적(논문 4절 참조)을 선택하여
이를 훈련 데이터로 패키징합니다.
이 스크립트는 에이전트의 생활 보상을 측정하고, 리턴과 어드밴티지를 계산하며,
어드밴티지가 가장 높은 궤적을 선택하고, 그 생성 트레이스를 훈련 세트로 수집합니다.

```bash
python scripts/build_rft_data.py --data-dir school_06031205 --top 0.25 
```

주요 인자:

- `--data-dir`(필수): `data/` 아래의 특정 시뮬레이션 실행 디렉터리로,
  `worldname_<runid>`(예: `school_06031205`)로 이름이 지정되며 — 베이스 월드 이름 `school`이 **아닙니다**.
- `--top`: 기간당 유지할 상위 궤적의 비율 (기본값은
  `config.json`의 `world.reward.rft_top_fraction`).
- `--n-year`: 선택 범위를 처음 N개의 시뮬레이션 연도로 제한합니다.

출력(`rft_data/` 아래):

- `rft_data/<data-dir>_Y<year>W<week>.jsonl` — 훈련 샘플
- `rft_data/<data-dir>_Y<year>W<week>.md` — 선택된 훈련 샘플에 대한 통계 보고서
- `rft_data/god_<data-dir>_Y<year>W<week>.jsonl` — 샘플링된 환경 모델
  생성 데이터 (`data/<data-dir>/god/`가 존재하는 경우에만)

## 분석 스킬

이 저장소는 완료된 실행을 점검하기 위한 [Claude Code](https://claude.com/claude-code) 스킬 모음을
`.claude/skills/` 아래에 함께 제공합니다. Claude Code에서 작업할 때는 이름으로
스킬을 호출합니다 (예: `analyze run school_06031205`). 각 스킬은 `SKILL.md` 프런트매터에도
트리거 문구를 나열하고 있습니다.

| 스킬 | 기능 |
|---|---|
| `analyze-run` | 실행에 대한 정성적 심층 분석 — 에이전트의 경험, 내면의 여정, 인격 성장 — 을 수행하여 `data/<run>/run_analysis/` 아래에 시스템 수준 및 에이전트별 보고서를 생성합니다. |
| `run-metrics` | 실행에 대한 정량 지표 (토큰, 연락, 활동, 지출, 기술, 충족감, 사회적 평가). `scripts/compute_metrics.py`를 래핑하며, `analysis/results/<run>_metrics.json`에 기록합니다. |
| `analyze-activity` | `analyze-activity/PRINCIPLES.md`의 기준에 따라 에이전트의 활동 단계 발화가 실제 사람처럼 읽히는지 점검합니다. `scripts/extract_activity_dialogues.py`를 사용합니다. |
| `time-analysis` | `logs/<run>/world.log`에서 파싱한, 실행의 주별 실제 소요 시간을 보고합니다. `scripts/time_analysis.py`를 래핑합니다. |

이 스킬들은 선택적인 분석 도우미이며, 시뮬레이션을 실행하는 데 필수는 아닙니다.

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다.
