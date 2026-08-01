# CSCR 재구현 프로젝트 — 진행 상황 & 컨텍스트

이 문서는 이 fork(`prectal123/cscr_re`)에서 진행 중인 연구/재구현 작업의 전체 맥락을 담고 있습니다.
**새로운 로컬 환경이나 새 Claude Code 세션에서 이 프로젝트를 이어갈 때는 이 파일부터 읽으면 됩니다** — Claude Code의 세션/메모리는 기기 간 동기화가 안 되기 때문에, 이 파일이 유일하게 확실한 컨텍스트 전달 수단입니다.

---

## 0. TL;DR — 2026-08-02 세션 종료 시점, 다음은 여기서부터 (최신)

**가장 최근 세션 요약은 17번 섹션(LOO unseen-model 실험) 참고, EmbedLLM pool 규격화/v1.2 프로토타입은 18번 섹션 참고(별도 세션에서 병행 진행됨).** 핵심: (1) RouterBench(11개 모델)가 너무 작아서 collapse가 생기는지 확인하려고 **LLMRouterBench**(33개 모델, 22개 태스크 카테고리 확보 가능)로 pool을 3배 확장 — parametric LOO는 여전히 유의미한 개선 없음, pool 크기 자체는 원인이 아님을 확인. (2) probe 선정이 flagship/lightweight tier gap에 지배당하는 버그 발견 → **flagship 13개를 아예 제외하고 lightweight 20개 모델로 피벗**(22개 카테고리 전부 확보). (3) PCA로 Ceiling FP를 분해해서 **핵심 메커니즘 규명**: 신호의 28.5%가 "이 모델이 전반적으로 얼마나 센가"라는 단일 coarse 축이고, 이걸 제거하면 성능이 완전히 붕괴함 — Ceiling이 세밀한 도메인 매칭이 아니라 이 coarse 축 덕분에 이겼다는 뜻. (4) **카테고리 단위로 집계한(개별 probe 아닌) Ceiling FP(Ceiling V2)**가 uniform/Perplexity/기존 Ceiling(V1) 셋 다 유의미하게 이기는 첫 깨끗한 승리 기록. (5) lightweight-20 pool의 **parametric LOO에서도 Ceiling V1이 Perplexity를 유의미하게 이김**(3-시드 평균 delta+0.109, p=0.0024로 확정) — 33개 풀에서는 안 됐던 게, tier gap을 없애니 실제 학습 단계에서도 처음으로 재현됨. (6) 33개 풀에서 Ceiling FP의 지배축이 사실상 tier(플래그십/경량) 분리축임을 직접 확인 → **"Dual-Tier 라우팅"(coarse 축=tier 게이트, 나머지=tier 내부 도메인 매칭) 후속 연구 아이디어 도출, 검증에는 EmbedLLM 규모 필요.** **다음 할 일: Ceiling V2 parametric LOO 완료 확인, EmbedLLM 자원 확보(18번 섹션 이미 진행된 pool 규격화 작업 활용), Dual-Tier 설계 착수.**

---

## 0-old. TL;DR — 2026-07-29 세션 종료 시점 (이전 기록, 참고용)

**중간 발표(랩미팅)용 정리 문서가 따로 있음** — 재구현+Observation만 깔끔하게 정리한 건 `MIDTERM_SUMMARY.md` 참고(개선 방안은 의도적으로 제외돼 있음). 새 FP 방법론 아이디어(v1.1 Lexical FP, v1.2 LLM 백본 전문성 임베딩)는 `FP_IDEAS.md`에 별도 기록.

**이번 세션(07-30)에 새로 완료한 것 — 자세한 내용은 15번 섹션**:
1. **EmbedLLM pool을 하드웨어 제약별로 필터링**: 3090 2장/4장, Colab 16GB 단일 GPU, "적당히 작은 50개" 등 여러 규격으로 `experts/pool-embedllm-*.json` 생성. 그 과정에서 registry의 MoE 모델 n_params 오표기(`Mixtral-8x7B`가 7B로 잘못 기재, 실제 47B) 및 `dolly-v2-12b` 저장소 삭제 이슈를 EmbedLLM 쪽에서도 재확인해서 전부 제외 처리. `src/router/utils.py`의 `SUS-Chat-72B` 로딩 분기가 `four_bit`를 무시하던 버그도 수정(안 고치면 4x3090에서도 OOM).
2. **v1.2 FP 방법론 프로토타입 검증(핵심)**: 실제 probe 응답을 읽고 강점/성능/하자/특징 JSON을 작성 → MiniLM 임베딩 → capability vector와 RSA 비교. **7개 pool: rho=+0.434(p=0.241), 11개로 확장: rho=+0.318(p=0.190)** — 지금까지 나온 어떤 descriptor(Logit, Perplexity)보다 방향(양의 상관)과 크기 모두 낫지만, 여전히 통계적으로 유의하진 않음. **평균 순위 이동량(5.24/21, p=0.155)이 이 프로젝트에서 나온 모든 비교 중 가장 낮은 p값** — 무작위 기준선(6.98/21)보다 확실히 덜 흔들림. 상세 수치·주의점은 15번 섹션.
3. **⚠️ 방법론 교훈**: 산점도를 고정 [0,1] 축으로 그리면(기존 `plot_rsa_scatter.py` 관행) 약한 상관관계가 실제보다 타이트해 보이는 착시가 생김 — "관계가 있다"를 주장하는 차트(v1.2 관련)는 반드시 실제 데이터 범위로 축을 확대해서 그릴 것. `scripts/plot_rsa_scatter_v12.py`에 반영함.
4. **오늘 밤 진행 예정**: `colab/test_v12_backbones.py` — 지금까지의 프로토타입은 Claude가 직접 요약문을 쓴 것(상한선 테스트)이라, 실제 가벼운 백본 모델(Phi-3-mini, Mistral-7B-Instruct) × instruction 3종 조합으로 진짜 파이프라인이 같은 신호를 내는지 검증 예정. 아직 미실행.

**이전 세션들에서 이어지는 미완료 과제**:
1. **Multi-seed 검증** (13번 섹션) — 논문 명시 하이퍼파라미터 그대로 baseline 재현 후 ablation. 환경 준비 완료, 아직 실행 전.
2. **GPU 서버 접속해서 pool 확장** (14번 섹션) — `moss-moon-003-sft`, `baize` logit descriptor 시도. 아직 미실행.
3. **EmbedLLM descriptor 계산** (15번 섹션) — pool 규격은 정했지만 Colab Pro/Pro+ 등 실행 환경 결정 및 실제 계산은 아직.

**기숙아 노트북(4GB VRAM)에서 재개하는 법**: `git clone`/`pull`만 하면 descriptor·probe·학습된 MLP(경량판)까지 다 딸려옴. MLP 재학습 없이 바로 쓰려면:
```python
from scripts.load_slim_encoder import load_slim_encoder
encoder = load_slim_encoder("local_checkpoints/slim/query-encoder-logit")
```
GPU 자체는 실질적으로 거의 필요 없음(descriptor 계산 제외 전부 CPU로 충분).

---

## 1. 프로젝트 개요

- 원본 논문/리포: **CSCR** ("Cost-Aware Contrastive Routing for LLMs", NeurIPS 2025 spotlight, arXiv:2508.12491), 원본 GitHub: https://github.com/rezashkv/cscr
- 목적: 고려대 랩 세미나용 재구현 + 논문 방법론에 대한 독자적 비판/제안. **3주 뒤 "Observation" 발표** 예정 (아직 최종 결과 발표가 아니라 예비 관찰 발표 — 새 방법론을 완전히 검증까지 할 필요는 없음).
- git 리모트 구성: `origin` = 이 fork(push 대상), `upstream` = 원본 `rezashkv/cscr`(참고용, push 안 함)
- 실험 환경: 랩실 GPU 서버 스펙 미확정이라 일단 **Google Colab (무료 티어)**으로 진행 중. 로컬 컴퓨터는 RAM 32GB / GPU 4GB(1650 Ti 추정)라 이 프로젝트의 실제 모델(6B+)은 못 돌림 — 코드 디버깅용으로만 유용.

---

## 2. 연구 논지 (이 재구현의 진짜 목적)

논문 Section 4.3.1 "Descriptor Choice"(Table 4)를 정독하다가 발견한 문제의식에서 출발:

**논문 주장**: MixInstruct(둘 다 계산 가능한 유일한 벤치마크)에서, logit descriptor 6개 + perplexity descriptor 5개를 **같은 FAISS pool 안에서 섞어도**(Mixed row) AUDC가 거의 안 변함(Logit 0.0461 → Perp 0.0467 → Mixed 0.0473) → "두 descriptor는 섞어 써도 문제없다"는 근거로 사용, "unified metric" 논지를 뒷받침.

**이 재구현이 반박/보완하려는 지점**:
1. 이 주장은 **단일 무작위 6/5 분할, 미미한 효과 크기, 데이터셋 1개**(그마저도 전부 오픈웨이트라 진짜 "강제 혼합" 상황이 아님)에만 근거함. 반복 시드 실험도 없음.
2. Logit descriptor와 perplexity descriptor는 **측정 대상과 측정 주체가 완전히 다른 프로세스**:
   - Logit: 타겟 모델 자신이 top-k vocab 토큰에 부여하는 확률 (축 = 특정 토큰)
   - Perplexity: 외부 고정 judge(GPT2)가 타겟 모델의 완성된 출력 텍스트에 매기는 surprisal (축 = 특정 probe)
   - 이걸 같은 FAISS 벡터공간에서 코사인 유사도로 비교하는 게 방법론적으로 의심스러움
3. (부차적 결함) `_get_shared_vocab_topk()`가 top-k 타겟 토큰을 **모델 자신의 tokenizer로** 뽑아서, 서로 다른 tokenizer를 쓰는 모델 간엔 "공유 vocab"이라는 이름과 달리 실제로 안 shared됨.

**제안하는 새 방법론 — "Lexical Fingerprint"**: 두 descriptor를 하나의 통일된 측정 대상으로 대체:
- **측정 대상**: 프롬프트에 대한 첫 T개 생성 토큰의 분포 (모두 동일)
- **추정 방식은 접근 권한에 따라 다름**:
  - 화이트박스(로컬, 로짓 접근 가능): 1회 forward pass로 정확한 softmax 분포 (기존 logit descriptor와 동일 비용)
  - 블랙박스(API 등, 로짓 접근 불가): temperature>0로 N회 반복 샘플링해서 경험적 빈도 분포로 근사 (Monte Carlo)
- 핵심 통찰: "확률"은 접근 권한에 따라 있을 수도 없을 수도 있지만, "샘플"은 어떤 모델이든 항상 뽑을 수 있음 — 이게 "측정 대상은 하나, 추정 전략만 접근권한별로 다름"이라는 진짜 unification.
- 세 가지 목표(Robustness, Lightness, Compatibility)를 검토한 결과: 화이트박스는 완전 만족, 블랙박스는 judge 모델을 거치지 않는다는 점에서 오히려 perplexity보다 더 robust하지만, N회 반복 샘플링 비용은 이론적 하한선(정보이론적으로 못 피함) — 다만 실측 결과 다운로드 시간이 훨씬 큰 병목이라 이 비용은 무시할 만한 수준으로 확인됨.

---

## 3. 원본 리포 구조에서 파악한 것들

- Apache-2.0 라이선스. 파이프라인: probe 생성 → descriptor 계산 → FAISS 인덱스 → contrastive query-encoder 학습 + UMR → 라우팅 평가(AUDC).
- 세 데이터셋 트랙: **EmbedLLM**(115개 모델, 대부분 대형), **MixInstruct**(11개 모델, 지금 쓰는 것), **RouterBench**(11개 모델, API 전용+오픈웨이트 혼합 — 나중에 시도할 후보로 언급됨, 아직 미착수).
- README quickstart가 `scripts/run_router_eval.py`를 언급하는데 **실제로는 존재하지 않음** — 진짜 진입점은 `scripts/run_audc_eval.py`(다른 인자 구조). README를 곧이곧대로 믿지 말 것.
- Descriptor 계산은 다운스트림(FAISS, encoder 학습, 라우팅)과 `.npy` 파일 인터페이스로 완전히 분리돼 있음 — `train_query_encoder.py`가 `proj_dim=E.size(1)`로 인코더 출력 차원을 descriptor 차원에 자동으로 맞춤. 즉 **descriptor 계산 방식을 바꿔도 다운스트림 코드는 안 건드려도 됨** (Lexical Fingerprint 구현 시 유리한 지점).
- 사전 계산된 결과물(artifacts/checkpoints/descriptors)은 전혀 없음 — 전부 처음부터 계산해야 함.

---

## 4. Colab 환경 (`colab/repro_mixinstruct.py`)

작업 스크립트: 리포 루트의 `colab/repro_mixinstruct.py`. `# %%`로 셀 구분돼 있어서, 각 블록을 순서대로 Colab 셀에 복붙해서 씀.

**핵심 설계**:
- 모델 하나씩 다운로드 → 4bit 양자화(`BitsAndBytesConfig`) 로드 → descriptor 계산 → **로컬 캐시 삭제** → 다음 모델. (11개 모델 원본 정밀도 다운로드 총합이 200GB+라 동시 보관 불가능해서 이렇게 설계함)
- 결과물(probe json, descriptor `.npy`)은 전부 **Google Drive**(`/content/drive/MyDrive/cscr_repro/`)에 저장 — Colab 런타임이 초기화돼도 안전.
- 모델 가중치 캐시(`HF_HOME`)는 Colab 로컬 디스크에만 있고 세션마다 휘발됨.
- Cell 8(logit) 루프는 **이미 완성된 모델은 스킵**, 진행 상황은 Drive의 `logit_progress.log`에 실시간 기록(크래시 나도 어디까지 갔는지 확인 가능).
- **주의**: 4bit 양자화는 VRAM만 줄이지 **다운로드 용량은 안 줄임** — `from_pretrained`가 원본 정밀도(fp16/fp32) 체크포인트를 그대로 받은 뒤 로드하면서 양자화함.

**겪었던 환경 이슈들 (전부 해결됨)**:
- Colab 런타임 유형을 GPU(T4)로 반드시 설정해야 함 — CPU로 두면 양자화가 아예 작동 안 하고 전체 모델을 RAM에 올리려다 크래시남.
- `device_map="auto"`가 사전-양자화 크기 기준으로 CPU/디스크 오프로드를 잘못 결정해서 "Some modules are dispatched on the CPU or disk" 에러 발생 → `device_map={"": 0}`로 강제 고정해서 해결.
- `.bin`(구형 포맷) 전용 체크포인트는 로딩 시 시스템 RAM을 많이 잡아먹음 → `low_cpu_mem_usage=True` 추가.
- LLaMA 계열 토크나이저는 `sentencepiece` 패키지 필요 (Cell 2에 추가함).
- pip install 후 `transformers`가 이미 import된 상태면 새로 설치한 패키지 인식이 안 됨 → **커널(런타임) 재시작 필요**, 단순 재실행으로는 해결 안 됨.
- 실패해도 다운로드 캐시는 지우지 않도록 변경(`free_model_cache`는 성공했을 때만 호출) — 로딩 단계 실패로 긴 다운로드를 낭비하지 않게.

---

## 5. 발견하고 수정한 버그 2개 (원본 리포 자체의 결함, 우리 환경 문제 아님)

### 5.1 Padding 버그 (`src/router/descriptors.py`)
`compute_logit_descriptor()`가 `tokenizer.pad_token`은 설정하지만 `tokenizer.padding_side`는 설정 안 함 → 기본값(right-padding) 사용. `batch_size>1`일 때(Cell 8은 4 사용), causal LM의 배치 생성에서 **오른쪽 패딩은 실제 버그**임 — 배치 안에서 제일 긴 문장 빼고는 실제 마지막 토큰이 아니라 PAD 토큰 위치에서 생성이 시작돼서 확률값이 왜곡됨.
**수정**: `tokenizer.padding_side = "left"` 추가. (커밋 `a46d24d`)

### 5.2 Perplexity descriptor의 NaN 오염 버그 (`scripts/compute_descriptors_perplexity.py`)
Cell 9(perplexity, mix-instruct) 최초 실행 시 크래시 발생 → 원인 분석:
- 응답 텍스트가 GPT2 tokenizer 기준 **1토큰 이하**면(`next-token loss` 계산 대상이 없음) `cross_entropy_fingerprint()`가 NaN 반환 → `perplexity_fingerprint()`가 이를 `inf`로 변환.
- descriptor 벡터 정규화(`/ np.linalg.norm(...)`) 시, 한 모델의 벡터에 `inf`가 하나라도 있으면 그 벡터의 norm 자체가 `inf`가 됨 → **나머지 정상 값들은 전부 0으로 소실**되고, 문제였던 값은 `inf/inf = NaN`으로 남음 → 모델 하나의 descriptor 전체가 사실상 파괴됨.
- 이 NaN이 `cosine_similarity()`(`--plot` 옵션)를 크래시시키는데, 이 체크가 **파일 저장 루프보다 먼저 실행**되기 때문에 크래시 나면 **11개 모델 전부 저장 안 됨** (정상 계산된 것들까지).
- **중요한 정정**: 실제로 확인해보니 "빈 응답"이 아니라 **짧지만 정답인 응답**들이었음 (`"400"`, `"Yes"`, `"paper"`, `"Amazon"` 등) — 특히 `flan-t5-xxl`이 QA 스타일 질문에 간결하게 단답하는 성향 때문에 9개 중 6개가 이 모델에서 나옴. 즉 **무작위 엣지케이스가 아니라, 간결하게 답하는 모델일수록 더 자주 걸리는 구조적 편향**.
- **Git 히스토리 확인 결과**: 이 버그는 파일 최초 커밋("add skeleton", 2025-09-06)부터 있었고 이후 두 번의 "bug fixes" 커밋에서도 안 고쳐짐. 저자들 자신의 `end_to_end.sh`도 RouterBench 계산 시 `--plot`을 그대로 씀 → 그들의 "정석" 파이프라인도 같은 상황에서 크래시 났어야 함. **원 논문의 공개 코드가 실제 논문 결과를 만든 코드와 다를 수 있다는 정황 증거** (확정은 아님).
- **수정**: `inf`가 하나라도 있는 probe(열)를 결측치로 취급해 pool 전체에서 제외 (한 모델의 이상치를 큰 상수로 대체하는 방식은 그 축이 벡터 전체를 지배해버려서 기각함). 정규화 분모에 epsilon(`1e-12`)도 추가. (커밋 `dbdd200`)
- 이 발견은 NLG 평가 문헌에서 이미 알려진 "perplexity 기반 자동 평가는 짧은 텍스트에서 불안정하다"는 한계와도 연결됨 — 발표 시 이 문헌과 엮어서 제시하면 좋음.

---

## 6. MixInstruct 모델 Pool 현황 — 최종 7개 확정 (2026-07-24 갱신)

원본 11개 중 4개는 실제 시도로 확인된 이유로 제외, 1개(chatglm-6b)는 legacy 환경 구축으로 살려냄:

| 상태 | 모델 | 비고 |
|---|---|---|
| ✅ 완료 | `eachadea/vicuna-13b-1.1` | Colab에서 계산, 이후 로컬로도 재계산(TOPK=192) |
| ✅ 완료 | `chavinlo/alpaca-native` | 상동 |
| ✅ 완료 | `stabilityai/stablelm-tuned-alpha-7b` | fp32 전용(31.75GB), 대체 정밀도 없음 |
| ✅ 완료 | `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` | 기본 revision은 안전한 `.bin`뿐이라 Windows에서 로딩 시 `[Errno 22]`남 → `revision="refs/pr/6"`(safetensors 변환 PR)로 우회 |
| ✅ 완료 | `TheBloke/koala-7B-HF` | |
| ✅ 완료 | `google/flan-t5-xxl` | 45GB로 제일 크지만 실제로 잘 됨 |
| ✅ 완료 | `THUDM/chatglm-6b` | 최신 `transformers`에서 토크나이저 로딩 버그(`get_vocab()`/`sp_tokenizer` 순서 문제) — `transformers==4.33.0` 전용 venv(`.venv-legacy`)로 해결, 로컬 5060 Ti에서 계산 |
| ❌ 제외(영구, 2026-07-25 HF API 재확인) | `databricks/dolly-v2-12b` | HF Hub에서 저장소 자체가 완전히 삭제됨(`401 Repository Not Found`) — GPU 서버로도 복구 불가능 |
| ❌ 제외(영구, 2026-07-25 HF API 재확인) | `mosaicml/mpt-7b-instruct` | dolly와 동일(`401 Repository Not Found`) — GPU 서버로도 복구 불가능 |
| 🔄 GPU 서버에서 재시도 예정 | `fnlp/moss-moon-003-sft` | chatglm과 같은 계열 코드 비호환(`is_tf_available` import 에러, 원본 코드에 우회책 있음) + **16B라 로컬 8GB VRAM은 못 감당** — 순수 VRAM 문제라 GPU 서버면 해결 가능, 14번 섹션 참고 |
| 🔄 GPU 서버에서 재시도 예정(성공 불확실) | `mosesjun0h/llama-7b-hf-baize-lora-bf16` | 저장소는 존재하나 토크나이저 파일이 없음(불완전한 커뮤니티 업로드) — 코드의 Llama-2 토크나이저 폴백으로 살아날 가능성 있음, 14번 섹션 참고 |

이 마모(model rot) 자체도 "2023년식 개인/소규모 팀 업로드 위주의 pool이라 시간이 지나며 자연 마모된 것" — 발표에 넣을 만한 관찰 포인트. **2026-07-25 GPU 서버 준비하면서 재확인한 결과**: dolly-v2-12b·mpt-7b-instruct 2개는 저장소 자체가 사라져서 영구 제외 확정, 나머지 2개(moss-moon-003-sft, baize)는 순수 하드웨어/사소한 파일 누락 문제라 GPU 서버에서 재시도 가치 있음(14번 섹션). "코드 호환성만의 문제"였던 chatglm-6b는 이미 살려냈고, moss-moon-003-sft도 비슷한 성격(VRAM만 있으면 됨)이라 GPU 서버에서 8~9개까지 pool을 키울 수 있을 전망.

**Probe 개수와 TOPK 모두 192로 통일**(2026-07-24) — 이유는 아래 9번 참고.

---

## 7. 다음 단계 — 5단계 계획 진행 중 (2026-07-24 사용자 확정)

1. ✅ **완료**: Logit·perplexity descriptor 둘 다 192차원으로 7개 모델 전부 재계산 (`local_descriptors/mix-instruct-logit/`, `local_descriptors/mix-instruct-perplexity/`)
2. ✅ **완료**: 벡터 분포 분석 — 11번 섹션 참고
3. ✅ **완료**: 7개 모델 pool로 벤치마크 필터링 — 12번 섹션 참고
4. ✅ **완료**: Query encoder(MLP) 학습(logit descriptor 기준) — 13번 섹션 참고
5. ✅ **완료(1차)**: Deferral curve 재현 — 13번 섹션 참고. **재현하다가 논문에는 없던 새로운 실패 모드(cost-band 붕괴)를 발견해서, 이게 오히려 발표의 핵심 내용이 될 가능성이 큼.** Multi-seed 검증은 다음 세션 과제로 남음.

**부가 트랙(스코프 밖, 참고용)**: RouterBench(perplexity는 무료), 다중 시드 Table 4 재현, Lexical Fingerprint 실제 구현(화이트박스는 기존 logit 코드 재사용, 블랙박스는 temperature=1.0 반복 샘플링 — 다른 온도값은 원 분포를 편향시켜서 안 됨). **3주 발표 스코프상 이것들까지 완전히 할 필요는 없음.**

---

## 8. 이 문서 사용법

새 로컬/새 세션에서 시작할 때:
1. 이 리포(`prectal123/cscr_re`)를 clone
2. 이 파일을 읽고 현재 어느 단계인지 파악
3. 아래 10번(로컬 인프라) 참고해서 환경 재구성
4. 위 "7. 다음 단계"부터 이어서 진행

---

## 9. 논문 자체의 내적 불일치 사례들 (코드 버그와는 별개, 논문 텍스트/설정 자체의 문제)

이 섹션은 5번(원본 리포의 코드 버그)과 다른 카테고리 — **논문이 스스로 적어놓은 것과 실제로 한 것/증명한 것이 어긋나는 경우**들. 코드 버그보다 반박하기 어려운 더 강한 증거.

### 9.1 "Unified metric space" 조건(N=K)을 논문 스스로 안 지킴
Section 3.1.2가 "N = K로 맞춰야 두 descriptor가 같은 단위구(unit sphere)에 놓인다"(logit의 top-k 토큰 개수 = perplexity의 probe 개수)고 명시. 근데 Section 4.1/D.1 실제 실험 설정은 **K=256(logit), N=192(probe)로 서로 다름** — MixInstruct Table 4 "Mixed" 실험(AUDC 0.0473)이 바로 이 설정에서 나온 결과. 즉 논문이 자기 조건을 자기 실험에서 안 지킨 것으로 보임.
**대응**: 우리 재구현은 논문이 말한 대로 N=K를 맞춤 — 이미 계산해둔 `N_PROBES=192`에 맞춰 `TOPK`를 256→192로 낮춤(함수/CLI 기본값 256은 그대로 두고 호출부에서만 `topk=192`로 오버라이드). Perplexity의 inf 이상치도 열 전체 드롭 대신 0-fill로 바꿔서 차원이 항상 192로 고정되게 함(0은 코사인 내적에서 "그 축 무시" 효과라 완벽한 중립값은 아니지만 실용적으로 허용 가능한 수준, 발생 빈도도 낮음 — 2304개 중 26개).

### 9.2 Cost-Spectrum InfoNCE — 본문 Eq.8과 Appendix B.3(Lemma 5.3 증명)의 공식이 다름
- **본문 Eq.8**(Section 3.4): 분모가 `exp((q^⊤e_m' − γc_m')/τ_k)` — 유사도와 비용 페널티를 **함께** τ_k로 나눔.
- **Appendix B.3**(Lemma 5.3 "Directional alignment" 증명에 실제로 쓰이는 식): `Sim = q^⊤e_m/τ_k`로 먼저 정의한 뒤 `exp(Sim_m' − γc_m')` — 비용 페널티 `γc_m'`은 **τ_k로 안 나눔**.
- 실제 코드(`scripts/train_query_encoder.py`의 `cost_spectrum_info_nce()`, `logits_k = (sim_k - cost_pen) / tau_b`)는 **본문 Eq.8 버전**을 구현함.
- τ_k가 band마다 다르므로(비싼 band일수록 τ_k가 큼), 코드대로면 비싼 band일수록 비용 페널티가 온도로 나눠져서 약해지는데, 이건 **Lemma 5.3 증명이 전제한 상황이 아님** — 엄밀히는 그 이론적 보장이 실제 구현된 loss에 대해 증명된 게 아닐 수 있음.
- **대응 방침**: 우리 학습(4단계)은 **코드(Eq.8 버전) 그대로 사용** — 코드가 이미 존재하고 실제로 쓰이는 버전이므로. 이 불일치는 발표에서 "논문 자체의 이론-구현 간 괴리" 사례로만 언급.

---

## 10. 로컬 인프라 (Colab 의존 없이 재구현+향후 실험 가능하게 구축, 2026-07-24)

- **`.venv-legacy`** (Python 3.10 + `transformers==4.33.0` 등 그 시절 조합, 정상 `pip install`로 설치됨): chatglm-6b 전용.
- **`.venv-modern`** (Python 3.10 + 최신 transformers/torch/bitsandbytes): 나머지 6개 모델 + perplexity 계산 + 기하학 분석(scipy/sklearn/matplotlib) 전부 여기서.
- **`local_models_4bit/`**: 6개 모델(chatglm-6b 제외, 옛 transformers가 4bit 모델 `save_pretrained()` 미지원) 4bit 양자화 버전 저장, 총 ~38GB — 재다운로드/재양자화 없이 즉시 로드 가능. chatglm-6b는 fp16 캐시에서 ~9초면 재양자화되니 문제없음.
- **GPU**: RTX 5060 Ti 8GB(5060 아님, 8GB 하위 모델). torch는 `cu128` 빌드(2.11.0+) 필요 — sm_120(Blackwell) 미지원인 `cu124`는 커널 없음 에러남.
- **`descriptors.py` 메모리 누수 버그(3번째 발견, 커밋 `28939e1`)**: `compute_logit_descriptor()`의 배치 루프가 `enc/gen/logits/probs`를 안 지워서 CUDA 캐시 파편화 누적 → N=32(8배치)에선 안 터지다 N=192(48배치)에서 flan-t5-xxl이 Colab T4(14.56GB)에서 OOM. 배치마다 `del` + `torch.cuda.empty_cache()` 추가로 해결.
- Colab 무료 티어 GPU 사용량 제한에 실제로 걸린 적 있음(재설정 시점 예측 불가, 구글 비공개) — 이게 로컬 인프라 구축의 직접 계기. **지금은 7개 모델 전부 로컬만으로 재계산 가능한 상태.**

---

## 11. 2단계 — 벡터 분포/기하학 분석 결과 (2026-07-24)

`local_descriptors/analysis/`에 스크립트 결과 저장(`cross_type_pca.png`, `rsa_scatter.png`, `similarity_heatmaps.png`). 7개 모델 기준.

**A. 타입 간 분리도** (logit 벡터 7개 + perplexity 벡터 7개를 같은 공간에 놓고 봄):
- Silhouette score = 0.123 (낮은 편, 완전 분리도 완전 혼합도 아님)
- MMD = 0.289, **정확 순열검정(14개를 7:7로 나누는 전체 3432가지 경우) p = 0.0012** — 통계적으로 유의미하게 구분됨

**B. 구조 재현도(RSA) — 더 중요한 결과**:
- Logit 기준 7×7 유사도 행렬 vs Perplexity 기준 7×7 유사도 행렬의 Spearman 상관 **rho = −0.079** (사실상 0)
- **정확 Mantel 검정(7! = 5040가지 순열 전부) p = 0.762** — 우연 수준, 유의미한 상관 없음
- 즉 "logit에서 가까운 모델 쌍이 perplexity에서도 가깝다"는 관계가 **전혀 없음**
- 극적인 예시: `koala-7B`↔`vicuna-13b` (logit 0.964 vs perp 0.212), `stablelm`↔`oasst-pythia` (logit 0.902 vs perp 0.048), `alpaca-native`↔`oasst-pythia` (logit 0.094 vs perp 0.890, 완전 반전)
- **해석 (manifold보다 정확한 표현)**: 회전/재배열이었다면 RSA rho가 1에 가까웠어야 함(등거리 변환은 상대적 유사도 순서를 보존하니까). rho≈0이라는 건 "같은 공간을 다르게 좌표화한 것"이 아니라 **"모델 간 유사도 관계 구조 자체가 두 descriptor 사이에 독립적"**이라는 뜻 — 훨씬 강한 주장이고 논문의 unified metric 주장에 대한 직접 반박 근거.
- **향후 검증 아이디어**: Lexical Fingerprint(화이트박스 vs 블랙박스)로 같은 RSA 테스트를 돌렸을 때 rho가 높게 나오면, "같은 대상을 다르게 추정"이라는 설계가 실제로 통일된 측정이라는 걸 증명하는 대조군이 됨.

---

## 12. 3단계 — 7개 모델 pool로 벤치마크 필터링 (2026-07-24)

**배경**: MixInstruct 원본 데이터셋은 11개 모델의 응답/점수를 담고 있는데, 우리는 로컬에서 descriptor를 계산한 **7개 모델**만 가지고 MLP(query encoder)를 학습해야 함. pool 밖 4개 모델(`dolly-v2-12b`, `moss-moon-003-sft`, `mpt-7b-instruct`, `baize`)을 가리키는 라벨/점수가 섞여 있으면 학습 라벨 차원이 안 맞거나 존재하지 않는 descriptor를 참조하게 됨.

**핵심 발견**: 이 필터링 기능은 **이미 코드베이스에 구현되어 있었음** — 새 코드 작성이 아니라 기존 옵션을 사용하는 것만으로 해결됨.

- `src/router/utils.py`의 `load_descriptors(desc_dir, pool=...)`: `pool` 리스트에 없는 `.npy` 파일은 로드 시 건너뜀.
- `src/router/mix_instruct.py`의 `MixInstructOracle(expert_names=...)`: 생성자에서 `expert_names`로 `name_to_idx`를 만들고, 데이터셋 순회 중 `expert_names`에 없는 모델의 응답/점수는 스킵. 모든 후보가 pool 밖이라 점수가 하나도 안 남는 프롬프트는 그 샘플 자체를 스킵. 라벨 벡터 길이는 `len(expert_names)`(=7)로 고정.
- `scripts/train_query_encoder.py`는 `--pool <json경로>` CLI 인자를 지원 — JSON에서 pool 리스트를 읽어 `load_descriptors(..., pool=pool)`에 전달하고, 그 결과 얻은 `desc_names`를 그대로 `MixInstructOracle(desc_names, ...)`에 넘김.

**한 일**: `experts/pool-mix-instruct-7.json` 파일 생성(HF 스타일 라벨 7개, `registry.json`/descriptor 파일명과 동일한 형식):
```json
[
  "eachadea__vicuna-13b-1.1",
  "chavinlo__alpaca-native",
  "TheBloke__koala-7B-HF",
  "stabilityai__stablelm-tuned-alpha-7b",
  "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5",
  "google__flan-t5-xxl",
  "THUDM__chatglm-6b"
]
```

**검증 결과** (`.venv-modern`, `PYTHONPATH=src/`로 직접 로드 테스트):
- Descriptor 7개 정상 로드, 각 shape=(192,)
- Cost dict 7개 모델 전부 정상 조회됨
- `MixInstructOracle` train split: **100,000 샘플**, validation split: **5,000 샘플** (전부 pool 밖 모델만 언급하는 원본 샘플은 자동 스킵된 결과)
- 라벨 벡터 길이 = 7 (pool 모델 수와 일치) 확인
- 예시 아이템: `("...tattoo...", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], None)` — 7차원 라벨

**다른 로컬(기숙사 컴퓨터)에서 재현하는 법**:
1. `git clone`/`pull`로 `experts/pool-mix-instruct-7.json`, `local_descriptors/`, `local_data/`를 그대로 받음(전부 git에 커밋되어 있어 수동 파일 이동 불필요, 총 ~1MB).
2. 4단계(MLP 학습)에 `--pool experts/pool-mix-instruct-7.json --desc_dir local_descriptors/mix-instruct-logit`(또는 `-perplexity`) 옵션만 넘기면 자동으로 7개 pool 기준으로 필터링된 데이터셋이 만들어짐 — 원본 11개 모델 데이터셋 파일 자체를 건드리거나 별도 전처리 스크립트를 돌릴 필요 없음.
3. 이 단계는 GPU가 전혀 필요 없음(descriptor는 이미 계산되어 있고, MixInstructOracle 필터링은 순수 CPU 텍스트/라벨 처리) — 4GB VRAM 노트북에서도 문제없이 재현 가능.

---

## 13. 4-5단계 — Query encoder 학습 + Deferral curve 재현, 그리고 예상 못한 발견 (2026-07-25)

### 4단계: MLP 학습

`scripts/train_query_encoder.py`로 logit descriptor 기준 학습. 아키텍처: frozen `sentence-transformers/all-MiniLM-L6-v2` CLS 토큰 → 2-layer MLP(학습 대상, 22만 파라미터) → L2 정규화. Loss는 논문의 Cost-Spectrum InfoNCE(`cost_spectrum_info_nce`, Eq.8 버전, 9번 섹션 참고).

**실측 성능** (RTX 5060 Ti 8GB): 데이터 로드 19초 + 인코더 로드 7초 + 학습 자체 2 epoch에 66초(step당 21ms). **GPU 메모리 111MB**로 4GB 노트북에서도 CPU만으로도 충분히 재현 가능한 수준.

### 5단계: Deferral curve 재현 + 겪은 버그들

`scripts/run_audc_eval.py`로 knn(우리 학습된 라우터)/random/oracle 비교. 겪은 이슈들:

1. **`--k 1`로 처음 돌렸더니 KNN 커브가 완전 수평선** — `KNNRouter`가 FAISS에서 top-k만 후보로 뽑은 뒤 그 안에서 λ(비용)로 저울질하는데, k=1이면 후보가 하나뿐이라 저울질 자체가 불가능. **`--k 7`(pool 전체)로 바꿔야 실제 cost-accuracy trade-off가 나타남.**
2. **플롯이 논문과 다르게 x축 간격이 고르다는 사용자 지적** — `run_audc_eval.py`의 내장 플롯은 AUDC 적분용으로 만든 인위적 균등 그리드(`build_cost_grid`+`interp_to_grid`)를 그리고 있었음. `--save_curve`가 저장하는 pickle은 보간 전 raw `(cost, acc)`를 담고 있어서, 그걸로 다시 그리면 논문처럼 불규칙한 뭉침이 나옴(주로 `λ`가 로그 간격이라 저λ 구간에 점들이 몰림). x축 단위도 `n_params × 0.03`이라 raw B로 보려면 0.03으로 나눠야 함.
3. **Windows 콘솔 유니코드 크래시** — `run_audc_eval.py`의 4곳(em-dash `—`, Δ 포함 print문)이 cp949에서 크래시. ASCII로 교체(`DeltaAUDC`, 일반 하이픈)해서 수정, 유의성 검정(bootstrap) 로직 자체는 정상 작동 확인됨.

### 핵심 발견: Cost-band가 작은 pool에서 완전히 붕괴함

AUDC 결과 (`k=7`, `n_bands=5`, 2 epoch): knn=0.0386 vs random=0.0377 (+2.4%, bootstrap p=0.004로 통계적 유의). 근데 raw curve를 뜯어보니 **중간 비용 구간(~7B)에서 KNN이 random보다 오히려 낮음**(0.027 vs 0.037).

원인을 진단한 결과(`k=7`, 검증셋 5000개 프롬프트에서 특정 λ의 라우팅 분포 확인): 학습된 라우터가 **프롬프트별 판단을 전혀 안 하고, 그 순간 (유사도−λ·비용) 점수가 제일 높은 전문가 단 하나에 전체 프롬프트를 통째로 몰아줌**:
- λ≈0 (비용압박 없음): oasst-pythia-12b 72%
- λ=0.616 (딥 지점): stablelm 98.7%
- λ=48.3 (최대압박): chatglm-6b 99.9%

**`n_bands` sweep(5→3→2→1)으로 원인 추가 확인**: 각기 다른 밴드 수로 재학습했더니 AUDC가 5(+2.4%) → 3(-21%) → 2(-17%) → 1(+9.5%)로 들쭉날쭉. 처음엔 "밴드가 적을수록 좋다"로 오해했으나, 라우팅 분포를 까보니 **밴드 개수가 원인이 아니라 각 학습이 우연히 어떤 전문가 하나에 수렴했는지가 전부**였음:
- n_bands=3, 2: 둘 다 독립적으로 **flan-t5-xxl**(94~99.6%)에 붕괴 — 이 모델의 평균 정확도가 낮아서(0.018~0.019) AUDC가 나쁘게 나옴
- n_bands=1: **alpaca-native**(99.6%)에 붕괴 — 이 모델은 평균 정확도가 높아서(0.047) AUDC가 좋게 나옴
- n_bands=5(원본): **oasst-pythia-12b**(72%)에 붕괴, 역시 평균 정확도 높음(0.048)

flan-t5-xxl이 두 번이나 독립적으로 붕괴 대상이 된 건 우연이 아닐 수 있음 — 유일한 인코더-디코더(T5) 구조라 descriptor 공간에서 나머지 6개(전부 디코더 전용)와 구조적으로 떨어진 outlier였음(11번 섹션의 gap 분석에서도 flan-t5-xxl 관련 쌍들이 유독 낮은 유사도를 보였음).

**결론(발표 핵심 논지 후보)**: `n_bands` 설정이나 pool 크기 자체보다, **Cost-Spectrum InfoNCE가 7개짜리 작은 pool에서는 프롬프트 조건부 라우팅을 전혀 학습 못 하고, 매 학습마다(랜덤 시드에 따라) 서로 다른 단일 "favorite" 전문가로 붕괴한다**는 게 핵심 관찰. AUDC가 좋게 나오는지 나쁘게 나오는지는 그 붕괴 대상이 우연히 성능 좋은 모델이었는지에 달려있어서, 재현성이 없음. **다음 검증 과제**: 같은 설정으로 시드만 바꿔 3~5회 반복해서 이 붕괴 패턴이 진짜 일관된 현상인지 확인 필요(아직 미완료).

### 결과물 위치
- `local_descriptors/analysis/deferral_curve_logit_nbands{5,3,2,1}_raw.png`: 각 n_bands의 raw curve
- `local_descriptors/analysis/deferral_curve_logit_nbands_comparison.png`: 4개 비교 한 그래프
- `local_checkpoints/deferral_curve_logit*.pkl`: raw (cost, acc) 데이터
- `local_checkpoints/slim/query-encoder-logit*/`: 학습된 MLP 체크포인트 경량판(`proj.pt`+`config.json`만, 각 ~900KB). Frozen base(MiniLM, 87MB)는 4개 런 전부 동일해서 git에 중복 저장 안 하고, `scripts/load_slim_encoder.py`로 base를 HF에서 새로 받아 재조립함:
  ```python
  from scripts.load_slim_encoder import load_slim_encoder
  encoder = load_slim_encoder("local_checkpoints/slim/query-encoder-logit")
  ```
  기숙사 컴퓨터에서는 이걸로 재학습 없이 바로 라우터 복원 가능(인터넷으로 MiniLM 90MB 한 번만 받으면 됨).

---

## 14. GPU 서버에서 실험 이어가기 (2026-07-25)

**목표**: 로컬(8GB VRAM)에서 못 돌린 모델들의 **logit descriptor**를 마저 계산해서 7개 pool을 최대한 키우는 것. (**Perplexity descriptor는 이미 11개 전부 로컬에 있음** — 응답 텍스트가 MixInstruct 데이터셋에 이미 포함돼 있어서 모델 실행 자체가 필요 없었기 때문. `local_descriptors/mix-instruct-perplexity/`에 11개 `.npy` 파일 확인됨. 하지만 이건 logit descriptor가 못 도는 이유와는 무관함 — 아래 참고.)

**⚠️ 중요한 정정 (2026-07-25 재확인)**: 처음에 "나머지 4개 계산하면 11개 다 채워진다"고 GPU 스크립트를 짰다가, 6번 섹션(이전 세션에 이미 확인된 제외 사유)이랑 안 맞는 걸 뒤늦게 발견해서 HF Hub API로 다시 확인함:
- `databricks/dolly-v2-12b`, `mosaicml/mpt-7b-instruct`: **저장소 자체가 HF Hub에서 완전히 삭제됨** (`401 Repository Not Found`, 2026-07-25 API로 재확인) — GPU/VRAM을 아무리 늘려도 존재하지 않는 저장소는 못 받음. **영구 제외.**
- `mosesjun0h/llama-7b-hf-baize-lora-bf16`: 저장소는 존재하지만 토크나이저 파일이 없는 게 재확인됨 — 다만 `load_model_and_tokenizer()`에 `AutoTokenizer` 로딩 실패 시 Llama-2 토크나이저로 폴백하는 기존 코드가 있어서(baize가 llama-7b 기반 LoRA라 vocab 호환 가능성 있음), **시도는 해볼 가치 있지만 성공 보장은 안 됨.**
- `fnlp/moss-moon-003-sft`(16B): 저장소·토크나이저 파일 다 있음. **진짜 VRAM 문제였던 유일한 모델 — GPU 서버로 확실히 해결 가능.**

**현실적 목표치는 11개가 아니라 9개**(현재 7개 + moss-moon-003-sft 확정 + baize 성공하면 +1). GPU 스크립트도 이에 맞춰 dolly/mpt-7b-instruct는 아예 시도 안 하도록 수정함.

**사용법 (SSH 접속 → git clone → 스크립트 1개 실행)**:
```bash
git clone https://github.com/prectal123/cscr_re.git
cd cscr_re
bash scripts/setup_and_run_gpu_server.sh
```
이 스크립트(`scripts/setup_and_run_gpu_server.sh`)가 하는 일:
1. `.venv` 생성 + `pip install -e .` + 로컬에서 필요했던 추가 패키지(`bitsandbytes`, `sentencepiece`, `protobuf`) 설치
2. `nvidia-smi`로 VRAM 확인 → **40GB 미만이면 자동으로 4bit 양자화**(`--four_bit` 플래그) 적용, 그 이상이면 bf16 그대로
3. torch가 GPU를 제대로 보는지 확인(안 보이면 CUDA 버전 안내 메시지와 함께 중단)
4. `moss-moon-003-sft`, `baize` 두 모델의 logit descriptor를 순서대로 계산해서 `local_descriptors/mix-instruct-logit/`에 저장

**재실행 안전함**: 이미 계산된 `.npy` 파일이 있으면 건너뛰고, 중간에 실패한 모델만 재시도 — 다운로드 도중 끊겨도 처음부터 다시 할 필요 없음.

**구현 세부사항**: 원본 `load_model_and_tokenizer()`(`src/router/utils.py`)는 기본 bf16 로드만 지원했음(4bit 양자화 없음) — `four_bit: bool = False` 파라미터를 추가하고, `True`일 때 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`를 써서 로컬에서 썼던 것과 동일한 NF4 양자화 경로를 추가함(`scripts/compute_descriptors.py --four_bit`로 노출). 원본 코드에 moss-moon-003-sft용 토크나이저 우회(`revision="refs/pr/6"`)가 이미 들어있어서 호환성 이슈는 낮을 것으로 예상.

**스크립트가 자동으로 안 하는 것(판단이 필요해서 의도적으로 남겨둠)**:
- `experts/pool-mix-instruct-7.json`을 8~9개로 확장할지 여부, FAISS 인덱스 재구성 — pool이 커지면 학습 pool이 커져서 cost-band 붕괴 문제(13번 섹션)가 완화될 수도 있음, 확인해볼 가치 있음.
- Multi-seed n_bands 재검증(13번 섹션 미완료 과제)이나 RouterBench mistral-7b-chat descriptor 계산 — 서버 여유 되면 같이 진행 가능.

---

## 15. LOO unseen-model 실험 — collapse 발견, mean-pooling 버그 발견, RouterBench로 확장 (2026-07-30~31)

**배경**: "FP가 capability에 정렬되면 unseen 모델 routing이 잘 된다"는 명제를 검증하려고, Perplexity/Ceiling(BartFP)/V1.2 FP 각각에 대해 leave-one-out(11개 중 1개 제외 후 MLP 학습, 제외된 모델 descriptor를 사후에 추가해서 잘 찾아가는지) 실험을 기획. 로컬 GPU(GTX 1650 Ti, CUDA torch로 재설치) 사용.

### 15.1 발견한 것 — MLP 학습이 collapse함 (MixInstruct 기준)

`scripts/loo_unseen_recovery.py`로 실험하다가, loss가 baseline(랜덤 초기화) 수준에서 안 내려가는 현상 발견. 여러 단계로 원인 진단:

1. **MixInstructOracle 기본 `MARGIN=0.1`이 degenerate**: 라벨의 82~90%가 all-positive. `TRAIN_MARGIN=0.01`로 재보정(스코어 갭 median=0.0075 기준으로 선택) → avg positive/row 9.04→2.92, all-positive 82%→5.6%.
2. **`cost_info_nce` 손실 함수 자체는 논문 원본(`scripts/train_query_encoder.py`, upstream과 diff 없음 확인)과 완전히 동일** — 우리 코드 문제 아님.
3. **원 논문 `QueryEncoder.encode()`가 CLS 토큰 pooling을 씀** — 근데 `all-MiniLM-L6-v2`는 Mean Pooling으로 학습된 모델. CLS pooling 시 baseline anisotropy(무관한 프롬프트 200개 pairwise cosine sim) 0.62, Mean pooling으로 고치면 0.05로 10배 이상 개선. **논문 레퍼런스 코드 자체의 버그로 추정.** `loo_unseen_recovery.py`에 `mean_pool()` 헬퍼 추가해서 전부 교체.
4. **Mean pooling 적용해도 collapse는 부분적으로만 완화됨** (쏠림 68~75%→50.5%, 여전히 심함). Ceiling FP를 collinearity 고쳐서 다시 넣어도(아래 15.2) 거의 같은 3개 모델(oasst-pythia/chatglm/alpaca)로 collapse — **descriptor 기하구조를 바꿔도 collapse 대상이 거의 안 바뀜** → 라벨/비용 구조가 만드는 attractor로 추정.
5. **Positive control 테스트**(합성 E, 완전히 클린한 1:1 랜덤 라벨, cost=0)로도 collapse 재현 — loss/라벨 문제가 아니라 **MiniLM 인코더+헤드 조합이 프롬프트 조건부 학습보다 collapse를 더 쉬운 해로 찾는다**는 것 시사.
6. **환경 자체는 정상**: sklearn digits 데이터셋으로 완전히 무관한 10-class 분류기(같은 LR/optimizer/GPU)를 학습시켜보니 loss 2.26→0.35로 정상 수렴 — torch/CUDA/backward pass는 멀쩡함.

### 15.2 Ceiling FP(BartScore 기반) collinearity 버그 발견 + 수정

`build_ceiling_fp_clustered.py`로 만든 원래 Ceiling FP는 pairwise cosine sim mean=0.982(거의 구분 불가) — bartscore가 "프롬프트 난이도"라는 전 모델 공유 성분에 지배당해서, 실제 모델 간 실력 차이가 파묻힘. **`scripts/build_ceiling_fp_centered.py`**로 클러스터별 pool-mean을 빼고(mean-centering) 나서 L2-normalize → mean=-0.08, std=0.72로 극적으로 개선.

### 15.3 V1.2 FP(LLM 전문성 요약 임베딩) 실제 구현

`scripts/build_v12_fp.py`: 32개 probe의 (prompt, 실제 응답) 쌍을 모델별로 묶어서 Qwen2.5-1.5B-Instruct에게 "System X"(3인칭, self-reference 방지 프롬프트 필요 — 처음에 0.5B로 시도했을 때 모델이 자기 자신 얘기를 해버리는 문제 있었음, role 분리 프롬프트로 해결)의 전문성을 요약시키고, 그 요약을 MiniLM mean-pooling으로 임베딩. Collinearity: mean=0.82(Ceiling보다는 나은데 여전히 높음 — LLM이 생성하는 요약 문체가 다 비슷해서 디테일이 mean-pooling 과정에서 묻히는 것으로 추정).

### 15.4 MLP 학습 없는 kNN 기반 검증법 (collapse 우회)

MLP 학습 자체가 collapse에 취약해서 FP 품질 비교 자체가 오염된다는 문제 제기(사용자) → **`scripts/knn_unseen_recovery.py`**: 학습 없이, held-out 모델 M의 FP-공간 코사인 유사도로 나머지 10개 모델의 **실제 점수**를 가중평균해서 M의 실제 점수를 예측(FP-weighted proxy) vs 그냥 10개 평균(uniform proxy) — Spearman rho 비교. Gradient descent가 전혀 없어서 collapse 위험 자체가 없음.

**MixInstruct 결과** (11-fold paired t-test):
| FP | mean delta | p |
|---|---|---|
| Perplexity | -0.0076 | 0.032 (유의, 마이너스) |
| Ceiling | -0.0070 | 0.342 (유의 안 함, 노이즈) |
| V1.2 | -0.0003 | 0.028 (유의하지만 크기 무의미) |

**셋 다 uniform baseline을 못 이김.** 원인: uniform proxy 자체가 이미 rho=0.73으로 매우 강함 — bartscore가 "프롬프트 난이도" 공유 성분에 지배당해서, 모델 고유 실력차라는 잔차 신호가 거의 안 남음 (Ceiling FP의 collinearity 문제와 동일 계열의 confound).

### 15.5 RouterBench로 확장 — 여기서는 다른 결과

MixInstruct는 거의 균질한 instruction-following 프롬프트라 태스크별 강약 구조가 약함. **RouterBench**(`withmartian/routerbench`, HF Hub, 36,497행, 11개 모델 — GPT-4/Claude 계열/Mixtral/Llama-2-70B 등, **86개의 실제 이질적 태스크**(hellaswag/grade-school-math/mmlu-professional-law/arc-challenge 등), binary correct/incorrect 점수, 실제 응답 텍스트, 실제 $비용)로 같은 실험 반복.

- Ceiling FP: eval_name을 클러스터로 써서(우리 k-means 대신 진짜 태스크 카테고리 사용) mean-centering. `scripts/routerbench_knn_test.py`.
- Perplexity FP도 여기서 만들 수 있음이 뒤늦게 확인됨 — **Perplexity FP는 타겟 모델의 로그프롭이 아니라 고정된 GPT-2로 응답 텍스트의 cross-entropy를 재는 방식**(`scripts/compute_descriptors_perplexity.py`)이라 API 전용 모델(GPT-4/Claude)에도 적용 가능. `scripts/build_routerbench_perplexity_fp.py`로 별도 구현(원본 스크립트의 내부 split(seed=47, 90/10)이 우리 Set A/B split(seed=42, 80/20)과 안 맞아서 leakage 위험 있었음 — 그래서 우리 split에 맞춰 새로 작성).

**RouterBench kNN 결과** (11-fold paired t-test):
| FP | mean delta | p | 개선 fold |
|---|---|---|---|
| **Ceiling** | **+0.0982** | **0.0205 (유의)** | 8/11 |
| Perplexity | +0.0111 | 0.1626 (유의 안 함, 방향은 양수) | 8/11 |
| V1.2 | +0.0001 | 0.9726 (유의 안 함) | 4/11 |

**Ceiling FP가 RouterBench에서 통계적으로 유의한 개선을 보임.** MixInstruct와 다른 결과 — RouterBench의 진짜 태스크 다양성(86개 카테고리)이 "모델별 안정적 강약 패턴"을 만들어주고, Ceiling FP(카테고리별 pool-mean-centered 정확도)가 그 패턴을 실제로 포착한 것으로 해석됨.

**결론(잠정)**: "FP를 capability에 정렬시키면 unseen 라우팅이 좋아진다"는 명제는 무조건 참/거짓이 아니라, **벤치마크에 안정적인 태스크별 capability 구조가 존재해야 성립하는 조건부 명제**로 정제됨.

### 15.6 Ceiling FP의 잔여 문제 — architecture vs capability 혼동

RouterBench Ceiling 결과에서 개선 안 된 3개 fold(llama-2-70b-chat, mixtral-8x7b-chat, wizardlm-13b) 분석:
- **llama-2-70b-chat**: FP 최근접 이웃이 code-llama-34b(유사도 0.987!)인데 정확도는 0.20 vs 0.33으로 전혀 다름 — 둘 다 Meta LLaMA-2 계열이라 **아키텍처/계보 유사성이 capability 유사성으로 오염**된 것으로 추정. MixInstruct에서 flan-t5-xxl이 T5 구조라는 이유만으로 outlier가 됐던 것과 같은 패턴.
- **mixtral-8x7b-chat**(정확도 0.55, pool 평균 0.52보다 위 — 약한 모델 아님): FP 최근접 이웃이 claude-v1/gpt-3.5/claude-v2(정확도 0.60~0.64) — 실제보다 더 비슷하다고 착각.
- **wizardlm-13b**: 누구와도 유사도 0.39 이하로 낮음 — FP 공간에서 고립, 기댈 이웃 정보 자체가 빈약.

**즉 "성능이 낮아서 안 됨"이 아니라, mean-centering이 "카테고리별 상대적 강약 패턴(모양)"만 남기고 "전체 실력 수준"을 지워버려서, 패턴은 비슷해도 절대 실력이 다른 모델을 가깝다고 착각하는 문제**로 진단됨.

### 다음 세션(2026-07-31) 할 일
1. **RouterBench에서도 실제 MLP LOO 실험** (지금까지는 kNN 방식으로 학습 없이만 검증함 — 실제 쿼리 인코더가 RouterBench의 다양한 태스크 프롬프트에서도 collapse 없이 학습되는지 확인 필요. MixInstruct보다 프롬프트 다양성이 훨씬 높아서(태스크 86종) MiniLM 임베딩 anisotropy가 덜할 가능성 있음 — 아직 검증 안 됨).
2. **V1.3 아이디어 — architecture/capability 혼동 완화**: mean-centering이 "전체 실력 수준" 성분을 과도하게 지워버리는 문제. 패턴(mean-centered)과 절대 수준을 같이 보존하는 방식(예: 별도 차원 추가, 부분적 centering 등) 검토.
3. V1.2는 우선순위 낮춤(원시적 방법론으로 처음부터 인지하고 있었음, MixInstruct/RouterBench 둘 다 효과 없음 확인됨).
4. **RouterBench Ceiling kNN 결과 보강 (내일 면담 전 우선순위 높음)** — 목적: "capability-aligned FP가 unseen routing을 유의미하게 개선할 수 있다"는 명제 자체의 정량적 근거(V1.3/V1.4를 만들 당위성)를 단단하게 만들기. Ceiling FP의 "실전 배포 가능성"은 이미 논외로 합의됨(무거워도 상관없음 — 천장을 보여주는 게 목적).
   - **진짜 이론적 천장과 비교**: 지금 Ceiling FP(86개 카테고리로 뭉개고 mean-centering한 근사치)보다 더 강한 상한선 필요. Set A에서 모델 i,j의 **원본 프롬프트별 점수의 실제 상관관계**(카테고리 집계 없이)를 직접 유사도 가중치로 써서 같은 kNN 테스트 재실행 → "선형 가중평균으로 얻을 수 있는 절대적 최댓값" 산출. 지금 Ceiling delta(+0.098)가 이 진짜 천장의 몇 %를 잡고 있는지 비교 — 천장이 훨씬 높으면 "더 나은 FP 방법론을 만들 여지가 크다"는 강한 동기 부여 논거가 됨.
   - **Outlier 의존성 점검**: 지금 delta(+0.0982, p=0.02)가 11-fold 중 code-llama-34b(+0.339)/mistral-7b(+0.252) 2개 fold에 크게 쏠려있음(나머지는 +0.06~0.14 또는 소폭 마이너스). mean 대신 median delta도 같이 보고하거나, 이 2개 outlier를 뺐을 때도 방향성(양수)이 유지되는지 확인해서 "소수 이상치에 의존하는 결과 아니냐"는 반박에 미리 대비.

---

## 16. 실제 MLP LOO 실험 + Load-Balancing Loss로 collapse 완화 시도 (2026-07-31)

### 16.1 실제 MLP LOO — kNN 신호가 실제 학습으로 안 옮겨짐

`scripts/routerbench_loo_recovery.py`로 RouterBench에서 실제 쿼리 인코더 LOO 학습(Ceiling FP, `cost_info_nce`)을 돌려보니, MixInstruct와 똑같이 collapse가 재현됨 — 여러 fold에서 oracle_match_rate가 0에 가까움. **프롬프트 다양성이 높은 RouterBench에서도 collapse는 그대로**임을 확인(raw 임베딩 다양성 자체는 0.24로 양호했는데도).

`cost_spectrum_info_nce`(밴드 버전, 논문 Eq.8)로 바꿔서도 재현 — 오히려 한 fold는 200/200(100%) 단일 모델로 collapse. **LOO 없이 11개 전부 넣고 학습시켜도 동일한 3개 모델(Yi-34B/mixtral/gpt-3.5)로 쏠림 재현** — LOO 구조 자체의 문제가 아님을 확인.

### 16.2 원인 진단 — "어려운 문제만 걸러도" 안 통함, 완전 대칭 합성 실험으로 원인 확정

- RouterBench Set A에서 "1~2개 모델만 정답인 어려운 프롬프트"(3167개, 10.8%)만 걸러서 재학습해도 collapse 유지 — 그 안에서도 gpt-4가 47.1%로 여전히 압도적 1위(어려운 문제일수록 오히려 제일 똑똑한 모델만 맞히니까).
- **완전 대칭 합성 실험**(5개 실제 도메인, 각 도메인마다 정확히 1개 모델만 90% 전문가, 전반적 1등 없음, 비용도 균등)으로 정확히 같은 구조(`ProjHead`+`cost_info_nce`)를 학습시키니 **98% 정확도**로 정상 작동(chance 20%). → **loss/구조 자체는 정상. 원인은 "실제 모델 풀에는 전반적으로 센 모델이 존재한다"는 비대칭성** — MoE 문헌의 "Expert Collapse"(Shazeer et al. 2017)와 정확히 일치하는 현상.
- Top-3로 쏠린 프롬프트들만 따로 떼서, 그 안에서의 선택이 진짜 판단인지 검증 → **라우터의 3-way 정확도(56.6%)가 "그냥 제일 잘하는 애 하나만 고정으로 찍기" 베이스라인(66.55%)보다도 낮음** — 쏠림 안의 "분배"조차 노이즈였음이 확인됨.

### 16.3 Load-Balancing Loss 적용 — 부분적 완화 확인

MoE 문헌의 표준 해법(Switch Transformer/GShard 스타일)을 이식: `load_balance_loss(q, E) = M · Σ(P_i²)`, P_i = 배치 평균 라우팅 확률(softmax, **반드시 기존 loss와 같은 temperature로 스케일** — 처음에 온도 안 맞춰서 효과 0이었던 버그 있었음, `scripts/loo_unseen_recovery.py`에 구현).

**11개 전부 학습 + beta 스윕(Perplexity FP)**:
| beta | 0아닌 모델 수(/11) | top3 비율 | 정확도(λ=0) |
|---|---|---|---|
| 0(없음) | 3 | 97% | 67% |
| 0.05 | 7 | 90.5% | 65.9% |
| 0.2 | 10 | 81.5% | 65.7% |
| **1.0** | **11** | 71.5% | 62.3% |
| 2.0 | 11 | 65.0% | 60.7% |
| 4.0 | 10 | 61.5% | 58.8% |

beta=1.0을 "11개 전부 최소 1번은 선택되면서 정확도 손실이 아직 크지 않은" 지점으로 선택.

### 16.4 beta=1.0으로 Ceiling FP 11-fold LOO 재실행 결과

| held_out | n_oracle_is_M | oracle_match_rate |
|---|---|---|
| claude-instant-v1 | 162 | 55.6% |
| claude-v1 | 63 | 0% |
| claude-v2 | 50 | 0% |
| gpt-3.5-turbo | 156 | 25.6% |
| gpt-4 | 128 | 10.2% |
| code-llama-34b | 23 | 0% |
| llama-2-70b | 140 | 7.9% |
| mistral-7b | 1579 | 0% |
| mixtral-8x7b | 1057 | 34.5% |
| Yi-34B-Chat | 998 | 0.2% |
| WizardLM | 1240 | 48.4% |

- 단순 fold 평균: 16.6% (chance 9.1%)
- **n_oracle_is_M 가중평균(pooled)**: (90+0+0+40+13+0+11+0+365+2+600)/5596 ≈ **20.0%** (chance 대비 약 2.2배)
- n_oracle_is_M이 유독 큰 모델(mistral-7b 1579, mixtral 1057, Yi-34B 998, WizardLM 1240 — 아마 이 중 다수가 pool에서 가장 저렴해서 "cheapest-correct" 자격을 자주 얻는 것으로 추정)이 결과를 좌우함. 이 중 mistral-7b(0%)와 Yi-34B(0.2%)는 완전히 실패, mixtral(34.5%)과 WizardLM(48.4%)은 크게 성공 — 편차가 매우 큼.
- 결과 파일: `local_descriptors/routerbench-analysis/ceiling_loo_beta1.0_results.json`

**해석은 아직 미정리 상태** — "11개 학습 시 선호도가 높았던 모델일수록 LOO 복원도 잘 된다"는 가설을 세웠으나 완전히 깔끔하게 들어맞지는 않음(mixtral은 맞지만 Yi-34B는 안 맞음). 아래 16.5에서 Perplexity FP와 비교.

### 16.5 beta=1.0으로 Perplexity FP 11-fold LOO 재실행 + Ceiling과 비교

| held_out | n | Ceiling oracle_match_rate | Perplexity oracle_match_rate |
|---|---|---|---|
| claude-instant-v1 | 162 | 55.6% | 14.2% |
| claude-v1 | 63 | 0% | 0% |
| claude-v2 | 50 | 0% | 0% |
| gpt-3.5-turbo | 156 | 25.6% | 40.4% |
| gpt-4 | 128 | 10.2% | 11.7% |
| code-llama-34b | 23 | 0% | 0% |
| llama-2-70b | 140 | 7.9% | 5.0% |
| mistral-7b | 1579 | 0% | 14.2% |
| mixtral-8x7b | 1057 | 34.5% | 0.8% |
| Yi-34B-Chat | 998 | 0.2% | 0% |
| WizardLM | 1240 | 48.4% | 3.9% |
| **pooled(n-weighted), n=5596** | | **20.0%** | **6.9%** |
| chance | | 9.09% | 9.09% |

- Ceiling FP pooled(20.0%)는 chance(9.09%) 대비 약 2.2배로 유의미하게 위지만, **Perplexity FP pooled(6.9%)는 오히려 chance보다 낮음** — beta=1.0 load-balancing을 똑같이 적용했는데도 두 FP의 결과 방향이 다름.
- "capability-aligned FP(Ceiling)일수록 unseen 라우팅이 잘 된다"는 가설과 방향은 일치.
- 다만 모델별로 보면 완전히 뒤집히는 경우도 있음(mistral-7b: Ceiling 0% vs Perplexity 14.2%, mixtral: Ceiling 34.5% vs Perplexity 0.8%, WizardLM: Ceiling 48.4% vs Perplexity 3.9%) — 두 FP가 서로 다른 모델에서 신호를 잡고 있어서, "Ceiling이 전반적으로 우월하다" 이상의 설명은 아직 섣부름. pooled 숫자는 n이 큰 4개 모델(mistral-7b/mixtral/Yi-34B/WizardLM)에 크게 좌우되는 구조라는 점도 유의.
- 결과 파일: `local_descriptors/routerbench-analysis/perplexity_loo_beta1.0_results.json`

### 16.6 in-pool 쏠림 비율 vs LOO 복원율 — 모델별 비교

11개 전부 학습(beta=1.0)했을 때의 probe 쏠림 비율(200개 중)과, 그 모델을 LOO로 뺐을 때의 Ceiling oracle_match_rate를 나란히 보면:

| model | in-pool 쏠림 비율 | Ceiling LOO oracle_match_rate |
|---|---|---|
| WizardLM | 6.0% | 48.4% |
| claude-instant-v1 | 10.0% | 55.6% |
| gpt-3.5-turbo | 4.0% | 25.6% |
| gpt-4 | 3.5% | 10.2% |
| llama-2-70b | 9.5% | 7.9% |
| mixtral-8x7b | 41.5% | 34.5% |
| Yi-34B-Chat | 20.0% | 0.2% |
| mistral-7b | 3.5% | 0% |
| claude-v1/v2, code-llama | ~0.5~1% | 0% |

- **WizardLM/claude-instant/gpt-3.5**: 11개 같이 학습할 때는 collapse로 밀려서 쏠림 비율이 낮았는데(4~10%), held-out으로 빼면 오히려 잘 찾아감(26~56%) — collapse에 가려졌던 진짜 구분 가능한 신호가 있었다는 뜻으로 해석됨.
- **mixtral/llama-2-70b**: in-pool과 LOO 성적이 비슷한 수준 — 별 차이 없음.
- **Yi-34B**: in-pool에서는 mixtral 다음으로 많이 쏠렸던(20%) 모델인데, LOO로 빼면 0.2%로 완전히 실패 — in-pool 쏠림이 "rich-get-richer" 붕괴 자체의 산물이었을 뿐, 진짜 구분 가능한 실력 신호는 아니었던 것으로 추정.

### 16.7 Descriptor 공간 고립도 점검 — 고립도로는 설명 안 됨

`scripts/descriptor_isolation_check.py`로 Ceiling/Perplexity 각 FP에서 모델별 pairwise 코사인 유사도(다른 10개와의 평균, 최근접 이웃) 확인:
- **Ceiling FP**: llama-2-70b/code-llama-34b/mistral-7b가 자기들끼리 뭉친 별도 클러스터(서로 0.93~0.99)를 이루고 나머지 8개(claude/gpt/mixtral/Yi-34B)와는 거리가 있음(mean_sim -0.37~-0.39). 하지만 mistral-7b는 이 클러스터 안에서 전혀 고립돼있지 않음(최근접 code-llama-34b sim=0.93)에도 LOO 복원 0% — 클러스터 동료가 학습 pool에 있어도 실패함. Yi-34B는 mean_sim=+0.12로 딱히 고립되지 않았는데도(claude-v2와 0.74) 실패.
- **Perplexity FP**: claude-instant-v1이 pool에서 가장 고립된 모델(mean_sim=+0.42)인데 오히려 LOO 복원이 상대적으로 잘 됨(14.2%) — "고립될수록 실패"라는 방향과 정반대.
- **결론**: 단순 descriptor 공간 고립도로는 Yi-34B/mistral-7b의 실패를 설명 못 함. 16.6에서 세운 "collapse 승자였을 뿐 진짜 신호는 아니었다"는 가설 쪽이 더 유력.

### 16.8 Rank/Margin 진단 추가 + 멀티시드 검증 — 중요한 반전 발견

사용자 제안으로, oracle_match_rate(엄격한 argmax 일치)만으로는 "근소하게 졌다"와 "아예 경합이 안 됐다"를 구분 못 한다는 문제 제기 → `scripts/ceiling_multiseed_check.py`에 rank(held_out이 11개 중 유사도 몇 등인지, chance=6.0)와 margin(1등과의 유사도 차이) 진단 추가. Yi-34B/mistral-7b(실패 사례)와 WizardLM/claude-instant-v1(성공 사례) 4개 fold를 seed 0/1/2로 재실행(Ceiling FP, beta=1.0):

| held_out | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Yi-34B-Chat | 0.2% (rank 7.43) | 0.3% (rank 7.56) | 0.0% (rank 8.85) |
| mistral-7b | 0.0% (rank 7.98) | 2.1% (rank 6.34) | 0.3% (rank 8.18) |
| WizardLM | **48.4% (rank 2.84)** | 11.0% (rank 5.73) | 0.3% (rank 8.36) |
| claude-instant-v1 | **55.6% (rank 1.99)** | 1.2% (rank 6.04) | 0.0% (rank 8.69) |

(chance rank=6.0, chance oracle_match_rate≈9.1%)

**중요한 발견**: Yi-34B/mistral-7b의 실패는 시드와 무관하게 일관됨(안정적 현상). 반면 **WizardLM/claude-instant-v1의 "성공"은 seed=0에서만 나타나고, seed=1/2에서는 chance 수준이거나 그보다 나쁨으로 붕괴함** — 이전에 보고했던 "Ceiling FP pooled 20.0%, chance 대비 2.2배"라는 결과는 이 두 fold의 seed=0 우연한 호성적이 견인한 것일 가능성이 큼. **단일 seed=0 결과만으로 "Ceiling FP가 unseen 라우팅에 유의미하다"고 결론 내리는 것은 성급함 — 전체 11-fold를 멀티시드로 재검증해야 진짜 결론이 나옴.**

결과 파일: `local_descriptors/routerbench-analysis/ceiling_multiseed_check.json`

### 16.9 AUC/point-biserial 진단 추가 (seed=0, 11-fold 전체)

oracle_match_rate는 다른 10개 모델과의 경합(argmax)에 의존하는 지표라, "held_out 자신의 신호가 좋은데 경합에서만 진 경우"를 놓친다는 문제 제기(사용자) → `scripts/full_loo_with_rank.py`에 두 가지 학습-포함(하지만 경합-독립적인) 진단 추가:
- **AUC / point-biserial correlation**: `q·E_heldout`(쿼리와 held_out descriptor의 코사인 유사도)과 held_out의 실제 정답 여부(0/1)의 관계를, Set B 전체(7,300개)로 계산. 경쟁 없이 "이 descriptor 자체가 프롬프트별 강약 신호를 담고 있는가"만 순수하게 봄.
- **mean_correct_incorrect_gap**: held_out이 정답인 프롬프트에서, 나머지 10개 중 "같이 정답인 모델들"과 "오답인 모델들"에 대한 평균 유사도 차이 — held_out 특정 위치가 아니라 "맞는 동네"를 찾아가는지 봄.

**seed=0 11-fold 결과**: Ceiling은 10/11 모델에서 AUC>0.5, 대부분 p<0.0001로 강하게 유의(mistral-7b만 예외, AUC=0.475로 오히려 chance 이하). Perplexity는 절반 가까이 chance 이하로 뒤섞임. pooled oracle_match_rate는 기존과 동일(Ceiling 20.0%, Perplexity 6.9%). 결과: `local_descriptors/routerbench-analysis/full_loo_with_rank_results.json`, 테이블 이미지 `full_loo_with_rank_table_seed0.png`.

### 16.10 "공유 난이도 의존도" 가설 — Ceiling FP가 어떤 모델에서 약한지 부분적으로 설명

mistral-7b(seed=0에서 AUC 약함)를 진단하다가 나온 가설: Ceiling FP는 mean-centering으로 "pool 전체가 공유하는 난이도" 성분을 일부러 제거하는데, 원래 실력 패턴의 대부분이 이 공유 성분이었던 모델은 제거 후 남는 고유 신호가 얄팍해서 AUC가 약해질 것이라는 가설. `scripts/shared_difficulty_check.py`로 검증(각 모델의 raw 86차원 정확도 벡터와 pool-mean 벡터의 상관계수 계산):

| model | pool-mean과 상관 | mean-centering 후 residual std | 관측된 AUC 패턴 |
|---|---|---|---|
| claude-v2 | 0.969 (최고) | 0.068 (최저) | 불안정/약함 (0.625→0.499→0.473, 멀티시드) |
| WizardLM | 0.946 | 0.071 | 불안정/약함 (0.542→0.444→0.441) |
| gpt-4 | 0.889 | 0.113 | 강함/안정 (0.745/0.739/0.738) |
| llama-2-70b | 0.458 | 0.207 | 강함/안정 |
| code-llama-34b | 0.404 (최저) | 0.207 (최고) | 강함/안정 |
| mistral-7b | 0.807 | 0.118 | **약함 (설명 안 됨, 예외)** |

4/5는 가설과 정확히 맞아떨어짐(claude-v2·WizardLM: 고상관·저잔차→약함, llama-2-70b·code-llama-34b: 저상관·고잔차→강함). **mistral-7b는 중간 정도 상관(0.807, gpt-4와 비슷한 수준)인데도 약해서 여전히 미해결 예외.** 결과 이미지: `shared_difficulty_vs_auc_table.png`. 다음 세션 V1.3(15.6)에서 "절대 수준을 별도 차원으로 추가"할 때, 이 성분(공유 난이도)을 다시 넣으면 도움이 될 수 있다는 제안이 나왔으나 — 그 성분이 정확히 16.2에서 진단한 collapse 유발 성분(전반적으로 센 모델을 프롬프트 무관하게 미는 경향)과 같은 것이라, "AUC 개선"이 "진짜 조건부 라우팅 개선"인지 "collapse 경향 강화"인지 구분해서 봐야 하는 위험이 있음 — 시도할 가치는 있으나 해석 시 주의 필요.

### 16.11 전체 11-fold 멀티시드 최종 결과 — Ceiling 우위는 재현 안 됨 (2026-07-31 결론)

`scripts/full_loo_multiseed.py`(task #10)로 Ceiling+Perplexity 11-fold 전부를 seed 0/1/2로 완주(seed=0은 task #9 결과 재사용, seed=1,2만 재학습, beta=1.0 고정). **최종 pooled 결과**:

| seed | Ceiling mean AUC | Perplexity mean AUC | AUC delta | Ceiling pooled oracle_rate | Perplexity pooled oracle_rate | oracle_rate delta |
|---|---|---|---|---|---|---|
| 0 | 0.623 | 0.521 | +0.102 | 20.0% | 6.9% | +13.1%p |
| 1 | 0.594 | 0.629 | **-0.035** | 13.5% | 34.5% | **-21.0%p** |
| 2 | 0.611 | 0.571 | +0.040 | 4.2% | 3.9% | +0.3%p |
| **3-seed 평균** | | | **+0.036** | | | **-2.5%p** |

**결론(있는 그대로)**: seed=0 하나만 보면 "Ceiling FP가 Perplexity FP보다 unseen 라우팅에 확실히 낫다"는 인상이 강했지만(AUC delta +0.102, oracle_rate delta +13.1%p, 둘 다 방향 일치), **3개 시드를 평균 내면 AUC는 근소한 양수(+0.036, 방향조차 seed마다 뒤집힘)로 쪼그라들고, oracle_match_rate는 오히려 근소하게 음수(-2.5%p, Perplexity가 미세 우위)로 반전**된다. 개별 모델 단위로도 seed=0/1/2에 따라 어느 FP가 이기는지 자주 뒤집힘(예: claude-v1은 seed0에서 Ceiling +0.095, seed1/2에서 Perplexity가 이김).

**사용자 판단(2026-07-31)**: 이 파라메트릭 MLP 학습 기반 데이터는 "Ceiling FP가 Perplexity FP보다 unseen 라우팅에 낫다"는 가설을 뒷받침하는 정량적 근거로 **사용하지 않기로 결정**. 원인은 seed마다 달라지는 collapse 패턴(13번 섹션부터 반복 확인된 현상)이 이 작은 pool(11개) 파라메트릭 학습에 근본적으로 내재된 것으로 보임 — "가설이 틀렸다"가 아니라 "이 검증 도구가 가설을 깔끔하게 보여주기엔 너무 시끄럽다"는 해석. 학습-배제 kNN 테스트(15.5, rho +0.098, p=0.02, seed 개념 자체가 없어서 이 문제에서 자유로움)는 여전히 유효한 별도 근거로 남음.

결과 파일: `local_descriptors/routerbench-analysis/full_loo_multiseed_results.json` (11개 모델 × 2 FP × 3 seed 전체 원본 데이터 — oracle_match_rate, mean_rank, AUC, point_biserial 등 전부 포함).

**FP pool 혼동 가능성 점검(사용자 요청)**: seed=0(task #9, 별도 스크립트)과 seed=1,2(task #10)가 다른 스크립트에서 나와서 Ceiling/Perplexity가 뒤바뀌었을 가능성을 의심 → (1) seed=0 원본 파일의 키 구조 직접 확인, (2) merge 코드가 같은 변수(`fp_name`)로 읽기/쓰기해서 구조적으로 크로스 불가능함 확인, (3) seed=1,2 학습 루프도 `(fp_name, desc_dir)` 튜플이 같이 흐르는 구조 확인 — **3곳 다 점검해서 혼동 없음을 확인함.** 즉 위 반전 현상은 버그가 아니라 진짜 학습 결과.

### 16.12 3-seed 평균 테이블 + Mantel 테스트 기록 (순환논증 결함 포함)

- **3-seed 평균 Ceiling vs Perplexity**(단순평균, n-weighted pooled 아님): `seed_avg_ceiling_vs_perp_table.png`. AUC delta 평균 +0.0356, rate delta 평균 -2.0%p — 11.11절 pooled 수치와 방향은 같음(AUC 근소 양수, rate 근소 음수). 모델별로는 완전히 갈림: gpt-4(+0.166)·mixtral(+0.106)·llama-2-70b(+0.086)는 Ceiling이 뚜렷하게 우세, WizardLM(-0.046)·claude-v2(-0.035)·mistral-7b(-0.030)는 Perplexity가 우세.
- **Mantel/RSA 구조 검정 — 순환논증 결함 발견**: `scripts/mantel_rsa_test.py`로 Ceiling FP의 11×11 유사도 구조와 "진짜 capability 상관행렬"을 비교(rho=+0.714, p=0.00031, Perplexity는 rho=0.064, p=0.76). **그런데 사용자가 즉시 지적**: 두 행렬 다 Set A 데이터로 만들었고, Ceiling FP 자체가 Set A 정확도를 직접 mean-centering해서 만든 것이므로 **이 비교는 순환논증**임 — "capability 정보가 진짜 있다"가 아니라 "FP 생성 과정이 원래 신호를 파괴 안 했다" 정도만 확인해줌. held-out 개념 자체가 없어서(Set A vs Set A) 예측 타당성 검증이 아님. 결과는 `mantel_rsa_setA_circular_table.png`에 캐비어 명시해서 기록만 해두고, **가설 뒷받침 근거로는 사용 안 함**. 교정 버전(Set A로 만든 FP vs **Set B**로 계산한 진짜 capability 구조 비교)이 다음 할 일.

### 16.13 Seed 앙상블 파일럿 — 효과 없음 (부정적 결과)

가설: 3개 seed에서 각각 학습된 인코더의 코사인 유사도 점수(raw sims, 최종 지표가 아니라 스코어 자체)를 평균 내면, 개별 seed의 우연한 collapse 방향이 상쇄돼서 더 안정적인 추정치가 나올 것이다. `scripts/seed_ensemble_pilot.py`로 4개 모델(gpt-4=안정적강함, claude-v2=불안정, mistral-7b=일관되게 약함, WizardLM=불안정) 파일럿 실행 — head를 저장 안 해서 3 seed를 한 스크립트에서 동시에 학습/보관 후 eval 시점에 sims를 평균.

| model | seed 0/1/2 AUC | mean-of-individual AUC | **SCORE-LEVEL 앙상블 AUC** |
|---|---|---|---|
| gpt-4 (안정) | 0.745/0.739/0.738 | 0.7405 | 0.7394 (거의 동일) |
| claude-v2 (불안정) | 0.625/0.499/0.473 | 0.5325 | 0.5398 (미미하게 개선, 여전히 약함) |
| mistral-7b (일관 약함) | 0.475/0.497/0.452 | 0.4747 | **0.4637 (오히려 악화)** |
| WizardLM (불안정) | 0.542/0.444/0.441 | 0.4759 | **0.4585 (오히려 악화)** |

oracle_match_rate는 더 나쁨 — claude-v2·mistral-7b는 앙상블하면 0%로 떨어짐(mean-of-individual은 각각 12.0%, 0.8%였는데도).

**결론: 이 형태(raw cosine similarity 단순 평균)의 seed 앙상블은 효과가 없음 — 4개 중 1개(이미 안정적이던 gpt-4)는 무변화, 1개(claude-v2)는 미미한 개선이지만 여전히 chance 근처, 2개(mistral-7b, WizardLM)는 오히려 악화.** 일반적인 앙상블(배깅)이 가정하는 "각 모델이 진짜 신호 주변에서 노이즈만 다르게 갖는다"는 전제가, collapse로 인해 서로 완전히 다른 attractor로 수렴하는 이 상황에는 안 맞는 것으로 보임 — 서로 다른 collapse 방향의 유사도를 그냥 더하면 진짜 신호가 아니라 애매한 잡음으로 섞이는 것으로 추정. 결과 파일: `local_descriptors/routerbench-analysis/seed_ensemble_pilot_results.json`, 이미지: `seed_ensemble_pilot_table.png`.

### 16.14 오늘 세션 최종 정리 — 두 가지 후속 시도 모두 무산

멀티시드 완주(16.11) 이후 시도한 두 가지 보완책이 전부 원하는 결과를 못 냄:
- **Mantel/RSA 구조 검정**(16.12): Set A로 두 행렬을 다 만들어서 순환논증 — 근거로 못 씀.
- **Seed 앙상블**(16.13): 효과 없거나 오히려 악화 — 완화책으로 못 씀.

**현재 시점에서 남아있는, 실제로 유효한 근거는 15.5의 학습-배제 kNN 테스트(RouterBench, Ceiling FP rho delta +0.098, p=0.02)뿐**이다. 파라메트릭 MLP 기반 unseen-routing 실험(13번 섹션부터 오늘까지)은 collapse의 seed 의존성 때문에 정량적 우열 근거로는 끝내 못 씀 — 다만 그 자체가 "왜 안 되는지"에 대한 상세한 진단(MoE Expert Collapse 연결, load-balancing 부분적 효과, 공유 난이도 의존 가설 등)으로 발표에 쓸 수 있는 내용.

### 16.15 낮은 LR + 더 많은 epoch 시도 (사수님 조언 #4 관련) — collapse는 완화되지만 정확도 희생, load-balancing과 비슷한 트레이드오프

`scripts/full_pool_lr_epoch_test.py`: 11개 전부 학습(LOO 아님), Perplexity FP, beta=0(load-balancing 없음), LR을 5e-4→5e-5(10배 낮춤), epoch을 2→10(5배 늘림), seed=0.

| | baseline(LR 기본값, beta=0) | LR 낮춤+epoch 늘림(beta=0) |
|---|---|---|
| n_nonzero | 3/11 | 9/11 |
| top3_share | 0.97 | 0.870 |
| router_acc | 0.67 | 0.4297 |

**결론**: 학습을 천천히/오래 시키는 것만으로도 collapse 집중도가 실제로 크게 완화됨(3/11→9/11 모델이 쓰임, top3 쏠림 97%→87%) — load-balancing 보조 loss 없이도 이 정도 분산 효과가 나온다는 건 흥미로움. **다만 정확도가 0.67→0.43으로 크게 떨어짐** — beta=1.0 load-balancing 때와 정성적으로 똑같은 트레이드오프(분산 vs 정확도)가 재현됨. 즉 "천천히 학습시키면 공짜로 collapse가 해결된다"는 아니고, 결국 어떤 방법으로든 분산시키면 정확도를 내줘야 하는 구조로 보임 — 근본적인 fix는 아직 못 찾음.

### 16.16 (미완료 과제, 참고용 — 17번 섹션에서 다른 경로로 진행됨)
1. Set A vs Set B로 교정한 Mantel 테스트 — 16.12 순환논증 문제 교정판. 아직 미실행.
2. mistral-7b의 "공유 난이도 가설로도 설명 안 되는" 예외는 여전히 미해결.
3. V1.3(15.6, 절대 수준 성분 재도입) — 16.10 "collapse 경향 강화" 위험 때문에 보류.

---

## 17. LLMRouterBench로 pool 규모 확장 실험 — pool 크기가 아니라 tier 동질성이 핵심이었음 (2026-08-01)

**동기**: RouterBench가 11개 모델뿐이라 "unseen 모델 routing 신호"가 통계적으로 불안정한 게 아닌가 하는 의문 → pool을 3배로 키워서 재검증. 이 전체 챕터의 가설("capability-aligned Ceiling FP가 unseen routing에 도움된다")은 **CSCR 논문의 주장이 아니라 본인의 독자적 연구 주제**임 — kNN 파일럿에서 먼저 검증(긍정적) → 이 주제를 확정하고 실제 parametric 구현으로 넘어간 흐름.

### 17.1 LLMRouterBench 벤치마크 발굴 + 33개 모델 pool 구축
GitHub `ynulihao/LLMRouterBench`(arXiv 2601.07206), HF `NPULH/LLMRouterBench`. 33개 모델(경량 20 + 플래그십 13) × query/response/score/cost 전부 갖춘 400K+ 인스턴스. `scripts/llmrouterbench/common.py`에 로더 구축, 33개 모델이 동시에 깨끗한(결측 없는) 데이터셋은 8개(`aime, arenahard*4, gpqa, livecodebench, livemathbench`)로 확정 — mmlupro는 모델 1개 결측이라 제외.

Probe 선정: 카테고리별 stratified 고분산 probe 24개씩(8×24=192), **Ceiling과 Perplexity가 동일한 probe 세트 사용**(RouterBench 시절 있었던 confound 제거).

### 17.2 33개 풀 결과 — kNN은 강한 승리, parametric은 무승부
- **kNN 테스트**(`knn_test.py`): Ceiling vs uniform delta+0.037, p<0.0001, 31/33. Ceiling vs Perplexity **delta+0.0352, p<0.0001, 30/33 승** — 이 세션 전체에서 가장 깨끗했던 결과.
- **33-fold parametric LOO**(`full_loo.py`, seed=0, beta=1.0): Ceiling vs Perplexity 직접비교 **delta-0.0154, p=0.178(유의 안 함)**, 심지어 flagship 13개만 보면 delta-0.031, p=0.083(Ceiling이 오히려 나쁜 경향). **kNN에서 이겼던 게 parametric에서 재현이 안 됨.**
- flagship 6개 모델(deepseek-r1-0528, deepseek-v3-0324, gemini-2.5-flash/pro, gpt-5, gpt-5-chat)에서 Ceiling AUC가 유의미하게 0.5 **미만**으로 역전됨.

### 17.3 원인 진단 — probe 선정이 tier gap에 지배됨
Probe를 "33개 모델 전체 pooled variance"로 뽑았더니, 경량-플래그십 사이의 거대한 실력차가 분산을 지배해서 실제로는 tier gap만 잡아내는 probe가 뽑힘 → flagship 내부의 진짜 세밀한 차이는 거의 못 담김. **해결책**: `build_data_split_v2.py`로 `var_light + var_flag`(각 tier 내부 분산의 합)로 재선정. Ceiling FP v2 재구축은 성공(std 0.52→0.30, tier gap 의존도 감소 확인)했으나, **이 시점에서 사용자가 더 근본적인 방향 제안**(아래 17.4) — v2 probe로 재학습은 보류하고 그쪽으로 전환.

### 17.4 플래그십 제외, lightweight 20개 pool로 피벗 — 결정적 전환점
**사용자 제안**: "플래그십만 빼면 카테고리도 더 많이 쓸 수 있지 않을까?" → 확인해보니 lightweight 20개만 놓고 보면 **22개 카테고리 전부**(8개→22개, mbpp/humaneval/bbh/math500/medqa/arcc 등 대거 추가) 완전 커버리지 확보됨 — flagship이 대부분 데이터셋의 결측 원인이었음. 게다가 tier gap 자체가 없어지니 probe 선정 버그도 원천 해결.

`common_lite20.py`(20개 모델, 22개 데이터셋), `build_data_split_lite20.py`(pooled variance로 단순 재선정, tier gap 없으니 안전), Ceiling/Perplexity FP 재구축(528차원, 24 probe×22 dataset). `local_descriptors/llmrouterbench_lite20/`에 저장.

### 17.5 lightweight-20 kNN 결과 — 신호는 살아있지만 33개 때보다 약함
- Ceiling vs uniform: delta+0.0138, **p=0.0225**(유의), 15/20
- Perplexity vs uniform: delta+0.0065, p=0.0956(유의 안 함), 14/20
- **Ceiling vs Perplexity 직접비교: delta+0.0073, p=0.1468(유의 안 함)**, 13/20 — 33개 풀의 깨끗한 승리(p<0.0001)가 재현 안 됨

### 17.6 메커니즘 규명 — PC1 분해 (이 세션의 핵심 발견)
Ceiling vs Perplexity 격차가 왜 좁아졌는지 파고들다가, Ceiling 임베딩 공간의 최근접 이웃 구조를 직접 확인 — Qwen3-8B/DeepSeek-R1-0528-Qwen3-8B/GLM-Z1-9B-0414/NVIDIA-Nemotron-Nano-9B-v2/Intern-S1-mini/MiniCPM4.1-8B가 아키텍처와 무관하게 서로 강하게 뭉침(+0.4~0.52) — 공통점은 "reasoning-RL 튜닝을 세게 받았다"는 것.

**PCA로 검증**(mean-centered score matrix에 SVD): PC1이 전체 분산의 **28.5%**(PC2는 8.4%)를 차지하고, 그 로딩값이 위 클러스터와 거의 정확히 일치. **PC1을 descriptor에서 제거한("domain-purified") 버전으로 kNN을 다시 돌리자 uniform 대비 0/20 fold 전부 악화**(delta-0.1219, p<0.0001) — Ceiling의 예측력 대부분이 세밀한 도메인 매칭이 아니라 이 **coarse "전반적으로 얼마나 센가"라는 단일 축**에서 나온다는 것을 직접 확인. (`build_ceiling_fp_lite20_deflated.py`, `knn_test_lite20_deflated.py`)

**해석**: parametric 학습이 collapse하는 이유도 이걸로 설명됨 — 경사하강법 입장에서 "coarse 축 방향으로 예측을 몰아넣기"가 세밀한 도메인 신호를 학습하는 것보다 훨씬 쉽고 손실을 빨리 줄여주는 지름길(shortcut learning)이기 때문. kNN이 잘 되는 이유(학습이 없어서 이 지름길에 빠질 수가 없음)와 parametric이 안 되는 이유(학습 과정 자체가 지름길로 수렴)가 거울상 관계.

### 17.7 Mean-centering 제거 테스트 — 가설 기각
"mean-centering이 신호까지 같이 지우는 게 아닐까"라는 가설로 uncentered Ceiling FP도 만들어 테스트(`build_ceiling_fp_lite20_uncentered.py`). 결과: centered와 통계적으로 구분 안 됨(delta-0.004, p=0.399), Perplexity 대비 우위는 오히려 더 약해짐(centered p=0.147 → uncentered p=0.196). **가설 기각 — mean-centering을 없앤다고 나아지지 않음.**

### 17.8 카테고리 단위 집계 FP — 이 세션 kNN에서 가장 깨끗한 승리
528개 **개별 probe** 대신, 22개 **카테고리 전체의 평균 정답률**(모델당 22차원, 각 차원 = 그 카테고리 Set A 전체 평균 점수)로 Ceiling FP를 재구성(`build_ceiling_fp_lite20_categoryrate.py`) — 노이즈 심한 개별 쿼리 대신 수백~수천 개를 평균 낸 값이라 차원당 노이즈가 훨씬 낮음.

결과, **세 비교 전부 통계적으로 유의미하게 승리**:
| 비교 | delta | p | 개선 fold |
|---|---|---|---|
| vs Uniform | +0.0180 | **0.0005** | 16/20 |
| vs Perplexity | +0.0115 | **0.0132** | 15/20 |
| vs 기존 528-probe Ceiling | +0.0042 | **0.0346** | 13/20 |

### 17.9 lightweight-20 parametric LOO — 33개 풀과 다른 결과, 처음으로 parametric도 유의미하게 이김
`full_loo_lite20.py`(seed=0, beta=1.0, 528-probe 기존 Ceiling FP 기준, 20-fold × 2 FP = 40 fold, GPU로 ~70분):
- Perplexity: mean AUC=0.4759, 20 fold 중 sig above=6, **sig below(역전)=12**, not sig=2
- Ceiling: mean AUC=0.5856, sig above=**13**, sig below=6, not sig=1
- **직접비교(Ceiling−Perplexity): delta+0.1097, p=0.0038(유의), 13/20 fold Ceiling 승**

33개 풀 parametric LOO(17.2, p=0.178 유의 안 함)와 대비됨 — **pool 크기(11→33, 3배)는 collapse/불안정성을 못 고쳤지만, pool 동질성(flagship 제외)은 kNN뿐 아니라 실제 parametric 학습 단계에서도 유효했음.**

### 17.10 종합 결론 (발표 서사용)
1. **Pool 크기 자체는 원인이 아니었음** — 11→33(3배)으로도 parametric collapse/불안정성 그대로.
2. **Pool 동질성(tier gap 유무)이 핵심 변수였음** — flagship 제외한 20개 풀에서 kNN과 parametric 둘 다 처음으로 깨끗하게 유의미한 승리.
3. **메커니즘 규명**: Ceiling FP가 담는 신호의 상당 부분이 세밀한 도메인 매칭이 아니라 "이 모델이 전반적으로 얼마나 센가"라는 coarse 축(PC1, 분산의 28.5%)이며, 이게 parametric 학습에서 shortcut learning으로 작용해 collapse를 유발한다는 가설이 이 세션의 여러 관찰과 일관됨.
4. **집계 단위(개별 probe vs 카테고리 단위)가 노이즈에 큰 영향** — 카테고리 단위 집계가 이 세션에서 가장 깨끗한 kNN 승리를 만듦.
5. Mean-centering 제거는 도움 안 됨(가설 기각, 17.7).

### 17.11 Multi-seed 재검증 완료 — 17.9 결과 확정됨, Dual-Tier 후속 연구 구상 (2026-08-02)

**Multi-seed 결과** (`full_loo_lite20_multiseed.py`, seed=1,2 추가 실행, `full_loo_multiseed_results.json`):

| seed | Ceiling mean AUC | Perp mean AUC | delta | p | 승 |
|---|---|---|---|---|---|
| 0 | 0.5856 | 0.4759 | +0.1097 | 0.0038 | 13/20 |
| 1 | 0.5796 | 0.4718 | +0.1077 | 0.0034 | 14/20 |
| 2 | 0.5734 | 0.4634 | +0.1101 | 0.0015 | 15/20 |
| **3-시드 평균** | | | **+0.1092** | **0.0024** | **14/20** |

RouterBench(16.11)에서 봤던 시드 요동(claude-v2, WizardLM이 시드마다 크게 뒤집힘)이 전혀 없음 — 오히려 seed=2가 승수/p-value 둘 다 더 좋게 나옴. **17.9의 결과는 확정된 근거로 사용 가능.**

단, "승패 개수"만 보면(sign test) 13~15/20은 그 자체로는 p≈0.06~0.13이라 관례적 유의수준을 못 넘김 — 실제 유의성은 **승패의 "크기" 비대칭**에서 나옴(이긴 경우 평균 delta≈+0.18, 진 경우 평균≈-0.03, 약 7배 차이). 발표 시 "20개 전부에서 이긴다"가 아니라 **"일부 모델(약 13~15개)에서 크게 개선, 나머지는 거의 무해"**로 정직하게 표현할 것.

**부가 확인**: 33개 풀에서 Ceiling FP의 within-tier(같은 tier끼리) 평균 코사인 유사도=+0.367, between-tier(플래그십↔경량)=**-0.440**로 거의 정반대 방향 — Ceiling의 지배적 축(PC1)이 사실상 tier 분리축이었음을 직접 확인. 반면 Perplexity FP는 within=0.770 vs between=0.725로 거의 구분을 못 함(mean-centering이 없어서 진짜 실력차보다 둔감).

**후속 연구 아이디어 — "Dual-Tier 라우팅"**: PC1(coarse, tier/전반적 강함 축)을 "지워야 할 노이즈"가 아니라 **1단계 게이트**(무거운 tier가 필요한 쿼리인가)로 명시적으로 활용하고, 2단계에서 선택된 tier 내부에서만 도메인 세부 매칭을 하는 2단계 설계. 별도 descriptor를 새로 만들 필요 없이 Ceiling FP 하나를 두 가지 방식으로 해석하면 됨(1단계: PC1 성분/coarse 축, 2단계: 전체 벡터 또는 PC1-제거 residual).

**문제점**: 이 설계를 제대로 검증하려면 각 tier 내부에도 충분한 모델 수가 있어야 하는데(현재 플래그십 13개, 경량 20개 — tier 내부 fine routing을 하기엔 특히 플래그십 쪽이 너무 얇음), 지금 33개 모델 풀로는 어렵다. **EmbedLLM**(115개 모델, CSCR 원 논문의 세 트랙 중 하나, 3번 섹션에서 이미 후보로 언급됐었음)이 이 검증에 필요한 실제 규모. **교수님께 EmbedLLM 규모를 다룰 수 있는 GPU/컴퓨팅 환경("playground")을 요청할 필요가 있음** — 로컬(4GB VRAM)이나 Colab 무료 티어로는 어려운 스케일.

### 다음 할 일 (2026-08-02 세션 종료 시점, 최신)
1. **✅ 완료**: lightweight-20 parametric LOO multi-seed 재검증 — 17.11 참고, 결과 확정됨.
2. **Category-rate FP(Ceiling V2)로 parametric LOO 재검증** — 진행 중(`full_loo_lite20_categoryrate.py`, seed=0). Perplexity는 이미 확인된 3-시드 안정성(17.11)을 재사용, Ceiling V2만 새로 학습. 17.8의 kNN 승리가 parametric 단계에서도 재현되는지 확인.
3. **Category-rate FP로 PC1 분해** — 노이즈가 줄어든 상태에서도 coarse 축이 여전히 지배적인지, 아니면 residual이 이번엔 쓸모 있는지 확인.
4. **⭐ Dual-Tier 라우팅 설계 + EmbedLLM 규모 확보** — 17.11 참고. (a) 교수님께 EmbedLLM(115개 모델) 규모를 감당할 GPU 자원 요청, (b) 확보되면 tier 게이트(coarse) + tier 내부 도메인 매칭(fine) 2단계 설계를 실제로 구현/검증.
5. Set A vs Set B로 교정한 Mantel 테스트(16.12/16.16 이월) — 여전히 미실행.
6. **발표 준비**: 이 챕터 전체(11→33 pool 확장은 실패, tier 동질성으로 전환해서 성공, PC1으로 메커니즘 규명, multi-seed로 확정)를 "본인이 제안한 가설 → kNN 파일럿 검증 → parametric 확장 시도 → 실패 진단 → 원인 규명 및 재현 → 후속 연구(Dual-Tier) 제안" 서사로 정리. 이 프로젝트가 "CSCR 논문 재현"이 아니라 "본인의 독자적 연구 주제"라는 점을 명확히 할 것.

---

## 18. EmbedLLM pool 규격화 + v1.2 프로토타입 검증 (2026-07-30, 별도 세션에서 병행 진행)

**참고**: 이 섹션은 15~17번 섹션(LOO unseen-model 실험, LLMRouterBench 확장)과 다른 세션/기기에서 거의 같은 시기에 독립적으로 진행된 트랙 — 병합 시 순서상 여기(18번)에 배치함. **17.11에서 나온 "Dual-Tier 라우팅엔 EmbedLLM 규모가 필요하다"는 결론과 직접 연결됨** — 아래 pool 규격화 작업이 이미 상당 부분 준비되어 있으므로, 교수님께 자원 요청할 때 바로 활용 가능.

### 18.1 EmbedLLM — 하드웨어 제약별 pool 필터링

**배경**: MixInstruct는 pool 확장이 사실상 막힘(11개 중 7개 확정, 2개 영구 제외, 2개 GPU서버 대기) — 그래서 EmbedLLM(115개 모델)에서 pool을 넓히는 쪽으로 방향 전환. 다만 랩실 서버 스펙(디스크/GPU 대수)이 아직 불확실해서, 여러 시나리오별로 pool을 미리 만들어둠.

**VRAM 추정 공식**: 4bit NF4 기준 `n_params(B) × 0.6GB + 2.5GB` (로컬에서 실측한 "6개 모델 4bit ~38GB" 수치로 역산 검증됨).

| 파일 | 조건 | 개수 | 비고 |
|---|---|---|---|
| `experts/pool-embedllm-3090x2.json` | 2×RTX3090(48GB) | 105 | 대부분 통과, 4x로 확장해도 10개(70~72B급)만 추가됨 |
| `experts/pool-embedllm-3090x4.json` | 4×RTX3090(96GB) | 115 | 전체 |
| `experts/pool-embedllm-colab16gb.json` | Colab T4/P100 단일(16GB) | 80 | ≤14B로 사실상 결정됨 |
| `experts/pool-embedllm-colab16gb-stratified50.json` | 위 조건, 비용 스펙트럼 층화추출 | 50 | 0.5B~14B 고르게, 총 다운로드 ~807GB |
| `experts/pool-embedllm-small50-naive.json` | 단순 용량 최소 50개 | 50 | 총 ~620GB지만 50개 중 45개가 7B에 몰려서 **cost-band 다양성이 거의 없음 — 비추천** |
| `experts/pool-embedllm-small50-stratified.json` | 층화추출 50개(≤34B 포함) | 50 | 총 ~1,023GB, 34B급 4개 포함 |

**만들면서 발견한 registry 데이터 문제** (`experts/registry-embedllm.json`):
- `Mixtral-8x7B-Instruct-v0.1`이 `n_params: 7`로 잘못 기재됨 — HF API로 실측하니 실제 다운로드는 46.7GB(≈47B 총 파라미터). MoE 모델은 "전문가 1개 크기"를 등록해놓은 것으로 추정. `MixTAO-7Bx2-MoE`, `Plaban81/Moe-4x7b`도 같은 의심으로 전부 제외.
- `databricks/dolly-v2-12b`가 EmbedLLM 쪽에도 있었음(MixInstruct에서 이미 저장소 삭제 확인된 그 모델과 동일) — 제외.
- 위 두 문제 다 위 pool 파일들 생성 시 이미 제외 처리됨.

**코드 수정**: `src/router/utils.py`의 `load_model_and_tokenizer()` — `SUSTech/SUS-Chat-72B` 분기가 `four_bit` 파라미터를 무시하고 항상 bf16으로 로드하던 버그 수정(72B bf16 = ~144GB VRAM 필요, 4x3090(96GB)에서도 OOM 났을 것). 4bit 지원 추가.

**자동화**: `scripts/compute_embedllm_descriptors_batch.py` + `scripts/run_embedllm_batch.sh` — EmbedLLM은 MixInstruct와 달리 데이터셋에 응답 텍스트가 없어서(정답 라벨만 있음) perplexity를 "공짜로" 못 얻음 → logit descriptor 계산용 `model.generate()`의 생성 결과를 그대로 잡아채서 같은 pass에서 perplexity도 계산하도록 설계(모델을 두 번 안 돌림). 디스크 사전 점검(HF API로 예상 다운로드 용량 확인 후 여유 없으면 skip), 실패해도 계속 진행, 재실행 시 완료된 모델 스킵.

**미결정**: 실제 서버(디스크 용량, 관리자)나 Colab 등급(Free/Pro/Pro+, VRAM·디스크·백그라운드 실행 제약 상이) 확정 전이라 아직 실행 안 함.

### 18.2 v1.2 FP 방법론 — 첫 프로토타입 검증

**목적**: `FP_IDEAS.md`의 v1.2("LLM 백본으로 모델 전문성을 자연어 요약 → 임베딩") 아이디어가 실제로 capability(bartscore)와 정렬되는지 사전 점검. Router 학습(붕괴 문제, 13번 섹션)을 거치지 않고 descriptor 자체의 품질만 독립적으로 검증하는 방법론(11번 섹션에서 이미 쓰던 RSA 프레임 재사용).

**방법**: MixInstruct 데이터셋에서 각 모델의 실제 probe 응답(25개, 모델 실행 없이 데이터셋의 `candidates[].text`에서 직접 추출 — 다운로드/GPU 불필요)을 읽고, 강점(strengths)/성능(performance)/하자(flaws)/특징(traits) 4개 필드로 구성된 JSON을 모델당 1개씩 작성(**이번엔 Claude가 직접 요약 — 실제 v1.2가 의도한 "가벼운 백본 모델 1회 호출"이 아니라 파이프라인의 상한선을 테스트한 것임, 18.3의 Colab 스윕이 이 gap을 메움**). 필드 길이는 모델 간 45~66단어로 통일(bartscore의 길이 confound 재발 방지). 이 JSON을 `sentence-transformers/all-MiniLM-L6-v2`(QueryEncoder와 동일 백본)로 임베딩 → cosine similarity → RSA.

**결과 (exact/Monte Carlo Mantel, 기존과 동일 통계 프레임)**:

| 비교 | n=7 | n=11 |
|---|---|---|
| v1.2 vs Capability | rho=+0.434, p=0.241 | rho=+0.318, p=0.190 |
| v1.2 vs Logit | rho=-0.455, p=0.093 (6쌍 중 가장 유의에 근접) | (Logit descriptor가 11개 전부 없어서 미계산) |
| v1.2 vs Perplexity | rho=-0.138, p=0.604 | rho=-0.085, p=0.723 |
| (참고) Logit vs Capability | rho=-0.252, p=0.411 | — |
| (참고) Perplexity vs Capability | rho=-0.226, p=0.422 | rho=+0.021, p=0.936 (n=7일 때와 부호가 뒤집힘 — 이 표본 크기에서 결과가 얼마나 요동치는지 보여주는 방증) |

**보조 지표(순위 기반, rho의 극단치 민감성 보완)**: Kendall tau=+0.286(p=0.271, exact), **평균 순위 이동량=5.24/21(p=0.155)** — 무작위 기준선(6.98/21)보다 확실히 적게 움직임. **이 프로젝트에서 나온 모든 descriptor-vs-capability 비교 중 가장 낮은 p값.**

**해석 — 정직하게 정리**:
- v1.2는 지금까지 나온 세 방법론(Logit, Perplexity, v1.2) 중 유일하게 capability와 **양의 방향**으로, 그리고 **가장 큰 크기**로 정렬됨. n=7→11로 늘려도 방향은 유지(다만 크기는 0.434→0.318로 줄어듦 — pool 확장이 신호를 강화하기보다는 노이즈를 더한 것으로 보임, 새로 들어온 4개 모델의 응답이 유독 불안정했던 것과 연결지어 해석 가능).
- **통계적으로 확정된 결과는 아님**(p<0.05 없음, n이 작아 검정력 자체가 낮음).
- 산점도로 시각 확인 중 **중요한 방법론 문제 발견**: 축을 고정 [0,1]로 그리면(기존 `plot_rsa_scatter.py` 관행) 실제 데이터가 좁은 범위(v1.2 0.67~0.86, capability 0.59~0.80)에 몰려있는 걸 감안 안 해서, 약한 상관관계(rho=0.43, R²≈0.19)가 시각적으로 훨씬 타이트해 보이는 착시가 생김. **`scripts/plot_rsa_scatter_v12.py`는 실제 데이터 범위로 확대해서 그리도록 수정함** — 이 문제는 "관계가 있다"를 주장하려는 차트(v1.2)에만 해당하고, 기존 3-way 비교("관계 없다"를 보여주는 차트)에는 영향 없음(확대해도 결론이 안 바뀌거나 오히려 더 무작위로 보임).

**결과물**:
- `local_descriptors/mix-instruct-v12-summaries.json`: 11개 모델의 강점/성능/하자/특징 JSON(Claude 작성)
- `local_descriptors/mix-instruct-v12/*.npy`: MiniLM 임베딩(11개)
- `local_descriptors/mix-instruct-capability-11/*.npy`: 11개 모델용 capability vector(기존 7개짜리와 별도 디렉토리, 105,000 프롬프트 100% dense)
- `experts/pool-mix-instruct-11.json`: 11개 표준 이름 목록
- `local_descriptors/analysis/rank_bump_v12_vs_capability.png`, `rsa_scatter_v12_vs_capability.png`(축 확대판)
- 스크립트: `scripts/build_v12_embeddings.py`, `scripts/rsa_v12_alignment.py`, `scripts/plot_rank_bump_v12.py`, `scripts/plot_rsa_scatter_v12.py`
- `scripts/build_capability_vectors.py`에 `--pool`/`--out_dir` 인자 추가(기존 하드코딩 제거, 7개/11개 둘 다 이 스크립트 하나로 생성 가능하게 일반화)

### 18.3 다음 단계 — 실제 백본 모델로 검증 (오늘 밤 예정, 미실행)

15.2는 Claude가 직접 요약문을 쓴 프로토타입이라 파이프라인의 상한선만 확인한 것. `colab/test_v12_backbones.py` 작성 완료(미실행) — 실제 가벼운 backbone 후보 2개(`microsoft/Phi-3-mini-4k-instruct` 3.8B, `mistralai/Mistral-7B-Instruct-v0.2` 7B, 둘 다 ungated) × instruction 3종(자유서술 / 구조화 JSON / 신뢰성-중심— 15.2에서 실제 발견한 "domain보다 reliability/format 패턴이 더 의미 있었다"는 교훈을 반영해 세 번째 variant를 별도로 설계함) 조합 총 6가지를 11개 모델 전체에 대해 돌려서 RSA를 비교함. Colab T4로 충분(4bit 양자화), probe 응답은 모델 다운로드 없이 데이터셋에서 직접 추출. 결과는 `Drive/cscr_repro/v12_backbone_sweep/sweep_results.json`에 rho 내림차순으로 저장됨.
