# CSCR 재구현 프로젝트 — 진행 상황 & 컨텍스트

이 문서는 이 fork(`prectal123/cscr_re`)에서 진행 중인 연구/재구현 작업의 전체 맥락을 담고 있습니다.
**새로운 로컬 환경이나 새 Claude Code 세션에서 이 프로젝트를 이어갈 때는 이 파일부터 읽으면 됩니다** — Claude Code의 세션/메모리는 기기 간 동기화가 안 되기 때문에, 이 파일이 유일하게 확실한 컨텍스트 전달 수단입니다.

---

## 0. TL;DR — 다음은 여기서부터 (최신, 정확한 날짜는 다음 세션에서 확인/수정할 것)

**가장 최근 세션 요약은 23번 섹션(교수님 피드백 3건 — InfoNCE 분자 max/softmax, fair comparison 검증, Combined GRPO 수식화 — 및 대응 계획) 참고.** RESULTS_SUMMARY.md(Combined GRPO 확립, §21~22 결과물)를 정리해 보고한 뒤 받은 피드백. 핵심 결정: (1) Ceiling FP를 192로 축소하는 실험은 결과가 예측 가능해(이미 1,200개에서 all-seen 붕괴 확인됨) 보류, 대신 (2) CSCR류 Perplexity FP를 192/1800으로 확장하는 실험을 우선 진행 — 그 과정에서 `build_routerbench_perplexity_fp.py`의 `N_PROBES=32`가 프로젝트 표준(192)과 안 맞는 오래된 값이었다는 것도 발견함. (3) Combined GRPO 손실 함수 수식화는 아직 미착수, 다음 세션 최우선 과제.

**23번 섹션 이전 최근 세션 요약은 20번 섹션(EmbedLLM 규모 "new LLMs" 프로토콜로 논문 Table 2와 직접 비교, GRPO 회귀 방식 검증) 참고 — 이 세션은 커밋만 되고 PROGRESS.md 서술이 누락돼 있던 걸 2026-08-15에 뒤늦게 정리함.** 그 이전은 19번 섹션(V1.3 LLM-judge 시도 → probe-count ablation 진단 → domain+difficulty 이중 라우팅 설계) 참고. 17번 섹션(LOO unseen-model 실험) 이전, EmbedLLM pool 규격화/v1.2 프로토타입은 18번 섹션 참고(별도 세션에서 병행 진행됨).

**20번 섹션 한 줄 요약**: EmbedLLM(112개 모델) 규모, 논문의 실제 "new LLMs" 프로토콜(2/3 seen/1/3 unseen)로 CSCR 논문 Table 2(AUDC=0.4848)와 처음으로 직접 비교. GRPO 스타일 회귀 방식이 **seed 0에서는 논문을 이겼지만(AUDC 0.527)**, seed 1~3 추가하니 전부 논문보다 낮게 나와서 **4-seed 평균 AUDC=0.4772로 논문과 거의 대등(근소 열세)** — "단일 시드로 결론 내리면 안 된다"는 이 프로젝트의 반복된 교훈이 또 재현됨. kNN(학습 없음) 검증은 3-seed 전부 유의(Ceiling FP PCA-5 vs uniform, p<0.02) — EmbedLLM 규모에서도 신호 자체는 확인됨. Seed 4는 컴퓨터를 꺼야 해서 중단, 미완료.

**19번 섹션 한 줄 요약**: (1) Claude Sonnet 5를 통합 judge로 쓰는 V1.3 FP를 시도했으나(66 probe, 22카테고리×3, $2.85) kNN에서 Ceiling V2에 유의하게 패배(delta=-0.0196, p=0.0067). (2) 원인을 무료 GT 기반 probe-count ablation으로 진단: N=3/카테고리는 **완벽한 채점자(GT)를 써도** 유의성 미달(p=0.086), 유의성은 N=6~8부터 확보됨 — judge 품질이 아니라 표본 크기 문제였음을 확정. (3) "judge가 GT를 얼마나 잘 흉내내는가"는 인접 연구주제로 스코프 아웃 결정. (4) 최종 발표(2주 후)용으로 domain(FP latent space 코사인 유사도, 학습 불필요)+difficulty(도메인-정규화 스칼라, probe 최근접이웃 가중평균, 학습 불필요) 이중 신호 기반 캐스케이드 라우터 프로토타입을 설계, 다음 세션에서 구현 예정. 핵심: (1) RouterBench(11개 모델)가 너무 작아서 collapse가 생기는지 확인하려고 **LLMRouterBench**(33개 모델, 22개 태스크 카테고리 확보 가능)로 pool을 3배 확장 — parametric LOO는 여전히 유의미한 개선 없음, pool 크기 자체는 원인이 아님을 확인. (2) probe 선정이 flagship/lightweight tier gap에 지배당하는 버그 발견 → **flagship 13개를 아예 제외하고 lightweight 20개 모델로 피벗**(22개 카테고리 전부 확보). (3) PCA로 Ceiling FP를 분해해서 **핵심 메커니즘 규명**: 신호의 28.5%가 "이 모델이 전반적으로 얼마나 센가"라는 단일 coarse 축이고, 이걸 제거하면 성능이 완전히 붕괴함 — Ceiling이 세밀한 도메인 매칭이 아니라 이 coarse 축 덕분에 이겼다는 뜻. (4) **카테고리 단위로 집계한(개별 probe 아닌) Ceiling FP(Ceiling V2)**가 uniform/Perplexity/기존 Ceiling(V1) 셋 다 유의미하게 이기는 첫 깨끗한 승리 기록. (5) lightweight-20 pool의 **parametric LOO에서도 Ceiling V1이 Perplexity를 유의미하게 이김**(3-시드 평균 delta+0.109, p=0.0024로 확정) — 33개 풀에서는 안 됐던 게, tier gap을 없애니 실제 학습 단계에서도 처음으로 재현됨. (6) 33개 풀에서 Ceiling FP의 지배축이 사실상 tier(플래그십/경량) 분리축임을 직접 확인 → **"Dual-Tier 라우팅"(coarse 축=tier 게이트, 나머지=tier 내부 도메인 매칭) 후속 연구 아이디어 도출, 검증에는 EmbedLLM 규모 필요.** (7) V1/V2 둘 다 multi-seed(3개) 확정 완료 — V2가 방향상 V1보다 약간 우세하나 유의수준 미달(p=0.083), 둘 다 Perplexity는 확실히 이김. Random FP(순수 노이즈) negative control로 파이프라인 자체엔 편향 없음 확인. (8) 특정 6개 모델("reasoning-RL 클러스터")이 V1/V2/Perplexity 전부에서 AUC가 나쁜 이유를 추적 → **probe 선정 편향 발견**: 고분산 probe 528개만 보면 이 그룹 간 격차가 전체 모집단(19%p)의 2배 이상(41%p)으로 과장돼 있었음 — mean-centering이 못 잡는 새로운 종류의 confound. **다음 할 일: 발표 자료 작성 시작(17.14의 future-work 결론 포함), 여유 되면 EmbedLLM 자원 확보/Dual-Tier 설계.**

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
2. **`cost_info_nce` 손실 함수 자체는 논문 원본(`scripts/train_query_encoder.py`, upstream과 diff 없음 확인)과 완전히 동일** — 우리 코드 문제 아님. **[2026-08-02 정정]**: 이 claim은 부정확했음 — 논문 원본(Eq.8, byte-identical to upstream)은 사실 밴드 기반의 `cost_spectrum_info_nce`이고, `cost_info_nce`(밴드 없음 + 분자에 비용 가중 positive-averaging 추가)는 이후 세션에서 만들어진 **별개의 단순화 변형**임이 뒤늦게 확인됨. 자세한 경위와 이 변형을 채택한 근거는 17.19절 참고.
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

### 17.12 Ceiling V2(category-rate) parametric LOO 결과 — V1과 통계적으로 동일 (2026-08-02)

`full_loo_lite20_categoryrate.py`(seed=0) 완료. V1/V2/Perplexity 3자 비교(같은 20개 모델, 같은 seed=0):

| 비교 | delta | p | 승 |
|---|---|---|---|
| V2 vs V1 | -0.0011 | 0.876 (유의 안 함) | 10/20 |
| V2 vs Perplexity | +0.1086 | 0.0030 | 14/20 |

kNN에서는 V2가 V1을 근소하게 유의미하게 이겼지만(17.8, delta+0.0042, p=0.0346), **parametric 단계에서는 그 우위가 재현 안 되고 사실상 동률**. 둘 다 Perplexity는 여전히 확실하게 이김(V1 delta+0.1097, V2 delta+0.1086 — 거의 동일한 마진). **발표 프레이밍(사용자 확정)**: V2(카테고리 집계)를 이상적 capability oracle에 더 가까운 주 방법론으로 세우고, V1(개별 probe-response)은 대안으로 함께 검증 — kNN에서는 V2가 근소 우세, LOO에서는 거의 동일이라는 점을 "방법론 선택에 결과가 민감하지 않다(robustness)"는 근거로 제시.

### 17.13 Ceiling V2 multi-seed 재검증 완료 + Random FP negative control (2026-08-02, 다음날 세션)

**V2 multi-seed**(seed=1,2 추가): V1과 V2를 직접 페어드 비교하면 seed=0에서는 완전 동률(delta-0.0011, p=0.876)이었지만, 3-시드 전체로 보면 **V2가 방향은 일관되게 앞섬**:

| seed | V2 mean AUC | V1 mean AUC | V2-V1 delta | p | V2 vs Perp delta | p |
|---|---|---|---|---|---|---|
| 0 | 0.5845 | 0.5856 | -0.0011 | 0.876 | +0.1086 | 0.0030 |
| 1 | 0.5973 | 0.5796 | +0.0177 | 0.0354 | +0.1255 | 0.0011 |
| 2 | 0.5958 | 0.5734 | +0.0223 | 0.0845 | +0.1324 | 0.0009 |
| **3-시드 평균** | | | **+0.0130** | **0.0829** | **+0.1222** | **0.0013** |

V2 vs V1 직접비교는 3-시드 평균 p=0.083으로 **관례적 유의수준(0.05) 미달 — "V2가 확실히 낫다"고 단정하면 안 됨**, 다만 seed=0만 봤을 때의 "완전 동률" 결론보다는 V2가 약간 우세한 쪽으로 방향이 잡힘. V2 vs Perplexity는 3-시드 전부 확실히 유의(p<0.005), 마진(+0.122)이 V1의 마진(+0.109)보다 약간 큼.

**Random FP negative control** (`build_random_fp_lite20.py` + `full_loo_lite20_random.py`, 22차원 순수 가우시안 노이즈, seed=0): mean AUC=**0.5139**, 0.5와 통계적으로 구분 안 됨(one-sample t-test p=0.349), 유의미하게 위/아래로 갈린 fold 수도 7/7로 대칭. **파이프라인 자체가 무의미한 descriptor로도 좋은 결과를 만들어내는 편향은 없다는 것을 확인** — Ceiling의 유의미한 승리와 Perplexity의 유의미한 역전이 전부 descriptor 내 실제 정보 때문이라는 근거가 하나 더 확보됨.

### 17.14 왜 특정 6개 모델은 세 방법(V1/V2/Perplexity) 전부에서 AUC가 나쁜가 — probe 선정 편향 발견 (2026-08-02)

세 테이블(V1/V2/Random 각각 vs Perplexity) 모두에서 **같은 6개 모델**(Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, GLM-Z1-9B-0414, NVIDIA-Nemotron-Nano-9B-v2, Intern-S1-mini, MiniCPM4.1-8B — 17.6의 PC1 "reasoning-RL 클러스터"와 정확히 일치)이 V1/V2/Perplexity 전부에서 AUC 0.5 미만으로 나오는 걸 발견 → outlier가 아니라 방법론 무관하게 일관된 현상.

**가설 검증 과정** (둘 다 기각됨, 아래 것만 지지됨):
1. ~~"이 6개가 서로 너무 비슷해서 구분이 안 된다"~~ — 기각. 실제 쿼리별 정답 상관관계(0.53~0.61)가 오히려 FP 코사인 유사도(0.29~0.52)보다 높음.
2. ~~"이 그룹이 원래 약한 모델들이다"~~ — 기각. 전체 Set A 기준 이 그룹 정답률(61.9%)이 비교군("plain instruct" 7개, 43.0%)보다 오히려 높음. 어려운 문제를 이 그룹만 유일하게 맞추는 비율도 15.9% vs 3.9%로 압도적.
3. **✅ 지지됨 — probe 선정 편향**: Ceiling FP를 만드는 528개 고분산 probe만 따로 보면, 이 두 그룹의 격차가 **전체 모집단(19%p)보다 2배 이상 과장**(41%p: reasoning-RL 70.1% vs plain-instruct 29.0%)됨. "고분산 probe를 뽑는다"는 기준 자체가 특정 서브그룹 간 격차를 실제보다 과장해서 반영하는 부작용이 있었음.

**메커니즘(가설, 완전 검증은 아님)**: descriptor(타겟)는 이 과장된 528개 probe로 만들어지는데, 실제 MLP 학습은 Set A 전체(~12,000개, 모집단 비율)로 이뤄짐 → 인코더가 "일반 쿼리 → 과장된 타겟 관계"를 학습 → Set B(모집단 대표 분포) 평가에서 이 관계가 깨짐. 다만 이게 단순 calibration 오차(기대치보다 낮음)를 넘어 AUC 0.5 미만(완전 역전)까지 가는 정확한 경로는 아직 다 못 밝힘 — 발표에서 깊게 안 파고들어도 되는 수준.

**⭐ Future work 결론 (사용자 확정, 발표에 포함)**: "Capability를 평가해서 FP로 변환하는 과정에서 사용된 probe prompt의 종류·양상, 그리고 그게 model pool 생태계 내에서 어떤 위치를 갖는지가 결과에 중대한 영향을 미친다. 따라서 향후에는 (a) coarse하지 않고 fine-grained하면서도 (b) 특정 서브그룹에 편향되지 않는 universal한 채점/probe 선정 기법이 중요한 연구 방향이다." — mean-centering이 잡아내는 "공유 난이도" 편향과는 별개로, "probe 선정 자체의 서브그룹 편향"이라는 새로운 종류의 confound를 이번 세션에서 추가로 발견한 셈.

### 17.15 예상 Q&A 대비 — "결국 유리한 probe만 고른 거 아니냐"는 질문에 대한 방어 논리

17.14의 probe 선정 편향 발견을 발표하면, "그럼 너희가 한 것도 결국 골라서 좋은 결과 낸 거 아니냐(cherry-picking)"는 질문이 나올 수 있음. 대비한 답변 정리:

1. **선정 기준은 결과-무관·사전 확정 알고리즘이었다**: 어떤 FP가 이기는지 보고 probe를 고른 게 아니라, "분산이 가장 높은 probe"라는 사전에 정해둔 알고리즘 기준을 결과와 무관하게 그대로 적용함. 이게 진짜 cherry-picking(원하는 결론이 나올 때까지 반복 선택)과 다른 점.
   - **다만 이렇게만 말하면 안 됨**: "그 기준 자체에서 방금 편향을 발견하지 않았냐"고 반박당할 수 있음. 그러니 **"우리 방법엔 편향이 없다"가 아니라 "선정 기준은 원칙적이었지만, 사후에 그 기준이 의도치 않게 특정 subgroup 격차를 왜곡시킨다는 걸 발견했고 이를 투명하게 보고한다"**는 프레이밍이 더 방어력 있음(편향을 스스로 찾아내 정직하게 보고한 것 자체가 신뢰도를 높이는 포인트).
2. **Ceiling V2(category-rate)는 애초에 선정 단계가 없다**: probe를 고르는 절차 자체가 없이 Set A 전체 평균을 쓰기 때문에, "특정 probe를 골랐다"는 비판이 원천적으로 성립 안 함. V1에 대한 비판을 V2가 방어해주는 구도.
3. **"섬세하게 조정하면 나머지도 chance를 넘길 것"은 검증 안 된 가설**: 6개 모델(Intern-S1-mini, GLM-Z1-9B-0414, Qwen3-8B, NVIDIA-Nemotron-Nano-9B-v2, MiniCPM4.1-8B, DeepSeek-R1-0528-Qwen3-8B) 얘기임(4개 아님). 메커니즘상 그럴 것으로 예상되지만 **아직 실제로 교정된 probe 선정으로 재실험해본 적은 없음** — 발표에서는 확정된 결과처럼 말하지 말고 "다음 검증 과제"로 명시할 것.

### ⚠️ 17.16 네이밍 변경 공지 (2026-08-02) — "Ceiling V1/V2" → "Pseudo Ceiling / Ceiling FP"

**사용자 확정, 이 시점 이후 새로 작성되는 내용부터 적용**:
- **"Ceiling V2"(category-rate, 22차원, Set A 카테고리 전체 평균) → "Ceiling FP"**로 개명. 앞으로는 이게 주(canonical) 방법론 이름.
- **"Ceiling V1"(528-probe, 개별 고분산 probe 528개) → "Pseudo Ceiling"**으로 개명. probe 선정을 거친 근사치라는 의미를 이름에 담음(17.14/17.15에서 다룬 probe 선정 편향 문제와도 이름이 자연스럽게 연결됨).

**주의**: 17.1~17.15 등 **이 공지 이전에 작성된 본문은 소급 수정하지 않음**(그 시점 작성 당시의 "V1/V2" 표기 그대로 유지 — 세션 기록의 시간순 정합성 보존 목적). 과거 섹션을 읽을 땐 "V1=Pseudo Ceiling, V2=Ceiling FP"로 치환해서 이해할 것. **이 공지 이후 작성되는 섹션(발표자료 포함)부터는 새 이름(Pseudo Ceiling / Ceiling FP)을 사용.** 발표 자료·이미지(테이블/막대그래프)도 새 이름으로 재생성함 — `local_descriptors/llmrouterbench_lite20/table_v1_vs_perp.png` 등 기존 파일명은 유지하되(스크립트 재실행 편의상), 이미지 안의 라벨/제목 텍스트는 새 이름으로 교체.

### 17.17 3-벤치마크 비교표 완성 — MixInstruct/RouterBench에 Pseudo Ceiling 신규 실험 추가 (2026-08-02)

발표용으로 MixInstruct/RouterBench/LLMRouterBench 3개 벤치마크를 나란히 비교하는 표를 만들다가, 기존 MixInstruct·RouterBench의 "Ceiling"이 이미 **카테고리(클러스터) 평균 기반**(=새 이름으로 "Ceiling FP")이었다는 걸 재확인 — 이 두 벤치마크에 진짜로 없던 건 **개별 고분산 probe 선정 기반**("Pseudo Ceiling")뿐이었음. 새로 만들어서 채움:

- **RouterBench Pseudo Ceiling**(`scripts/routerbench_pseudo_ceiling.py`): 86개 eval_name 카테고리별로 고분산 probe 6개씩 층화추출(512개 총), mean-centered, L2-normalized. 결과: **delta+0.1012, p=0.0200, 9/11 fold 개선** — 기존 Ceiling FP(+0.0982, p=0.0205)와 거의 동급.
- **MixInstruct Pseudo Ceiling**(`scripts/mixinstruct_pseudo_ceiling.py`): 진짜 카테고리가 없어서(k-means pseudo-cluster만 있음) 층화추출 없이 Set A 84,000개 프롬프트 전체에서 고분산 512개를 그냥 뽑음(방법론적 단순화, 명시적으로 기록). 결과: **delta-0.0575, p=0.0132, 2/11 fold만 개선** — 유의미하게 uniform보다 나쁨. 기존 MixInstruct 결과들(Ceiling FP -0.0070, Perplexity -0.0076)과 같은 패턴 재확인.

**최종 3-벤치마크 비교표**(`local_descriptors/llmrouterbench_lite20/three_benchmark_comparison_table.png`, V1.2는 발표 스코프에서 제외):

| Benchmark | Pseudo Ceiling | Ceiling FP | Perplexity |
|---|---|---|---|
| MixInstruct | -0.0575* | -0.0070 | -0.0076* |
| RouterBench | **+0.1012*** | +0.0982* | +0.0111 |
| LLMRouterBench | +0.0138* | **+0.0180*** | +0.0065 |

**MixInstruct에서 전체적으로 성능이 안 좋은 이유(정리)**: uniform baseline 자체가 rho=0.73으로 극단적으로 강함 — BartScore가 "이 프롬프트가 원래 쉬운가 어려운가"라는 전 모델 공유 성분에 지배당해서, 아무 가중치 없이 나머지 모델 평균만 내도 이미 그 공유 신호를 다 잡아냄. 게다가 MixInstruct는 거의 균질한 instruction-following 프롬프트라 태스크 다양성이 거의 없어서, 모델별 도메인 차별화가 드러날 여지 자체가 부족함. → **"태스크 다양성이 있어야 capability-alignment FP가 의미를 가질 여지가 생긴다"**는 이 세션 전체의 핵심 메시지와 정확히 일관됨(MixInstruct < RouterBench < LLMRouterBench 순으로 다양성도, 결과도 좋아지는 패턴).

### 17.18 멘토 피드백 대응 — Perplexity 채점 모델(GPT2) 크기를 키워도 결과 그대로, kNN·parametric LOO 둘 다 multi-seed 확정 (2026-08-02, 다음날 세션)

**멘토 피드백**: Perplexity FP가 GPT2(1.24억 파라미터, 2019년 구형 모델)로 채점되는 반면 Ceiling FP는 사실상 벤치마크 정답(완벽한 채점자) 기반이라 공정성(fairness) 문제로 비칠 수 있음 — 채점 모델을 더 무겁게 바꿔도 결과가 그대로인지 확인 필요. (V1.2처럼 새 방법론을 도입할 땐 LLM judge를 진짜 70B급이나 유료 API로 쓰는 것도 고려하라는 의견도 있었음 — V1.2가 발표 스코프에서 빠져서 실행은 안 하고 향후 과제로만 기록.)

**실험**: `build_perplexity_fp_lite20_heavier.py`로 같은 528개 probe를 gpt2(1.24억) 대신 **gpt2-large(7.74억, 6배)**로 cross-entropy 재계산(통제 비교, 나머지 전부 동일).

**kNN 테스트** (`local_descriptors/llmrouterbench_lite20/perplexity_gpt2large/`): GPT2-large Perp vs 원본 GPT2 Perp, delta=-0.0008, p=0.1828 — 사실상 동일. 단, 이 kNN 테스트 자체에서는 원래도 Ceiling FP vs Perplexity 직접비교가 유의하지 않았음(17.5, p=0.1468)에 주의 — 그래서 좀 더 결정적인 검증을 위해 parametric LOO(Perplexity가 Ceiling FP에 확실히 진 곳, 17.9/17.11)에서도 재확인.

**Parametric LOO 3-시드 확정** (`full_loo_lite20_gpt2large.py`, `full_loo_gpt2large_results.json`):

| seed | GPT2-large Perp mean AUC | 원본 GPT2 Perp mean AUC | Ceiling FP mean AUC |
|---|---|---|---|
| 0 | 0.4811 | 0.4759 | 0.5856 |
| 1 | 0.4725 | 0.4718 | 0.5796 |
| 2 | 0.4687 | 0.4634 | 0.5734 |
| **3-시드 평균** | **0.4741** | **0.4704** | **0.579** |

GPT2-large vs 원본 GPT2 직접비교(3시드×20모델 풀링, n=60): **delta+0.0037, p=0.3451 — 완전히 무의미한 차이.**

**결론**: 채점 모델을 6배(1.24억→7.74억) 키워도 kNN·parametric LOO 둘 다에서 결과가 사실상 그대로였고, Ceiling FP와의 격차(0.47대 vs 0.58대)는 전혀 안 좁혀짐(multi-seed로 확정). "채점 모델이 작아서 Perplexity가 불리했다"는 fairness 가설은 **기각**됨 — Perplexity(surprisal/유창성) 신호 자체가 Ceiling FP(실제 정답 기반) 신호만큼 capability를 못 담는다는 게 더 근본적인 방법론적 한계라는 근거가 강화됨. 멘토 피드백에 대한 명확하고 확정된 답변으로 발표에 포함할 것.

### 17.19 Loss 함수 정정 + "Altered Loss" 채택 근거 정리 — 이 세션 최종 발표 프레이밍 (2026-08-02)

**배경 정정**: 15.1절의 "`cost_info_nce`가 논문 원본과 동일하다"는 기존 claim은 부정확했음. 실제로는:
- **`cost_spectrum_info_nce`**(밴드 기반, `n_bands`개 비용 구간마다 별도 temperature $\tau_b = \tau_{min}+\alpha\bar{c}_b$ 사용) = 논문 Eq.8 원본, `scripts/train_query_encoder.py`에서 byte-identical to upstream으로 확인됨.
- **`cost_info_nce`**(밴드 없음, 단일 temperature + 분자에 비용 가중 positive-averaging $w^{pos}_m \propto (1-c_m)$ 추가) = 이 프로젝트가 LOO 실험(15번 섹션)을 위해 만든 **별개의 단순화 변형**. 분자의 비용 가중 평균 항은 논문 코드 어디에도 없음 — 정확한 외부 출처는 못 찾음.

**"Altered Loss"로 프레이밍하는 근거 3가지** (발표 메인 서사에 포함, 사용자 확정):

1. **방법론적 근거 — 밴드 구조가 이 pool 규모에서 수학적으로 붕괴함**: $\tau_b$는 band 소속 모델들의 평균 비용인데, `n_bands=5`·모델 20개면 band당 3~4개뿐 → $\bar{c}_b$가 사실상 그 소수 멤버 개개인의 비용과 거의 같아짐. 극단(`n_bands`=모델 수)에서는 $\tau_b$가 문자 그대로 "모델별 개별 temperature"가 됨 — 즉 논문이 의도한 "굵직한 비용 계층"이라는 구조가, 이 규모(11~33개 모델)에서는 이미 그 퇴화 극단에 가까워져 있음. **밴드 없는 단일 temperature 버전이 오히려 이 규모에 더 적합한 선택.**
2. **실증적 근거 — 어느 loss를 쓰든 핵심 발견(collapse)은 동일함**: collapse는 애초에 **논문 원본 loss로 먼저 발견**됐음(13번 섹션, 2026-07-25). 이후 RouterBench에서 `cost_info_nce`→`cost_spectrum_info_nce`로 직접 바꿔서도 재검증(16번 섹션) — collapse 동일하게(오히려 한 fold는 더 심하게, 100% 단일모델 쏠림) 재현됨. 완전 대칭 합성 pool에서는 `cost_info_nce`로 98%(chance 20%) 정상 작동 확인 — **loss 변형이 문제를 만든 게 아니라, 실제 pool의 비대칭성(일부 모델이 전반적으로 강함)이 원인**(MoE 문헌의 Expert Collapse, Shazeer et al. 2017과 일치).
3. **직접 재현 검증 — 논문 원본 loss로도 핵심 결과가 (마진은 줄어도) 유지됨, multi-seed 확정**: `full_loo_lite20_paperloss.py`(별도 디렉토리 `local_descriptors/llmrouterbench_lite20_paperloss/`, `n_bands=5`)로 lightweight-20 pool Ceiling vs Perplexity를 seed 0/1/2 세 번 재실행. **⚠️ 정정(2026-08-03): 이 스크립트는 `loo.CEILING_DIR`(= `local_descriptors/llmrouterbench_lite20/ceiling/`, 528차원)을 그대로 재사용함 — 즉 여기서 재현된 건 Ceiling FP(V2, category-rate, `ceiling_categoryrate/`)가 아니라 원래 17.9~17.11에서 먼저 유의미했던 headline 결과인 Pseudo Ceiling(V1, 528-probe)임. 아래 표의 "Ceiling"은 전부 Pseudo Ceiling을 가리킴.**

| seed | Perplexity mean AUC | Ceiling mean AUC | delta | p(seed) |
|---|---|---|---|---|
| 0 | 0.5189 | 0.5686 | +0.0498 | 0.0304 |
| 1 | 0.5397 | 0.5629 | +0.0232 | 0.2466 |
| 2 | 0.5193 | 0.5674 | +0.0481 | 0.0073 |
| **3-시드 pooled (n=60)** | **0.5260** | **0.5663** | **+0.0403** | **0.0005** |

| | 3-시드 pooled Ceiling AUC | 3-시드 pooled Perplexity AUC | delta | p |
|---|---|---|---|---|
| `cost_info_nce`(단순화, 주 결과, 3-시드 pooled, n=60) | 0.5796 | 0.4704 | +0.1092 | 1.05e-7 |
| `cost_spectrum_info_nce`(논문 원본, n_bands=5, 3-시드 pooled, n=60) | 0.5663 | 0.5260 | +0.0403 | 0.0005 |

**⚠️ 정정(2026-08-03)**: 이 표의 `cost_info_nce` 행 수치는 원래 0.579/+0.1222/p=0.0013으로 잘못 기록되어 있었음 — `full_loo_results.json`(seed0) + `full_loo_multiseed_results.json`(seed1,2)에서 직접 재계산한 정확한 값(0.5796/+0.1092/p=1.05e-7, sign test 42/60)으로 정정.

**반전 없음 — seed=1 단독으로는 유의하지 않았지만(p=0.2466), 3개 시드를 pooling하면 논문 원본 loss로도 유의미(p=0.0005)하게 Ceiling(Pseudo Ceiling) 승리.** 효과 크기는 단순화 loss의 대략 1/3 수준(+0.0403 vs +0.1092, 주로 Perplexity 평균이 0.47대→0.53대로 오른 영향, 특히 17.6/17.14의 "reasoning-RL 클러스터" 역전이 완화됨)인데, 이건 정확히 1번 근거("밴드가 이 규모에서 신호를 흐린다")와 일관됨 — 격차를 없애는 게 아니라 흐리는 정도. **"loss를 단순화한 이유가 결과를 유리하게 만들려던 것 아니냐"는 의심에 대한 직접 답: 논문 원본 loss로도 핵심 결론은 (세 시드 모두 필요하긴 하지만) 유의하게 재현된다.**

**최종 발표 프레이밍(사용자 확정)**: "이 프로젝트는 collapse 현상과 좁은 model pool 문제를 극복하기 위해 논문의 cost-aware contrastive loss를 의도적으로 변형(altered)해서 실험을 진행했다 — 밴드 구조 없이 단일 temperature로 단순화. 이 선택은 (a) 밴드 구조가 이 규모 pool에서 수학적으로 개별-모델 temperature에 가깝게 퇴화한다는 것, (b) 어느 버전을 쓰든 collapse라는 핵심 현상은 동일하게 재현된다는 통제 실험, (c) 논문 원본 loss로 핵심 발견(Ceiling > Perplexity, 여기서는 Pseudo Ceiling 기준)을 직접 재검증해 마진은 줄어도 방향은 유지됨을 확인한 것, 세 가지로 뒷받침된다. 논문 원본 loss 결과는 부차적 재현 검증(secondary robustness check)으로 별도 제시."

### 17.20 Collapse degree 정량 비교 + 메커니즘 + "Collapse가 사실 정답 Oracle 아니냐" Q&A 검증 (2026-08-03)

발표에서 가장 두려운 두 질문에 대한 사전 방어 논리를 정리함: **(1) "논문 원본 loss로 바꿨더니 결과가 안 좋게 나오니까, 결과 좋게 나온 단순화 loss로 그냥 밀어붙인 거 아니냐"**, **(2) "지금 관찰되는 collapse가 사실 정답 오라클을 정확히 반영한 건강한 라우팅 아니냐."** 17.19가 (1)에 대한 답이고, 아래는 (1)의 정량적 근거 보강과 (2)에 대한 실증 검증.

**Collapse 정도(top3_share, 200개 고정 probe의 nearest-FP-neighbor 라우팅 배정 중 top3 모델이 차지하는 비율, chance=0.15) 정량 비교**:

| Loss 변형 | FP | top3_share 평균 | n |
|---|---|---|---|
| `cost_info_nce`(단순화, seed=0) | Ceiling(Pseudo Ceiling) | 0.3183 (std 0.0484) | 20 |
| `cost_info_nce`(단순화, seed=0) | Perplexity | 0.3445 (std 0.0368) | 20 |
| `cost_spectrum_info_nce`(논문 원본, 3-시드 pooled) | Ceiling(Pseudo Ceiling) | 0.3911 (std 0.0928) | 60 |
| `cost_spectrum_info_nce`(논문 원본, 3-시드 pooled) | Perplexity | 0.4302 (std 0.0716) | 60 |

**(위 표는 모두 Pseudo Ceiling(V1, 528-probe) 기준 — Ceiling FP(V2, category-rate)는 이 ablation을 아직 안 돌려봤음, 미검증.)**

Welch t-test: Ceiling(Pseudo Ceiling)에서 논문 원본 loss가 단순화 loss보다 collapse가 유의미하게 더 심함(diff +0.0728, **p=0.000038**). Perplexity에서도 동일(diff +0.0857, **p<0.000001**). → **단순화 loss를 채택한 근거가 "결과가 좋아서"가 아니라 "collapse가 덜해서"라는 걸 결과와 독립적인 지표(top3_share)로 재확인.** 이 지표는 Ceiling vs Perplexity 비교(우리가 밀고 싶은 결론)와 무관하게, 두 FP 모두에서 일관되게 같은 방향(단순화<원본)이라 "원하는 결론에 맞춰 고른 지표"라는 반박이 성립하기 어려움.

**메커니즘 — temperature mismatch (코드로 확인)**: `cost_spectrum_info_nce`는 밴드별 가변 온도 $\tau_b = \tau_{min} + \alpha\bar c_b$(`loo_unseen_recovery.py:306`)를 쓰는데, 같이 걸리는 보조 손실 `load_balance_loss(q, E)`(MoE expert-collapse 방지용, Shazeer et al. 2017)는 `tau=None` 기본값이라 **고정 전역 temperature 0.05**(`TEMPERATURE = 0.05`, 파일 56번 줄)를 그대로 씀(`loo_recovery_lite20_paperloss.py:63-64`에서 두 손실을 이렇게 같이 호출). 즉 주 손실의 실제 temperature가 밴드마다 바뀌는데, collapse를 막아야 할 보조 손실은 그 변화를 전혀 모른 채 고정된 값으로 작동함 — 이 함수 자신의 docstring도 "primary loss와 같은 temperature를 써야 한다"고 명시해 둔 조건을 논문 원본 loss 조합에서는 실제로 어기고 있는 셈. 단순화 loss(`cost_info_nce`)는 애초에 전역 단일 temperature 하나만 쓰므로 이 불일치가 구조적으로 발생하지 않음.

**reasoning-RL 클러스터(17.14의 6개 모델: Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, GLM-Z1-9B-0414, NVIDIA-Nemotron-Nano-9B-v2, Intern-S1-mini, MiniCPM4.1-8B) 비용 재확인**: 이 6개의 평균 정규화 비용이 나머지 14개 평균의 **7.38배**(cost_dict 기준 실측, 이전에 언급했던 배수와 다름 — 재계산으로 정정). 비용이 높은 모델일수록 코사인 유사도 페널티(`cost_pen = gamma·cost_norm`)가 커지고 손실 지형이 가팔라지는데, 여기에 위 temperature mismatch까지 겹치면 이 클러스터가 유독 불안정해지는 방향과 일치함(다만 이건 정성적 정합성이지 별도로 통제된 인과 검증은 아님 — 발표에서는 "정합적인 가설"로만 제시할 것).

**(2) True Oracle 검증 — collapse가 정답 분포를 반영한 결과인가**: Set B 3,115개 쿼리 중 정답 가능한 2,731개에 대해, "정답을 맞춘 모델 중 가장 저렴한 모델"(=true oracle)을 모델별로 집계하고, 라우터의 실제 쏠림(collapse_nearest_dist, `cost_info_nce`/Ceiling(Pseudo Ceiling)/seed=0/20-fold 합산)과 비교:

| | true oracle 점유율 | 라우터 실제 쏠림 점유율 |
|---|---|---|
| GLM-Z1-9B-0414 | **68.3%** | 7.0% |
| MiniCPM4.1-8B | 9.6% | 4.9% |
| MiMo-7B-RL-0530 | 4.0% | 4.0% |
| Qwen2.5-Coder-7B-Instruct | 3.7% | **9.8%**(라우터 1위) |
| (나머지 16개) | 각 <3% | 각 1.6~6.6% |

- true oracle의 top3 점유율은 **0.8184**(chance 0.15) — 실제 정답 자체가 극도로 GLM-Z1-9B-0414 한 모델에 쏠려있음.
- 20개 모델 전체에 대해 (true oracle 점유율) vs (라우터 쏠림 점유율)의 **Spearman rho=0.2611, p=0.2662 — 통계적으로 무상관**.
- 결정적 근거: true oracle의 68.3%를 차지하는 GLM-Z1-9B-0414에게 라우터는 겨우 7.0%만 배정함(오라클 몫의 1/10 수준). 반대로 라우터가 가장 많이 배정한 Qwen2.5-Coder-7B-Instruct는 true oracle 점유율이 3.7%에 불과함.

**결론(두 방향을 다 인정하는 정직한 프레이밍)**: 정답 분포 자체가 극단적으로 쏠려있다는 건 사실이라, "쏠림 자체가 비정상"이라는 주장은 할 수 없음. 하지만 **라우터가 관찰하는 collapse는 이 실제 오라클 쏠림과 통계적으로 무관하게(rho=0.26, ns) 별도 모델(Qwen2.5-Coder 등)에 쏠려있음** — 즉 "우리 라우터의 collapse가 정답 구조를 정확히 학습해서 생긴 건강한 현상"이라는 반박은 데이터로 성립하지 않음. 오히려 라우터가 진짜 oracle 구조를 제대로 못 잡아내고 있다는 뜻(under-fitting 쪽에 가까움)이며, 이는 collapse가 "의도된 정답 반영"이 아니라 "학습 과정에서 생긴 편향(온도 불일치, 좁은 pool에서의 상대적 강자로의 쏠림)"이라는 이 세션 전체의 진단과 일관됨.

### 17.21 "AUC 말고 실제 라우팅 성능도 확인했냐" Q&A 대비 — router_overall_accuracy 검증 + AUC의 역할 재정리 (2026-08-03)

**예상 질문**: "지금까지 AUC만 보여줬는데, 실제 라우팅 성능(라우터가 고른 모델이 진짜 정답을 맞추는 비율)은 확인해봤냐? 새 loss의 라우팅 성능이 실제로 좋아지는지 확인했냐?"

**답: 이미 매 fold 결과에 `router_overall_accuracy`(라우터의 top-1 선택이 Set B에서 실제로 정답인 비율)가 저장돼 있었음 — 지금까지 AUC만 집계하고 이걸 따로 안 봤을 뿐. 지금 집계함.**

| Loss | Ceiling router_overall_accuracy | Perplexity router_overall_accuracy | delta | p |
|---|---|---|---|---|
| `cost_info_nce`(단순화, 3-시드 pooled, n=60) | 0.6172 | 0.6158 | +0.0014 | 0.2982 (ns) |
| `cost_spectrum_info_nce`(논문 원본, 3-시드 pooled, n=60) | 0.5784 | 0.5693 | +0.0091 | **0.000019** |

**AUC 격차만큼 실제 라우팅 정확도 격차가 나지는 않음(특히 단순화 loss에서는 거의 무의미, p=0.30). 이건 나쁜 소식이 아니라 AUC가 원래 재려는 것과 router_overall_accuracy가 재는 것이 다르기 때문**: AUC(`auc_heldout_correctness`)는 held-out(unseen) 모델 **1개**의 FP 유사도가 그 모델의 실제 정답 여부를 얼마나 잘 rank-order하는지만 재는 지표 — 이 프로젝트의 실제 연구 질문("재학습 없이 descriptor 하나만으로 unseen 모델을 얼마나 잘 복원하는가")을 정확히 겨냥한 지표임. 반면 `router_overall_accuracy`는 20개 후보 중 19개가 이미 학습에 쓰인 seen 모델이라 그 라우팅 품질에 압도적으로(19:1) 좌우됨.

**이 두 지표를 같이 보면 오히려 프레이밍이 명확해짐**: router_overall_accuracy가 FP에 따라 거의 안 갈린다는 건 "이미 아는 19개 모델 사이의 라우팅은 어느 FP를 쓰든 이미 잘 되고 있다(포화 상태)"는 뜻이고, AUC가 크게 갈린다는 건 "그 위에 새로 추가되는 unseen 모델을 얼마나 잘 이해하느냐"에서만 Ceiling FP가 확실히 앞선다는 뜻. 즉 **전체 라우팅 성능을 깎아먹지 않으면서 정확히 어려운 부분(unseen 복원)에서만 이긴다** — AUC를 주 지표로 쓴 선택이 자의적이지 않고 연구 질문에 정확히 부합함을 보여줌.

**보조 검증 시도, 결과는 애매함(정직하게 기록)**: `oracle_match_rate`(held-out 모델이 특정 쿼리의 진짜 oracle일 때 라우터가 실제로 그걸 골랐는지 비율)로 더 직접적인 확인을 시도했으나, fold당 해당 이벤트 수가 너무 적어(n_oracle_is_M이 작음) 노이즈가 커서 방향이 깨끗하게 안 나옴(단순화 loss pooled: Ceiling 0.0678 vs Perplexity 0.0354, delta+0.0324, p=0.147; 논문 loss pooled: delta+0.0111, p=0.730). **이 지표는 추가 증거로 쓰지 않음** — 위 두 지표(AUC + router_overall_accuracy)로만 설명하는 게 정직함.

**아직 안 채운 구멍(정직하게 표시)**: 위 비교는 전부 raw accuracy 기준이고, 비용을 고려한 진짜 cost-accuracy tradeoff(AUDC 등)는 이 lightweight-20 pool에서 한 번도 계산해본 적 없음. 참고로 비용을 완전히 무시한 "static-best"(항상 정확도 1위 모델만 사용, Qwen3-8B, 정확도 0.6533) baseline이 라우터의 raw accuracy(0.578~0.617)보다 높게 나오는데, Qwen3-8B는 비용 순위 18/20(거의 최고가 축)이라 이건 공정한 비교가 아님(라우터는 비용까지 고려한 트레이드오프를 하는 것). 하지만 이 트레이드오프를 정량적으로 보여주는 지표(AUDC)는 아직 미검증 — "라우팅 성능이 실제로 좋아지냐"는 질문에 raw accuracy로는 답했지만, cost-accuracy tradeoff까지 완전히 검증된 건 아님을 발표에서 숨기지 말 것.

### 17.22 AUDC(cost-accuracy tradeoff) full-pool 실험 — 17.21의 미검증 구멍을 채움 (2026-08-03)

**배경**: 17.21에서 "라우팅 성능이 실제로 좋아지냐" Q&A에 raw accuracy(router_overall_accuracy)로는 답했지만, 진짜 cost-accuracy tradeoff(AUDC)는 이 lightweight-20 pool에서 한 번도 계산 안 해봤다는 구멍을 남겨뒀음. 이번에 채움.

**방법** (`scripts/llmrouterbench/audc_fullpool_lite20.py`, 신규): LOO(held-out) 방식이 아니라 **20개 모델 전체로 한 번에 학습**(재학습 필요 없이 fold 20번 대신 FP당 1번만 학습하면 되는 구조 — Ceiling FP, Perplexity FP 각각). 이미 학습된 임베딩 공간에서 inference 시점 결정 규칙을 $\text{argmax}(sim - \lambda \cdot c_m)$로 바꾸고, $\lambda$를 0(비용 완전 무시)부터 50까지 41개 지점으로 sweep — 지점마다 (평균 비용, 정확도) 쌍을 구해서 cost-accuracy curve를 그리고, 두 FP를 **공통 cost grid**(정규화 안 하면 Ceiling·Perplexity가 도달하는 cost 범위 자체가 달라서 직접 비교가 안 됨 — 처음 계산할 때 이 실수를 했다가 재계산함)에 맞춰 step-function으로 보간한 뒤 trapezoidal 적분해서 AUDC를 구함. `cost_info_nce`(단순화 loss), seed 0/1/2.

**결과**:

| seed | Perplexity AUDC | Ceiling AUDC | delta |
|---|---|---|---|
| 0 | 0.6170 | 0.6042 | -0.0128 |
| 1 | 0.6133 | 0.6135 | +0.0002 |
| 2 | 0.6133 | 0.6176 | +0.0043 |
| **3-시드 평균** | **0.6145** | **0.6118** | **-0.0028** |

**paired t-test (n=3): p=0.6478 — 방향도 seed마다 음/거의0/양으로 흔들리고, 통계적으로 완전히 무의미한 차이.** (n=3이라 검정력 자체가 낮다는 것도 명시할 것 — 이 프로젝트 다른 검정(LOO fold 단위, n=60)만큼 신뢰도 높은 검정은 아님.) $\lambda=0$(비용 무시) 정확도로도 delta=-0.0017, p=0.6458로 동일하게 무의미.

**해석**: 나쁜 소식이 아니라 17.21 프레이밍을 한 번 더 확인해줌 — held-out 없이 20개 모델 전체로 학습했을 때(=순수 "이미 아는 모델들 사이의 라우팅" 성능), cost-accuracy tradeoff는 Ceiling FP와 Perplexity FP 사이에 유의미한 차이가 없음. Ceiling FP의 강점은 AUC로 잡히는 **unseen 모델 복원**에 있지, 이미 아는 모델들의 cost-aware 라우팅 품질 자체에는 없다는 것 — 17.21의 "전체 라우팅 성능을 깎아먹지 않으면서 어려운 부분에서만 이긴다"는 결론과 정확히 일관됨.

**결과물**: `local_descriptors/llmrouterbench_lite20/audc_fullpool_results_seed{0,1,2}.json`, `audc_fullpool_multiseed_table.png`.

### 17.23 (급히 기록) load_balance_loss OFF 상태의 base collapse 수준 — paper loss(cost_spectrum_info_nce), full pool, seed=0 (2026-08-04)

`scripts/llmrouterbench/collapse_fullpool_paperloss_nobalance.py`로 확인. β=1.0(기존)일 때 top3_share ~0.39-0.43이었는데, **β=0(load_balance_loss 완전히 끔)이면 top3_share가 Perplexity 0.9750, Ceiling 0.9800로 거의 완전 collapse**함 — Intern-S1-mini + DeepSeek-R1-Distill-Qwen-7B 두 모델이 200개 probe 중 188~193개를 독점.

**결정적으로, 이 두 모델의 true oracle 점유율은 각각 0.44%뿐**(진짜 주역인 GLM-Z1-9B-0414는 68.25%인데 무시당함) — "collapse가 oracle과 무관하다"는 17.20 결론이 base 상태(방어장치 없음)에서는 훨씬 더 극단적으로 나타남. load_balance_loss는 (temperature mismatch로 최적은 아니어도) top3_share를 98%→40%대로 확실히 줄이는 실질적 효과가 있음이 이걸로 확인됨.

또한 CSCR 논문 원본 코드(`train_query_encoder.py`, upstream과 byte-identical 확인됨, 커밋 작성자 Reza Shirkavand 본인)엔 `load_balance_loss`가 아예 없음 — `cost_info_nce`와 `cost_spectrum_info_nce` 둘 다 논문 저자 코드에 실존하지만(전자는 논문 본문에 라벨링 안 된 버전), load-balancing 보조항은 이 프로젝트가 자체 추가한 것. 즉 "논문 원본 방법론은 애초에 collapse 방어 장치가 없다"가 정확한 설명.

**미해결 가설(발표에는 말로만 언급, 검증 안 됨)**: 멘토가 이전에 진단한 "multi-positive 가중평균이 여러 정답 모델을 동시에 만족시키는 '제너럴리스트 유인자'를 만들어 collapse를 조장한다"는 가설(`cost_info_nce_cheapest`의 docstring, PROGRESS.md 16.2 참고) — 검증용 pilot(`scripts/cheapest_loss_pilot.py`, 4개 모델)은 실행됐으나 결과가 저장 안 돼 있어 확인 불가. 다음 세션 후보 작업.

**결과물**: 별도 JSON 저장 없음(콘솔 출력만, 시간 급해서 스크립트 실행 결과만 기록). 재현하려면 위 스크립트 그대로 재실행.

### 다음 할 일 (2026-08-03 세션 종료 시점, 최신)
1. **✅ 완료**: lightweight-20 parametric LOO multi-seed 재검증(V1, V2 둘 다) — 17.11, 17.13 참고.
2. **✅ 완료**: Random FP negative control — 17.13 참고, 파이프라인 편향 없음 확인.
3. **✅ 완료**: 6개 모델 AUC 저조 현상 원인 규명 — 17.14 참고, probe 선정 편향 발견 + future work 방향 확정.
4. **✅ 완료**: "Altered Loss" 채택 근거 3종 + multi-seed 확정 + collapse degree 정량 비교 + true oracle 무상관성 검증 — 17.19, 17.20 참고. Q&A 방어 논리 확정.
5. **✅ 완료**: "AUC 말고 실제 라우팅 성능 확인했냐" Q&A 대비 — 17.21 참고. router_overall_accuracy 집계 완료, AUC=unseen 복원 전용 지표라는 프레이밍 확정.
6. **✅ 완료**: AUDC(cost-accuracy tradeoff) full-pool 실험, multi-seed — 17.22 참고. 17.21의 마지막 미검증 구멍 채움, 유의미한 차이 없음으로 확정(프레이밍과 일관).
7. **발표 자료 작성 시작**(사용자 확정) — 서사: "본인이 제안한 가설 → kNN 파일럿 검증 → parametric 확장 시도 → 실패 진단(tier gap) → 원인 규명 및 재현(V1/V2/Random 전부 multi-seed 확정) → 세부 원인 추가 규명(probe 선정 편향) → loss 변형 근거 방어(17.19/17.20) → 실제 라우팅 성능 방어(17.21/17.22) → 후속 연구(Dual-Tier + fine-grained universal scoring) 제안". 이 프로젝트가 "CSCR 논문 재현"이 아니라 "본인의 독자적 연구 주제"라는 점을 명확히 할 것.
8. **Dual-Tier 라우팅 설계 + EmbedLLM 규모 확보** — 17.11/18번 섹션 참고. 교수님께 EmbedLLM 규모 GPU 자원 요청 필요 (우선순위 낮춤, 발표 이후).
9. Set A vs Set B로 교정한 Mantel 테스트(16.12/16.16 이월) — 여전히 미실행, 우선순위 낮음.
10. (선택) probe 선정 편향(17.14)을 교정한 버전 재시도 — 서브그룹별로도 분산을 균형있게 반영하는 probe 선정 기법 설계. 발표 이후 여유 있으면.
10. **(미검증, 발표 전 여유 있으면 우선순위 고려)** cost-accuracy tradeoff 지표(AUDC 등)를 lightweight-20 pool에서 한 번도 계산 안 함 — 17.21 참고. static-best(비용 무시, Qwen3-8B)가 raw accuracy로는 라우터를 이기는데, 비용까지 고려하면 어떤지는 미확인.

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

---

## 19. V1.3(LLM-judge) 시도 → probe-count ablation 진단 → domain+difficulty 이중 라우팅 설계 (2026-08-11)

### 19.1 용어 정리 (세션 중 혼동 발생, 코드로 확정)

- **Ceiling V1** = `build_ceiling_fp_lite20.py`, 528차원(22카테고리 × 카테고리당 24개, `PROBES_PER_DATASET=24`, pooled-variance 상위 선택), mean-center + L2-normalize.
- **Ceiling V2** = `build_ceiling_fp_lite20_categoryrate.py`, 22차원(카테고리당 Set A **전체** 평균), probe 선택 과정 자체가 없음.
- **"Pseudo Ceiling"**(3-benchmark 비교표에 쓰인 용어, `scripts/routerbench_pseudo_ceiling.py`/`scripts/mixinstruct_pseudo_ceiling.py`) = **Ceiling V1과 동일한 방법론**(카테고리별 고분산 probe 여러 개를 개별 차원으로 유지, 평균 안 냄)을 MixInstruct/RouterBench에도 적용한 버전. "카테고리당 1개만 뽑은 22차원 실험"은 코드베이스에 존재한 적 없음(세션 중 실제로 검색해서 확인) — 이전에 이 이름으로 발표했던 실험은 Ceiling V1/Pseudo Ceiling(528차원, 카테고리당 24개)이 맞았고, 단지 "22개 카테고리"라는 숫자가 "22차원"으로 기억에서 단순화된 것이었음.

### 19.2 V1.3 FP — Claude Sonnet 5 통합 judge 시도

**동기**: 22개 카테고리가 원래 서로 다른 채점 방식(exact-match, 코드 실행, arenahard 계열은 정체불명의 외부 judge)을 쓰고 있어 측정 방식이 카테고리 간 이질적(incommensurable) — 하나의 일관된 judge로 전부 다시 채점하면 이 문제를 해결하면서, 동시에 probe-selection bias(17번 섹션에서 발견한 41%p vs 실제 19%p 왜곡)도 균형 잡힌 표본 선택으로 같이 해결할 수 있을 것으로 기대.

**설계**: 22카테고리×3개 균형 선택(`v1_3_probe_selection.json`, 66 probe) → 정답 있는 18개 카테고리는 "prediction이 ground_truth와 동치인가" 루브릭, arenahard 계열 4개는 절대 품질 루브릭 → Claude Sonnet 5 Batch API로 1320콜(66×20) 채점 → 22차원 category-mean으로 집계, mean-center + L2-normalize.

**실행 이슈(재현용 기록)**:
- Batch API `custom_id`는 `^[a-zA-Z0-9_-]{1,64}$` 패턴만 허용 — 모델명에 마침표가 들어간 경우(`Llama-3.1-8B-Instruct` 등) 400 에러. `.replace(".", "-")`로 해결.
- 1320콜 중 10개는 max_tokens=1024가 빠듯해서 JSON이 중간에 끊김(parse error) — probe 행 평균으로 fallback 처리, 0.76%라 무시 가능한 수준.
- 코드 3개 카테고리(humaneval/mbpp/livecodebench)는 재채점하지 않고 기존 실행-기반 score를 1-10 스케일로 재조정해 대체(텍스트만 읽는 judge보다 실제 테스트 실행이 더 신뢰도 높다는 판단) — 이 로직을 배치 **제출 스크립트**엔 반영을 깜빡해서 1320콜 전체가 나갔음(계획은 1140콜) — 결과 수거 스크립트에서 사후에 correction 적용. 다음에 비슷한 배치 짤 때 제출 전 체크리스트에 넣을 것.
- **API 키를 채팅에 직접 붙여넣으면 안 됨**(Claude Code 안전 규칙, 사용자 동의해도 예외 없음) — 로컬 파일 경로(`C:\Users\user\anthropic_key.txt`)로 우회, 스크립트가 런타임에 파일을 읽어 환경변수로 설정(Claude가 파일 내용을 직접 열람하지 않음).

**결과**: V1.3 vs CeilingV2 직접 비교, delta=-0.0196, **p=0.0067**(V1.3이 유의하게 열세, 6/20만 개선). V1.3 vs uniform은 delta=-0.0016, p=0.857(사실상 무의미). V1.3 vs Perplexity는 delta=-0.0081, p=0.289(유의하지 않음, 비슷한 수준).

### 19.3 진단 — 무료 GT 기반 probe-count ablation (`scripts/llmrouterbench/probe_count_ablation.py`)

V1.3이 왜 안 됐는지(judge 품질 문제 vs 표본 크기 문제)를 판별하기 위해, **LLM judge 호출 없이** 기존 GT 점수를 "완벽한 judge의 채점"으로 간주하고 카테고리당 probe 개수(N)를 1~24로 스윕하며 같은 kNN 테스트 반복. `probe_info.json`(이미 카테고리 내 분산 내림차순 정렬됨)의 top-N만 사용.

| N/카테고리 | 전체 probe | delta | p-value |
|---|---|---|---|
| 1 | 22 | +0.0050 | 0.456 |
| 2 | 44 | +0.0104 | 0.107 |
| **3(V1.3과 동일)** | **66** | **+0.0106** | **0.086(미달)** |
| 6 | 132 | +0.0124 | **0.043(첫 유의)** |
| 8 | 176 | +0.0136 | 0.025 |
| 24(=Ceiling V1/Pseudo Ceiling) | 528 | +0.0138 | 0.0225 |

N=24 결과(+0.0138, p=0.0225)가 기존에 발표했던 Pseudo Ceiling 수치(+0.0138, p=0.022)와 정확히 일치 — ablation 스크립트가 검증된 방법론을 올바르게 재현함을 확인.

**결론**: N=3은 **완벽한 채점자(GT)를 써도** 유의성 미달 — V1.3의 실패는 judge 품질이 아니라 순수 표본 크기 문제였음이 확정됨. 유의성은 N=6~8부터 확보, N=12 이후는 체감수익 감소(거의 정체). **향후 LLM-judge FP를 다시 시도한다면 N=6~8/카테고리($8~10 선)가 합리적 타겟이지, N=24($31.6)까지 갈 필요는 없어 보임.**

### 19.4 스코프 결정 — judge 충실도 검증은 프로젝트 범위 밖

"LLM judge가 GT를 얼마나 잘 근사하는가"는 LLM-as-judge calibration/agreement 쪽에 이미 상당한 별도 문헌이 있는 인접 연구 주제로 판단 — 본 연구의 핵심("capability score를 어떻게 재분배·활용하는가")과 어긋나므로, V1.3/judge 확장 시도는 19.2~19.3 결과로 마무리하고 추가 투자하지 않기로 결정. 19.3의 ablation 자체는 GT만 쓴 거라 judge 품질과 무관하며, "capability descriptor 구축에 카테고리당 신호가 몇 개 필요한가"라는 본 연구 범위 안의 질문에 대한 유효한 결과로 남음.

### 19.5 최종 발표(2026-08-25 전후 예상)용 계획 — Domain+Difficulty 이중 라우팅 캐스케이드 프로토타입

**배경**: Ceiling V1/V2가 이미 작동함을 보인 것만으로는 최종 발표에서 "그래서 뭐가 되는가"가 약하다는 문제의식. PC1(일반 실력 축, 17번 섹션에서 발견)을 "제거해야 할 confound"가 아니라 "명시적 게이트로 쓰면 유용한 신호"로 재해석하는 Dual-Tier 아이디어(17번 섹션 후속 아이디어)를 실제로 작게 구현해서 캡스톤으로 제시하기로 함. 새 데이터/API 비용 없이 기존 자원만으로 구현 가능.

**설계 (사용자 확정, 2026-08-11)**:

1. **도메인 신호 (벡터, latent space 코사인 유사도, 학습 불필요)**:
   ```
   domain_score(쿼리, 모델) = cosine_sim( encoder(쿼리), domain_FP[모델] )
   ```
   기존 query encoder(MiniLM+projection, 이미 훈련됨) + Ceiling V2류 도메인 FP를 그대로 재사용. 별도의 "도메인 라벨" 분류 단계 없이, 모델 20개 각각에 대한 연속적 적합도 점수로 바로 계산.

2. **난이도 신호 (스칼라, 도메인과 orthogonal, 양자화 없이 연속값, 학습 불필요)**:
   ```
   difficulty(쿼리) = Σ w_i · domain_normalized_difficulty(probe_i),  w_i ∝ cosine_sim(encoder(쿼리), encoder(probe_i))
   ```
   도메인 신호와 **완전히 같은 kNN 가중평균 메커니즘**을 재사용 — 비교 대상만 "모델"에서 "이미 난이도를 아는 probe들"로 바뀜. `domain_normalized_difficulty`는 카테고리별 population 평균 정답률(pop_ease)을 카테고리 내에서 z-score 정규화한 값(16.x/17.x 섹션에서 발견한 probe-selection bias/subgroup distortion과 같은 이유로, 카테고리 간 난이도 베이스라인 차이가 신호에 안 섞이게 함).
   - MLP 기반 text-to-scalar 회귀도 검토했으나 기각: 학습 파이프라인이 추가로 필요하고(Colab GPU 필요, 튜닝/검증 부담), 2주 타임라인에 부담. kNN 방식은 기존 `knn_test_lite20.py`의 가중평균 로직을 거의 그대로 재사용 가능하고 로컬에서 즉시 실행 가능.
   - **사용자가 명시한 우려(2026-08-11)**: 이 kNN 난이도 추정의 정확도는 결국 "이웃 probe들의 난이도 라벨이 얼마나 정확한가"에 크게 좌우됨 — 구현 시 이 라벨 신뢰도를 어떻게 확보/검증할지가 미해결 과제로 남음.

### 19.7 domain_score용 query encoder 학습 -- 1차 트라이얼 결과 (2026-08-11, 같은 세션 후반)

**로컬 GPU 사용 가능으로 확인됨**(GTX 1650 Ti, 4GB VRAM, CUDA 12.5) -- 이전 세션들의 "학습은 Colab GPU로" 방침은 이 특정 노트북 한정으로는 완화됨. torch(cu121)+transformers 로컬 설치 완료.

**학습 목표 설계 (GRPO 스타일)**: 원래 CSCR류처럼 raw 성공/실패(0/1) 라벨로 contrastive 학습시키면 PC1(난이도/일반실력) 오염이 재발할 위험이 있어 채택 안 함. 대신 **쿼리별로 20개 모델 점수를 그 쿼리 안에서 mean-center + std로 정규화**(`(score - query_mean) / (query_std + eps)`, GRPO의 group-relative advantage와 동일 원리 -- 그룹 평균을 baseline으로 써서 별도 난이도 추정 없이 난이도 성분을 상쇄)한 값을 타겟으로 사용. 목표: `cosine_sim(encoder(쿼리), domain_FP[모델])`가 이 GRPO 타겟을 예측하도록 MSE로 projection head만 학습(MiniLM 백본은 얼림, domain_FP도 고정 앵커).

**스크립트**: `scripts/llmrouterbench/train_domain_encoder_trial.py`. `src/router/query_encoder.py`의 `QueryEncoder`를 import하지 않고 인라인 복사해서 씀(`router/__init__.py`가 faiss/matplotlib 등 무관한 무거운 의존성을 전부 끌고 옴 -- 다음에 정식 버전 만들 때는 이 import 체인부터 가볍게 정리할 것).

**1차 소규모 트라이얼**(3000 쿼리 서브셋, 3 epoch): holdout(450개 중 유효 399개) mean per-query Spearman rho = **0.2963**(std=0.254) -- 방향성 검증 목적 달성, 실제로 유의미한 양의 신호 확인.

**2차 확대 트라이얼**(Set A 전체 12,426 쿼리, train 10,563/holdout 1,863, 10 epoch 계획) epoch별 결과:

| epoch | train MSE | holdout rho |
|---|---|---|
| 1 | 0.7603 | 0.3253 |
| 2 | 0.7371 | **0.3338 (peak)** |
| 3 | 0.7203 | 0.3310 |
| 4 | 0.7049 | 0.3280 |
| 5 | 0.6904 | 0.3067 |

**epoch 2에서 정점 찍고 이후 하락 -- train MSE는 계속 감소하는데 holdout rho는 떨어지는 전형적 과적합 패턴 확인, epoch 5에서 사용자 판단으로 중단(프로세스 kill, PID 확인 후 PowerShell Stop-Process).** 스크립트가 맨 마지막에만 저장하는 구조라(중간 epoch 저장 안 함) 이번 실행은 **저장된 체크포인트가 없음** -- 중단 시점까지 아무 산출물도 안 남음.

**해석 참고치**: rho=0.33 수준은 절대적으로는 "약함~중간" 경계지만(교과서 기준), 이 프로젝트에서 나온 다른 descriptor-alignment 비교(v1.2: rho=0.32~0.43, Logit/Perplexity vs Capability: rho≈-0.25~0)와 비교하면 상당히 준수한 축에 속함. 다만 타겟 자체가 쿼리당 20개 raw 0/1 점수에서 나온 거라 원래 노이즈가 많아 상한선 자체가 1.0보다 훨씬 낮을 것으로 추정(정확한 상한선 미검증).

**다음 세션에서 반드시 고칠 것**:
1. **매 epoch마다(혹은 best-rho 갱신 시마다) 체크포인트 저장** -- 지금처럼 맨 끝에만 저장하면 이번처럼 조기 종료 시 전부 유실됨. best-epoch 저장 로직 추가 필수.
2. 위 결과 기준 **epoch 2 근처가 최적**으로 보임 -- 다음 실행은 처음부터 `N_EPOCHS=3~4` 정도로 줄이고, 매 epoch 저장하면서 best만 남기는 방식으로.
3. `router/__init__.py` 무거운 import 체인 문제 -- `QueryEncoder`를 별도 경량 모듈로 분리하거나, 인라인 버전을 정식 위치로 승격하는 것 검토.
4. 이 encoder가 실제로 준비되면, 19.5의 캐스케이드 라우터 프로토타입(domain_score 계산 부분)에 바로 연결.
   - (참고, 17.x번 섹션 관련) 모델 측 "난이도 내성" 스칼라는 easy/hard tertile 격차(slope)가 아니라 **그냥 raw 평균 레벨**을 쓸 것 — PCA 체크에서 slope-PC1과 generic capability의 상관이 0.915로 사실상 같은 축임을 확인했으므로(같은 세션, 미기록 상세는 대화 로그 참고), 별도 축으로 취급할 근거가 없음.

3. **캐스케이드 결정 로직**: 모델을 비용 오름차순 정렬 → domain_score + difficulty 조건을 둘 다 통과하는 **가장 싼 모델**을 순서대로 탐색해 선택 → 아무도 통과 못 하면 최강 모델로 escalate. 기존 kNN/MLP 라우팅(전 모델 비교 후 argmax/가중평균)과는 다른, 순차적 threshold 방식.

**다음 세션에서 할 일**: 위 설계를 실제로 구현 — (a) 도메인 코사인 유사도 계산 파이프라인, (b) 난이도 kNN 추정기, (c) 캐스케이드 결정 로직, (d) Set B로 "비용 절감 vs 정확도 손실" 정량 평가. 플래그십 확장(9/22 카테고리는 이미 무료로 존재, 나머지 13개는 멀티 프로바이더 API 비용 필요 — 별도 판단 보류 중)과 V1.3 스케일업은 이번 최종 발표 스코프에서 명시적으로 제외.

### 19.6 domain_score용 query encoder 재사용 여부 (확인 필요, 미해결)

domain_score(쿼리, 모델) = cosine_sim(encoder(쿼리), domain_FP[모델])의 `encoder`는 순수 임베딩이 아니라 학습된 projection head(`QueryEncoder.proj`, MiniLM 위에 얹은 것) — 이 학습 자체는 새로 설계할 필요 없이 기존 스크립트(`full_loo_lite20_categoryrate.py`의 `loo.train_fold`)를 그대로 재사용 가능함을 확인.

**단, 저장된 체크포인트는 전부 LOO fold용**(`train_fold(pool_19, ...)` — 매 fold마다 모델 1개씩 빼고 19개로 학습, 20개 fold가 각각 다른 헤드) — "20개 모델을 전부 아는 하나의 헤드"는 현재 없을 가능성이 높음. 캐스케이드 프로토타입은 unseen 시뮬레이션이 아니라 실제 20개 풀 전체를 다루는 거라, **held-out 없이 20개 전부로 딱 1번** 같은 스크립트로 학습 필요(20-fold 재학습 아님, 기존 프로젝트 관행상 이 정도 학습은 가볍다고 판단됨 — 13번 섹션 참고). 다음 세션에서 기존 체크포인트 디렉토리 확인 후, 없으면 이 1회 학습부터 진행할 것.

---

## 20. EmbedLLM 규모로 논문 Table 2와 직접 비교 + GRPO 회귀 방식 검증 (2026-08-14, 2026-08-15에 소급 정리)

**⚠️ 기록 경위**: 이 세션은 2026-08-14에 커밋(`b867531`, `28281eb`)까지 됐지만 PROGRESS.md 본문 서술이 누락된 채로 남아있었음 — 결과 JSON 파일과 커밋 메시지만 존재하고 서사가 없는 상태였다가, 2026-08-15 세션에서 다시 pull 받은 뒤 발견해서 뒤늦게 정리함. 코드/데이터는 이미 git에 다 있었으므로 손실된 건 없음, 문서화만 밀렸던 것.

### 20.1 배경 및 목표

19.5~19.7(도메인+난이도 캐스케이드 설계, GRPO 스타일 회귀)까지의 아이디어를 MixInstruct/LLMRouterBench 소규모 pool이 아니라 **EmbedLLM(112개 모델, 80개 카테고리)** 규모에서 검증하고, 동시에 **CSCR 논문이 실제로 보고한 "new LLMs" 프로토콜**(전체 pool을 2/3 seen(훈련 시 아는 모델) / 1/3 unseen(훈련 후 새로 추가되는 모델)로 나눠 AUDC를 재는 방식, 논문 Table 2 AUDC=0.4848, Peak=0.565)로 **처음으로 논문 수치와 직접 비교**하는 실험.

### 20.2 인프라 준비

- **빠른 평가 하네스**: `run_audc_eval.py`를 그대로 쓰면 seed당 15~20분 걸리는데(모델 재로드가 λ마다 반복되는 구조, 13번 섹션에서 이미 지적했던 비효율), batch-encode + 결과 caching으로 재구현해서 seed당 **~40초**로 단축. 공식 스크립트 대비 AUDC 오차 ±0.004 이내로 검증 완료(`scripts/embedllm_newllm_fast_eval.py`).
- **dtype 버그 발견+수정**: 논문 원본 `cost_spectrum_info_nce`(`scripts/train_query_encoder.py`)가 내부적으로 `~pos_mask`(비트단위 NOT)를 쓰는데, label을 float32로 넘기면 타입 에러/오동작 — EmbedLLM 규모에서 이 함수를 직접 재사용하려다 발견함. `label.bool()`로 넘기도록 수정(`scripts/embedllm_newllm_train_encoder_csinfonce.py:111-115`에 주석으로 기록). **결과에 영향 없음** — 버그가 있던 `neg_k` 변수는 그 함수 안에서 실제로 안 쓰이는 dead code라 순수 타입 호환성 수정임.
- `.gitignore`에 `local_checkpoints/` 추가(재생성 가능한 모델 체크포인트, ~2.3GB — 이전 세션(13번 섹션)에서 손으로 slim 버전만 골라 커밋했던 것과 별개로, 이 세션부터는 아예 전체를 git 추적 대상에서 제외).

### 20.3 kNN 검증(학습 없음) — Ceiling FP(PCA-5) vs uniform, EmbedLLM 규모

`scripts/build_embedllm_ceiling_fp.py` + 그룹 홀드아웃 kNN 테스트, 3 seed, unseen 22개 모델:

| seed | Ceiling FP rho | uniform rho | delta | p | 개선 |
|---|---|---|---|---|---|
| 0 | 0.664 | 0.643 | +0.0203 | 0.0174 | 12/22 |
| 1 | 0.704 | 0.682 | +0.0215 | 0.0040 | 14/22 |
| 2 | 0.639 | 0.612 | +0.0279 | 0.0026 | 18/22 |

**3-seed 전부 유의** — 이전 소규모 pool(11~20개)보다 훨씬 안정적인 유의성. 80차원 전체와 PCA-5(80차원을 5차원으로 압축) 결과가 거의 동일(delta 소수점 4자리까지 일치) — **PCA-5가 원본 신호를 거의 다 담고 있음을 확인**, 이후 실험은 전부 PCA-5 기준으로 진행.

### 20.4 Group LOO(contrastive, `multi_positive_info_nce`) — seen/unseen 격차 재확인

`scripts/embedllm_group_loo.py`, seen 90개로만 학습 후 unseen 22개 포함 전체 pool에서 라우팅:

| seed | overall hit rate | seen 선택 시 정답률 | unseen 선택 시 정답률 | unseen 선택 비율 |
|---|---|---|---|---|
| 0 | 56.0% | 75.1% | 32.7% | 45.2% |
| 1 | 64.8% | 68.9% | 21.4% | 8.6% |
| 2 | 63.0% | 64.3% | 47.9% | 7.9% |

**예상된 패턴**: seen 모델을 골랐을 때 정답률(64~75%)이 unseen을 골랐을 때(21~48%)보다 항상 높음 — 학습 때 못 본 모델에 대한 판단이 구조적으로 더 약함. seed마다 unseen 선택 비율 자체가 크게 요동(45%→9%→8%) — 이것도 13~17번 섹션에서 반복 확인된 collapse/시드 불안정성의 연장선.

### 20.5 GRPO 회귀 방식 — 논문 Table 2와 직접 비교, multi-seed (이 세션의 핵심 결과)

`scripts/embedllm_newllm_grpo_train.py` + `embedllm_newllm_grpo_multiseed.py`: 19.7에서 설계한 GRPO 스타일(쿼리별 mean-center+std 정규화 타겟, MSE 회귀, per-epoch holdout-rho 기준 best-checkpoint 선택, 10 epoch 중 최적점 자동 선택) 방식을 그대로 EmbedLLM 2/3-seen/1/3-unseen 프로토콜에 적용.

| seed | best epoch | holdout rho | AUDC | Peak | 논문(AUDC 0.4848) 대비 |
|---|---|---|---|---|---|
| 0 | 3 | 0.194 | **0.527** | 0.567 | **이김** (AUDC·Peak 둘 다) |
| 1 | 6 | 0.167 | 0.466 | 0.475 | 짐 |
| 2 | 8 | 0.157 | 0.449 | 0.466 | 짐 |
| 3 | 7 | 0.163 | 0.467 | 0.485 | 짐 |
| **4-seed 평균** | | | **0.477** | 0.498 | **4개 중 1개만 이김** |

**seed 4는 실행 중 사용자가 컴퓨터를 꺼야 해서 중단 — 미완료.**

**비교군**(contrastive, `cost_spectrum_info_nce` 2 epoch, 5 seed): mean AUDC=0.468, std=0.029, 5개 중 1개만 논문 이김. GRPO 회귀(0.477)가 contrastive(0.468)보다 근소하게 높지만 **이 차이 자체를 통계적으로 확정 지은 검정은 없음** — 방향만 참고할 것.

**해석**: seed 0만 보면 "우리 방법이 논문을 이겼다"는 인상이 강하지만(AUDC +8.6%, Peak도 근소 우위), **4-seed 평균으로 보면 논문과 거의 대등한 수준(근소 열세)으로 수렴** — 15~17번 섹션에서 여러 차례 반복된 "단일 시드 결과를 과대 해석하지 말 것" 교훈이 이번에도 정확히 재현됨. 다만 긍정적으로 보면: **이 프로젝트에서 나온 모든 실험 중 논문이 보고한 수치에 가장 근접하게 붙은 결과**(대부분의 다른 실험은 논문 주장을 반박하거나 재현이 안 됐던 것과 대비됨) — EmbedLLM처럼 실제 pool 규모·다양성이 충분히 크면, 우리 방법(GRPO 회귀, capability-aligned FP)이 논문 수준의 성능에 근접할 수 있다는 긍정적 신호로 해석 가능.

### 20.6 Ceiling FP probe 개수 ablation (N=6/12/24 probes/category) — 매우 불안정

`scripts/embedllm_newllm_probe_sampling.py`, 각 N당 5 seed:

| N | delta 부호/유의성 패턴(5 seed) |
|---|---|
| 6 | -0.112(ns) / +0.010(ns) / -0.064(ns) / **+0.020(p=0.025)** / -0.056(ns) |
| 12 | +0.035(p<0.001) / +0.094(p<0.001) / NaN(계산 실패) / +0.105(p<0.001) / **-0.073(p=1.0, 역전)** |
| 24 | **-0.102(p=1.0, 역전)** / +0.010(ns) / +0.109(p<0.001) / +0.037(p<0.001) / +0.043(p<0.001) |

**N을 늘려도 안정성이 확보되지 않음** — 부호 자체가 세 그룹 모두에서 시드에 따라 자주 뒤집힘(N=24조차 seed 0에서 역전). 19.3(LLMRouterBench 규모)에서 확인한 "probe 개수를 늘리면 유의성이 확보된다"는 패턴이 EmbedLLM 규모/이 특정 ablation 설계에서는 깨끗하게 재현되지 않음 — **원인 미분석, 다음 세션 과제.** (seed 2의 N=12 NaN은 계산 오류로 추정, 원인 미확인.)

### 20.7 Scale(모델 크기) 분석 — PC1=tier축 패턴, 훨씬 큰 표본으로 재확인

`scripts/embedllm_scale_analysis.py`: EmbedLLM은 순수 오픈웨이트만 있어서(GPT-4/Claude 같은 독점 flagship 없음) "브랜드" 아닌 순수 **모델 크기(파라미터 수)** 축을 봄. 112개 중 98개 크기 파싱 성공, 중앙값(8B) 기준 대형/소형 50:48 분할.

- **PC1(Ceiling FP 최상위 주성분)과 log(모델 크기)의 Spearman rho = -0.397, p=5.06e-5** — 17.6(LLMRouterBench, 33개 모델)에서 발견한 "PC1=전반적 실력/tier 축" 패턴이, 표본이 3배 이상 큰 EmbedLLM(112개)에서도 훨씬 강한 통계적 검정력으로 재확인됨.
- 대형/소형 그룹 간극에 대한 순열검정(permutation test) percentile = 0.998 — 이 크기-실력 축 분리가 우연이 아님을 뒷받침.
- PCA 차원 ablation(`pca_dim_ablation`): k=1(주성분 1개)만으로 이미 누적분산 87.8%, k=2로 91.4% — Ceiling FP의 정보 대부분이 극소수 축(사실상 PC1 하나)에 몰려있다는 17.6의 관찰이 이 규모에서도 그대로.

### 20.8 Selection diagnostic — 결론 없음

`scripts/embedllm_selection_diagnostic.py`: unseen 모델의 (선택 빈도) vs (선택됐을 때 정답률) 상관, rho=-0.238, p=0.57 — **통계적으로 무의미**, 뚜렷한 해석 없음. 다음 세션에서 우선순위 낮음.

### 20.9 종합 정리 + 다음 할 일

1. **긍정적 신호**: EmbedLLM 규모·GRPO 회귀 방식 조합이 이 프로젝트에서 논문 보고 수치에 가장 근접(4-seed 평균 AUDC 0.477 vs 논문 0.4848)했고, kNN 검증(20.3)은 3-seed 전부 깨끗하게 유의함 — capability-aligned FP 가설이 pool이 충분히 크고 다양하면 실제로 통한다는 이 프로젝트 전체의 결론(17.10)과 일관됨.
2. **여전한 경고**: GRPO 회귀도 시드 의존성에서 자유롭지 않음(seed 0만 논문을 이김, probe-count ablation은 부호 자체가 요동) — "단일 시드/단일 설정으로 결론 내지 말 것"이라는 교훈이 이 스케일에서도 유효.
3. **미완료 항목**:
   - GRPO seed 4 (중단됨, 재실행 필요)
   - Probe-count ablation의 불안정성 원인 분석 (20.6)
   - Group LOO 결과(20.4)를 발표에 어떻게 쓸지 미정
   - 19.5~19.7의 도메인+난이도 캐스케이드 라우터는 이 세션에서 진행 안 됨 — 여전히 다음 세션 과제로 남아있음(19번 섹션 참고).

---

## 21. Outlier-Drag 진단 및 해결 시도 (2026-08-15)

**배경**: 20번 섹션의 GRPO 회귀가 여전히 시드마다 편차가 컸음(seed0=0.529, 나머지 3개는 0.45~0.47대). 사용자가 원인으로 "한 쿼리에 정답 모델이 여럿일 때, 그 advantage-가중 평균 타겟이 outlier 모델 때문에 흔들려서 빈 공간에 착탄하는 게 아닐까"라는 가설(outlier-drag)을 제기 — 이번 섹션은 이 가설의 진단과 두 가지 해결 시도를 다룸.

### Observation (관찰)

#### 21.1 Outlier-drag 현상 직접 확인
`scripts/embedllm_outlier_blend_check.py` — GPU 없이 기존 FP·라벨 데이터만으로 검증. 학습 쿼리 5000개 샘플에서, "정답(양의 advantage) seen 모델들이 FP 공간에서 얼마나 흩어져 있는지(spread)"와 "그 advantage-가중 평균 타겟이 실제 존재하는 모델로부터 얼마나 먼가(dist_to_nearest_real)"의 관계를 측정.
- **Spearman rho=0.52, p≈0** — spread가 넓을수록 타겟이 빈 공간에 착탄. spread 넓은 절반이 좁은 절반보다 착탄거리가 **2.24배** 더 멂.
- 실제 사례: `deepseek-math-7b-instruct`(수학 특화 소형) + `falcon-40b-instruct`(범용 대형)처럼 성격이 다른 두 모델이 같은 쿼리에서 동시에 정답일 때, 블렌드 타겟이 실제 존재하는 어떤 모델로부터도 0.565만큼(전형적 모델간 거리 1.23의 거의 절반) 떨어진 빈 공간에 착탄.

#### 21.2 Contrastive(softmax) 방식이 구조적으로 덜 취약함을 확인 — 그러나 그게 해법은 아니었음
`scripts/embedllm_outlier_drag_loss_comparison.py` — 같은 5000개 쿼리에서, linear(MSE 회귀가 암묵적으로 향하는 방향) vs softmax(contrastive loss의 gradient가 수렴하는 fixed point, Weiszfeld류 반복으로 시뮬레이션) 두 집계 방식을 직접 비교.
- softmax 집계가 linear보다 실제 모델에 **3.05배 더 가깝게** 착지(0.089 vs 0.271), 83.4%의 쿼리에서 더 정확(Wilcoxon p≈0).
- 결정적으로 **rho(spread, 착탄거리)가 linear=+0.52 vs softmax=-0.11** — softmax는 positive set이 흩어져도 안 흔들림.
- **하지만 "그럼 contrastive로 갈아타면 된다"는 결론은 아니었음**: 이미 존재하던 `multi_positive_info_nce`(순수 contrastive, `scripts/embedllm_newllm_train_encoder.py`) 결과를 `newllm_curves.pkl`에서 직접 재계산해보니 AUDC=0.4649로, GRPO(seed0=0.529, 4-seed 평균 0.477)보다 오히려 낮았음. Contrastive는 outlier-drag엔 안 취약하지만, 이 프로젝트 전체에서 반복 확인된 "하나의 지배적 모드로 스냅해버리는 collapse"(13~17번 섹션)라는 별개의 병을 그대로 앓고 있어서, 두 효과가 상쇄된 것으로 추정 — 그래서 MSE 회귀 틀은 유지하면서 outlier-drag만 고치는 방향으로 전환.

#### 21.3 정답 모델이 여럿인 상황 자체가 매우 흔함
카테고리 필터 구현 중 확인(21.5): 학습 쿼리 27,940개 중 **94.3%(26,355개)**가 seen 모델 2개 초과가 동시에 정답이었음 — "여러 정답 중 outlier가 낀다"는 상황이 드문 예외가 아니라 거의 항상 존재하는 조건이었음.

### 해결 시도 (Fix attempts)

#### 21.4 Min-over-positives (`scripts/embedllm_newllm_grpo_train_minpos.py`)
MSE 회귀 틀은 유지하되, "여러 정답 모델의 평균에 맞추기"(AND) 대신 "그 중 지금 제일 가까운 하나만 맞으면 됨"(OR)으로 loss를 바꿈:
```
loss_pos = min_{m: t_m>0} (cos_sim(q,E_m) - t_m)²
loss_neg = mean_{m: t_m<=0} (cos_sim(q,E_m) - t_m)²   (오답 쪽은 불변 — "전부"로부터 멀어져야 하니 AND 그대로)
```
**사용자가 사전에 제기한 우려**("winner-take-all 방식은 학습 중 '제일 가까운 쪽'이 계속 바뀌면서 불안정해질 수 있다" — Multiple Choice Learning 문헌에 알려진 문제)와 달리, 실제로는 매우 매끄럽게 수렴함(epoch-to-epoch |Δholdout_rho| 평균 0.004, epoch 2~3에서 정점).

#### 21.5 카테고리 트랙레코드 필터링 (`scripts/embedllm_newllm_grpo_train_catfilter.py`, 사용자 아이디어)
같은 쿼리에서 여러 모델이 동시에 정답이어도, 그 쿼리의 **카테고리에서 역대 성적(Set A 전체 기준 raw 정답률, `train.csv`의 `category` 컬럼 이용)이 Top-2인 모델만** 정답 타겟으로 인정하고, 나머지("우연히 얻어걸린" outlier 후보)는 loss에서 아예 제외(오답으로 flip하지 않음 — 실제로 맞혔으니).
- **한계로 지적됨(사용자, 미해결)**: Top-K 고정 컷은 `[100%,95%,90%,89%,10%]` 같은 분포에서 90%·89%처럼 충분히 신뢰할 만한 모델까지 3등 밖이라는 이유로 잘라버릴 수 있음 — "1등 대비 상대적 격차(margin)"로 자르는 방식이 개선안으로 제안됐으나 아직 미구현.

#### 21.6 Load-balancing은 기각됨 (참고용, 이번 섹션 이전 시도)
`scripts/embedllm_newllm_grpo_train_balanced.py` + beta 스윕(0.05~1.0, `scripts/embedllm_newllm_grpo_beta_sweep.py`) — outlier-drag의 "증상"(라우팅 쏠림)만 억지로 펴는 방식으로 먼저 시도했으나, **어떤 beta 값에서도 AUDC와 선택-실제정확도 상관(rho) 둘 다 악화**됨(beta=0.05만으로도 rho 유의성 상실, p=0.012→0.25). 원인(타겟 구성 자체의 결함)을 안 고치고 결과(쏠림)만 펴면 약한 신호마저 흐트러진다는 것을 확인 — 21.1의 근본 원인 진단으로 이어진 계기.

#### 21.7 Multi-seed 최종 비교
`scripts/embedllm_newllm_grpo_variant_multiseed.py`, seed 0-3, EmbedLLM "new LLMs" 프로토콜(2/3 seen/1/3 unseen), CSCR 논문 Table 2 AUDC=0.4848 기준:

| variant | seed0 | seed1 | seed2 | seed3 | 평균 | CSCR 이김 |
|---|---|---|---|---|---|---|
| GRPO 원본(20번 섹션) | 0.529 | 0.466 | 0.449 | 0.467 | 0.478 | 1/4 |
| **min-pos** | 0.510 | 0.512 | 0.502 | 0.530 | **0.513** | **4/4** |
| catfilter(Top-2) | 0.525 | 0.505 | 0.460 | 0.547 | 0.509 | 3/4 |

**결론**: 두 수정 다 원본 대비 확실한 개선(+6.5~7.4%)이고, **min-pos가 성능·안정성 둘 다 우세**(4/4 전원 CSCR 이김, 스프레드 0.028로 원본의 1/3 이하). Outlier-drag 가설과 그 해결 방향이 실증적으로 검증됨. Category-filter는 평균은 비슷하나 seed2에서 CSCR한테 짐 — Top-K 고정 컷의 한계(21.5)와 관련 있을 가능성. (catfilter-seed3의 bootstrap 유의성 검정에서 NaN 발생, QNC도 1.92로 유독 높았음 — AUDC 값 자체는 정상 계산됐으나 그 시드의 비용 분포가 특이했을 가능성, 원인 미조사.)

**다음 검증 후보**: catfilter를 margin 기반으로 개선, min-pos + category-filter 결합, all-seen(21.8) 결과와 종합.
4. **문서화 프로세스 교훈**: 이번처럼 커밋은 됐는데 PROGRESS.md 서술이 누락되는 일이 재발하지 않도록, **작업 세션을 마칠 때 커밋 메시지에 적은 내용은 반드시 PROGRESS.md에도 같은 세션 안에서 옮겨 적을 것.**

---

## 22. Cost 버그 수정, Catfilter 방법론 확정, Combined GRPO 확립, FP 유형별 일반화 검증 (2026-08-15)

**배경**: 21번 섹션 이후 "치팅 없는지 점검해보자"는 요청으로 파이프라인 전체를 감사하다가 진짜 버그를 발견했고, 그걸 고친 뒤 catfilter의 컷오프 방법론을 확정하고, min-pos+catfilter를 결합한 Combined GRPO를 멀티시드로 검증하고, 마지막으로 이 개선이 FP 종류에 무관하게 통하는지(Ceiling V1/Perplexity FP) 테스트한 하루.

### 22.1 Cost 계산 버그 발견 및 수정

- `src/router/cost_models.py`의 `get_param_count()`가 `int()` 캐스팅을 써서 n_params<1인 모델(예: Qwen1.5-0.5B-Chat)의 비용이 전부 0으로 계산되던 버그. `float()`로 수정.
- `experts/registry.json`에 모델 3개가 아예 없었음: `microsoft__phi-1_5`, `cloudyu__Mixtral_11Bx2_MoE_19B`(HF API로 n_params 확인 후 추가, 151개로 증가), `JaeyeonKang__CCK_Asura_v1`(HF API가 `401 Repository Not Found` 반환 — 삭제된 phantom repo, 등록 대신 전 실험에서 영구 제외).
- 이 버그로 all-seen AUDC가 약 4% 부풀려져 있었음(min-pos all-seen: 버그 상태 0.5824 → 수정 후 0.5587, 3/3 CSCR 이김 결론 자체는 유지).
- 별개로 `run_audc_eval.py`의 `paired_bootstrap_audc_cached()`가 정렬 안 된 cost/accuracy 배열을 받아 `delta=+nan`이 가끔 나오는 기존 버그도 발견(AUDC/QNC/Peak 자체는 별도의 정렬된 경로라 영향 없음) — RouterBench 스크립트들에서 `np.argsort` 정렬을 추가해 해결.

### 22.2 Margin catfilter + Combined 첫 시도 (seed0 단발 확인)

`scripts/embedllm_margin_combined_seed0.py` — margin=0.15 기반 catfilter(1등 대비 격차가 margin 이내인 정답만 유지, Top-K 고정 컷의 "89%도 3등이면 잘림" 문제를 해결하려는 시도)와, 그 마스크를 min-pos 로스에 먹인 combined를 seed0만 빠르게 확인:

| 프로토콜 | margin-catfilter | combined(min-pos+margin-catfilter) |
|---|---|---|
| unseen (CSCR 0.4848) | 0.5307 | 0.5233 |
| all-seen (CSCR 0.541) | 0.5381 (짐) | **0.5663** (당시까지 all-seen 최고) |

combined가 seed0만으로도 유망해 보여 이후 멀티시드로 확장(22.5).

### 22.3 RouterBench 일반화 확인 (vanilla/min-pos/catfilter-top2, all-seen, 3시드)

`scripts/routerbench_grpo_variants_multiseed.py`, CSCR 논문 RouterBench AUDC=0.711 기준:

| variant | seed0 | seed1 | seed2 | 평균 | std | CSCR 이김 |
|---|---|---|---|---|---|---|
| vanilla-grpo | 0.7420 | 0.7359 | 0.7391 | 0.7390 | 0.0025 | 3/3 |
| min-pos | 0.7196 | 0.7214 | 0.7209 | 0.7206 | 0.0007 | 3/3 |
| catfilter-top2 | 0.7397 | 0.7399 | 0.7402 | 0.7400 | 0.0002 | 3/3 |

셋 다 CSCR은 이기지만, **순위가 EmbedLLM과 정반대**(RouterBench에서는 min-pos가 제일 약함) — 모델 풀이 작으면(11개) min-pos의 "가장 가까운 정답 하나만 맞추면 됨" 방식이 오히려 손해라는 신호로 해석.

### 22.4 Catfilter 컷오프 방법론 확정: Top-50%

사용자의 원래 아이디어는 "상위 50% 유지"(percentile)였는데, 실제 구현은 Top-2(고정 개수)와 margin(임계값)이 먼저 나왔던 상태 — 3가지를 전부 하나의 `compute_keep_idx(pos_idx, scores, mode)` 함수로 통일 구현해(모드만 다르게) 오류 위험을 없애고, EmbedLLM all-seen에서 3시드로 단독 비교(`scripts/embedllm_catfilter_multiseed_costfixed.py`):

| 방법론 | seed0 | seed1 | seed2 | 평균 | std | CSCR(0.541) 이김 |
|---|---|---|---|---|---|---|
| top2 | 0.5430 | 0.5310 | 0.5297 | 0.5346 | 0.0060 | 1/3 |
| **top50%** | 0.5432 | 0.5392 | 0.5339 | **0.5388** | **0.0038** | 1/3 |
| margin | 0.5381 | 0.5323 | 0.5302 | 0.5335 | 0.0033 | 0/3 |

**top50%가 평균 최고 + margin과 안정성 거의 동급 → 이후 모든 combined 실험의 catfilter는 top50%로 확정.** (재밌게도 top50%는 사용자의 원래 직관이었고, top2/margin은 이후에 추가된 변형이었음.)

### 22.5 Combined(min-pos + top50%-catfilter) 확립 — EmbedLLM 11시드

`scripts/embedllm_combined_multiseed.py`(seed 0-2) + `scripts/embedllm_generate_splits_5to10.py`(seed 5-10 신규 split 생성, 기존 `build_split()` 로직 그대로 재사용해 시드 1-4와 방법론적으로 동일) + SEEDS 확장 재실행(seed 3-10)으로 총 11시드(0~10) 확보:

| 프로토콜 | 11시드 평균 | std | CSCR 이김 |
|---|---|---|---|
| unseen (CSCR 0.4848) | **0.5175** | 0.0485 | **11/11** |
| all-seen (CSCR 0.541) | **0.5581** | **0.0036** | **11/11** |

unseen이 마진이 더 크고(+6.8%) all-seen은 마진이 얇지만(+3.2%) 11시드 전원 안 짐 — "unseen(본선)에서 확실히 이기고 all-seen(곁다리)에서도 안 진다"는 프레이밍으로 정리. (CSCR 자체가 FP 데이터셋을 비공개해서 정확한 재현이 불가능하다는 점, 그리고 우리 자체 cost 버그 하나가 4% 가까이 결과를 흔들었던 전례를 감안하면, "CSCR을 이겼다"는 주장은 통계적으로 방어하기 약하고 — 파이프라인 내부의 vanilla-GRPO 대비 개선이 훨씬 rigorous한 주장이라는 점을 사용자와 합의함.)

### 22.6 Ceiling FP 구축 방식의 스코프 한계 (검토 후 결론: 문제 아님)

Ceiling FP는 "unseen" 모델이라도 그 모델의 FP 좌표 자체는 Set A(train.csv) 실제 정확도로 만들어짐(`build_embedllm_ceiling_fp.py`) — 즉 unseen 프로토콜은 "쿼리-라벨 페어를 안 보여준 것"이지 "성능 이력이 아예 없는 신규 모델 라우팅"을 테스트하는 게 아님. 발표 시 "unseen"이라는 단어의 스코프를 명확히 할 필요.

### 22.7 Ceiling V1(probe-sampled) 에서도 Combined 유지되는지 검증

배경: `scripts/embedllm_newllm_probe_sampling.py`로 예전에 만든 vanilla GRPO 결과를 보면, 오라클급 정보량인 Ceiling V2(카테고리당 중앙값 177개, 최대 3,454개 프롬프트 평균)와 달리, Ceiling V1(카테고리당 top-variance 24개 프롬프트만 사용, 최대 **140배** 적은 정보량)에서는 vanilla GRPO가 N=24에서도 5시드 평균 0.4421로 **0/5 (CSCR 못 이김)**, N=6에서는 random보다도 못한 시드까지 있었음.

`scripts/embedllm_probeN24_combined_multiseed.py`(unseen, 3시드, N=24)로 combined를 이 조건에서 테스트:

| | seed0 | seed1 | seed2 | 평균 | std | CSCR 이김 |
|---|---|---|---|---|---|---|
| vanilla GRPO (N=24, 5시드) | 0.4161 | 0.4701 | 0.4507 | 0.4421(5시드) | - | 0/5 |
| **combined (N=24, 3시드)** | 0.5185 | 0.5224 | 0.5103 | **0.5171** | 0.0050 | **3/3** |

같은 3시드로 비교한 V2 combined 평균(0.5163, seed0-2)과 **차이가 0.0008**로 사실상 동일 — FP의 정보량이 7~140배 줄어도 combined의 효과는 거의 손실되지 않음.

### 22.8 FP 유형(Capability vs Perplexity) 의존성 검증 — RouterBench

질문: Combined GRPO가 FP 종류와 무관하게 통하는 범용 로스 기법인가, 아니면 capability-encoded FP와의 시너지인가? EmbedLLM은 모델의 실제 응답 텍스트(response text)를 공개하지 않아 Perplexity FP를 만들 수 없음(라벨만 있음, `build_routerbench_perplexity_fp.py` 방식 적용 불가) — RouterBench는 `{model}|model_response` 컬럼이 있어 기존에 구축된 32차원 GPT-2 cross-entropy 기반 Perplexity FP(`local_descriptors/routerbench-perplexity/`, 정확도 정보 전혀 없음)가 존재.

`scripts/routerbench_perplexity_combined.py`(all-seen, 3시드)로 같은 데이터/같은 combined 로스, FP만 교체해 비교:

| FP | seed0 | seed1 | seed2 | 평균 | std | CSCR(0.711) 이김 |
|---|---|---|---|---|---|---|
| Ceiling(capability) | 0.7216 | 0.7238 | 0.7223 | **0.7226** | 0.0009 | **3/3** |
| Perplexity(GPT-2 CE) | 0.6916 | 0.7003 | 0.6888 | 0.6936 | 0.0049 | **0/3** |

Perplexity 쪽도 std가 크게 넓지 않아(0.0049) 노이즈로 보기 어려움 — **체계적인 차이**. 결론: Combined GRPO는 범용이 아니라 **capability-encoded FP와 시너지가 있는 방법**. Perplexity FP는 random보다는 유의미하게 낫지만(bootstrap delta 전부 p<0.001) CSCR급까지는 못 올라감.

**RouterBench 해석 시 주의**: Oracle(11개 모델 중 프롬프트별 최고 성능, 비용 무시) 정확도 0.9636, 정적으로 GPT-4만 계속 쓸 때 0.8418, 우리 라우터의 최고 지점(Peak)은 0.776~0.778로 **정적 GPT-4 단독보다도 낮음**. AUDC(0.72)는 오라클과 비교 대상이 아니라 cost-lambda 스윕 전체의 적분 평균이라 이 격차 자체는 정상이지만, RouterBench는 모델 풀이 작고(11개) GPT-4가 압도적으로 강해서 라우팅으로 얻을 수 있는 이득 자체가 원래 작은 벤치마크라는 점은 발표 시 짚어야 함.

### 22.9 종합 결론 (2026-08-15 세션 기준, 사용자 정리 및 검증)

1. Ceiling V2(이상적 capability FP)에 CSCR류 baseline(contrastive InfoNCE)만 쓰면 CSCR을 못 이김.
2. min-pos·category-filter(top50%)로 Combined GRPO 형태를 갖추면 CSCR을 소폭 상회(unseen +6.8%, all-seen +3.2%, 11시드 전원).
3. unseen에서 효과가 가장 크고, all-seen에서도 지지 않음.
4. FP가 capability(정확도) 정보를 담고 있는 것이 핵심 — Perplexity FP(비-capability 신호)에서는 CSCR을 못 넘김(RouterBench, 0/3).
5. Ceiling V1처럼 프롬프트를 크게 줄여 샘플링해도(최대 140배 적은 정보량) 효과는 V2 대비 거의 떨어지지 않음(차이 0.0008).

**다음 방향(논의만, 미착수)**: (a) 적응적 trim(soft-min, temperature를 학습해 min-pos/catfilter를 쿼리마다 얼마나 강하게 걸지 모델이 직접 정하게 하는 방향), (b) 추론 시점에 top-K cosine similarity margin을 불확실성 신호로 써서 outlier-drag 상황을 라우팅 결정에도 반영하는 방향(현재는 min-pos/catfilter가 학습 시그널에만 관여하고 추론 코드는 vanilla와 완전히 동일), (c) 3개 시드 체크포인트 앙상블(거의 공짜, 미시도).

**미커밋 상태**: `experts/registry.json`, `src/router/cost_models.py`(cost 버그 수정), 이번 섹션에서 언급된 모든 신규 스크립트와 결과 JSON — 다음 커밋에서 정리 필요.

### 22.10 발표용 정리 중 발견: catfilter 컷 pct=0.3이 top50%보다 확실히 우세 (같은 날 후속)

22.9의 결론 5개를 논문 형식(`RESULTS_SUMMARY.md`)으로 재정리하던 중, ablation 표를 보고 "catfilter가 min-pos에 얹었을 때 거의 기여가 없어 보인다"는 의문이 제기됨 — all-seen에서 combined(0.5581)이 min-pos 단독(0.5587)과 거의 동일했기 때문. 원인 가설: top50% 컷이 너무 관대해서, min-pos가 어차피 고르지 않았을 후보만 잘라내는 경우가 대부분이라 catfilter가 실질적으로 개입할 여지가 적었을 것.

**percentile을 10/20/30/50%로 스윕(seed0)** — RouterBench(11개 모델)는 정답 후보가 2~3개인 쿼리가 대부분이라 `ceil(n*0.1)=ceil(n*0.2)=ceil(n*0.3)=1`로 컷들이 서로 구별이 안 돼 무효 판정, EmbedLLM(111개 모델)으로 옮겨서 재시도:

| pct | AUDC(seed0, all-seen) |
|---|---|
| 0.1 | 0.5510 |
| 0.2 | 0.5511 |
| **0.3** | **0.5659** |
| 0.5(=기존 확정치) | 0.5574 |

0.3에서 뚜렷한 비단조 정점 발견 → 3시드로 확정(mean=**0.5652**, std=**0.0005**, 3/3 CSCR 승) — min-pos 단독(0.5587)보다 +0.0065, 기존 top50%-combined(0.5581)보다 +0.0071. std도 지금까지 나온 all-seen 결과 중 제일 좁음. **EmbedLLM all-seen의 새로운 확정 combined 설정은 pct=0.3.**

**unseen에서는 재현 안 됨**: 같은 pct=0.3을 unseen(학습에 쓰는 seen 모델 74개)에서 3시드 재검증하니 평균 0.5162(std=0.0132) — 기존 top50%의 0.5163과 사실상 동일. **catfilter 컷 민감도 자체가 모델 풀 크기에 비례**하는 것으로 보임: RouterBench(11개, 무감) < unseen 학습 풀(74개, 무감) < all-seen(111개, 뚜렷) — §17.10 이래 이 프로젝트 전체에 반복된 "모델 풀이 커야 라우팅 개입의 여지가 생긴다"는 결론과 같은 방향.

관련 신규 스크립트: `routerbench_catfilter_pct_sweep.py`(무효 판정, 참고용), `embedllm_catfilter_pct_sweep.py`(all-seen seed0 스윕), `embedllm_pct30_combined_multiseed.py`(all-seen 3시드 확정), `embedllm_pct30_unseen_multiseed.py`(unseen 3시드 재검증). `RESULTS_SUMMARY.md`의 §1.2/§2.3/§2.5에 전체 정리 반영됨.

### 22.11 Ablation 표 완결 (같은 날 후속) — vanilla all-seen과 catfilter 단독(pct=0.3) 채움

`RESULTS_SUMMARY.md` §2.2/§2.3을 "제대로" 정리하기 위해, 남아있던 3개 gap을 `embedllm_ablation_gapfill.py`(3시드씩, 총 9 run)로 채움:

- **all-seen vanilla GRPO(신규)**: 0.5056/0.4664/0.4685, 평균 **0.4802, 0/3** — unseen에서 1/4만 이겼던 것과 같은 방향으로, GRPO 원본은 all-seen에서도 CSCR을 하나도 못 이김.
- **all-seen catfilter(pct=0.3) 단독(mean 집계, 신규)**: 0.5351/0.5386/0.5355, 평균 **0.5364, 0/3** — 기존 top50% 단독(0.5388, 1/3)보다 오히려 낮음.
- **unseen catfilter(pct=0.3) 단독(신규)**: 0.5224/0.5078/0.4808, 평균 **0.5037, 2/3** — 기존 Top-2 단독(0.5092, 3/4)과 비슷한 수준.

**핵심 발견**: catfilter 단독으로는 top50%가 pct=0.3보다 낫지만(0.5388 > 0.5364), min-pos와 결합하면 정반대로 pct=0.3이 top50%보다 확실히 나음(0.5652 > 0.5581, §22.10). **"catfilter에 최적인 컷"과 "min-pos와 결합했을 때 최적인 컷"이 다르다**는 비대칭 — 처음 combined(top50%)가 min-pos 단독과 거의 같아서 "catfilter가 기여 없다"고 봤던 게, 사실은 컷 비율과 집계 방식(mean vs min) 사이의 상호작용을 놓친 결과였음. 메커니즘은 아직 가설 수준("min-pos가 이미 outlier를 어느 정도 거르므로, catfilter는 min-pos가 놓치는 부분만 노려 더 공격적으로 잘라야 시너지")이고 정식 검증은 안 됨 — `RESULTS_SUMMARY.md` 데이터 공백 7번 참고.

이걸로 §2.2(unseen)/§2.3(all-seen) ablation 표가 vanilla/min-pos단독/catfilter단독/combined 4-way로 완결됨(§2.4 RouterBench와 동일한 형태). `RESULTS_SUMMARY.md` 전체 반영됨.

### 22.12 "한 스텝 더" 탐색 — Inference/FP 개선 시도 두 건 (같은 날 후속)

**시도 1 (기각): 쿼리→모델 상호 견인 (`embedllm_fp_refine_seed0.py`)** — 학습 중 매 배치마다 catfilter 통과 정답 모델의 FP를 현재 쿼리 착탄점 쪽으로(거리 반비례 감쇠 스텝, alpha=0.02) 살짝 당기는 방식. unseen seed0에서 AUDC=0.4310으로 combined(pct=0.3) 기준(0.5264) 대비 **-18% 확실한 실패**. 원인: "아주 살짝"을 의도했지만 10 epoch × 수백 배치 누적으로 net drift가 평균 1.21/최대 1.95(최대 가능 2.0)까지 폭주 — seen 모델이 원래 오라클 위치에서 사실상 이탈해 unseen(고정) 모델과의 좌표계 정합성이 깨짐. holdout_rho도 epoch3 이후 요동치며 하락.

**시도 2 (효과는 있으나 미미, 최종안 미포함): capability 기반 FP 스무딩 (`embedllm_fp_smooth_seed0.py`, `embedllm_fp_smooth_beta_sweep.py`)** — 학습 시작 "전에 딱 한 번", 결정론적으로 카테고리별 정확도 프로필이 비슷한 **seen 모델끼리만**(unseen은 이웃이 될 수 없음 — `build_embedllm_ceiling_fp.py`의 기존 "unseen-unseen 리크 방지" 원칙 그대로 적용) K=5개 평균 쪽으로 FP를 blend(`(1-β)·원본 + β·이웃평균`). 시도1과 달리 seen/unseen 전부 동일 규칙으로 처리되고, 학습 신호와 무관한 1회성 전처리라 drift가 통제됨(β=0.15: 평균 0.0344/최대 0.1493, 최대 가능 2.0 대비).

β 스윕(K=5 고정, seed0, unseen): 0.15→0.5305, 0.25→0.5327(최고), 0.35→0.5311, 0.5→0.5323 — 전부 무보정 기준(0.5264) 대비 **+0.4~1.2%의 작은 개선**, β=0.25 근처에서 빠르게 정체(비단조, 더 키워도 안 좋아짐). 단일 시드라 노이즈 가능성을 배제 못하고, 설령 진짜여도 pct=0.3 발견(+4.5~6.8%)에 비하면 규모가 작아 **추가 복잡도(K/β 하이퍼파라미터, unseen 처리 로직) 대비 이득이 낮다고 판단, 최종 확정 방법론(combined pct=0.3)에는 포함하지 않기로 함.**

**다음 방향 재확인**: 오늘 나온 패턴(작은 손잡이 조정은 효과 미미, 로스/타겟 구조 자체를 바꾼 것만 큰 폭 개선)에 따라, 다음으로 시도할 "과감한" 후보는 여전히 §22.9에서 언급한 **적응적 trim(soft-min, learnable temperature)** — 하드코딩된 min-pos/percentile-catfilter 규칙 자체를 학습 가능하게 만드는 구조적 확장. 미분 가능한 트리밍 메커니즘 설계가 필요해 다음 세션 과제로 남김.

### 22.13 앙상블(정직하게 기각) + Ceiling V1의 all-seen 재검증 (같은 날 후속)

**앙상블 시도**: 이미 학습된 pct=0.3 combined 체크포인트들의 similarity 점수를 평균(재학습 없음). All-seen(3체크포인트, 111개 모델 — 시드마다 모델 풀이 동일해 앙상블에 구조적 문제 없음): AUDC=**0.5713**, 최고 단일 시드(0.5659)보다도 높음 — 진짜 개선. 그러나 **"이번 방법론에서 편법 같다"는 사용자 판단으로 최종안에서 제외** — 메커니즘 설명이 없는 범용 분산 감소 트릭이라 min-pos/catfilter처럼 이 프로젝트의 "방법론"이라 부르기 어렵고, 성과 귀인이 흐려진다는 이유. (참고로 unseen 앙상블은 애초에 시도했다가 **방법론적 결함**을 발견 — 시드마다 seen/unseen 모델 구성이 달라서, 다른 시드의 unseen 폴더로 여러 체크포인트를 평가하면 seen-모델 정보 누수가 생김(seed0 unseen 35개 중 23개가 seed1의 seen 세트에 포함). all-seen에는 이 문제가 없음. 이 버그는 앙상블 스크립트에만 있었고 다른 모든 기존 결과는 각 시드가 자기 자신의 split만 쓰도록 정확히 구현돼있음을 재확인함.)

**Ceiling V1(N=24)의 all-seen 재검증**: unseen에서는 V1이 V2와 사실상 동일했지만(§22.7), all-seen은 처음 시도 — probeN24 FP 디렉토리가 레지스트리 수정 이전에 만들어져 2개 모델(phi-1_5, Mixtral)이 빠져있어 109개 모델(V2 111개와의 교집합)로 3시드 진행:

| | seed0 | seed1 | seed2 | 평균 | std | CSCR(0.541) 승 |
|---|---|---|---|---|---|---|
| Ceiling V1 all-seen combined(pct=0.3) | 0.5548 | 0.5450 | 0.5447 | **0.5481** | 0.0047 | 3/3 |

CSCR은 여전히 이기지만(+1.3%) V2(0.5652, 111개 모델 기준)보다 **-3.0%, 세 시드 모두 일관되게 낮음**(격차가 -0.0111→-0.0198→-0.0212로 점점 벌어지는 추세). **unseen과 달리 all-seen에서는 저비용 FP(V1)의 성능 손실이 실재함** — 후보 풀이 클수록(all-seen 109개 vs unseen 평가시 35~37개) probe 샘플링 노이즈가 랭킹에 영향을 줄 기회가 늘고, 학습 타겟 차원도 커져서(74→109) 노이즈가 누적되는 것으로 해석. §22.10/22.11에서 반복 확인된 "모델 풀 크기가 개입의 여지를 결정한다"는 결론의 반대쪽 얼굴(이득뿐 아니라 손실도 커짐).

**결론 정정**: "V1은 V2와 완전히 동등하다" → "**V1은 unseen(주 타겟)에서는 손실 없이 동등하고, all-seen(부차 타겟)에서는 모델 풀이 클수록 커지는 작지만 일관된 대가가 있다.**" 논문 서사의 핵심(Combined GRPO가 CSCR을 이긴다)은 전혀 손상되지 않음 — V1은 애초에 §4의 보너스 질문(비용 절감)이었고, 그 답이 "완전 공짜"에서 "주 타겟은 공짜, 부차 타겟은 조건부"로 더 정교해진 것뿐.

**다음 탐색 후보(미착수, 다음 세션)**: probe 선택 휴리스틱 개선(현재는 카테고리별 분산 top-N만 봄, 프롬프트 간 다양성/중복 미고려) 또는 all-seen처럼 풀이 큰 상황에 한해 N을 늘리는 이원화 전략.

관련 신규 스크립트: `embedllm_ensemble_eval.py`(unseen, 결함 발견용 참고 자료로 남김), `embedllm_allseen_ensemble_eval.py`(all-seen, 유효), `embedllm_probeN24_allseen_pct30_multiseed.py`(Ceiling V1 all-seen). `RESULTS_SUMMARY.md` §4 반영됨.

### 22.14 Probe 재분배(PCA loading 가중) 시도 — unseen은 유효, all-seen은 명확히 실패 (다음날)

카테고리별 균등 24개 대신, V2 전체 데이터로 계산한 PCA loading(5개 주성분 기여도)에 비례해 probe를 차등 배분(sqrt(importance) 스케일, 바닥값 6개, 총 1,200개 — 균등 1,920개 대비 37.5% 절감) 시도.

먼저 카테고리 중요도 자체를 계산해보니(`embedllm_pca_loading_analysis.py`) 예상보다 덜 집중돼있었음 — PC1이 87.8%를 차지하는데도, 상위 20/80 카테고리가 중요도의 50%만 차지(90% 담으려면 45/80 필요). "소수만 남겨도 된다"는 극단적 가설은 기각.

3시드로 unseen/all-seen 둘 다 검증:
- **unseen**: 0.5406/0.5263/0.5044, 평균 0.5238(std=0.0149) — V1 균등(0.5171)·V2(~0.517)와 대등, 3/3 CSCR 승. 37.5% 적은 데이터로 손실 없음.
- **all-seen**: 0.5397/0.5378/0.5403, 평균 **0.5393(std=0.0011, 매우 일관됨)** — V1 균등(0.5481)보다도 낮고 V2(0.5652) 대비 -4.6%, **0/3 CSCR 패배**.

**원인 추정**: 중요도를 "풀 전체의 집합적 분산 설명력"으로 계산했는데, all-seen처럼 109개 모델을 전부 구별해야 하는 상황에서는 전체 분산엔 기여 적어도 특정 모델 쌍 구별엔 결정적인 카테고리가 있을 수 있고, 그런 카테고리가 바닥값(6개)으로 밀려나면서 랭킹 품질이 떨어진 것으로 보임 — §22.13의 "모델 풀이 클수록 저비용 FP의 노이즈 민감도도 커진다"는 진단과 같은 방향, 더 강하게 재확인됨.

**검토된 대안(미착수)**: 바닥값 상향(6→12~15), 균등 베이스+가중 보너스 하이브리드, 재분배를 unseen 전용으로 스코프 한정, 중요도 기준을 "전체 분산 설명력" 대신 "가장 비슷한 모델 쌍 구별력"으로 재정의.

관련 신규 스크립트: `embedllm_pca_loading_analysis.py`, `embedllm_build_pca_weighted_probe_fp.py`, `embedllm_pcaweighted_combined_multiseed.py`(unseen), `embedllm_pcaweighted_allseen_pct30_multiseed.py`(all-seen). `RESULTS_SUMMARY.md` §4.1 반영됨.

### 22.15 바닥값 상향(6→15)으로 재분배 성공 + 대안 2개 기각 (같은 날 후속)

바닥값을 6→15로 올려(총 1,800개, 균등 1,920개 대비 6.2% 절감) 재검증:

| | probe 수 | unseen(3시드) | all-seen(5시드, seed0-4) |
|---|---|---|---|
| V1(균등) | 1,920 | 0.5171 | 0.5481 |
| **V1.5(가중, 바닥15)** | **1,800** | **0.5121** | **0.5561** |

**V1.5가 V1보다 probe를 더 적게 쓰면서도 all-seen에서 확실히 앞섬**(0.5561 > 0.5481, 5/5 CSCR 승, V2 대비 -1.6%로 격차도 절반 이하로 줄어듦). 바닥값 6(§22.14)에서는 all-seen 0/3으로 완패했는데 15에서는 성공 — "전체 분산 기여는 적어도 특정 모델 쌍 구별에 필요한 카테고리"의 최소 안전마진이 핵심이었다는 가설이 실증적으로 확인됨.

같은 scale에서 바닥값만 분리해서 올린 순수 실험(상위 카테고리 배분 1,200-probe 버전과 완전 동일, 총 1,442개)도 시도했으나, seed1에서 V2 대비 -4.3%로 나빠 사용자가 "원하는 방향이 아니다"라고 판단해 중단.

**기각된 대안 2건**:
1. **프롬프트 단위 직접 PCA**(카테고리 안 거치고 29,673차원에서 SVD, 축당 상위 200개): 커버리지 버그(12개 카테고리 probe 0개) 발견했고, 더 근본적으로 프롬프트 단위 PCA는 상위 5축이 분산의 **32.27%**만 설명(카테고리 단위 94.66%의 1/3) — 카테고리 평균이 신호 희석이 아니라 노이즈 감소 역할도 하고 있었다는 뜻. 사용자 판단으로 폐기.
2. **모델 쌍 구별력 기반 probe 선별**: 사용자 스스로 "체리피킹 같다"고 기각 — 현재 모델 풀의 특정 혼동 쌍에 과적합돼 신규 모델에 일반화 안 됨.

**대화 중 나온 별도 통찰**: CSCR이 probe를 거의 안 쓸 수도 있다는 우려에 대해, (a) CSCR FP가 비공개라 정확한 비교 불가, (b) 핵심 기여(Combined GRPO)는 이미 존재하는 공개 벤치마크 데이터로 검증됐으므로 FP 구축 비용과 무관하게 유효, (c) FP 비용 절감은 "미래의 새 모델 onboarding"이라는 별도의 부차적 질문임을 재확인 — 사용자의 "핵심 기여가 의미없어 보인다"는 우려에 대한 정리.

**다음 세션 후보**: 공개 벤치마크 점수(gsm8k, mmlu 등 모델 카드에 이미 보고된 점수) 재사용 — probe 자체를 아예 안 돌리는 방향, 미착수.

관련 신규 스크립트: `embedllm_build_pca_weighted_probe_fp_floorablation.py`(분리 실험, 폐기), `embedllm_prompt_level_pca_probe_selection.py`(프롬프트 단위 PCA, 폐기). `RESULTS_SUMMARY.md` §4.1 반영됨.

---

## 23. 교수님 피드백 반영 계획 — Fair comparison 검증 + Combined GRPO 수식화 (최신 세션, 날짜 확인 필요)

`RESULTS_SUMMARY.md`(Combined GRPO 확립) 이후 교수님께 중간 결과 피드백을 받음. 세 가지 지적사항과, 이번 세션에서 정리한 대응 계획.

### 23.1 받은 피드백 3가지
1. **InfoNCE 분자를 sum 대신 max 또는 softmax로 바꿔볼 것.**
2. **Fair comparison 문제**: CSCR은 probe를 192개만 쓰는데, 우리(Ceiling FP, 특히 V1.5/V1)는 1,800~1,920개를 씀 — 비교가 공정하지 않을 수 있음.
3. **Combined GRPO의 손실 함수를 정식 수식으로 정리할 것** (발표/논문용).

### 23.2 피드백 1 — InfoNCE 분자 sum→max/softmax

**22.9/22.12에서 이미 제안했던 "적응적 trim(soft-min, learnable temperature)" 방향과 정확히 일치**하는 제안임을 확인. 현재 `multi_positive_info_nce`(20.4/21.2에서 이미 구현·테스트됨)의 분자는 정답 후보 전체를 sum으로 묶는데:
- **Sum**(현재): outlier-drag에 취약(21.1)하지만 collapse엔 상대적으로 강함(21.2에서 확인 — 순수 contrastive/sum이 GRPO보다 collapse는 덜함)
- **Max**: `exp(max_p sim_p / τ)` — MSE 회귀에서 이미 검증된 min-pos(21.4)의 InfoNCE 버전. outlier-drag에도 안 취약할 것으로 예상
- **Softmax**(LogSumExp, 온도 τ₂): sum(τ₂→∞)과 max(τ₂→0) 사이를 잇는 연속적 중간 지점 — "학습 가능한 트림 강도" 아이디어의 구체화

**가설**: max/softmax가 sum의 collapse-저항성과 min-pos의 outlier-drag-저항성을 동시에 잡을 수 있다면, 지금까지 나온 조합(GRPO+min-pos+catfilter) 이상으로 개선될 가능성. `multi_positive_info_nce`의 분자 계산부만 3버전으로 스위치 가능하게 구현하면 됨(min-pos 만들 때와 비슷한 작업량). **미착수.**

### 23.3 피드백 2 — Fair comparison (probe 개수)

**검토한 두 방향**:
- (a) Ceiling FP를 192로 축소해서 비교
- (b) CSCR류(Perplexity FP)를 1,800으로 확장해서 비교

**(a)는 우선순위에서 제외하기로 결정** — 이미 있는 데이터로 결과가 어느 정도 예측됨:
- EmbedLLM all-seen은 PCA-weighted 1,200개(바닥6)에서 이미 균등 1,920개(V1, 0.5481)보다 낮은 0.5393까지 떨어짐(§22.14) — 192는 그 6분의 1도 안 됨
- LLMRouterBench(19.3, 22개 카테고리) 기준 "카테고리당 probe 2개"(44개 총량)는 p=0.107로 무의미했음 — EmbedLLM(80개 카테고리)에 192개를 균등 배분하면 카테고리당 2.4개로 밀도가 비슷해서, 유의성 확보가 어려울 것으로 예상
- 즉 돌려봐도 "무너진다"는 예측 가능한 결과라 새로운 정보가 적음

**(b)를 채택** — CSCR 쪽 예산을 늘리는 실험을 우선 진행하기로 함. 이유:
1. 교수님이 실제로 주장한 명제("probe를 늘리면 CSCR도 좋아질 것")를 가장 직접적으로 검증함
2. "우리 예산을 깎아도 우리가 무너진다"보다 "CSCR에 우리와 같거나 더 많은 예산을 줘도 못 따라온다"는 게 fairness 반박으로서 더 완결적임
3. 결과 자체가 (a)와 달리 사전에 확정적이지 않은, 진짜 열린 질문임

**부가 발견 (중요)**: `scripts/build_routerbench_perplexity_fp.py`의 `N_PROBES=32`가 git 최초 커밋(`68fc46b`)부터 고정된 값이고, 이후 한 번도 수정 안 됨. 이 프로젝트의 확정 표준은 N=192(9.1 섹션, 논문의 N=K 조건에 맞춰 의도적으로 선택)인데, 이 스크립트만 그 표준을 안 따르고 있었음. MixInstruct용 Colab 스크립트에 있던 "디버그용 32로 먼저 검증 → 실제 실행은 192로" 관례(원본 주석: "Start small to validate the full pipeline runs cleanly, then bump N up to ~150-192")가 이 RouterBench 스크립트에서는 반영이 안 된 것으로 추정 — **즉 22.8의 기존 Perplexity FP vs Ceiling FP 비교 자체가, 애초에 논문 기준보다도 6배 적은 예산에서 나온 결과였을 가능성.**

**확정 계획**: `N_PROBES ∈ {32(현재/추정 버그), 192(논문·프로젝트 표준), 1800(Combined 예산)}` 3점 스윕.
- **데이터 가용성 확인 완료**: RouterBench Set A(seed=42, 80%) = 29,197행, 11개 모델 응답 텍스트 100% 밀도 — 1,800개 추출에 전혀 문제없음(Set A의 6.2%만 사용)
- **구현**: `build_routerbench_perplexity_fp.py`의 `N_PROBES` 상수만 바꾸면 나머지 로직(Set A 샘플링, GPT-2 채점, 저장) 그대로 재사용 가능. 코드 수정 거의 없음.
- **예측**: 32→192 구간에서 이미 개선되면(reliability 문제였다는 뜻) 22.8 결과를 스스로 시정한 셈이 되어 신뢰도에 플러스. 192→1800까지 가도 여전히 Ceiling에 못 미치면, 격차가 probe 예산이 아니라 **FP가 capability 정보를 담고 있는가(validity)의 문제**라는 기존 결론(22.8)이 더 탄탄해짐.
- unseen/all-seen 결과가 다를 가능성 있음 — probe 축소에 all-seen이 훨씬 민감했던 기존 패턴(22.7/22.13/22.15)을 감안하면, probe 증가의 효과도 all-seen 쪽이 더 클 수 있음(반대 방향 대칭) — 실행 후 비교 필요.

**⚠️ 실행 전 발견한 추가 confound — "표본 수"와 "차원 수"가 지금 1:1로 묶여 있음**: `build_routerbench_perplexity_fp.py`는 probe 1개 = 벡터 1차원(GPT-2 cross-entropy 스칼라)으로 대응되므로, N_PROBES를 1,800으로 올리면 자동으로 **1,800차원** 벡터가 됨. 문제는 **RouterBench가 모델 11개뿐**이라는 것 — 11개 점을 1,800차원에 흩뿌리면 실제 필요한 자유도(최대 10차원)보다 압도적으로 많은 차원을 주는 셈이라, "probe를 늘려서 노이즈가 줄었다"는 진짜 reliability 효과와 "차원이 극단적으로 많아져서 다운스트림 학습(query encoder projection head)이 11개 타겟을 사실상 과적합/암기해버릴 여지가 생겼다"는 순수 차원-팽창 효과가 뒤섞임. (참고: EmbedLLM의 Ceiling V1/V1.5도 probe-indexed라 비슷한 문제가 원리상 있지만, 거긴 모델이 80~112개라 비율(1800/80≈22)이 RouterBench(1800/11≈164)보다 훨씬 덜 극단적이라 상대적으로 덜 위험했음.)

**대응 계획 — 표본 수(reliability)와 차원 수(representational capacity)를 분리해서 스윕**:
1. **고정 32차원 + 표본만 증가(메인)**: 기존 32차원은 유지하되, 각 차원을 probe 1개가 아니라 여러 개(예: 56개씩 32묶음 = 1,800개)의 평균으로 채움 — "차원 수는 그대로, 각 차원의 추정치만 더 안정적"이라 순수 reliability 효과만 봄. 교수님 주장을 가장 정직하게 검증하는 버전.
2. **1,800차원 그대로(보너스/참고용)**: probe=차원=1,800, Ceiling FP의 probe-indexed 방식과 형식적으로 일치. **이 버전에서 성능이 크게 오르면, 반드시 같은 차원의 랜덤 벡터(negative control, 17.11에서 이미 쓴 방식과 동일)로도 돌려서 "차원 수 자체의 효과"와 "진짜 신호"를 분리 확인할 것.**
- 구현 시 `N_PROBES`(표본 수)와 `N_DIMS`(차원 수)를 별도 파라미터로 분리해야 함 — 현재 스크립트는 이 둘이 암묵적으로 묶여 있어 수정 필요.

**미착수**, GPU 있는 다른 로컬에서 이어서 실행 예정.

### 23.4 피드백 3 — Combined GRPO 수식화, 그리고 명명 체계 확정

**명명 체계 (2026-08-18 확정)**: "Combined GRPO"는 실제 GRPO(RL 알고리즘)가 아니라는 멘토 피드백에 따라 전면 재명명. 방법론을 2단계 계층으로 구분:

- **COMPAR** (Capability-Oriented Multi-Positive Adaptive Routing) — descriptor(Ceiling FP) + loss + kNN/bandit 라우팅을 포함한 **전체 방법론**을 가리키는 상위 이름. "Capability-Oriented"는 정답 여부 기반 descriptor(CSCR의 Perplexity, 즉 유창함 기반 descriptor와의 핵심 차별점)를, "Multi-Positive"는 한 쿼리에 정답 모델이 여러 개일 수 있다는 구조(outlier-drag/min-pos/catfilter 설계 전체의 전제)를 가리킴. "Adaptive"는 발음 가능한 약어(COMPAR, "compare"처럼 읽힘 — 결국 여러 모델의 능력치를 비교/라우팅한다는 의미와도 부합)를 만들기 위해 추가, 쿼리 인코더가 쿼리별로 적응적으로 판단한다는 의미도 자연스럽게 부합.
- **TAR** (Trimmed Advantage Regression) — COMPAR가 쓰는 **loss 함수 자체**를 가리키는 하위 이름(구 "Combined GRPO"). min-pos + top-K%/percentile 카테고리 필터(§21~22에서 확립)를 가리킴.

이후 문서/발표에서 "COMPAR가 쓰는 loss가 TAR"라는 계층으로 서술.

### 23.4-a TAR(Trimmed Advantage Regression) 수식화

**표기**: 쿼리 $i$, 전문가 풀 $\{1,\dots,M\}$, 쿼리 임베딩 $q_i\in\mathbb{R}^d$(L2 정규화), 전문가 descriptor $E\in\mathbb{R}^{M\times d}$(L2 정규화, 고정), 코사인 유사도 $s_{i,m}=q_i\cdot E_m$, 정답 여부 $\ell_{i,m}\in\{0,1\}$, 쿼리 카테고리 $c(i)$.

**① GRPO mean-centered 타겟**:
$$\bar\ell_i=\frac{1}{M}\sum_m \ell_{i,m},\quad \sigma_i=\sqrt{\bar\ell_i(1-\bar\ell_i)},\quad z_{i,m}=\frac{\ell_{i,m}-\bar\ell_i}{\sigma_i+\epsilon}$$
($\sigma_i$는 이진 라벨의 표준편차라 $\sqrt{p_i(1-p_i)}$로 닫힌 형태 — $\sigma_i\approx0$인 쿼리(전원 정답/오답)는 제외. 이진값이라 한 쿼리 안에서 정답은 전부 같은 값 $z_i^{(1)}=\sqrt{(1-p_i)/p_i}$, 오답도 전부 같은 값 $z_i^{(0)}=-\sqrt{p_i/(1-p_i)}$만 가짐, $p_i=\bar\ell_i$.)

**② Category-filter (top-$\rho$ percentile trim, $\rho=0.5$)**: $a_{m,c}$ = Set A에서 모델 $m$의 카테고리 $c$ 평균 정답률(사전 계산, 고정).
$$P_i^{raw}=\{m:\ell_{i,m}=1\},\quad P_i=\mathrm{top}_{\lceil\rho|P_i^{raw}|\rceil}\big(P_i^{raw};\text{rank by }a_{m,c(i)}\big)$$
$$\mu_{i,m}=\begin{cases}1 & \ell_{i,m}=0\\ 1 & \ell_{i,m}=1\text{ and }m\in P_i\\ 0 & \ell_{i,m}=1\text{ and }m\notin P_i\end{cases}$$
(오답은 항상 마스킹 없음. 정답 중 해당 카테고리 트랙레코드가 하위인 것만 강등되어 손실에서 제외 — "우연히 맞춘" 정답을 걸러냄.)

**③ Min-pos (trimmed advantage regression)**:
$$P_i^{keep}=\{m\in P_i:\mu_{i,m}=1\},\quad N_i=\{m:\ell_{i,m}=0\}$$
$$\mathcal{L}_i^{pos}=\begin{cases}\min_{m\in P_i^{keep}}(s_{i,m}-z_{i,m})^2 & P_i^{keep}\neq\varnothing\\0&\text{otherwise}\end{cases}\qquad \mathcal{L}_i^{neg}=\frac{1}{|N_i|}\sum_{m\in N_i}(s_{i,m}-z_{i,m})^2$$

**최종 (배치 $\mathcal{B}$)**:
$$\mathcal{L}_{Combined}=\underbrace{\frac{\sum_{i\in\mathcal{B}}\mathcal{L}_i^{pos}\cdot\mathbb{1}[P_i^{keep}\neq\varnothing]}{\sum_{i\in\mathcal{B}}\mathbb{1}[P_i^{keep}\neq\varnothing]}}_{\text{min-pos}}+\underbrace{\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\mathcal{L}_i^{neg}}_{\text{오답 평균(기존 GRPO와 동일)}}$$

**핵심 직관**: $z_i$ 벡터 전체를 하나의 목표로 회귀하는 게 아니라(바닐라 GRPO가 이 방식이고 이게 outlier-drag 원인), 정답 쪽은 여러 후보 중 **가장 맞히기 쉬운 것 하나만**(min, implicit OR) 골라 맞히면 되고, 오답 쪽은 원래대로 전부 평균 내어 밀어냄(여러 오답을 동시에 피하는 건 기하학적으로 항상 가능하므로 평균을 내도 outlier-drag가 안 생김). "advantage 벡터를 통째로 회귀"가 아니라 "일부(정답 중 최근접 1개+오답 전체)만 trim해서 쓴다"는 뜻에서 Trimmed Advantage Regression으로 재명명 제안(멘토 피드백 — "Combined GRPO"는 실제 GRPO/RL이 아니므로 이름 변경 필요).

### 23.5 다음 할 일 (우선순위 순)
1. **Combined GRPO 손실 함수 수식화** (발표/논문에 필요, 아직 아무 데도 정리 안 돼 있음)
2. RouterBench Perplexity FP `N_PROBES ∈ {32, 192, 1800}` 스윕 실행 — **표본 수(N_PROBES)와 차원 수(N_DIMS)를 분리한 버전**을 메인으로(고정 32차원 + 표본만 1800으로 증가), probe=차원=1800인 버전은 negative control(랜덤 벡터)과 같이 보너스로. 데이터 가용성 확인 완료, GPU 있는 다른 로컬에서 실행 예정.
3. InfoNCE 분자 sum/max/softmax 비교 실험 (`multi_positive_info_nce` 확장, 다른 로컬에서 실행)
4. (참고, 보류) Ceiling FP를 192로 축소하는 실험 — 결과가 예측 가능해 우선순위 낮음, 필요시 나중에

## 24. Fair-probe-budget 검증 완료 — RouterBench + LLMRouterBench, CSCR 실제 loss로 재검증

23.3의 계획을 실행. 두 단계 시행착오를 거쳐 최종 방법론을 확정하고, RouterBench·LLMRouterBench 두 벤치마크 모두에서 CSCR에게 우리와 같은 probe 예산을 줘도 격차가 안 줄어드는 것을 확인.

### 24.1 시행착오 1 — 처음엔 Perplexity FP에도 Combined loss를 잘못 적용함

`routerbench_perplexity_probesweep_combined.py`(N_PROBES=32/192/1800, 고정 32차원)를 만들어 실행했으나, Perplexity FP 쪽에도 우리 Combined(min-pos+catfilter) loss를 썼음 — 이건 22.4("FP-type 의존성 확인")와 같은 질문이지 fairness 검증이 아님. 사용자 지적으로 정정: **fairness 검증은 CSCR의 실제 논문 loss(`cost_spectrum_info_nce`, 9번 섹션에서 이미 upstream과 byte-identical 확인됨)를 Perplexity FP에 그대로 써야 함.** 실행 중이던 잘못된 job 중단.

### 24.2 시행착오 2 — 매 epoch마다 frozen encoder를 다시 인코딩하던 속도 버그

기존 `routerbench_perplexity_combined.py`의 `train()`은 frozen MiniLM(gradient 없음)의 CLS 임베딩을 매 epoch, 매 배치마다 다시 계산 — 실제로는 텍스트가 고정이니 결과도 매번 동일한데도 재계산. 1개 시드에 1207초(20분) 걸림 확인, 9개 조합(3 FP × 3 시드) 예상 3시간+. **고정**: 학습 전에 frozen 임베딩을 한 번만 계산해서 캐싱, 매 epoch은 작은 projection head(2-layer MLP)만 캐싱된 벡터 위에서 학습 — `scripts/routerbench_fair_probe1800_multiseed.py`. 결과: 시드당 20분 → 1분 미만으로 단축(`embedllm_newllm_fast_eval.py`에서 썼던 것과 같은 캐싱 원칙).

### 24.3 RouterBench 최종 결과 (all-seen, 3시드)

- **Ceiling V1.5**(`build_routerbench_ceiling_fp_v15.py`, PCA 중요도 가중 probe 배분, 1761개 probe, 86차원 그대로): 0.7285 / 0.7303 / 0.7257, 평균 **0.7282**(std 0.0019) — CSCR 논문(0.711) 3/3 이김, 기존 풀예산 Ceiling V2(0.7226)와 동급이거나 근소 우위.
- **Perplexity 1800-probe + CSCR 실제 loss**(`cost_spectrum_info_nce`, 32차원 고정 N_DIMS/N_PROBES 분리): 0.6382 / 0.6352 / 0.6371, 평균 **0.6368**(std 0.0012) — CSCR 논문(0.711) 0/3, N_PROBES=32였던 기존 결과(22.8, 0.6936)보다도 오히려 낮음. probe를 늘려도 CSCR 자체 loss로는 개선이 없다는 뜻 — 16번 섹션에서 이미 문서화된 `cost_spectrum_info_nce`의 소규모 pool collapse 경향과 일관됨.

결과 파일: `local_descriptors/routerbench-analysis/fair_probe1800_multiseed_results.json`

### 24.4 LLMRouterBench로 확장 — probe 예산 재검토 (900으로 조정)

사용자 요청으로 LLMRouterBench(33개 모델)에도 같은 검증 확장. RouterBench와 달리 LLMRouterBench는 Set A 전체가 8개 데이터셋 합쳐 **2,345행뿐**(RouterBench는 29,197행) — 1800 probe는 Set A의 77%를 써버리는 셈이라 의미가 퇴색됨. **900개로 조정**(Set A의 약 38%), PCA 중요도 기반으로 8개 데이터셋에 비균등 배분(`scripts/llmrouterbench/build_fp_v15_900.py`):

| dataset | importance | probes (cap) |
|---|---|---|
| arenahard_creative_writing | 0.1913 | 159 (200) |
| aime | 0.1780 | 48 (48, 캡) |
| arenahard_coding | 0.1467 | 139 (202) |
| arenahard | 0.1354 | 133 (600) |
| livecodebench | 0.1036 | 117 (844) |
| arenahard_math | 0.0897 | 109 (197) |
| livemathbench | 0.0798 | 96 (96, 캡) |
| gpqa | 0.0754 | 99 (158) |

LLMRouterBench는 이 프로젝트의 기존 관례상(`build_perplexity_fp.py`) Ceiling·Perplexity 두 FP가 **같은 probe**를 공유하므로, 900개를 그대로 양쪽 FP 빌드에 재사용. 출력 차원은 기존 192차원(8×24) 관례 유지 — probe 수만 192→900으로 늘고, 각 24차원 슬롯에 들어가는 probe를 평균내는 방식(RouterBench의 N_DIMS 고정 트릭과 동일).

**GPT-2 스코어링 성능 이슈 2건 발견·수정**:
1. 처음엔 텍스트 1개씩 순차 처리(호출당 batch=1) — 900×33=29,700회 예상 45분+. 배치 처리(batch_size=32)로 전환.
2. batch_size=32에서 GPU 메모리 3893/4096MB로 OOM 위험 확인 → batch_size=8로 하향.
3. 그래도 예상보다 느림(aime 하나에 4.7분) — 원인 진단: `aime`(추론 CoT) 텍스트가 평균 9777 토큰, 최대 37516 토큰이라 `max_length=512`로 truncate해도 토크나이저가 truncate 이전에 전체 문자열을 먼저 인코딩해야 해서 병목. **문자 단위 사전 절단**(`char_cap = max_len * 6`, 토큰화 전에 raw string을 미리 잘라냄) 추가로 해결 — 이후 데이터셋 종류 무관하게 일관된 속도(~0.178초/probe-model) 확보. 전체 900-probe 빌드 완료까지 약 96분 소요(1회성 빌드 비용).

### 24.5 LLMRouterBench 최종 결과 (all-seen, 3시드) — Combined GRPO 최초 적용

이 pool에 Combined GRPO(min-pos+top50pct catfilter, 카테고리=8개 데이터셋)를 적용한 것은 이번이 처음. `scripts/llmrouterbench/fair_probe900_multiseed.py`.

- **Ceiling-900 + Combined GRPO**: 0.7627 / 0.7642 / 0.7662, 평균 **0.7644**(std 0.0014)
- **Perplexity-900 + CSCR 실제 loss**: 0.6887 / 0.6621 / 0.6629, 평균 **0.6712**(std 0.0123)

격차 약 9.3%p, Ceiling 쪽이 편차도 훨씬 작음(std 0.0014 vs 0.0123). 이 벤치마크는 CSCR 논문에 없는 독자 구축 pool이라 외부 기준값이 없음 — 두 방법론의 직접 head-to-head 비교임.

**주목할 점**: 17.2에서 이 33개 pool은 **옛날 loss(cost_info_nce/cost_spectrum_info_nce)로는 parametric 학습 단계에서 Ceiling이 Perplexity를 유의미하게 못 이겼음**(delta=-0.0154, p=0.178, kNN에서는 이기는데 학습시키면 무승부) — 원인은 Ceiling FP 신호의 28.5%가 "이 모델이 전반적으로 센가"라는 coarse 축(PC1)이라 경사하강법이 세밀한 도메인 매칭 대신 이 축으로 shortcut-collapse하기 때문(17.2 상세 분석). 이번엔 Combined GRPO로 3시드 다 깔끔하게, 낮은 분산으로 이겼음 — min-pos(평균 대신 최근접 정답만 매칭)+catfilter(카테고리 내 상위 트랙레코드만 유지)가 outlier-drag뿐 아니라 이 pool 특유의 tier-axis collapse도 완화했을 가능성. **확정하려면 추가 진단(예: collapse_diagnostic로 실제 라우팅 분포가 소수 모델에 쏠렸는지 확인) 필요 — 아직 가설 단계.**

결과 파일: `local_descriptors/llmrouterbench_v15_900/fair_probe900_multiseed_results.json`, `local_descriptors/llmrouterbench_v15_900/allocation.json`

### 24.6 "Naive 차원" 버전 (probe=차원 그대로) — RouterBench에서 랜덤 노이즈가 CSCR 논문값을 이겨버림

24.3의 "차원 고정(32)+표본만 증가" 버전과 별개로, 사용자가 "교수님이 상상하는 fairness가 이게 아닐 수도 있다"고 지적 — 그냥 probe 개수만큼 차원을 그대로 늘린(binning 없음) naive 버전도 확인. RouterBench는 이미 만들어둔 보너스 아티팩트(`routerbench-perplexity-dim1800/`, 진짜 probe=차원 1800차원, + 같은 차원의 랜덤벡터 negative control `routerbench-perplexity-dim1800-randomcontrol/`, 17.11 방식과 동일)를 그대로 재사용해서 CSCR 실제 loss로 학습(`scripts/routerbench_naive_dim1800_cscrloss.py`).

| FP (1800차원) | seed0 | seed1 | seed2 | 평균 | std | CSCR 논문(0.711) 이김 |
|---|---|---|---|---|---|---|
| Perplexity-dim1800(진짜) + CSCR loss | 0.6352 | 0.6279 | 0.6388 | **0.6340** | 0.0045 | 0/3 |
| 1800차원 랜덤벡터(노이즈) + CSCR loss | 0.7202 | 0.7073 | 0.7145 | **0.7140** | 0.0053 | 2/3 |

**진짜 신호(0.6340)보다 순수 랜덤 노이즈(0.7140)가 더 잘 나옴 — 심지어 노이즈가 CSCR 논문값(0.711)을 2/3 시드에서 이김.** 11개 모델짜리 pool에 1800차원을 주면, 내용이 뭐든 상관없이 "11개 타겟에 과적합할 자유도"만으로 성능이 올라간다는 뜻. 23.3에서 미리 우려했던 capacity-vs-reliability confound가 실측으로 확인된 것.

**해석 (사용자 제안, 발표에서 구두로 언급 예정)**: CSCR 논문/코드가 실제로 192차원 정도의 상대적으로 작은 차원을 유지한 게, 어쩌면 우연이 아니라 바로 이 과적합 함정을 피하려는 의도적 설계였을 수 있음. 그렇다면 "CSCR한테 probe를 늘려서 fair하게 해달라"는 요청을 문자 그대로(naive하게 차원까지 같이 늘려서) 따르는 건 오히려 CSCR을 인위적으로 유리하게 만드는 방법론적 실수가 됐을 것 — 24.3의 "차원 고정, 표본만 증가" 버전이 이 요청에 대한 올바른 해석이었고, 이 naive 버전은 정반대 사례(이렇게 하면 안 되는 이유)로서의 반증 자료.

결과 파일: `local_descriptors/routerbench-analysis/naive_dim1800_cscrloss_results.json`

**LLMRouterBench도 완료** (RouterBench와 GPU 경합 있었던 첫 시도는 중단, 단독 재실행): `build_perplexity_naive_dim900.py`(fp16, batch=16, max_len=256, 총 5278초=88분, 원시값 `raw_perplexity_values_900.npz`로 캐싱됨) + `naive_dim900_cscrloss.py`로 CSCR 실제 loss 학습.

| FP (900차원) | seed0 | seed1 | seed2 | 평균 | std |
|---|---|---|---|---|---|
| Perplexity-dim900(진짜) + CSCR loss | 0.6814 | 0.6865 | 0.6079 | **0.6586** | 0.0359 |
| 900차원 랜덤벡터 + CSCR loss | 0.6607 | 0.6166 | 0.6492 | **0.6421** | 0.0187 |

RouterBench(랜덤이 역전)만큼 극적이진 않음 — 진짜 신호가 근소하게 앞서지만 std 범위 안에 들어가 사실상 유의차 없음. 33모델/900차원(비율 ~27)이 RouterBench의 11모델/1800차원(비율 ~164)보다 과적합 압력이 훨씬 약한 게 원인으로 추정. 그래도 "naive 차원 확장이 안정적 이득을 못 준다"는 결론은 동일 — 오히려 24.5의 컨트롤 버전(192차원 고정+900표본평균, 평균 0.6712)이 이 naive 900차원 버전(0.6586)보다 나음, RouterBench와 같은 방향.

결과 파일: `local_descriptors/llmrouterbench_v15_900/naive_dim900_cscrloss_results.json`

### 24.7 EmbedLLM에서 Ceiling V1.5를 192-probe로 축소 (반대 방향 fairness 테스트)

사용자 요청으로 반대 방향도 확인: CSCR에게 probe를 늘려주는 대신, **Ceiling V1.5를 CSCR의 원래 예산(192개)으로 줄이면 어떻게 되는가.** 23.3에서 예측 가능하다고 후순위로 미뤘던 실험이지만, PCA 가중 배분이 예상보다 잘 버틸 가능성을 확인하기 위해 실행.

`scripts/embedllm_build_pca_weighted_probe_fp_192.py`(기존 1800-probe 레시피 재사용, MIN_PROBES만 15→1로 하향 — 80개 카테고리에 floor=15면 최소 1200개라 192 목표 자체가 불가능해서): 192개 중 top 카테고리(gsm8k)가 5개, 대부분 3~4개, 23/80개 카테고리는 floor인 1개씩. `scripts/embedllm_pcaweighted_192_allseen_multiseed.py`(기존 pct=0.3 combined 학습 재사용 + 임베딩 캐싱 속도 개선 적용)로 all-seen 3시드 학습 중.

**결과 — 예상과 달리 훨씬 잘 버팀**:

| 방법 | probe 수 | all-seen AUDC |
|---|---|---|
| V2 (전체 데이터, 카테고리 집계) | 전체 | 0.5652 |
| V1.5 (PCA 가중, floor=15) | 1800 | ~0.556 |
| V1 (균등 배분) | 1920 | 0.5481 |
| **V1.5 (PCA 가중, floor=1, 이번 실험)** | **192** | **0.5444**(std 0.0025), seed 0.5409/0.5457/0.5467, CSCR 이김 2/3 |
| CSCR 논문값(all-seen) | 192 | 0.541 |

1800→192개(90% 감소)인데 AUDC는 0.556→0.5444로 겨우 ~2%만 하락, 여전히 CSCR 논문값과 대등/근소 우위(2/3 시드). 사용자가 예상했던 "처참한 점수"는 아니었음 — PCA 중요도 기반 배분이 중요 카테고리(gsm8k, mmlu 계열)에 예산을 집중하고 나머지는 floor=1로 최소 커버만 유지하는 전략이 극단적 저예산에서도 핵심 신호를 상당 부분 보존한 것으로 보임. §23.3에서 "예측 가능해서 새 정보 적음"이라 후순위로 미뤘던 판단은 재고할 만함 — 실제로는 예측(무너진다)과 다른 결과가 나왔으므로.

결과 파일: `local_descriptors/embedllm-analysis/pcaweighted_allseen_pct30_multiseed_results_192probes.json`, 배분표 `local_descriptors/embedllm-analysis/pca_weighted_probe_allocation_192.json`

### 24.9 명명 체계 확정 — COMPAR / TAR

"Combined GRPO"가 실제 GRPO(RL 알고리즘)가 아니라는 멘토 피드백(§23.4 참고)에 따라 최종 명명 확정:
- **COMPAR** (Capability-Oriented Multi-Positive Adaptive Routing) — descriptor(Ceiling FP) + loss + kNN/bandit 라우팅을 포함한 전체 방법론의 이름. "Capability-Oriented"=정답 기반 descriptor(CSCR의 Perplexity와의 차별점), "Multi-Positive"=한 쿼리에 정답 모델이 여러 개일 수 있다는 구조적 전제(outlier-drag의 근본 원인), "Adaptive"=발음 가능한 약어(COMPAR, "compare"처럼 읽힘)를 위해 추가, 쿼리별 적응적 판단이라는 의미도 부합.
- **TAR** (Trimmed Advantage Regression) — COMPAR가 쓰는 loss 함수(구 "Combined GRPO", §23.4 수식 참고)의 이름.

### 24.10 멘토 피드백 — Min-pos를 Min-top-K로 일반화 (K=1→3, 유의미한 추가 개선 확인)

멘토 피드백: min-pos(K=1, 가장 가까운 정답 1개만)가 아니라 상위 K개 평균으로 일반화해보라는 제안. $K=1$이면 기존 min-pos와 동일, $K=$전체면 바닐라 GRPO(outlier-drag 재발)로 자연스럽게 양 극단을 포함하는 일반화:
$$\mathcal{L}_i^{pos,K}=\frac{1}{\min(K,|P_i^{keep}|)}\sum_{m\in\mathrm{bottomK}_K(P_i^{keep};(s_{i,m}-z_{i,m})^2)}(s_{i,m}-z_{i,m})^2$$

EmbedLLM V2 FP(80차원, 기존 K=1 all-seen 기준값 0.5652를 낸 바로 그 FP)에 K=3로 재테스트(`scripts/embedllm_mintopk_allseen_multiseed.py`, pct=0.3 catfilter는 그대로 유지, 캐싱 파이프라인으로 빠르게 실행):

| K | seed0 | seed1 | seed2 | 평균 | std | CSCR(0.541) 이김 |
|---|---|---|---|---|---|---|
| K=1(기존 min-pos) | – | – | – | 0.5652 | – | – |
| **K=3** | 0.5897 | 0.5853 | 0.5839 | **0.5863** | 0.0025 | 3/3 |

**K=3가 K=1보다 +0.0211(약 3.7% 상대 개선), 편차도 작음(std 0.0025).** min-pos가 "가장 가까운 정답 1개"만 보는 게 정보를 너무 적게 쓰고 있었을 가능성 — top-3 정도로 완화하니 outlier-drag 저항력은 유지하면서 더 많은 정답 신호를 활용하는 균형점을 찾은 것으로 보임. 결과 파일: `local_descriptors/embedllm-analysis/mintop3_allseen_multiseed_results.json`.

**후속(완료)**: 고정 K=3 대신 카테고리 필터와 같은 방식(top-pct%)으로 "정답 후보 중 상위 30%"를 쓰는 `minpct_loss`(pct=0.3, `scripts/embedllm_minpct_allseen_multiseed.py`) 결과:

| 버전 | seed0 | seed1 | seed2 | 평균 | std |
|---|---|---|---|---|---|
| K=1(기존 min-pos) | – | – | – | 0.5652 | – |
| K=3(고정 개수) | 0.5897 | 0.5853 | 0.5839 | 0.5863 | 0.0025 |
| **pct=30%(비율 기반)** | 0.5858 | 0.5900 | 0.5816 | **0.5858** | 0.0034 |

고정 개수(K=3)와 비율 기반(pct=30%)이 사실상 동률(0.5863 vs 0.5858, 오차 범위 안) — 파라미터화 방식과 무관하게 "정답 후보 1개만 보던 min-pos를 살짝 완화하면 도움이 된다"는 효과가 안정적으로 재현됨. 결과 파일: `local_descriptors/embedllm-analysis/minpct0.3_allseen_multiseed_results.json`.

**추가 후속**: pct=30%가 정답 후보가 많은 쿼리에서 K를 과도하게 키울 수 있다는 우려(사용자 지적)로, `min(pct=0.3, K_CAP=3)`(비율과 고정 상한 중 더 작은 쪽, `scripts/embedllm_minpctcap3_allseen_multiseed.py`)도 확인. (선택 자체는 `sort`/`argmin`이라 이산적이지만 max-pooling과 동일한 패턴 — 선택된 원소의 값에는 그래디언트가 정상적으로 흐르므로 전체적으로 미분 가능.)

| 버전 | seed0 | seed1 | seed2 | 평균 | std |
|---|---|---|---|---|---|
| K=1(min-pos) | – | – | – | 0.5652 | – |
| K=3(고정) | 0.5897 | 0.5853 | 0.5839 | 0.5863 | 0.0025 |
| pct=30%(비율) | 0.5858 | 0.5900 | 0.5816 | 0.5858 | 0.0034 |
| **min(30%, 3)** | 0.5886 | 0.5890 | 0.5824 | **0.5867** | 0.0030 |

세 일반화 버전(K=3/pct30/min(30%,3)) 전부 0.586대로 사실상 동률 — 파라미터화 방식은 결과에 큰 영향이 없고, "정답 후보를 1개에서 2~3개 수준으로 완화"라는 핵심 효과 자체가 안정적. 결과 파일: `local_descriptors/embedllm-analysis/minpct0.3cap3_allseen_multiseed_results.json`.

**§21.4의 min-pos(K=1) 확립 당시엔 K를 스윕하지 않고 바로 K=1로 확정했었음** — 이번 결과로 보면 K=1이 로컬 최적이 아니었을 가능성.

사용자가 "min(30%,3)"을 최종 버전으로 선택(가장 안정적) — 이후 실험은 전부 이 버전으로 진행.

### 24.12 min(30%,3)을 세 벤치마크에 전면 적용 — 모델 풀 크기에 따라 효과가 완전히 갈림

사용자 요청으로 이미 만들어둔 FP(GPT-2 스코어링 등 비싼 부분은 이미 끝나 있음, loss만 교체)를 이용해 RouterBench·LLMRouterBench의 기존 all-seen COMPAR 결과를 min(30%,3)으로 재학습, EmbedLLM 4-way ablation의 빠진 칸(min(30,3) 단독, catfilter 없이)도 채움. RouterBench-Unseen은 구조적으로 불가능(11개 모델뿐, CSCR 논문도 시도 안 함)이라 제외.

**RouterBench·LLMRouterBench all-seen(기존 V1.5 FP 재학습)**:

| 벤치마크(모델 풀) | K=1 | min(30%,3) | 개선폭 |
|---|---|---|---|
| EmbedLLM(111개) | 0.5652 | 0.5867 | **+3.7%** |
| RouterBench(11개) | 0.7282 | 0.7289 (0.7288/0.7304/0.7276, std 0.0011) | +0.1%(오차범위) |
| LLMRouterBench(33개) | 0.7644 | 0.7652 (0.7609/0.7659/0.7687) | +0.1%(오차범위) |

**모델 풀이 클수록 min-top-K 완화 효과가 커짐, 작을수록 거의 없음** — 작은 풀은 쿼리당 정답 후보 수 자체가 보통 1~2개뿐이라(§2.5 패턴과 동일) K=3/pct=30% 확장이 실질적으로 K=1과 같아지는 경우가 대부분. EmbedLLM처럼 정답 후보가 풍부한 큰 풀에서만 완화가 유의미.

결과 파일: `local_descriptors/routerbench-analysis/v15_minpctcap3_multiseed_results.json`, `local_descriptors/llmrouterbench_v15_900/v15_minpctcap3_multiseed_results.json`

**EmbedLLM all-seen 완전한 4-way ablation (min(30,3) 버전)**:

| 방법 | 평균 | std | CSCR(0.541) 이김 |
|---|---|---|---|
| vanilla GRPO | 0.4802 | 0.0180 | 0/3 |
| K=1(min-pos)만 | 0.5587 | 0.0036 | 3/3 |
| **min(30,3)만(catfilter 없음)** | **0.5725** | 0.0049 | 3/3 |
| catfilter만(pct=0.3) | 0.5364 | 0.0015 | 0/3 |
| K=1 + catfilter(기존 combined) | 0.5652 | 0.0005 | 3/3 |
| **min(30,3) + catfilter(신규 combined)** | **0.5867** | 0.0030 | 3/3 |

min-loss 자체를 K=1→min(30,3)으로 바꾸는 것만으로도 catfilter 유무와 무관하게 항상 개선(0.5587→0.5725, 0.5652→0.5867) — 두 메커니즘(min-loss 완화, catfilter)이 독립적으로 기여하고 있다는 근거. 결과 파일: `local_descriptors/embedllm-analysis/minpct0.3cap3_nocatfilter_allseen_multiseed_results.json`.

**LLMRouterBench Unseen (이 프로젝트에서 최초 시도, 완료)**: 33개 모델 풀이라 진짜 seen/unseen 분할이 가능한데(RouterBench는 불가), 지금까지 all-seen만 검증했었음. 기존 Ceiling-900 FP가 모델별 독립 파일이라 재구축 없이 seen/unseen만 나누면 재사용 가능 — 시드마다 다른 랜덤 분할(22 seen/11 unseen, `scripts/llmrouterbench/minpctcap3_unseen_multiseed.py`).

| seed | seen/unseen | AUDC |
|---|---|---|
| 0 | 22/11 | 0.6927 |
| 1 | 22/11 | 0.7166 |
| 2 | 22/11 | 0.7091 |
| **평균** | | **0.7061** |

이 벤치마크에서 unseen을 처음 시도한 거라 비교할 K=1 기준값은 없음 — all-seen(0.7652)보다는 낮지만(예상대로, 학습에 안 쓰인 모델로의 라우팅이 더 어려움), 0.69~0.72대로 붕괴 없이 안정적. EmbedLLM unseen(K=1: 0.5162, min(30,3): 0.5238)에 비해 all-seen 대비 낙폭이 훨씬 작음(LLMRouterBench: 0.7652→0.7061, -7.7% / EmbedLLM: 0.5867→0.5238, -10.7%) — 절대적 라우팅 난이도 차이는 있지만 unseen 일반화 자체는 두 벤치마크 모두에서 확인됨. 결과 파일: `local_descriptors/llmrouterbench_v15_900/unseen_minpctcap3_multiseed_results.json`.

### 24.14 Fairness — EmbedLLM 192-probe도 min(30%,3)로 재검증, 최종 종합표

`scripts/embedllm_pcaweighted_192_minpctcap3_multiseed.py`(§24.7의 K=1 버전에서 loss만 교체):

| 버전 | seed0 | seed1 | seed2 | 평균 | std | CSCR(0.541) 이김 |
|---|---|---|---|---|---|---|
| K=1(min-pos) 192-probe | 0.5409 | 0.5457 | 0.5467 | 0.5444 | 0.0025 | 2/3 |
| min(30,3) 192-probe | 0.5490 | 0.5291 | 0.5526 | 0.5436 | 0.0103 | 2/3 |

평균은 K=1과 사실상 동일(오히려 미세하게 낮음), 편차만 4배 커짐(probe가 극단적으로 적은 상황이라 후보 풀 자체가 얕아서, 2~3개까지 보게 하면 노이즈 낀 후보까지 끌어들이는 것으로 추정). **사용자 판단(확정)**: 이 실험의 목적은 "CSCR 원래 예산으로 줄여도 안 무너진다"는 fairness 반박 그 자체이지 이 극단적 저예산 상황에서의 실전 배포가 아니므로, 편차 증가는 페널티로 취급하지 않음 — 2/3 CSCR 이김 + K=1과 동급 평균이면 fairness 목적은 충분히 달성.

**최종 종합 — 세 벤치마크 fairness 결과 (min(30%,3) 전면 반영)**:

| 벤치마크 | 방향 | Probe | 평균 AUDC | 비고 |
|---|---|---|---|---|
| RouterBench | CSCR↑ (Perplexity에 예산 증가) | 1761 | 0.7289 | CSCR 논문(0.711)·재현(0.6368) 모두 이김 |
| LLMRouterBench | CSCR↑ | 900 | 0.7652 | CSCR 재현(0.6712) 이김(논문값 없음) |
| EmbedLLM | Ceiling↓ (우리 예산을 CSCR 수준으로 축소) | 192 | 0.5436 | CSCR 논문(0.541)과 대등(2/3 이김) |

세 벤치마크, 양방향(CSCR에 더 주기 / 우리를 줄이기) 전부에서 결론 동일: **probe 예산을 어느 쪽으로 맞춰도 COMPAR가 CSCR을 넘어선다.**

### 24.15 중요 발견 — "1800-probe로 돌리고 있다"고 착각했던 실험들이 사실은 전체 데이터(V2급) 버전이었음

§24.10~24.14의 EmbedLLM all-seen Combined(0.5867), ablation(0.5725), unseen(0.5238) 전부 **실제로는 V1.5(1800-probe, floor=15)가 아니라 `embedllm-ceiling`/`embedllm-ceiling-pca5`(probe 제한 없음, 전체 Set A 사용)를 쓰고 있었음** — 사용자가 뒤늦게 지적해서 발견. 진짜 V1.5는 `embedllm-ceiling-pcaweighted-pca5`(별도 디렉토리)였고, 오늘 min(0.3,3) 실험 중 단 한 번도 이 디렉토리를 쓴 적이 없었음.

**COMPAR 정규 주장용으로 진짜 V1.5(1800-probe, floor=15)를 재검증**(`scripts/embedllm_v15_1800_minpctcap3_allseen_multiseed.py`, `scripts/embedllm_v15_1800_minpctcap3_unseen_multiseed.py` — 기존 V1.5 FP는 112개 모델 전부로 이미 올바르게 만들어져 있어 재구축 불필요, unseen도 별도 unseen-only 서브셋 없이 같은 디렉토리에서 unseen 모델만 골라 씀). 처음 3시드(0,1,2) 확인 후, "중요한 실험이니 시드를 늘리자"는 사용자 요청으로 3~9 시드 추가, **총 10시드로 확정**:

**All-seen** (seed0-2: 0.5511/0.5547/0.5549, seed3-9: 0.5419/0.5505/0.5487/0.5511/0.5566/0.5429/0.5466):

| 통계 | 값 |
|---|---|
| 10시드 평균 | **0.5499** |
| std | 0.0047 |
| CSCR(0.541) 대비 | +1.65% |
| 승률 | 10/10 |

**Unseen** (seed0-2: 0.5060/0.5173/0.5017, seed3-9: 0.5137/0.5054/0.5238/0.5131/0.5146/0.5065/0.5256):

| 통계 | 값 |
|---|---|
| 10시드 평균 | **0.5128** |
| std | 0.0076 |
| CSCR(0.4848) 대비 | +5.77% |
| 승률 | 10/10 |

기존 K=1 버전의 V1.5 all-seen(≈0.556, 구버전 기록)과 비교하면 min(0.3,3)의 all-seen 개선 효과는 거의 없음(오히려 미세 하락) — RouterBench·LLMRouterBench에서 이미 봤던 "descriptor 공간이 작고 거칠수록 min-top-K 완화 효과가 사라진다"는 패턴이 EmbedLLM V1.5(probe로 제한된 5차원 PCA 공간)의 all-seen에서도 재현됨. 다만 **unseen 쪽은 마진이 오히려 더 좋음(+5.77%, all-seen +1.65%보다 큼)** — all-seen에서 안 보이던 min(0.3,3)의 이득이 unseen에서는 살아있는 것으로 보임(정확한 원인은 미분석). 두 프로토콜 다 10/10 완승이라 "1800-probe로도 COMPAR가 CSCR을 이긴다"는 핵심 주장은 통계적으로 탄탄하게 확정됨.

결과 파일: `local_descriptors/embedllm-analysis/pcaweighted_allseen_minpctcap3_multiseed_results_v15_1800probes{,_seeds3to9}.json`, `local_descriptors/embedllm-analysis/unseen_minpctcap3_multiseed_results_v15_1800probes{,_seeds3to9}.json`

**사용자 최종 방침(대화 중 확정)**: V1.5 숫자는 여전히 필요함 — "probe 예산 fairness"라는 멘토 요구사항에 대한 답이 이거이기 때문(교수님이 1800을 "평범한 수준"이라 언급했다고 해서 불필요해지는 게 아님, 별개의 질문). 반면 "probe 수에 따른 scalability" 스토리는 V1.5의 절대 수치와 무관하게 V2를 상단 앵커로 써도 무방 — 두 스토리가 서로 다른 질문에 답하는 것이므로 어느 하나가 다른 하나를 대체하지 않음.

### 24.16 부수 발견 — Probe-scale sweep 스크립트에서 phantom 모델 제외 시점 버그

96~4200 probe sweep(§24.14 이전 작업, 사용자가 "Deferral curve까지 그려보자"고 요청)을 만들 때, phantom 모델(`JaeyeonKang__CCK_Asura_v1`)을 **FP 생성 전에 미리 제외**해버려서, PCA/mean-centering이 111개 모델 기준으로 계산됨 — 기존 스크립트들(FP는 112개로 만들고 나중에 필터링)과 달라서 같은 seed=0/같은 probe 수인데도 결과가 달랐음(target=192에서 0.5343 vs 올바른 값 0.5490). **원인**: pool 평균과 PCA 축(주성분 방향) 자체가 "몇 개 모델을 기준으로 계산하느냐"에 따라 바뀌는 통계량이라, phantom 모델 하나를 빼고 계산하면 나머지 111개 전부의 좌표가 미묘하게 달라짐(주성분 축이 회전함). `all_models`(112)로 FP를 만들고 `models`(111, 필터링됨)로 로드하도록 수정 후 재실행 — target=192 재현값(0.5490)이 §24.14의 3시드 fairness 결과 seed0(0.5490)과 정확히 일치해 수정 확인.

수정된 sweep 최종 결과(seed=0만, 단일 시드 참고용):

| target | 실제 probe 수 | AUDC | Peak |
|---|---|---|---|
| 96 | 96 | 0.5570 | 0.5870 |
| 192 | 192 | 0.5490 | 0.5867 |
| 300 | 300 | 0.5390 | 0.5603 |
| 500 | 500 | 0.5409 | 0.5673 |
| 800 | 800 | 0.5493 | 0.5770 |
| 1200 | 1200 | 0.5511 | 0.5753 |
| 1800 | 1800 | 0.5436 | 0.5753 |
| 2800 | 2800 | 0.5382 | 0.5713 |
| 4200 | 4200 | 0.5358 | 0.5837 |
| V2(전체) | – | 0.5886 | 0.6260 |

단일 시드라 비단조적으로 흔들림(300/2800/4200이 CSCR 0.541보다 낮게 나옴) — 사용자 확정 방침: 이 sweep은 fairness 확정 주장이 아니라 "대체적 추세" 참고용 시각자료이므로 멀티시드 재검증 없이 진행. Deferral curve 이미지: `local_descriptors/embedllm-analysis/probe_scale_sweep_deferral_curves.png`. 데이터: `local_descriptors/embedllm-analysis/probe_scale_sweep_minpctcap3_results.json`.

### 24.17 다음 할 일
1. `Final_result_Summary.MD` §1.1/1.2를 진짜 V1.5(1800-probe) 숫자로 갱신하거나 라벨 정정(§24.15) — §25에서 처리, `Final_Result_Summary2.MD`로 별도 작성
2. 사용자 요청: probe-scale sweep을 1800~30000 구간으로 확장 — §25.2에서 처리
3. 24.5의 tier-axis collapse 완화 가설을 `collapse_diagnostic`으로 직접 검증 — 여전히 미착수
4. InfoNCE 분자 sum/max/softmax 비교(§23.1 피드백 1) — 여전히 미착수
5. RouterBench·LLMRouterBench 둘 다 all-seen만 검증됨(RouterBench는 구조적으로 불가, LLMRouterBench는 §24.12에서 별도로 unseen 완료)

## 25. PCA-5 압축이 병목이었음을 발견 — 무압축+균등 할당으로 파이프라인 전환 (2026-08-19)

### 25.1 발단 — Probe-scale sweep을 넓은 범위로 재설계하다가 압축 자체를 의심하게 됨

24.17의 "sweep을 30000까지 확장" 요청에 따라 MAX_PROBES 캡을 풀고(3500) 넓은 범위로 재설계했으나, 사용자가 "노이즈가 너무 심하고 작은 probe는 의미 없다"며 카테고리별 **균등 할당(uniform)**으로 바꿀 것을 요청. PCA 중요도 가중 할당은 고중요도 카테고리가 실제 보유 데이터를 빨리 소진해버려 target을 올려도 실제 확보 probe 수가 정체되는 문제가 있었음(target=25000인데 실제 12441개만 확보) — 균등 할당으로 바꾸니 목표치와 실제 확보량이 거의 정확히 일치(24999/25000 등).

### 25.2 균등 할당으로도 AUDC가 평평함 — 압축 자체를 의심

균등 할당(PCA-5 압축은 유지)으로도 1760~24999 probe 전 구간에서 AUDC가 0.547~0.560 사이로 평평(probe 수와 무관). 사용자가 "5차원 압축이 병목 아니냐"는 가설 제기 → PCA-5 투영 단계를 생략한 "무압축(80차원 그대로)" 버전을 만들어 같은 스윕 재실행.

**결과: 무압축은 훨씬 높고(0.579~0.587), 1800개만 써도 V2(전체 데이터, 29673개)의 0.5867과 거의 동률(25000개에서 0.5867로 완전 일치).** 압축 손실을 probe 수와 완전히 분리해서 격리 측정(`embedllm_compression_cost_isolation.py`, 양쪽 다 전체 데이터 100% 사용): 무압축 0.5867 vs 압축 0.5558 → **압축만으로 -0.0309 AUDC 손실**, 1800-probe 스케일에서 본 손실(-0.0314)과 거의 동일 — 압축 손실은 probe 수와 무관하게 일정.

원인 분석(§20.3 참고): 예전에 "80차원이나 PCA-5나 결과가 거의 같다"고 확인했던 건 **학습 없는 직접 kNN 검증**이었음 — FP 자체의 정보 보존은 맞았지만, 이후 도입된 **학습되는 projection head**가 5차원이라는 훨씬 좁은 목표 공간을 정확히 맞추기가 80차원보다 구조적으로 어려움(같은 절대 오차가 더 적은 차원에 집중되어 코사인 유사도에 미치는 영향이 커짐) — "FP의 정보 보존"과 "그 정보를 학습으로 재현하는 난이도"는 다른 문제였음.

### 25.3 Unseen에서도 동일한 손실 재현 — 압축은 seen/unseen 무관하게 순손실

All-seen만의 암기 효과(사용자 의심: "카테고리 80개로 이 정도 안 나올 리 없다, 과적합 아니냐")를 배제하기 위해 Unseen 프로토콜로도 같은 압축/무압축 비교 실행(`embedllm_probe_scale_sweep_unseen_multiseed.py`, 71~73 seen/35~36 unseen, 기존 `newllm_split*.json` 재사용). 압축 vs 무압축 격차가 Unseen에서도 비슷한 크기로 유지됨(1800-probe: +0.0158, All-seen의 +0.0314와 같은 방향·비슷한 비율) — **압축이 순수 암기 효과의 산물이 아니라 실제 일반화 능력 자체를 깎아먹는다는 근거**.

Unseen에서 probe 수와의 상관관계도 확인: 압축 버전은 약한 상관(Spearman ρ=0.70, 유의하지 않음, 8000에서 한 번 꺾임), 무압축 버전은 96~25000(300배) 전 구간에서 **완벽한 단조 증가**(Spearman ρ=1.0, Pearson r=0.97, p=0.001) — 다만 효과 크기는 작음(+0.013 AUDC).

### 25.4 300-probe 지점 이상치 발견 → 재분배(resplit)로 반증

무압축 all-seen 스윕에서 target=300(실제 320 probe)이 이웃 지점보다 뚜렷하게 높게 나옴(All-seen 0.6004 vs 96=0.5798·800=0.5909, std=0.0006으로 우연이라 보기엔 너무 타이트; Unseen에서도 0.5345로 동일 지점이 재현). 사용자가 "자연 현상 같은 스윗스팟은 있을 수 없다"며 강하게 의심 → 순차적으로 검증:
1. Set A(train.csv)/Set B(test.csv) 프롬프트 중복 0건 확인(ID·텍스트 모두 대조)
2. probe 할당 로직 정상(카테고리당 정확히 4개씩, 불균형 없음)
3. FP의 V2(정답) 근사도 측정 — 300은 오히려 1800·4000보다 정답에서 더 멂(코사인 유사도 0.55 vs 0.74/0.82) → "우연히 정답에 가까워서" 가설 기각
4. 학습 내부 holdout rho도 96~1200 전 구간 평평(0.40~0.41) → 학습이 300에서 특별히 잘 됐다는 신호 없음
5. **train.csv+test.csv를 합쳐 완전히 새로운 무작위 분할(seed=777)로 재현** — All-seen·Unseen 둘 다 300 스파이크 소멸(재분배 후 300은 이웃 지점들과 std 범위 내에서 구분 불가, 오히려 1800이 제일 높게 나오기도 함). **결론: HF 공식 train/test split과 이 특정 probe 부분집합 사이의 우연한 정렬이었을 뿐, 재현 가능한 현상이 아님.**

### 25.5 `Final_Result_Summary2.MD` 작성

무압축+균등 할당을 새 기준으로 EmbedLLM 결과를 재정리한 신규 문서 작성(v1인 `Final_result_Summary.MD`는 그대로 보존, 비교용). 헤드라인: All-seen 0.5787(+7.0% vs CSCR), Unseen 0.5232(+7.9% vs CSCR), 둘 다 1800-probe·3시드. RouterBench·LLMRouterBench는 이번 세션에 무압축+균등 할당으로 재검증하지 않음(원래도 무압축이었으나 할당 방식이 아직 PCA가중 — 전환 필요, §25.6 참고).

### 25.6 다음 할 일 (25.6-5, 25.6-6 등 완료 항목은 §26 참고)
1. ~~EmbedLLM Ablation(catfilter 없음)·Fairness(192-probe)~~ → §26.1에서 완료
2. ~~RouterBench·LLMRouterBench 균등 할당 재검증~~ → §26.2에서 Pure V2 기준으로 완료 (probe-budget 버전은 §26.2/26.5 일부만)
3. ~~Unseen 스윕 Deferral Curve 시각화~~ → §26.3에서 완료
4. 헤드라인 수치(1800-probe All-seen/Unseen)를 3시드→10시드로 확장할지 — 여전히 미결정
5. §24.17의 3~4번(collapse_diagnostic, InfoNCE 분자 비교)은 여전히 미착수
6. min(0.3,3) 단독 효과는 미미했고(All-seen 기준 K=1 대비 오히려 미세 하락), §25의 개선(+0.0227 all-seen)은 사실상 전부 압축 제거+균등 할당 전환에서 옴 — 논문/발표에서 "손실 함수 개선"과 "표현 방식(압축/할당) 개선"을 구분해서 서술할 것

## 26. Random 대조군으로 방법론 재검증 + CSCR FP 교차 실험 + Deferral Curve 확정 (2026-08-19, 같은 날 후반)

### 26.1 EmbedLLM 남은 항목 채우기

무압축+균등 할당 기준으로 미착수였던 것들 완료:
- **Fairness(192-probe)**: catfilter 있음 0.5883(std 0.0049, +8.7%), 없음 0.5696(std 0.0047) — catfilter 효과 +0.0187로 **probe가 적을수록 catfilter가 유의미해짐**(1800-probe에서는 +0.0005로 거의 없었음) — probe 희소성에 반비례하는 안전장치라는 패턴 확인.
- **Ablation(catfilter 유무, 1800-probe)**: 있음 0.5787 vs 없음 0.5792(+0.0005, 사실상 무의미) — 무압축 80차원에서는 catfilter 단독 기여가 거의 사라짐.

### 26.2 사용자의 근본적 의심 — "probe가 96개인데 왜 V2급으로 잘 나오지?" → Random 대조군 연쇄 검증

사용자가 극단적으로 적은 probe(카테고리당 1개)에서도 All-seen/Unseen 성적이 V2(전체 데이터)에 육박하는 것을 "과적합/의심스럽다"고 지적, 일련의 negative control 실행:

1. **Random probe 선택**(target=96, top-variance 대신 완전 무작위): All-seen 0.5909 — top-variance 선택(0.5798)과 동등하거나 오히려 높음. "똑똑한 probe 선택이 이유"라는 가설 기각.
2. **순수 랜덤 노이즈 FP**(진짜 데이터 전혀 없음, 112개 모델×80차원 무작위 벡터): All-seen 0.5769(실제 FP와 거의 동일) / **Unseen 0.3721(CSCR 0.4848보다도 낮음, 완전 붕괴)**. → **All-seen은 "학습 중 라벨을 본 111개 닫힌 클래스에 대한 분류 문제로 퇴화"하는 경향이 있어 FP 내용이 거의 안 중요하지만, Unseen은 이 암기 메커니즘이 불가능해서 노이즈에서 확실히 무너짐 — Unseen이 방법론의 핵심 주장(텍스트→능력 매핑 학습)을 검증하는 진짜 지표.**
3. **카테고리 셔플 FP**(80개 슬롯·실제 데이터는 그대로, 슬롯의 카테고리 소속만 무작위로 뒤섞음, 모델 간엔 동일 배정 유지): All-seen 0.5730, **Unseen 0.5261(std 0.0008, 실제 FP의 0.5232와 사실상 동일)**. → 카테고리 "의미 정렬"은 Unseen에 불필요, 필요한 건 "진짜 데이터로 채워진 서로 다른 real signal 축이 있다는 것" 자체.

**결론(사용자 정리, 확인됨)**: Unseen을 위해 필요했던 건 FP의 (의미 정렬이 아니라) 데이터 진위 — Catfilter/min(0.3,3)은 각각 probe 희소성(§26.1)/모델 풀 크기(§24.12)에 반응하는 노이즈 대응 장치.

### 26.3 RouterBench·LLMRouterBench Pure V2 + Fairness 채우기, Deferral Curve 시각화

- **RouterBench**: 기존 `routerbench-ceiling`(86차원, 전체 Set A, 원래부터 무압축)을 Pure V2로 사용, All-seen 3-way ablation(Combined/min(0.3,3)만/Catfilter만) → 세 조건 거의 동일(0.7202~0.7225) — 모델 풀 작아 개별 기여 희석. 이후 사용자 요청으로 ablation 대신 **CSCR 비교로 단순화**: Combined 0.7205 vs CSCR 논문값 0.711(+1.3%).
- **LLMRouterBench**: 신규 `llmrouterbench-ceiling-purev2`(8차원, 전체 Set A 2345행, probe 제한·binning 없음) 빌드. All-seen 0.7349, Unseen 0.6719. 이후 **900-probe 균등할당·무압축 버전**도 신규 빌드(`llmrouterbench-ceiling-900-uniform-nocompress`, 8차원)해서 정식 Fairness 수치 채움: All-seen 0.7186(+7.1% vs CSCR 재현치 0.6712), Unseen 0.6586.
- **Unseen Deferral Curve**: 기존 스윕 스크립트가 스칼라만 저장했던 걸 곡선 저장하도록 수정 후 재실행(`embedllm_probe_scale_sweep_unseen_multiseed_withcurves.py`) — 압축/무압축 전체 지점 곡선 확보, 시각화 완료.
- **# Params(B) 스케일 재시각화**: `compute_cost(..., "n_params") = raw_params(B) * 0.03`임을 확인, cost를 0.03으로 나눠 원 논문 스타일 차트(# Params vs Accuracy)와 동일 스케일로 V1.5(1800)/V2 커브 재작성 (All-seen, Unseen 둘 다). 사용자가 이걸 원작 baseline 비교차트(UMR/SOFTMOE/THOMPSON/CSCR 등, `scripts/plot_curves.py`가 원본 생성 스크립트로 확인됨) 위에 직접 오버레이 — COMPAR 커브가 좌상단(적은 파라미터로 고정확도)에 위치, CSCR 자체 곡선까지 상회.
- **AUDC 계산 방식 일치 검증**: `plot_curves.py`(원작자)의 `build_cost_grid`/`interp_to_grid`가 `run_audc_eval.py`(이번 세션 전체가 사용)와 **완전히 동일한 코드**(N_grid=20, `np.trapz`, 동일 정규화)임을 직접 대조로 확인 — cost 범위도 겹침 확인(사용자) → 오버레이 비교 유효성 확보.

### 26.4 CSCR FP + COMPAR loss 교차 실험 — "FP와 loss는 짝을 맞춰야 한다"

COMPAR의 loss(Catfilter+min(0.3,3))를 CSCR의 실제 FP(Perplexity)에 그대로 적용해서, 우위가 loss 때문인지 FP 때문인지 분리:
- **RouterBench**(Perplexity dim=32): COMPAR loss+CSCR FP = 0.6170(std 0.0006) — CSCR 자체 loss+같은 FP(0.6368)보다도 낮음.
- **LLMRouterBench**(Perplexity dim=192): COMPAR loss+CSCR FP = All-seen **0.2594**(std 0.0375), Unseen 0.4183(std 0.0679) — CSCR 자체 loss(0.6712)보다 훨씬 낮음, 완전 붕괴 수준. (시스템 재부팅 직후 실행이라 결과 의심 → 재실행, 결정론적 파이프라인이라 완전히 동일한 값 재현 확인, 시스템 불안정 때문이 아니라 진짜 결과임을 확정.)

**결론**: COMPAR의 우위는 loss 단독 효과가 아니라 "정밀한 FP(Ceiling, 직접 측정) + 그 정밀함을 전제로 설계된 loss(catfilter의 track-record 필터링, min(0.3,3)의 상위 k개 정밀 매칭)"가 함께 있어야 나옴 — FP 차원이 클수록(32→192) 손해가 더 커지는 것으로 보아, 노이즈 낀 FP일수록 COMPAR의 정교한 loss가 오히려 역효과를 낼 수 있음.

### 26.6 원본 CSCR 논문·공식 repo 대조 검증

사용자가 원본 논문(arXiv:2508.12491v3, `2508.12491v3.pdf`)과 공식 GitHub repo(`github.com/rezashkv/cscr`)를 제공 — 직접 대조:

- **CSCR_ALLSEEN=0.541, CSCR_UNSEEN=0.4848 둘 다 논문 Table 1/Table 2(EmbedLLM) 원본 수치와 정확히 일치 확인** — 이 세션 내내 상수로 써온 값들의 출처가 확정됨.
- **AUDC 계산 방식 완전 일치 확인**: 공식 repo의 `scripts/run_audc_eval.py`, `scripts/plot_curves.py`가 이 프로젝트의 동명 파일과 `build_cost_grid`/`interp_to_grid`(N_grid=20, `np.trapz`, `(grid[-1]-grid[0])` 정규화) 코드까지 바이트 단위로 동일 — 이 프로젝트의 AUDC 계산 스크립트가 원작자의 실제 파일일 가능성이 매우 높음.
- **논문 §4.2.1 "Generalization to New LLMs"**: "select two-thirds... training... **testing is conducted exclusively on the unseen LLM pool**" — 우리의 `newllm_split.json`(seen 학습 전용, unseen만 후보로 평가) 구현과 방법론적으로 일치.
- **공식 repo에 New-LLM 전용 스크립트는 없음** — 대신 범용 `eval_router(router, data, name, candidates=None, ...)` 함수가 `candidates` 파라미터를 받아 `router.route(..., candidates=candidates, ...)`로 그대로 전달하는 범용 필터링 메커니즘 확인. 즉 New LLM 실험은 별도 스크립트가 아니라 이 범용 함수에 `candidates=unseen_리스트`를 넘겨서 실행했을 것 — 우리의 `load_embedllm(candidates=unseen_models)` 방식과 동일한 메커니즘.
- **QNC 단위 확인**: `compute_cost(..., "n_params") = raw_params(B) * 0.03`. 논문 Table 1/2의 QNC는 raw params(B) 단위 — 우리 QNC(0.03-스케일)를 0.03으로 나누면 직접 비교 가능. 실측: All-seen에서 COMPAR(~44.3B)가 CSCR(43.28B)보다 근소하게 비쌈, **Unseen에서는 COMPAR(~37.6B)가 CSCR(70.0B, 사실상 최대 모델)의 절반 수준 cost로 거의 같은 Peak(0.554 vs 0.565)를 찍고 AUDC에서 크게 앞섬(0.523 vs 0.485)** — CSCR의 Unseen 대응이 "가장 비싼 모델로 억지로 정점 찍기"에 가까워 보임.
- **# Params(B) 스케일로 V1.5(1800)/V2 커브를 원작 baseline 차트(UMR/SOFTMOE/THOMPSON/CSCR) 위에 직접 오버레이 — COMPAR가 CSCR 자체 곡선보다도 위(좌상단)에 위치, 같은 벤치마크·같은 AUDC 공식으로 확인된 유효한 비교.**

### 26.7 Unseen 평가의 엄격성 재검토 — 혼합 풀(Mixed-pool) 테스트

사용자 질문: "SetB에서 seen 모델로 라우팅되는 것도 허용하는가?" → 코드 확인 결과 **아니오** — `E_unseen`(코사인 유사도 행렬 자체)에 seen 모델이 아예 안 들어있어 라우팅 후보에서 원천 배제. 즉:
1. unseen 모델 전원이 오답인 쿼리 → 무조건 실패 처리.
2. 정답이 10개 중 6개(5 seen+1 unseen)여도, 성공하려면 그 1개의 unseen 모델을 정확히 찍어야 함(seen 5개는 애초에 후보에도 없음).

이게 평가 설계 자체의 가혹함인지 확인하기 위해 **혼합 풀 테스트**(`embedllm_unseen_mixedpool_1800_multiseed.py`) 실행 — 학습은 그대로(seen만), 평가 시 후보를 seen+unseen 전체로 개방, 쿼리를 4개 버킷(seen_only/unseen_only/both/neither 각각 "정답이 어디 있는가")으로 나눠 버킷별 top-1 정확도 측정:

| 버킷 | Top-1 정확도(3시드 합산) |
|---|---|
| seen_only | 12.7~18.9% |
| **unseen_only** | **0/206 (0.0%, 3시드 전부)** |
| both | 62.2~67.3% |
| neither | 0%(자명) |

전체 AUDC=0.5590(All-seen 0.5787에 근접)이지만, **이는 전적으로 "both" 버킷(전체 쿼리의 ~93%)에서 나온 것 — 정답이 unseen에만 있는 쿼리에서는 3시드 합쳐 206개 중 단 한 번도 못 맞힘.** 사용자가 사전에 정확히 이 결과를 예측함.

**해석(사용자·모델 합의)**: 이건 별개의 새 약점이 아니라 §26.2의 "분류 퇴화" 메커니즘의 논리적 귀결 — 라우터는 학습 중 타겟으로 한 번도 쓰인 적 없는 unseen 좌표를 "암기"할 방법이 없으므로, seen 옵션이 함께 있으면 항상 암기된 seen 쪽이 이김. 기존 "Unseen만 후보" 프로토콜(0.5232, 논문과 동일 방식)이 오히려 이 암기 우회를 원천 차단하는 더 정직한 지표였다는 게 재확인됨. 혼합 풀 결과는 논문에 없는 이 세션만의 추가 탐구이며, "Unseen 일반화"를 논문 그대로 주장할 때는 "제한된 후보군 안에서"라는 조건을 명시해야 함(발표 한계점 섹션에 적합).

### 26.8 다음 할 일
1. RouterBench·LLMRouterBench에도 §26.2/26.7과 같은 negative control(노이즈 FP, 셔플 카테고리, 혼합 풀) 미실행
2. RouterBench probe-budget(fairness) 버전은 Pure V2로 대체되어 이번엔 미실행
3. §25.6-4/5(10시드 확장, collapse_diagnostic, InfoNCE 비교)는 여전히 미착수
4. 사용자가 언급한 "더 검증하고 싶었던 것들"이 남아있음 — 다음 세션 시작 시 확인 필요
5. 발표/논문에서 Unseen 관련 주장 시, §26.7의 혼합 풀 결과(unseen_only 버킷 0%)를 한계점으로 명시할지 결정 필요

## 27. Collapse diagnostic 최초 실행 — 원본 GRPO 대비 TAR/COMPAR의 개선 확인 (2026-08-19, Claude Code 세션)

24.17-3/25.6-5/26.8-3에서 세 번 연속 "미착수"로 남아있던 collapse_diagnostic을 처음으로 현재 방법론 체크포인트에 대해 실행. (진짜 최신인 무압축+균등할당 헤드라인 설정(§25-26)은 저장된 체크포인트가 없어서 — 해당 스윕 스크립트들이 `enc.save()`를 호출하지 않음 — 대신 V1.5(1800-probe, PCA-weighted floor=15, PCA-5) + min(0.3,3)+catfilter 체크포인트(`embedllm-newllm-encoder-minpctcap3-v15-1800-seed0`, §24.15의 10-seed unseen 결과에 쓰인 바로 그 체크포인트)로 실행. 스크립트: `scripts/embedllm_v15_minpctcap3_collapse_check.py`(기존 `embedllm_newllm_grpo_collapse_check.py`를 V1.5 FP 경로에 맞게 포크).

**원본 vanilla-GRPO(§21 이전, seed0) vs 현재 TAR/COMPAR(V1.5, seed0) 비교**:

| 지표 | vanilla GRPO | TAR/COMPAR (V1.5) |
|---|---|---|
| top3_share | 0.853 | **0.769** |
| 최다 선택 모델 점유율 | 67.1% | **42.2%** |
| 사용된 모델 수 | 12/35 | 10/35 |
| Spearman rho(선택빈도, true_acc) | 0.42 (p=0.012) | **0.69 (p<0.0001)** |
| 최다 선택 모델 true_acc | 0.579 | 0.538 |
| 실제 1위 모델(Llama-3-70B, 0.605) 선택 비율 | 0.2% | 3.0% |

**해석**: raw collapse(top3_share)는 완전히 사라지지 않았지만(0.853→0.769, 여전히 chance 대비 9배), 최다 선택 모델의 독점률은 뚜렷이 낮아졌고(67%→42%), 무엇보다 **선택-실제정확도 상관관계가 0.42→0.69로 크게 개선**됐음 — 이번 세션 초반 사용자가 정리한 "irrational collapse(정답 아닌 모델로 쏠림, 예: 20.4/26.7의 group_loo 결과) → rational collapse(GRPO 도입 후, 정답 쪽으로 쏠림) → 지금(TAR/COMPAR)은 그 rational한 성질이 한층 더 강해짐"이라는 서사가 실측으로 뒷받침됨. 다만 여전히 진짜 최고 모델(true_acc 1위)은 거의 안 뽑힘(3%) — "쏠리되 대충 괜찮은 쪽으로 쏠린다"는 성질 자체는 완전히 해소되지 않았고, 완전한 fine-grained 도메인 매칭까지는 못 감.

**남은 gap**: 이 결과는 V1.5(1800-probe, PCA-5 압축) 체크포인트 기준 — §25에서 확인된 대로 PCA-5 압축 자체가 성능을 깎아먹는 요인이므로(-0.03 AUDC), 압축을 안 쓰는 최종 헤드라인 설정(§25-26, 무압축+균등할당)에서 collapse가 더 개선되는지는 별도 확인 필요(그 파이프라인에 `enc.save()` 추가 후 재학습해야 확인 가능 — 아직 미착수).

결과 파일: `local_descriptors/embedllm-analysis/newllm_grpo_collapse_check_v15_minpctcap3_seed0.json`

### 27.1 무압축 헤드라인 체크포인트로 재검증 (같은 세션, 사용자 지적 반영)

사용자가 "PCA 압축은 더 이상 안 쓰는 거고, 압축 버전으로 뭔가 했으면 무압축으로 다시 해야 한다"고 지적 — 위 27번의 V1.5(압축) 결과는 이제 деprecated 파이프라인 기준이므로, 진짜 헤드라인 FP(`embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-1800`, 80차원, 무압축+균등할당, Unseen AUDC=0.5232를 낸 바로 그 FP)로 재학습 후 재검증. `embedllm_unseen_mixedpool_1800_multiseed.py`(§26.7 혼합 풀 실험에 쓰인, 헤드라인과 정확히 같은 학습 레시피)의 학습 코드를 그대로 재사용해 seed=0 하나를 학습(기존 헤드라인 스크립트들은 체크포인트를 저장하지 않으므로, 이번에 처음으로 `torch.save`) — `scripts/embedllm_uncompressed_headline_collapse_check.py`, 체크포인트는 `local_checkpoints/embedllm-newllm-encoder-uncompressed-headline-seed0/proj.pt`.

| 지표 | V1.5(압축, 27번) | **무압축 헤드라인** |
|---|---|---|
| top3_share | 0.769 | 0.767 (거의 동일) |
| 사용된 모델 수 | 10/35 | 9/35 |
| rho(선택빈도, true_acc) | 0.690 | 0.669 (거의 동일, 단일 시드 노이즈 범위) |
| 최다 선택 모델(SUS-Chat-34B) 점유율 | 42.2% | 40.8% |
| **실제 1위 모델(Llama-3-70B, true_acc=0.605) 선택 비율** | **3.0%** | **19.0%** |

**핵심 발견**: top3_share·rho 같은 집계 지표는 압축 유무와 거의 차이가 없었지만(둘 다 오차범위 안), **진짜 최고 성능 모델이 받는 라우팅 비중은 3%→19%로 크게 늘었음** — Llama-3-70B가 압축 버전에서는 5위(3.0%)였는데 무압축에서는 2위(19.0%)로 올라옴. 즉 압축 제거가 "collapse의 폭"을 줄이진 않았지만 **"collapse가 향하는 대상의 질"은 개선함** — §25에서 이미 확인된 "압축이 AUDC를 깎아먹는다"는 결론과 같은 방향의 또 다른 증거. rho 같은 순위상관 지표만 보면 이 개선이 안 잡히고 "1위 모델 선택 비율"처럼 더 구체적인 지표에서만 드러난다는 점도 기록해둘 만함.

결과 파일: `local_descriptors/embedllm-analysis/newllm_grpo_collapse_check_uncompressed_headline_seed0.json`. 참고로 이 학습에서 나온 홀드아웃 rho(0.3869, best_epoch=2)는 압축 버전들에서 반복적으로 보이던 0.15~0.29대보다 뚜렷이 높음 — 이 역시 압축 제거의 효과와 일치.

### 27.2 무압축 기준 4-way ablation — min-loss/catfilter가 collapse에 미치는 영향 (같은 세션 후속)

사용자 질문: "무압축에서 min()이나 catfilter가 (AUDC가 아니라) collapse 측면에서 도움이 되는가?" — `scripts/embedllm_uncompressed_ablation_collapse_check.py`(seed=0), 무압축 헤드라인 FP 그대로, catfilter 유무(build_items의 category-track-record 데모션)와 min-loss 유무(minpctcap_loss의 top-k trim)를 독립적으로 켜고 끔:

| variant | top3_share | 사용 모델 수 | rho(선택,true_acc) | 1위모델 선택비율 |
|---|---|---|---|---|
| vanilla(둘 다 off) | **0.6857**(최저) | **23/35**(최다) | 0.4472 | 0.1460 |
| catfilter만 | 0.8163(최고) | 15/35 | 0.5150 | 0.2127 |
| min-loss만 | 0.7603 | 13/35 | 0.6632 | **0.2597**(최고) |
| combined(현재 TAR) | 0.7670 | 9/35(최소) | **0.6687**(최고) | 0.1900 |

**압축 버전에서의 통념과 반대되는 발견**: vanilla(아무 trim도 없음)가 top3_share·사용 모델 수 기준으로 **가장 덜 collapse함**. catfilter/min-loss를 추가할수록 raw collapse(top3_share)는 오히려 커지고 사용 모델 수는 줄어듦 — **즉 이 두 메커니즘은 "collapse의 폭을 줄이는" 효과가 없고, 오히려 좁힘.** 대신 rho(선택이 실제 정확도와 얼마나 맞는가)는 vanilla 0.447 → combined 0.669로 뚜렷이 개선 — **"더 좁게 쏠리지만, 쏠리는 대상은 더 정확해진다"**는 트레이드오프가 명확히 드러남.

부가 관찰: combined가 min-loss 단독보다 rho는 근소 우위(0.669 vs 0.663)지만 **1위 모델 선택 비율은 오히려 낮음**(19.0% vs 26.0%) — catfilter를 min-loss에 추가하는 게 "진짜 1등 모델에 더 집중"이 아니라 "소수의 상위권 모델들 사이에서 좀 더 골고루" 쪽으로 작동하는 것으로 보임(사용 모델 수는 13→9로 오히려 줄었는데도). 단일 시드라 이 미세한 역전이 노이즈일 가능성 배제 못함 — 확정하려면 멀티시드 필요.

**해석**: §27/27.1까지의 "collapse가 점점 더 rational해진다"는 서사는 유지되지만, "collapse 자체를 줄인다"는 주장은 성립하지 않음 — TAR(min-loss+catfilter)는 diversity(모델 풀을 넓게 쓰는가)가 아니라 accuracy-of-collapse(좁게 쓰더라도 맞는 쪽으로 쓰는가)를 개선하는 메커니즘으로 재규정해야 함. 발표에서 "collapse를 해결했다"고 하면 부정확 — "collapse가 향하는 방향을 교정했다"가 정확한 서술.

**왜 이런 결과가 나오는가(사용자, 메커니즘 설명)**: catfilter도 min-loss도 본질적으로 "학습 타겟에서 후보를 솎아내는" 필터임 — catfilter는 트랙레코드 나쁜 정답 후보를 타겟에서 제외하고, min-loss는 남은 정답 후보 중 제일 가까운 것 하나(또는 소수)만 맞추면 되도록 loss를 구성함. 즉 둘 다 "학습이 봐야 할 정답 후보의 폭을 의도적으로 좁히는" 설계이므로, 그 결과 추론 시 라우팅 폭도 좁아지는 것은 부작용이 아니라 **설계가 의도대로 작동한 당연한 귀결** — "collapse를 줄이려다 실패한 것"이 아니라 애초에 "정답 후보 풀을 순수하게 좁히는" 것이 이 두 메커니즘의 목적이었고, 그 좁힘이 라우팅 다양성 감소로 그대로 이어진 것.

결과 파일: `local_descriptors/embedllm-analysis/uncompressed_4way_ablation_collapse_check_seed0.json`

## 28. LLMRouterBench probe 스윕 + 192-probe fairness 보강 + FP×Loss 2x2 그리드 완성 (2026-08-19, 같은 날 세션 후속)

### 28.1 LLMRouterBench probe-count 스윕 — All-seen + Unseen (균등 할당, 무압축, 3시드)

EmbedLLM에서 확인된 "probe 수 무관하게 Unseen 성능 유지" 패턴이 모델 수(33 vs 111)·카테고리 수(8 vs 80)가 훨씬 작은 벤치마크에서도 재현되는지 확인. `scripts/llmrouterbench/build_ceiling_probesweep.py`(target 64~1800, 균등 할당, dim=8 무압축)로 FP 8세트 빌드, `scripts/llmrouterbench/probe_scale_sweep_allseen_unseen_multiseed.py`로 학습(캐싱된 임베딩을 모든 target에서 재사용 — target별로 바뀌는 건 FP뿐).

| target | 실제 probe | All-seen AUDC | Unseen AUDC |
|---|---|---|---|
| 64 | 64 | 0.7009±0.0015 | 0.6628±0.0178 |
| 160 | 160 | 0.7151±0.0071 | 0.6557±0.0307 |
| 320 | 320 | 0.7232±0.0037 | 0.6573±0.0364 |
| 480 | 482 | 0.7248±0.0068 | 0.6554±0.0343 |
| 640 | 643 | 0.7205±0.0124 | 0.6581±0.0364 |
| 900 | 900 | 0.7225±0.0042 | 0.6640±0.0294 |
| 1200 | 1202 | 0.7214±0.0060 | 0.6624±0.0294 |
| 1800 | 1799 | 0.7217±0.0045 | 0.6711±0.0236 |
| full | 2345 | 0.7349±0.0061 | 0.6719±0.0357 |

**EmbedLLM보다도 더 심하게 평평함** — Unseen은 시작점(64=카테고리당 8개)부터 이미 0.655~0.672 사이, 변동폭(~0.017)이 시드 표준편차(0.018~0.036)보다 작아 사실상 유의미한 추세 없음. 카테고리가 8개뿐이라 최소 할당만으로도 이미 "일반 실력 축"을 잡아내기 충분한 것으로 해석 — 카테고리 수가 적을수록 이 현상이 더 빨리 나타난다는 뜻으로, 기존 메커니즘 설명(클래시피케이션 콜랩스 + 일반 실력 축 지배)과 정합적.

**부가 확인 — 랜덤 대비 delta**: 모든 target×seed에서 delta=+0.17~+0.39, p=0.000999(부트스트랩 최대 유의성) — "probe 수가 안 중요하다"가 "Unseen이 안 된다"는 뜻은 아님, 랜덤 라우팅은 압도적으로 이김. 다만 **seed=0의 unseen 분할이 모든 probe target에서 일관되게 가장 어려움**(AUDC 0.607~0.640, 다른 시드는 0.664~0.702) — 분산의 대부분이 probe 수가 아니라 "어떤 11개 모델이 하필 unseen으로 빠졌는가"에서 나옴.

결과: `local_descriptors/llmrouterbench-analysis/probe_scale_sweep_allseen_unseen_results.json`

**실수 기록(경로 버그 재발)**: `build_ceiling_probesweep.py`를 `scripts/llmrouterbench/` 안에서 실행해 FP가 `scripts/llmrouterbench/local_descriptors/...`에 잘못 생성됨(§24 이전에도 반복된 동일 패턴) — 레포 루트로 `mv`해서 해결. 항상 레포 루트에서 실행할 것.

### 28.2 EmbedLLM 192-probe fairness — Unseen (기존 §1.10 All-seen의 짝, 이번 세션 신규)

기존에 All-seen만 있었던 CSCR 예산(192) 매칭 fairness 포인트를 Unseen에도 채움. `scripts/embedllm_uniform_nocompress_192_fairness_unseen_multiseed.py`(FP는 §1.10과 동일한 `embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-192` 재사용).

| seed | AUDC | Peak |
|---|---|---|
| 0 | 0.5141 | 0.5540 |
| 1 | 0.5373 | 0.5637 |
| 2 | 0.5240 | 0.5640 |
| **평균** | **0.5251 (std 0.0095)** | 0.5606 |

3/3 시드 모두 CSCR-unseen(0.4848) 이김(+8.3%), 1800-probe 헤드라인(0.5232)보다도 근소하게 높음 — All-seen(§1.10, +8.7%)과 함께 "CSCR 예산 그대로 맞춰도 두 프로토콜 다 이긴다"는 주장이 EmbedLLM에서 완결됨.

### 28.3 RouterBench 192-probe fairness (COMPAR 자체 probe-limited 버전 — 이 벤치마크에서 최초 실행)

RouterBench는 지금까지 Pure V2(무제한)와 "CSCR에게 1800 줬을 때" 비교만 있었고, COMPAR 자신의 probe-limited(균등 할당+무압축) fairness 숫자가 없었음. `scripts/build_routerbench_ceiling_192.py`(86개 eval_name 카테고리, target=192, 실제=172)로 FP 신규 빌드 → `scripts/routerbench_192_fairness_allseen_multiseed.py`.

| seed | AUDC | delta(vs random) | p |
|---|---|---|---|
| 0 | 0.7757 | — | — |
| 1 | 0.7773 | — | — |
| 2 | 0.7712 | +0.2698 | 0.000999 |
| **평균** | **0.7747 (std 0.0026)** | | |

3/3 시드 CSCR(0.711) 이김(+9.0%) — **기존 1800-probe 헤드라인(0.7205)보다도 높음**, RouterBench도 §1.6/1.7과 같은 "probe 늘려도 이득 없음" 패턴 재확인.

### 28.4 EmbedLLM Unseen — catfilter·min(0.3,3) 둘 다 제거한 vanilla loss로 전체 probe 스윕

"probe 수 무관"이 TAR(catfilter+min) 로스의 인공물인지, FP 자체 특성인지 분리하기 위해 vanilla MSE loss(카테고리 데모션도, top-k 정답 강조도 없음, 모든 (cos_sim, target) 쌍에 동일 가중치)로 동일한 probe 스윕을 재실행. `scripts/embedllm_nocatfilter_nominpctcap_unseen_sweep_multiseed.py`.

| target | 실제 probe | Combined(TAR) | Vanilla |
|---|---|---|---|
| 96 | 80 | 0.5200 | 0.5243 |
| 192 | 160 | 0.5251 | 0.5404 |
| 300 | 320 | 0.5345† | 0.5215 |
| 1800 | 1760 | 0.5232 | 0.5090 |
| 4000 | 4000 | 0.5248 | 0.5211 |
| 8000 | 8031 | 0.5260 | 0.5289 |
| 15000 | 14998 | 0.5298 | 0.5263 |
| 25000 | 24999 | 0.5330 | 0.5362 |
| V2(전체) | 29673 | 0.5299 | 0.5372 |

† 300은 §1.8에서 이상치로 확인됨.

**평균 성능은 승자가 지점마다 바뀜(노이즈 범위)** — vanilla가 오히려 96/192/8000/25000/V2-full에서 더 높음. 로스 종류와 무관하게 "평평함" 자체는 그대로 유지 — probe-count 무관성은 FP의 특성이지 TAR 로스의 인공물이 아님.

**다만 지점 간 분산은 뚜렷이 다름**(300 이상치 제외 8개 지점 기준): Combined std=0.0039, Vanilla std=0.0096 — **TAR가 평균 성능을 올리진 않지만, probe 예산에 따른 결과 변동을 ~2.4배 줄여줌.** RouterBench/LLMRouterBench Pure V2 3-way ablation(§26 근방 기존 데이터: combined≈min-only≈catfilter-only, 전부 노이즈 범위, 명확한 승자 없음)과 함께, **"TAR는 평균 AUDC를 올리는 장치가 아니라 결과의 안정성/신뢰성을 높이는 장치"**로 재정의.

**옆 워크스페이스 세션(§27.2)과의 수렴**: 그 세션은 "TAR의 후보-필터링 메커니즘이 probe 감소로 인한 FP 좌표 노이즈까지 흡수해서 분산을 줄이는 것 아닐까"라는 가설을 미착수 상태로 남겨뒀는데, 이번 세션의 위 분산 비교가 (다른 목적으로 이미 돌린 실험이지만) 그 가설과 정확히 일치하는 방향의 데이터를 제공함 — 서로 독립적으로 도달한 결론이라 신뢰도가 높음. 다만 "분산이 작다"는 가설과 일치하는 정황 증거일 뿐 직접 증명은 아님(다른 설명도 가능).

**실수 기록(결과 파일 덮어씀)**: 스모크테스트(target=96 하나만) 실행 시 원본 withcurves 스윕 스크립트의 out_path(`probe_scale_sweep_unseen_multiseed_withcurves_results.json`)를 그대로 물려받아, 기존 TAR 로스 8-포인트 스윕의 raw curve JSON을 96-포인트 단독 결과로 덮어씀. 스칼라 AUDC 값 자체는 Final_Result_Summary2.MD §1.7에 이미 기록돼 있어 소실 없지만, 그 원본 스윕의 곡선 데이터(costs_mean_curve/accs_mean_curve)는 재실행 전까진 복구 불가. 이후 vanilla 스크립트의 out_path를 `probe_scale_sweep_unseen_nocatfilter_nominpctcap_results.json`으로 분리해 재발 방지.

### 28.5 FP×Loss 2x2 그리드 완성 — RouterBench + EmbedLLM에서 "격차의 원인은 FP"를 직접 증명

기존엔 "COMPAR loss(TAR) + CSCR FP(Perplexity)가 CSCR 자체보다 나쁨"(§26.4, RouterBench 0.6170, LLMRouterBench 유사)만 확인돼 있었음 — 2x2의 대각선 반대 칸("CSCR loss + Ceiling FP")이 비어있었음. 이번 세션에 양쪽 벤치마크에서 채움.

**RouterBench** (`scripts/routerbench_ceilingfp_cscrloss_purev2_allseen_multiseed.py`, Ceiling FP=Pure V2 `routerbench-ceiling`, CSCR의 `cost_spectrum_info_nce` 그대로 사용):

| | CSCR loss | TAR loss |
|---|---|---|
| CSCR FP (Perplexity) | 0.711 (기준) | 0.6170 (크게 나쁨) |
| Ceiling FP | **0.7147±0.0012 (3/3 기준 이김, NEW)** | 0.7205 (헤드라인) |

**EmbedLLM Unseen** (`scripts/embedllm_ceilingfp_cscrloss_v2_unseen_multiseed.py`, Ceiling FP=V2 무압축 `embedllm-ceiling`):

| seed | AUDC |
|---|---|
| 0 | 0.5158 |
| 1 | 0.5255 |
| 2 | 0.5165 |
| **평균** | **0.5193 (std 0.0044)** |

3/3 시드 CSCR-unseen(0.4848) 이김. **역사적 기록과의 정합성 확인**: 메모리 파일에 남아있던 과거 실행(`embedllm_newllm_train_encoder_csinfonce.py`, PCA-5 **압축된** Ceiling FP + CSCR loss, 2 epoch 고정, holdout 조기종료 없음)의 5시드 평균은 0.468 — CSCR(0.4848)보다 **낮았음**. 오늘 같은 로스를 무압축 FP + holdout 기반 학습(현재 컨벤션)으로 재실행하니 0.468→0.5193(+0.051)로 뒤집힘. **원인은 §25에서 나중에 발견된 PCA-5 압축 손실(~0.03 AUDC)** — 그 당시엔 압축 버그를 모른 채 손상된 FP로 테스트했던 것으로 확인. "FP가 안 중요하다"는 과거의 오해가 실은 "그때 FP가 압축 때문에 훼손돼 있었다"였던 것.

**결론(양쪽 벤치마크 공통)**: 로스를 CSCR 것 그대로 써도 FP만 Ceiling으로 바꾸면 이기고(FP 단독 효과, 확인됨), FP를 CSCR 것 그대로 두고 로스만 TAR로 바꾸면 오히려 크게 나빠짐(로스 단독 효과, 기존 확인). **격차의 주된 원인은 FP 구성(Capability-Oriented Fingerprinting)이고, TAR 로스는 이미 이긴 상태 위에서 소폭 다듬어주는 역할** — §28.4의 "TAR=안정성 장치" 결론과 정확히 같은 방향.

**버그 발견+수정**: `eval_proj`/`knn_curve`(routerbench_perplexity_combined.py, RouterBench용으로 작성됨)는 `label_maps[i][best_j]`처럼 **모델 리스트 순서에 맞춘 위치 인덱싱**을 기대하는데, EmbedLLM의 `load_embedllm()`은 `{model_name: label}` **딕셔너리**를 반환함 — 그대로 넘기면 `KeyError`. EmbedLLM 스크립트에서 RouterBench의 `eval_proj`를 재사용할 땐 `[[lm.get(m, 0) for m in models] for lm in label_maps]`로 변환 필요.

### 28.6 FP 분리도(separation geometry) — All-seen도 왜 여전히 FP에 민감한가

사용자 질문: All-seen은 "카테고리→아는 모델" 암기로 어느 정도 풀리는데(노이즈 FP로도 안 무너짐, §26.2), 그런데 왜 FP를 Perplexity→Ceiling으로 바꾸는 것만으로 All-seen까지 이렇게 크게 오르나?

답: "암기 가능하다"와 "제한된 학습 예산(고정 MiniLM 백본+작은 헤드+10 epoch) 안에서 잘 수렴한다"는 다른 문제. 이전에 측정된 사실(메모리 파일, 33-모델 풀 기준) — **Perplexity FP는 모델 간 코사인 유사도가 티어 내부 +0.770, 티어 간 +0.725로 거의 구분이 안 됨**(전부 비슷한 방향), 반면 **Ceiling FP는 티어 내부 +0.367, 티어 간 −0.440으로 뚜렷이 갈라짐**. 노이즈 FP가 오히려 All-seen을 거의 안 망가뜨렸던 이유도 같은 논리로 설명됨 — 고차원 랜덤 벡터는 자연히 서로 거의 직교(=잘 분리)하므로, "실제 신호가 없어도 잘 분리돼 있으면" 암기가 쉬움. Perplexity FP는 실제 신호는 있지만 분리도가 나빠 오히려 암기조차 어려운 타겟인 것.

**정리**: All-seen은 FP 내용의 "진위"엔 둔감하지만 벡터들의 "기하학적 분리도"엔 여전히 민감함. Ceiling FP는 진짜 실력차를 직접 반영해 자연히 잘 분리되므로, All-seen(암기 쉬움)과 Unseen(일반화 잘 됨) 양쪽에 좋은 이유가 사실 같은 원인(분리도)에서 나옴 — Unseen만 "의도한 개선"이고 All-seen은 "부수 효과"가 아니라, 둘 다 같은 메커니즘의 서로 다른 표현.

### 28.7 최종 서사 정리 (사용자 확정)

- **1800-probe 헤드라인은 "최적값"이 아니라 "여러 분석 간 비교 가능성을 위한 고정 실험 조건"으로 프레이밍** — 실제 최적 판단 근거는 §1.6/1.7/28.1의 스케일링 스윕(probe 수 거의 무관) 쪽.
- **TAR(catfilter+min(0.3,3))는 평균 AUDC를 올리는 장치가 아니라 안정성(§28.4 분산, §27.2 collapse 방향성)을 높이는 장치**로 재정의 — vanilla loss는 이 재정의를 뒷받침하는 진단 도구로만 쓰고 최종 발표 서사에선 언급하지 않기로 함(사용자 결정).
- **FP 구성(Capability-Oriented Fingerprinting)이 CSCR 대비 성능 격차의 주된 원인**이라는 게 RouterBench+EmbedLLM 양쪽에서 완성된 FP×Loss 2x2 그리드로 직접 증명됨(§28.5).
- 다음 단계: 발표 자료 준비 단계로 전환(사용자 확인).

## 29. Model-count 스케일링 스윕 + 카테고리 없는 상황 대응(Adaptive-Clustering FP) (2026-08-19 밤 ~ 2026-08-20)

### 29.1 Model-count 스윕, 1차 시도 (probe-count 스윕의 대칭축 — 이번엔 확인 결과 설계 결함 있었음)

사용자 제안: probe 수 대신 **모델 풀 크기**를 스윕하면 All-seen AUDC가 어떻게 변하는지 확인. 사전 예측(사용자+Claude 합의): probe 스윕처럼 로그형으로 오르다 어느 지점에서 포화될 것 — 근거는 §7의 FP 분리도 메커니즘(모델이 많아질수록 같은 실력 스펙트럼에 더 촘촘히 몰려 구분이 어려워짐).

`scripts/embedllm_modelcount_sweep_allseen_multiseed.py`: model_count ∈ {5,10,20,30,50,75,111}, 각 (count, seed)에서 전체 111개 중 랜덤 서브샘플, probe 예산은 1800 고정, 헤드라인 파이프라인(TAR loss) 그대로.

**1차 결과 (3시드)**:

| n_models | AUDC(평균±std) |
|---|---|
| 5 | 0.4880±0.0199 |
| 10 | 0.5552±0.0272 |
| 20 | 0.5121±0.0114 |
| 30 | 0.5433±0.0159 |
| 50 | 0.5702±0.0198 |
| 75 | **0.5830±0.0058 (최고점)** |
| 111 | 0.5786±0.0042 |

예측대로 로그형으로 올라 75에서 정점(111 전체보다도 높음) — probe 축과 같은 "포화" 패턴 재확인. 다만 **n=20이 이웃 지점(10, 30)보다 뚜렷이 낮은 dip**(3/3 시드 모두 CSCR 이하) — 이상 현상으로 판단, 추가 시드 5개(3,4,5,6,7) 투입해서 재확인.

**추가 5시드 결과**: n=20의 dip이 **재현됨**(8시드 합산: n=5 0.4749, n=10 0.5359, n=20 **0.5213**, n=30 0.5577) — §1.8의 300-probe 스파이크(재분배하니 소멸)와 반대로, 이번엔 독립 시행을 늘려도 사라지지 않음 → **진짜 재현 가능한 로컬 dip**으로 잠정 결론.

### 29.2 설계 결함 발견 (사용자 지적) — probe 선택이 모델 서브셋에 얽혀있었음

사용자 질문: "n=20이 이상한 게 probe set 때문 아닐까? 시드 바뀌면 probe set도 바뀌나?" — 코드 확인 결과 **맞음**: `build_probe_sampled_fp`가 probe를 고를 때 쓰는 분산(variance)이 **그 시드에서 뽑힌 모델 서브셋만으로 계산됨** — 즉 "어떤 모델이 뽑히는지"가 "어떤 probe가 선택되는지"까지 오염시키고 있었음. "모델 수만 격리된 변수"라는 전제가 깨져있었던 것.

**수정**: `scripts/embedllm_modelcount_sweep_fixedprobes_allseen_multiseed.py` — probe 선택을 **전체 111개 모델 기준으로 딱 한 번만** 수행(헤드라인과 동일한 고정 probe set), 이후 각 (count, seed)에서는 "그 고정된 probe들 중 샘플링된 모델들의 데이터만 집계"하는 방식으로 FP를 만듦 — 모델 수를 진짜로 유일한 변수로 격리.

n=111 seed=0으로 sanity check(0.5767, 기존 헤드라인 seed0=0.5760과 거의 일치) 통과. **8시드×7지점 전체 재실행은 사용자가 "중단하고 내일 하자"고 해서 스모크테스트만 완료, 본 실행은 다음 세션 과제로 남음.**

### 29.3 카테고리 없는 상황 대응 — Adaptive Clustering(K-Means) 기반 Ceiling FP

**동기**: 사용자가 밤에 위화감을 느낌 — 지금까지 세 벤치마크(EmbedLLM/RouterBench/LLMRouterBench) 전부 우연히 **카테고리가 메타데이터로 이미 주어져 있었음**. 실제 배포 상황에서 카테고리 라벨이 아예 없으면 Ceiling FP 구성 자체가 불가능한 것 아닌가 하는 걱정. 이어서 "이게 UMR 논문(CSCR이 baseline으로 이긴 클러스터링 기반 footprint 방법)이랑 너무 비슷해지는 거 아니냐"는 novelty 우려도 제기 — 논의 결과, COMPAR의 기여는 "클러스터링해서 footprint 만든다"는 아이디어 자체가 아니라 그 위의 capability 중심 구성 + TAR 로스 + 검증이므로, 카테고리 소스(사람 라벨 vs 자동 클러스터)를 바꾸는 건 핵심 기여를 훼손하지 않는다는 결론.

**설계**: `scripts/embedllm_kmeans_category_fp_allseen_unseen_multiseed.py` — `df["category"]` 컬럼을 **통째로 KMeans(K=80, MiniLM 임베딩 기준) 클러스터 라벨로 덮어씀**(K=80은 실제 카테고리 수와 맞춰 공정 비교). 이후 균등 할당·probe 선택·catfilter·FP 구성·TAR 학습 로직은 **코드 변경 없이 그대로 재사용**(전부 `category` 컬럼 값에 무관하게 동작하도록 이미 짜여 있었음). 시드마다 KMeans의 `random_state`도 다르게 줘서 클러스터링 자체도 시드별로 변동. All-seen은 전체 모델, Unseen은 기존 `newllm_split_seed{N}.json` 재사용.

**결과 (3시드, 1800-probe 예산, TAR loss)**:

| | K-Means(K=80) 평균 | 진짜 카테고리 헤드라인 | CSCR |
|---|---|---|---|
| All-seen | 0.5578±0.0050 (seed별 0.5511/0.5595/0.5629) | 0.5787 | 0.541 |
| Unseen | 0.5253±0.0044 (seed별 0.5292/0.5274/0.5192) | 0.5232 | 0.4848 |

**해석**: 카테고리 라벨이 전혀 없어도 두 프로토콜 다 CSCR을 3/3 이김. All-seen만 헤드라인 대비 뚜렷한 손실(−0.021)이 있고 Unseen은 사실상 무손실(오히려 근소 우위). §7의 분리도 메커니즘으로 설명됨 — K-Means는 "텍스트 임베딩 유사도" 축으로 묶는데, 이건 "모델 실력이 갈리는 축"과 완전히 일치하지 않음(주제는 같아도 난이도·요구 스킬은 다를 수 있음). All-seen은 이 분리 정밀도에 민감(§7)해서 손실이 나타나지만, Unseen은 §4의 카테고리 셔플 실험에서 이미 확인했듯 "정밀한 정렬"보다 "일관된 파티션 + 진짜 신호"만 있으면 충분해서 거의 무손실.

**카테고리 믹스업(§4)과의 차이 (사용자 질문, 명확화)**: 셔플은 진짜 카테고리 기준으로 이미 선택된 probe들의 **라벨(소속)만 재배치**(선택 과정 자체는 불변) — 반면 K-Means는 **그루핑 자체를 다른 신호(텍스트 유사도)로 새로 만들고 그 안에서 probe를 다시 선택**함. 셔플은 "이미 최적으로 뽑힌 probe들이 어느 축에 붙는지"만 흔드는 보수적 개입이고, K-Means는 "애초에 어떤 probe가 대표성 있는지"부터 다시 정하는 더 공격적인 개입 — 그래서 셔플보다 손실이 더 큰 게 자연스러움. 둘은 모순이 아니라 서로 다른 강도의 개입을 테스트한 것.

**다음 방향(미착수)**: 순수 텍스트 임베딩 클러스터링보다 "모델 간 정답 분산"까지 반영한 반복적/적응적 클러스터링, 또는 LLM에게 쿼리의 요구 스킬/난이도를 직접 분류시키는 방식이 K-Means보다 헤드라인에 더 가까이 붙을 가능성 — Appendix 발표 자료용 소재로 채택(사용자 결정).

결과 파일: `local_descriptors/embedllm-analysis/modelcount_sweep_allseen_results.json`, `modelcount_sweep_allseen_extraseeds_results.json`, `kmeans_category_fp_allseen_unseen_results.json`

## 30. 발표 자료 구성안 정리 (2026-08-20)

메인 발표 9단계(동기→방법론→헤드라인→Fairness→스케일링→FP×Loss 2x2→분리도 메커니즘→TAR의 역할→한계) + Appendix 항목(negative control, ablation 세부, collapse 진단, 300-probe 조사, 논문 대조, K-Means) 구조로 정리 — 전체 내용은 별도 파일 `Presentation_Outline.md`에 저장(주말 자료 제작용).

**미결정 항목 2개**: (1) Mixed-pool 한계(unseen_only 쿼리 0% 적중, §26.7) 메인/Appendix/비공개 중 어디로 갈지, (2) Model-count 스윕(§29, n=20 dip 미해결·fixed-probes 본 실행 미완료) 이번 발표에 넣을지 다음으로 미룰지 — `Presentation_Outline.md` 하단에 플래그해둠.

## 31. Catfilter/min(0.3,3) 메커니즘 직접 진단 — outlier-drag vs query-mislanding (2026-08-20)

**동기**: min(0.3,3)이 진짜 outlier-drag를 잡기 위한 설계였는지, 무압축(80차원) 전환 후에도 그 근거가 여전히 유효한지 불확실해짐(§25 압축 제거 이후 재검증한 적 없었음). "catfilter=outlier-drag 개선, min=query-mislanding 개선"이라는 역할 분담 가설을 직접 측정.

`scripts/embedllm_outlierdrag_mislanding_4way_check.py` — §27.2와 동일한 4-way(vanilla/catfilter만/min만/combined) 학습 셋업 재사용, sim 행렬에서 두 지표를 사후 계산(추가 forward pass 없음): (1) outlier-drag corr = spearman(정답 후보 spread, 착지점의 최근접 실제 모델까지 거리) — 원래 outlier-drag 진단(`embedllm_outlier_blend_check.py`, 압축 시절)과 유사하지만 **타겟 구성 단계가 아니라 학습 완료 후 실제 착지점**을 재는 다른 지표임(직접 비교 아님, 주의). (2) mislanding_rate = 정답 후보가 2개 이상인 쿼리 중, 착지한 모델이 (true_acc 기준) 최선이 아닌 비율.

**결과 (seed=0, 2647개 multi-positive 쿼리)**:

| variant | outlier_drag_corr | p | mislanding_rate |
|---|---|---|---|
| vanilla | −0.1492 | <0.0001 | 0.668(최악) |
| catfilter | −0.0928 | <0.0001 | 0.532 |
| minloss | +0.0134 | 0.49(무의미) | 0.541 |
| combined | −0.0403 | 0.038 | 0.544 |

**해석**: (1) outlier-drag corr가 전 variant에서 음수/무의미 — "spread↑→drag↑"라는 원 가설과 반대 방향. 다만 이 지표가 원래 진단(타겟 구성 시점)과 다른 것(학습 후 실제 착지점)을 재고 있어 직접 반박은 아님, 애매하게 남음. (2) mislanding_rate는 **catfilter만·min만이 거의 동일하게(0.53대) vanilla(0.668) 대비 개선되고, 둘을 합쳐도(combined 0.544) 추가 이득이 없음** — "각자 다른 문제를 전담한다"는 원래 서사와 다르게, 사실상 겹치는 효과.

**최종 결론(사용자 확정)**: catfilter/min을 "outlier-drag 전담/mislanding 전담"으로 깔끔히 나누는 서사는 오늘 증거로 뒷받침되지 않음 — 대신 §28.4(분산 2.4배 감소)와 오늘 결과(mislanding 중복 개선)를 합쳐 **"둘 다 probe/FP 저해상도로 인한 타겟 노이즈를 줄이는, 방식만 다른 동일 계열의 개입"**으로 재정의. 시간 제약상 새 휴리스틱 탐색은 보류, TAR는 현행 그대로 유지 — 정확한 메커니즘 분해보다 "노이즈 저감 장치"라는 느슨하지만 정직한 설명으로 발표.

결과 파일: `local_descriptors/embedllm-analysis/outlierdrag_mislanding_4way_check_seed0.json`

## 32. 발표용 결과표 8종 제작 + LLMRouterBench Unseen CSCR 기준선 신규 구축 (2026-08-21~24)

**동기**: 고려대 세미나 최종 발표(`COMPAR Final.pdf`)에 넣을 시각 자료를 기존 "Table 디자인 정책"(진남색 헤더 + 흰 볼드 텍스트, 본문은 항상 흰 배경·무강조, [[feedback_plain_white_tables]])을 그대로 따라 하나씩 제작. 표마다 스크립트 → PNG 1:1 매핑.

| 표/차트 | 스크립트 | 산출물 | 내용 |
|---|---|---|---|
| 헤드라인 결과표 | `scripts/plot_table_headline.py` | `local_descriptors/embedllm-analysis/table_headline.png` | EmbedLLM All-seen/Unseen(1800·Full) + RouterBench All-seen(Full) + LLMRouterBench All-seen/Unseen(Full), 총 7행. vs CSCR 마진 전부 명시 |
| Fairness 표 | `scripts/plot_table_fairness.py` | `local_descriptors/embedllm-analysis/table_fairness.png` | COMPAR을 CSCR 논문 예산(192-probe)으로 축소해도 이기는지 + CSCR을 COMPAR 예산(1800-probe)으로 늘려도 안 느는지, 역방향 fairness 행 포함 |
| FP×Loss 2x2 그리드 (RouterBench) | `scripts/plot_table_2x2grid.py` | `local_descriptors/embedllm-analysis/table_2x2_routerbench.png` | CSCR-FP/Ceiling-FP × CSCR-loss/TAR-loss 인과 분해 |
| FP×Loss 2x2 그리드 (EmbedLLM) | `scripts/plot_table_2x2grid.py` | `local_descriptors/embedllm-analysis/table_2x2_embedllm.png` | 동일 구조, EmbedLLM은 Perplexity 데이터가 없어 CSCR-FP 열이 구조적으로 N/A |
| 전 조건 마진 요약 (막대, 초판) | `scripts/plot_margin_summary.py` | `local_descriptors/embedllm-analysis/margin_summary.png` | COMPAR-vs-CSCR 마진 10개 조건 내림차순 — 이후 §아래 사유로 side-by-side로 대체 |
| COMPAR vs CSCR 나란히 비교 (최종) | `scripts/plot_sidebyside_summary.py` | `local_descriptors/embedllm-analysis/sidebyside_summary.png` | 마진(%)만 보여주면 LLMRouterBench Unseen의 +26.3%가 시드 편차 때문인지 실제 효과인지 안 보인다는 지적(사용자) → 실측 AUDC 값을 COMPAR(남색)/CSCR(빨강) 나란히 + std 에러바로 교체, 큰 편차가 있는 조건은 에러바가 겹치는 걸 그대로 노출 |
| Appendix — Collapse 진단 | `scripts/plot_table_collapse.py` | `local_descriptors/embedllm-analysis/table_collapse.png` | §27.2 4-way 붕괴 실험 + §31 outlier-drag/mislanding 진단을 합친 표 |
| Appendix — K-Means FP | `scripts/plot_table_kmeans.py` | `local_descriptors/embedllm-analysis/table_kmeans.png` | §29 카테고리 라벨 없는 상황 대응(K=80) vs 실제 카테고리 vs CSCR |
| Appendix — Negative Control | `scripts/plot_table_negcontrol.py` | `local_descriptors/embedllm-analysis/table_negcontrol.png` | 랜덤 probe/노이즈 FP/셔플 FP 대조군. **최종 포함 여부 미확정** — 사용자가 직접 판단하기로 함 |

**LLMRouterBench Unseen CSCR 기준선 신규 구축**: 헤드라인 표의 LLMRouterBench Unseen 행에 비교할 CSCR 재현값이 그동안 없었음(All-seen만 있었음, §28). `scripts/llmrouterbench/cscrfp_cscrloss_unseen_multiseed.py`를 새로 작성 — `fair_probe900_multiseed.py`의 `make_split`/`build_rows_with_dataset`/`build_cost_dict`/`build_setB_eval`/`build_items`와 `routerbench_fair_probe1800_multiseed.py`의 `precompute_cls`/`train_cscr_fast`/`eval_proj`를 재사용해 CSCR-loss + Perplexity-FP를 Unseen 프로토콜로 3시드 재현.

결과: seed0=0.5831, seed1=0.5751, seed2=0.4375 → **mean=0.5319 ± 0.0669** (시드 편차 큼 — 어느 11개 모델이 unseen으로 빠지느냐에 따라 크게 갈림). 이 값이 헤드라인 표의 LLMRouterBench Unseen 행 "vs CSCR +26.3%*" 마진의 기준선이 됨 (각주로 "공식 논문 기준선 아님, 900-probe 자체 재현" 명시). 결과 파일: `local_descriptors/llmrouterbench_v15_900/cscrfp_cscrloss_unseen_results.json`.

## 33. Deferral Curve 최종본 재정비 — Probe-count / Model-pool-size(Unseen+All-seen) sweep (2026-08-21~24)

**동기**: 기존 폴더에 있던 deferral curve 이미지들(`unseen_probe_sweep_deferral_curves.png`, `probe_scale_sweep_uniform_deferral_curves.png` 등)을 다시 열어보니 (1) 폐기된 PCA-5 압축 라인이 섞여 있거나 (2) 범례의 AUDC 값이 현재 확정된 공식 표 수치와 어긋남(예: 구버전 1760-probe=0.547 vs 현재 헤드라인 1800-probe=0.5787) — §25 무압축 전환 이전의 낡은 산출물로 판단, 재사용 대신 현재 데이터로 전부 재실행하기로 함.

**Probe-count sweep (Unseen)**: `scripts/embedllm_probe_scale_sweep_unseen_multiseed_withcurves.py`의 POINTS 리스트를 정리(폐기된 PCA-5 포인트 제거, 192-probe 포인트 추가)하고 재실행 — 96/192/300/1800/4000/8000/15000/25000/V2-full(29673) 9개 포인트, 3시드. 이전 세션에서 스모크 테스트가 같은 출력 파일명을 덮어써 96-probe만 남았던 문제(사고)를 이번엔 깨끗한 재실행으로 해결. AUDC가 공식 표(1800-probe=0.5232 등)와 정확히 일치함을 확인. 결과: `local_descriptors/embedllm-analysis/probe_scale_sweep_unseen_multiseed_withcurves_results.json` → `scripts/plot_deferral_probecount_unseen.py` → `local_descriptors/embedllm-analysis/deferral_probecount_unseen_FINAL.png`.

**Model-pool-size sweep (Unseen + All-seen, 커브 포함)**: 기존 §29 스윕에는 커브 데이터가 저장되어 있지 않아 신규 스크립트 작성.
- `scripts/embedllm_modelcount_sweep_unseen_withcurves_multiseed.py` — fixed-probe 설계(probe 선택은 전체 111개 모델 분산으로 1회 고정, FP를 받는 모델 집합만 변화), MODEL_COUNTS=[15,30,50,75,111], SEEDS=[0,1,2], seen(2/3)/unseen(1/3) 분할.
  결과: n=15 0.4752±0.047 / n=30 0.4942±0.005 / n=50 0.5160±0.017 / n=75 0.5116±0.009 / n=111 **0.5215±0.003** — 111에서 계속 상승.
- `scripts/embedllm_modelcount_sweep_allseen_withcurves_multiseed.py` — 동일 MODEL_COUNTS/SEEDS로 All-seen 버전(§29 결과 재확인 + 커브 저장).
  결과: n=15 0.5141±0.038 / n=30 0.5409±0.015 / n=50 0.5662±0.016 / n=75 **0.5833±0.007(피크)** / n=111 0.5793±0.003(하락) — §29의 "75에서 피크, 111에서 하락" 패턴 재확인.
- 결과 파일: `local_descriptors/embedllm-analysis/modelcount_sweep_unseen_withcurves_results.json`, `.../modelcount_sweep_allseen_withcurves_results.json`
- 플롯: `scripts/plot_deferral_modelcount_unseen.py` → `local_descriptors/embedllm-analysis/deferral_modelcount_unseen_FINAL.png` (Unseen 단독), `scripts/plot_deferral_modelcount_sidebyside.py` → `local_descriptors/embedllm-analysis/deferral_modelcount_sidebyside_FINAL.png` (All-seen vs Unseen 나란히 — 75-line이 All-seen에서만 111-line 위에 있는 걸 한눈에 보여주는 핵심 비주얼)

**해석 논의**: 사용자가 처음엔 "모델 풀이 넓을수록 정확도가 오르는 건 당연하다(선택지가 많아지니까), 가치 있는 관찰이 아니다"라고 지적 → All-seen 스윕이 정확히 반례(75에서 피크, 111에서 하락 — 선택지가 늘어도 안 좋아짐, §7 분리도-기하 논리와 동일 메커니즘)라는 걸 근거로 반박, Unseen의 단조 상승은 "선택지 증가"가 아니라 "학습 데이터 다양성 증가로 인한 판별력 향상"이라는 실질적 발견임을 설명 → 사용자 수긍("아 그렇군.. 이해했어"). CSCR과의 스케일링 비교는 EmbedLLM에 Perplexity FP 자체가 없어 구조적으로 불가능함을 설명, 사용자도 추가 요청 없이 수락.

## 34. Min(0.3,3) "raw-distance" 대안 설계 구현 및 검증 (2026-08-21~22)

**동기**: §22에서 확정한 Min(0.3,3)은 "각 모델 자신의 Z-score 타겟에 가장 가깝게(오차 최소) 맞춘 top-k"를 고르는데, 사용자가 원래 의도했던 설계는 "정답 라벨 모델들 중 쿼리 임베딩과 실제로 가장 가까운(원거리 최소) top-k"였음 — 두 설계가 다르다는 걸 뒤늦게(발표 직전) 발견. 이진(0/1) 라벨 체계에서는 같은 정답(positive) 모델들이 전부 동일한 Z-score 타겟을 공유하므로, "타겟-오차 최소"와 "원거리 최소"가 유사(undershoot 영역에서는 사실상 동일, overshoot 영역에서만 갈림)하다는 걸 사용자가 직접 수식으로 재도출 — 그래도 실제로 성능 차이가 나는지 빠르게 실증 검증하기로 함("한번 구현 바꿔서 다시 해볼 수 있을까? 빠르게").

`scripts/embedllm_minbydistance_1800_unseen_multiseed.py` — 기존 vanilla-loss unseen sweep 스크립트를 복사, catfilter의 카테고리 트랙레코드 강등 로직을 `build_items`에 복원, POINTS를 `uncompressed-1800` 하나로 축소, `minpos_bydistance_loss` 함수로 Min 선택 기준을 교체(정답 후보 중 raw cosine similarity 상위 top-k로 선택, 오차가 아니라 거리 기준). EmbedLLM Unseen 1800-probe 헤드라인 조건에서 3시드 실행.

**결과**: raw-distance 방식 mean=**0.5257±0.0071** (seed 0.5312/0.5302/0.5156) vs 기존 error-to-target 방식 mean=0.5232±0.0053 (seed 0.5261/0.5278/0.5158) — 차이 +0.0025, 노이즈 범위 내. 수학적 예측(undershoot 영역이 실전에서 지배적)과 정확히 부합.

**최종 결정(사용자)**: 시간 제약상("다시 다 돌리려면 한참 걸릴텐데") 전면 재검증/전환은 하지 않고 현행 구현(error-to-target) 유지 — "일단 두고보자구"로 마무리, 디펜스 질문 대비용으로 두 설계가 이진 라벨 하에서 수학적으로 거의 동치임을 설명할 수 있는 근거만 확보. 결과 파일: `local_descriptors/embedllm-analysis/minbydistance_1800_unseen_results.json`.

## 35. 최종 발표자료(COMPAR Final.pdf) 오탈자 검토 (2026-08-24)

`C:\Users\user\Downloads\COMPAR Final.pdf`(27페이지) 전체를 PyMuPDF로 페이지별 PNG 렌더링(poppler 미설치로 인한 우회) 후 육안 검토. 발견한 오탈자/이슈 8건 — 텍스트 오타(p.1 "Mulit-Positive"→"Multi-Positive", p.22 "Limitaition"→"Limitation", p.11 "per queries"→"per query"), CSCR 대소문자 불일치(p.7, p.16 "cscr"→"CSCR"), 표기 불일치(p.15 "Embed LLM"→"EmbedLLM"), 이미지 결함(p.16 우측 차트 범례 "C" 글자 잘림, p.20 하단에 미완성 텍스트 조각 "0.475 → 0.494 → 0.516 → 0.522," 방치 — 75-모델 값(~0.512) 누락). 데이터/수치는 전 페이지에서 §32~34의 확정 결과와 정확히 일치, 사실 오류는 없음. **사용자가 전부 직접 수정 완료.**

발표는 로컬 노트북을 들고 발표 장소로 이동해 진행 예정 — §32~34에서 만든 이미지 산출물은 전부 `local_descriptors/embedllm-analysis/`(+ LLMRouterBench 기준선은 `local_descriptors/llmrouterbench_v15_900/`)에 있고, 위 표에 스크립트→산출물 매핑을 정리해 둠.
