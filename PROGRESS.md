# CSCR 재구현 프로젝트 — 진행 상황 & 컨텍스트

이 문서는 이 fork(`prectal123/cscr_re`)에서 진행 중인 연구/재구현 작업의 전체 맥락을 담고 있습니다.
**새로운 로컬 환경이나 새 Claude Code 세션에서 이 프로젝트를 이어갈 때는 이 파일부터 읽으면 됩니다** — Claude Code의 세션/메모리는 기기 간 동기화가 안 되기 때문에, 이 파일이 유일하게 확실한 컨텍스트 전달 수단입니다.

---

## 0. TL;DR — 2026-08-14 세션 종료 시점, 다음은 여기서부터 (최신)

**가장 최근 세션 요약은 20번 섹션(EmbedLLM 규모 "new LLMs" 프로토콜로 논문 Table 2와 직접 비교, GRPO 회귀 방식 검증) 참고 — 이 세션은 커밋만 되고 PROGRESS.md 서술이 누락돼 있던 걸 2026-08-15에 뒤늦게 정리함.** 그 이전은 19번 섹션(V1.3 LLM-judge 시도 → probe-count ablation 진단 → domain+difficulty 이중 라우팅 설계) 참고. 17번 섹션(LOO unseen-model 실험) 이전, EmbedLLM pool 규격화/v1.2 프로토타입은 18번 섹션 참고(별도 세션에서 병행 진행됨).

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
