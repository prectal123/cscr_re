# CSCR 재구현 프로젝트 — 진행 상황 & 컨텍스트

이 문서는 이 fork(`prectal123/cscr_re`)에서 진행 중인 연구/재구현 작업의 전체 맥락을 담고 있습니다.
**새로운 로컬 환경이나 새 Claude Code 세션에서 이 프로젝트를 이어갈 때는 이 파일부터 읽으면 됩니다** — Claude Code의 세션/메모리는 기기 간 동기화가 안 되기 때문에, 이 파일이 유일하게 확실한 컨텍스트 전달 수단입니다.

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

## 6. MixInstruct 모델 Pool 현황 — 최종 6개 확정

원본 11개 중 5개는 실제 시도로 확인된 이유로 제외:

| 상태 | 모델 | 비고 |
|---|---|---|
| ✅ 완료 | `eachadea/vicuna-13b-1.1` | |
| ✅ 완료 | `chavinlo/alpaca-native` | |
| ✅ 완료 | `stabilityai/stablelm-tuned-alpha-7b` | fp32 전용(31.75GB), 대체 정밀도 없음 |
| ✅ 완료 | `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` | |
| ✅ 완료 | `TheBloke/koala-7B-HF` | |
| ✅ 완료 | `google/flan-t5-xxl` | 45GB로 제일 크지만 실제로 잘 됨(~25분) |
| ❌ 제외 | `databricks/dolly-v2-12b` | HF Hub에서 저장소 자체가 사라짐/제한됨(검색에도 안 뜸), gated 아님 |
| ❌ 제외 | `mosaicml/mpt-7b-instruct` | dolly와 동일한 실패 패턴, 마찬가지로 접근 불가로 추정 |
| ❌ 제외 | `THUDM/chatglm-6b` | 2023년식 `trust_remote_code`가 현재 `transformers` 버전과 비호환 (`'property' object cannot be interpreted as an integer`) |
| ❌ 제외 | `fnlp/moss-moon-003-sft` | 위와 동일 계열 문제 (`cannot import name 'is_tf_available'`) |
| ❌ 제외 | `mosesjun0h/llama-7b-hf-baize-lora-bf16` | 저장소에 토크나이저 파일 자체가 없음(불완전한 커뮤니티 업로드) — sentencepiece 문제 아니었음 |

이 마모(model rot) 자체도 "2023년식 개인/소규모 팀 업로드 위주의 pool이라 시간이 지나며 자연 마모된 것" — 발표에 넣을 만한 관찰 포인트로 정리해둠. 6개면 vector-diversity 분석엔 충분(t-SNE 등 시각화는 표본이 적어 약할 수 있음 → RSA/Mantel test 같은 정량적 방법을 메인 근거로 삼는 게 안전).

나중에 필요하면 `experts/registry.json`의 다른 소형 모델을 **descriptor/기하학 분석용으로만**(라우팅 평가용 라벨은 없음) 추가하는 것도 가능.

---

## 7. 다음 단계 후보 (아직 미착수)

1. **당장**: Cell 9(perplexity) 재실행해서 6개 모델 perplexity descriptor까지 마저 확보 — 오늘(이 세션 기준) 목표.
2. 벡터 diversity/기하학 분석 — 단순 시각화보다 **정량적 방법 우선**:
   - MMD 또는 Classifier Two-Sample Test로 "두 descriptor 타입이 실제로 분리되는가" 검정
   - RSA(Representational Similarity Analysis)/Mantel test로 "두 descriptor가 상대적 유사도 구조에 동의하는가" 검정 (이게 라우팅과 직결된 더 중요한 질문)
   - Silhouette score로 빠른 1차 확인
3. 논문 Table 4의 "Mixed" 실험을 **다중 랜덤 시드**로 재현해서 분산 확인 (paper의 단일 시행 주장에 대한 직접 반박 근거)
4. 여유 있으면 **RouterBench 트랙** 시도 — API 전용+오픈웨이트가 실제로 섞여있는 pool이라 "진짜 forced mixing" 시나리오이자, perplexity descriptor는 모델 재다운로드 없이 계산 가능(데이터셋에 응답이 이미 있음)해서 model rot 문제도 회피됨.
5. Lexical Fingerprint 방법론 실제 구현 — 화이트박스는 기존 logit descriptor 코드 재사용 가능, 블랙박스는 반복 샘플링 로직 신규 구현 필요. **3주 발표 스코프상 완전한 검증까지는 필수 아님** — "관찰 + 제안 + 방향성"까지만 보여줘도 충분.

---

## 8. 이 문서 사용법

새 로컬/새 세션에서 시작할 때:
1. 이 리포(`prectal123/cscr_re`)를 clone
2. 이 파일을 읽고 현재 어느 단계인지 파악
3. `colab/repro_mixinstruct.py`를 열어서 실제 코드 상태 확인 (이 문서는 요약이고, 정확한 최신 코드는 그 파일이 진실)
4. 위 "7. 다음 단계"부터 이어서 진행
