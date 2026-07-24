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
| ❌ 제외(확정) | `databricks/dolly-v2-12b` | HF Hub에서 저장소 자체가 사라짐/제한됨(검색에도 안 뜸), gated 아님 |
| ❌ 제외(확정) | `mosaicml/mpt-7b-instruct` | dolly와 동일한 실패 패턴, 마찬가지로 접근 불가로 추정 |
| ❌ 제외(확정) | `fnlp/moss-moon-003-sft` | chatglm과 같은 계열 코드 비호환(`is_tf_available` import 에러) + **16B라 로컬 8GB VRAM도 Colab 무료 티어도 감당 못 함** — 코드만 고쳐도 하드웨어 벽에 막힘, 임시 보류 아니라 확정 제외 |
| ❌ 제외(확정) | `mosesjun0h/llama-7b-hf-baize-lora-bf16` | 저장소에 토크나이저 파일 자체가 없음(불완전한 커뮤니티 업로드) |

이 마모(model rot) 자체도 "2023년식 개인/소규모 팀 업로드 위주의 pool이라 시간이 지나며 자연 마모된 것" — 발표에 넣을 만한 관찰 포인트. 남은 3개 제외 사유가 전부 "코드를 고쳐도 안 되는" 근본적인 것들(저장소 소실 2개, 업로드 불완전 1개)뿐이고, "코드 호환성만의 문제"였던 건 chatglm-6b 하나뿐이라 그건 실제로 살려냈다는 서사로 정리 가능.

**Probe 개수와 TOPK 모두 192로 통일**(2026-07-24) — 이유는 아래 9번 참고.

---

## 7. 다음 단계 — 5단계 계획 진행 중 (2026-07-24 사용자 확정)

1. ✅ **완료**: Logit·perplexity descriptor 둘 다 192차원으로 7개 모델 전부 재계산 (`local_descriptors/mix-instruct-logit/`, `local_descriptors/mix-instruct-perplexity/`)
2. ✅ **완료**: 벡터 분포 분석 — 11번 섹션 참고
3. ✅ **완료**: 7개 모델 pool로 벤치마크 필터링 — 12번 섹션 참고
4. **다음**: Query encoder(MLP) 학습 — `scripts/train_query_encoder.py --dataset mix-instruct --desc_dir local_descriptors/mix-instruct-logit --pool experts/pool-mix-instruct-7.json --out_dir <출력경로>` (perplexity로 학습하려면 `--desc_dir`만 바꾸면 됨)
5. Deferral curve 실험 결과 재현 — `scripts/run_audc_eval.py` + AUDC/QNC/Peak 지표로 논문 Table 1/4 형식과 비교

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
