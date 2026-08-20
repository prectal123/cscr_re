# COMPAR 전체 결과 레퍼런스 테이블

`Final_Result_Summary2.MD`(발표용 서술) + `PROGRESS.md` §28(오늘 추가분)의 모든 숫자를 실험 단위로 재정리. 각 행 = [숫자] + [이게 뭘 보여주는 실험인지 한 줄 해석] + [재현용 스크립트 경로].

CSCR 논문 기준값: **EmbedLLM All-seen 0.541 / Unseen 0.4848**, **RouterBench 0.711**(LLMRouterBench는 논문에 없는 독자 벤치마크라 기준값 없음). CSCR 실제 probe 예산은 세 벤치마크 전부 **192**(논문 원문 확인, §26.6).

---

## 1. 헤드라인 (공식 파이프라인: 무압축 + 균등 할당 + TAR 로스)

| 벤치마크 | 프로토콜 | probe | AUDC (3시드 평균±std) | CSCR 대비 | 해석 | 스크립트 |
|---|---|---|---|---|---|---|
| EmbedLLM | All-seen | 1760/1800 | **0.5787 ± 0.0014** | +7.0% | 공식 헤드라인 | `embedllm_probe_scale_sweep_uniform_nocompress_multiseed.py` |
| EmbedLLM | Unseen | 1760/1800 | **0.5232 ± 0.0053** | +7.9% | 공식 헤드라인 | `embedllm_probe_scale_sweep_unseen_multiseed.py` |
| EmbedLLM | All-seen | 전체(V2, 29673) | 0.5867 ± 0.0030 | +8.5% | probe 무제한 상한값 — 헤드라인과 거의 차이 없음(probe 수 무관 증거) | 없음(기존 FP `embedllm-ceiling` 재사용) |
| EmbedLLM | Unseen | 전체(V2, 29673) | 0.5299 ± 0.0080 | +9.3% | 상동 | 상동 |
| RouterBench | All-seen | 전체(V2, Unseen 구조상 불가) | **0.7205 ± 0.0013** | +1.3% | 11개 모델뿐이라 probe 제한이 원래 없음(V2=헤드라인) | `routerbench_purev2_ablation3way_allseen_multiseed.py` |
| LLMRouterBench | All-seen | 전체(V2, 2345) | **0.7349 ± 0.0061** | 기준값 없음(독자 pool) | CSCR 실제 loss 재현치(0.6712)는 상회 | `llmrouterbench/purev2_ablation3way_allseen_unseen_multiseed.py` |
| LLMRouterBench | Unseen | 전체(V2, 2345) | **0.6719 ± 0.0357** | 기준값 없음 | 33개 모델 pool 최초 Unseen 테스트 | 상동 |

---

## 2. Fairness — CSCR 자기 예산(192)에 정확히 맞췄을 때도 이기는가

| 벤치마크 | 프로토콜 | 실제 probe | AUDC | CSCR 대비 | 해석 | 스크립트 |
|---|---|---|---|---|---|---|
| EmbedLLM | All-seen | 160 | **0.5883 ± 0.0049** | +8.7% | 1800-probe보다 오히려 근소하게 높음 | `embedllm_uniform_nocompress_192_fairness_allseen_multiseed.py` |
| EmbedLLM | Unseen | 160 | **0.5251 ± 0.0095** | +8.3% | 1800(0.5232)과 사실상 동일 | `embedllm_uniform_nocompress_192_fairness_unseen_multiseed.py` (오늘 신규) |
| RouterBench | All-seen | 172 | **0.7747 ± 0.0026** | +9.0% | **1800-probe 헤드라인(0.7205)보다 오히려 높음** | `build_routerbench_ceiling_192.py` + `routerbench_192_fairness_allseen_multiseed.py` (오늘 신규) |

**해석**: 세 칸 다 CSCR의 실제 예산(192)만으로 이기고, 그중 2곳은 오히려 더 큰 예산(1800)보다 잘 나옴 — "COMPAR가 더 많은 probe를 써서 유리했다"는 가설을 완전히 기각.

---

## 3. Probe-count 스케일링 스윕 (probe 수가 성능에 영향을 주는가)

### EmbedLLM (80개 카테고리, TAR 로스)

| target | 실제 probe | All-seen | Unseen |
|---|---|---|---|
| 96 | 80 | 0.5798 | 0.5200 |
| 300† | 320 | 0.6004 | 0.5345 |
| 800 | 800 | 0.5909 | – |
| 1200 | 1200 | 0.5833 | – |
| 1800 | 1760 | 0.5787 | 0.5232 |
| 4000 | 4000 | 0.5852 | 0.5248 |
| 8000 | 8031 | 0.5793 | 0.5260 |
| 15000 | 14998 | 0.5847 | 0.5298 |
| 25000 | 24999 | 0.5867 | 0.5330 |
| V2(전체) | 29673 | 0.5867 | 0.5299 |

† 300 지점은 이상치 확인됨(§4 참고) — scalability 근거로 사용 금지.

**해석**: 80개부터 29673개(370배)까지 늘려도 All-seen은 사실상 무관(0.579~0.590), Unseen은 약하지만 유의미한 완만한 상승(Pearson r=0.97, 효과 크기는 작음, +0.013). "probe 많이 쓸수록 좋다"는 통념과 다름.

스크립트: `embedllm_probe_scale_sweep_wide_multiseed.py`, `embedllm_probe_scale_sweep_uniform_nocompress_multiseed.py`, `embedllm_probe_scale_sweep_uniform_nocompress_lowrange_multiseed.py`, `embedllm_probe_scale_sweep_unseen_multiseed_withcurves.py`

### LLMRouterBench (8개 카테고리, TAR 로스) — 오늘 신규

| target | 실제 probe | All-seen | Unseen |
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

**해석**: EmbedLLM보다 더 즉각적으로 평평함 — 카테고리가 8개뿐이라 최소 할당만으로도 이미 충분. 랜덤 라우팅 대비 delta는 항상 +0.17~+0.39, p=0.000999(모든 지점에서 유의미) — "무관함"이 "안 됨"을 뜻하지 않음. 다만 seed=0의 unseen 분할(어떤 11개 모델이 빠지는지)이 모든 probe 지점에서 일관되게 가장 어려움 — 분산의 원인은 probe 수가 아니라 분할 구성.

스크립트: `llmrouterbench/build_ceiling_probesweep.py`, `llmrouterbench/probe_scale_sweep_allseen_unseen_multiseed.py`

---

## 4. Negative Controls (진짜 신호가 필요한지 검증)

| 실험 | 벤치마크/프로토콜 | AUDC | 실제 FP 대비 | 해석 | 스크립트 |
|---|---|---|---|---|---|
| Probe 선택: top-variance vs 완전 무작위 | EmbedLLM All-seen (96) | 무작위 0.5909 vs top-var 0.5798 | 무작위가 오히려 높음 | "똑똑한 선택 방식" 자체는 핵심이 아님 | `embedllm_probe96_random_selection_negcontrol_allseen_multiseed.py` |
| 순수 노이즈 FP | EmbedLLM All-seen | 0.5769±0.0049 | 실제 FP(0.5787)와 거의 동일 | All-seen은 분류 퇴화(카테고리→아는 모델 암기)로 설명 가능, FP 내용과 무관 | `embedllm_randomnoise_fp_allseen_multiseed.py` |
| 순수 노이즈 FP | EmbedLLM **Unseen** | **0.3721±0.0329** | **완전 붕괴, CSCR(0.4848)보다도 낮음** | Unseen은 학습 중 라벨을 본 적 없어 암기 불가 — 노이즈 사용 시 무너짐이 "진짜 신호가 필요함"의 직접 증거 | `embedllm_randomnoise_fp_unseen_multiseed.py` |
| 카테고리 셔플 FP(진짜 데이터, 슬롯만 재배치) | EmbedLLM All-seen | 0.5730 | 실제(0.5787)와 거의 동일 | 카테고리 의미 정렬은 핵심이 아님 | `embedllm_shuffled_category_fp_allseen_unseen_multiseed.py` |
| 카테고리 셔플 FP | EmbedLLM Unseen | 0.5261±0.0008 | 실제(0.5232)와 거의 동일 | 진짜 데이터이기만 하면 카테고리별 정교한 정렬은 불필요 | 상동 |
| Mixed-pool(seen+unseen 후보 동시 제공) bucket 분석 | EmbedLLM Unseen | unseen_only 정답 쿼리 적중률 **0/206 (0%)** | – | seen 모델이 후보에 있으면 unseen 모델은 절대 안 뽑힘 — 분류 퇴화 메커니즘의 직접적 귀결, 표준 Unseen 프로토콜이 실제로는 매우 가혹한 조건임을 보여줌 | `embedllm_unseen_mixedpool_1800_multiseed.py` |

**종합 해석**: All-seen은 FP의 "진짜 신호"에 둔감(분류 퇴화), Unseen은 매우 민감 — 다만 Unseen도 "카테고리별 정교함"엔 둔감하고 "실제 데이터인가"에만 민감. 핵심은 벡터들의 분리도(§7)와 진짜 실력 정보의 존재 여부.

*(RouterBench/LLMRouterBench엔 아직 이 negative control들 미실행 — 남은 항목)*

---

## 5. Ablation — Catfilter / min(0.3,3)이 실제로 기여하는가

### 5-1. 평균 성능 기준 (3-way ablation, Pure V2 전체 데이터)

| 벤치마크 | 프로토콜 | Combined(TAR) | min만 | catfilter만 | 해석 |
|---|---|---|---|---|---|
| RouterBench | All-seen | 0.7205 | 0.7202 | **0.7225**(최고) | 셋 다 노이즈 범위, 명확한 승자 없음 |
| LLMRouterBench | All-seen | 0.7349 | **0.7351** | 0.7317 | 상동 |
| LLMRouterBench | Unseen | 0.6719 | **0.6797** | **0.6797** | 상동 |

스크립트: `routerbench_purev2_ablation3way_allseen_multiseed.py`, `llmrouterbench/purev2_ablation3way_allseen_unseen_multiseed.py`

### 5-2. 고정 probe 예산에서 catfilter 단독 효과 (EmbedLLM All-seen)

| probe | Catfilter 있음 | Catfilter 없음 | 차이 |
|---|---|---|---|
| 1800(실제 1760) | 0.5787 | 0.5792 | +0.0005(무의미) |
| 192(실제 160) | 0.5883 | 0.5696 | **+0.0187(유의미)** |

**해석**: catfilter는 probe가 풍부할 땐 무의미, **probe가 부족할 때만 의미 있게 기여** — 희소성에 반비례.

스크립트: `embedllm_uniform_nocompress_1800_ablation_nocatfilter_allseen_multiseed.py`, `embedllm_uniform_nocompress_192_ablation_nocatfilter_allseen_multiseed.py`

### 5-3. Vanilla(catfilter+min 둘 다 제거) vs Combined, 전체 probe 스윕 — 오늘 신규

| target | 실제 probe | Combined(TAR) | Vanilla |
|---|---|---|---|
| 96 | 80 | 0.5200 | 0.5243 |
| 192 | 160 | 0.5251 | 0.5404 |
| 300† | 320 | 0.5345 | 0.5215 |
| 1800 | 1760 | 0.5232 | 0.5090 |
| 4000 | 4000 | 0.5248 | 0.5211 |
| 8000 | 8031 | 0.5260 | 0.5289 |
| 15000 | 14998 | 0.5298 | 0.5263 |
| 25000 | 24999 | 0.5330 | 0.5362 |
| V2(전체) | 29673 | 0.5299 | 0.5372 |

**표준편차(300 이상치 제외 8개 지점)**: Combined **0.0039** vs Vanilla **0.0096** — **약 2.4배 차이**.

**해석**: 평균 성능은 승자가 지점마다 바뀜(로스가 평균을 안 올림) — 하지만 Combined이 probe 예산에 따른 결과 변동을 확실히 줄여줌. **TAR의 진짜 역할은 "평균 성능 향상"이 아니라 "안정성/신뢰성".**

스크립트: `embedllm_nocatfilter_nominpctcap_unseen_sweep_multiseed.py` (오늘 신규)

### 5-4. Collapse 관점 4-way ablation (다른 세션, PROGRESS.md §27.2)

| variant | top3_share | 사용 모델 수(35개 중) | rho(선택,정답) |
|---|---|---|---|
| vanilla | 0.686(최저) | 23(최다) | 0.447 |
| catfilter만 | 0.816 | 15 | 0.515 |
| min만 | 0.760 | 13 | 0.663 |
| combined(TAR) | 0.767 | 9(최소) | **0.669**(최고) |

**해석**: TAR는 collapse의 폭을 줄이지 않고(오히려 더 좁힘) collapse가 **향하는 방향을 더 정확하게** 만듦 — "diversity ↓, accuracy-of-collapse ↑" 트레이드오프. §5-3의 "안정성" 결론과 같은 방향.

스크립트: `embedllm_uncompressed_ablation_collapse_check.py`

---

## 6. FP × Loss 2x2 그리드 — 격차의 원인이 FP인지 로스인지 (오늘 완성)

### RouterBench (All-seen, Pure V2 FP)

| | CSCR loss | TAR loss |
|---|---|---|
| CSCR FP (Perplexity) | 0.711(기준) | 0.6170±0.0006(크게 나쁨) |
| Ceiling FP | **0.7147±0.0012**(기준 이김, 오늘 신규) | 0.7205±0.0013(헤드라인) |

스크립트: 기준=논문값, CSCR loss+Ceiling FP=`routerbench_ceilingfp_cscrloss_purev2_allseen_multiseed.py`(오늘), TAR loss+CSCR FP=`routerbench_cscrfp_comparloss_allseen_multiseed.py`, 헤드라인=`routerbench_purev2_ablation3way_allseen_multiseed.py`

### EmbedLLM (Unseen, V2 FP)

| | CSCR loss | TAR loss |
|---|---|---|
| CSCR FP (Perplexity) | 0.4848(기준, FP 자체가 EmbedLLM엔 없어서 미테스트) | – |
| Ceiling FP (무압축) | **0.5193±0.0044**(기준 이김, 오늘 신규) | 0.5299±0.0080(V2 헤드라인) |
| Ceiling FP (**압축**, 과거 기록) | 0.468(기준 **이하**, 과거 오해의 원인) | – |

스크립트: CSCR loss+Ceiling FP(무압축)=`embedllm_ceilingfp_cscrloss_v2_unseen_multiseed.py`(오늘), 과거 압축 버전=`embedllm_newllm_train_encoder_csinfonce.py`

**종합 결론**: 두 벤치마크 모두 "로스를 CSCR 것 그대로 둬도 FP만 바꾸면 이긴다" / "FP를 CSCR 것 그대로 두고 로스만 바꾸면 오히려 나빠진다"는 패턴이 동일 — **격차의 주된 원인은 FP 구성(Capability-Oriented Fingerprinting)**, TAR 로스는 이미 이긴 상태 위에서 안정성을 더하는 보조 요소(§5-3/5-4와 일관).

**과거 기록과의 정합성**: EmbedLLM에서 "Ceiling FP + CSCR loss"가 예전엔(PCA-5 압축, 2 epoch 고정, 5시드 평균 0.468) CSCR을 못 이겼던 게 혼란의 원인이었는데 — 압축 손실(§25, ~0.03 AUDC)이 원인이었음을 오늘 확인. FP가 안 중요했던 게 아니라 그때 FP가 압축으로 손상돼 있었던 것.

---

## 7. 메커니즘 진단 (왜 이런 결과가 나오는가)

| 진단 | 핵심 수치 | 해석 |
|---|---|---|
| FP 벡터 분리도 (33-모델 풀 기준) | Perplexity FP 코사인 유사도: 티어 내부 +0.770, 티어 간 +0.725(거의 안 갈라짐) / Ceiling FP: 티어 내부 +0.367, 티어 간 **−0.440**(뚜렷이 갈라짐) | All-seen도 왜 FP에 민감한지 설명 — 암기(All-seen)든 일반화(Unseen)든 타겟 공간이 잘 분리돼 있어야 학습이 잘 됨. 노이즈 FP가 All-seen을 안 망가뜨린 것도 같은 논리(고차원 랜덤 벡터는 자연히 잘 분리됨) |
| 분류 퇴화(classification collapse) | 노이즈 FP All-seen 0.5769 ≈ 실제 0.5787 | All-seen은 "카테고리→아는 모델" 암기로 상당 부분 설명 가능 |
| 일반 실력 축 지배 | 카테고리 셔플에도 Unseen 거의 안 떨어짐(§4) | 좁은 도메인 전문성보다 모델 간 전반적 실력차가 이 벤치마크들의 지배적 신호 |
| Collapse 방향 교정 | rho 0.447→0.669(§5-4) | TAR는 collapse를 줄이는 게 아니라 향하는 방향을 교정 |

---

## 8. Model-count 스케일링 스윕 (모델 수가 성능에 영향을 주는가 — probe 축의 대칭)

### 8-1. 1차 시도 (probe 선택이 모델 서브셋에 얽혀있던, 설계 결함 있는 버전)

| n_models | AUDC (3시드) | AUDC (8시드 합산, 노이즈 지점만) |
|---|---|---|
| 5 | 0.4880±0.0199 | 0.4749 |
| 10 | 0.5552±0.0272 | 0.5359 |
| 20 | 0.5121±0.0114 | **0.5213 (재현되는 dip)** |
| 30 | 0.5433±0.0159 | 0.5577 |
| 50 | 0.5702±0.0198 | – |
| 75 | **0.5830±0.0058 (최고점, 111보다 높음)** | – |
| 111 | 0.5786±0.0042 | – |

**해석**: probe 스윕처럼 로그형으로 오르다 75 근방에서 포화(오히려 111보다 높음) — §7의 분리도 메커니즘(모델이 많을수록 촘촘해져 구분이 어려워짐)과 일치. **n=20의 dip은 시드 5개를 더 넣어도 재현됨**(300-probe 스파이크와 달리 소멸하지 않음) — 진짜 로컬 현상으로 잠정 결론.

**설계 결함(사용자 지적) 및 수정**: probe 선택(top-variance)이 그 시드에서 샘플링된 모델 서브셋만으로 분산을 계산해서, "모델 수"와 "probe set"이 얽혀있었음. `embedllm_modelcount_sweep_fixedprobes_allseen_multiseed.py`로 수정(probe 선택을 전체 111개 기준으로 한 번만 고정) — n=111 sanity check(0.5767 ≈ 헤드라인 seed0 0.5760) 통과, **8시드×7지점 본 실행은 미완료(다음 세션 과제)**.

스크립트: `embedllm_modelcount_sweep_allseen_multiseed.py`(1차), `embedllm_modelcount_sweep_extraseeds_allseen_multiseed.py`(추가시드), `embedllm_modelcount_sweep_fixedprobes_allseen_multiseed.py`(수정판, 미완료)

---

## 9. 카테고리 없는 상황 대응 — Adaptive Clustering(K-Means) FP (Appendix 발표 자료용)

**동기**: 지금까지 세 벤치마크 전부 카테고리가 메타데이터로 주어져 있었음 — 실제로 카테고리 라벨이 없는 데이터라면 Ceiling FP를 어떻게 만드는가? K-Means(K=80, MiniLM 임베딩 기준)로 `category` 컬럼을 자동 생성된 클러스터 라벨로 완전히 대체하고, 나머지 파이프라인(균등 할당, probe 선택, catfilter, TAR)은 코드 변경 없이 그대로 재사용.

| 프로토콜 | K-Means(K=80) 평균 | 진짜 카테고리 헤드라인 | CSCR |
|---|---|---|---|
| All-seen | 0.5578±0.0050 | 0.5787 | 0.541 |
| Unseen | 0.5253±0.0044 | 0.5232 | 0.4848 |

**해석**: 카테고리 라벨이 전혀 없어도 두 프로토콜 다 CSCR을 3/3 이김. All-seen만 헤드라인 대비 뚜렷한 손실(−0.021)이 있고 Unseen은 무손실 — K-Means는 "텍스트 유사도" 축으로 묶는데 이게 "모델 실력이 갈리는 축"과 완전히 일치하진 않음(§7의 분리도 논리). All-seen은 이 정밀도에 민감하지만 Unseen은 §4의 카테고리 셔플 실험처럼 "일관된 파티션 + 진짜 신호"만 있으면 충분해서 거의 무손실.

**카테고리 셔플(§4)과의 차이**: 셔플은 진짜 카테고리로 이미 뽑힌 probe의 **소속 라벨만 재배치**(선택 과정 불변, 보수적 개입) — K-Means는 **그루핑 자체를 다른 신호로 새로 만들고 그 안에서 probe를 다시 선택**(공격적 개입). 그래서 셔플보다 손실이 큰 게 자연스러움 — 두 결과는 모순이 아니라 서로 다른 강도의 개입.

**향후 방향**: 순수 텍스트 유사도 대신 모델 간 정답 분산까지 반영한 적응적 클러스터링, 또는 LLM에게 쿼리의 요구 스킬/난이도를 직접 분류시키는 방식이 헤드라인에 더 가까울 가능성.

스크립트: `embedllm_kmeans_category_fp_allseen_unseen_multiseed.py`

---

## 10. 아직 안 한 것 (참고용)

- RouterBench/LLMRouterBench에 negative control(노이즈 FP, 셔플 FP, mixed-pool) 미실행 — EmbedLLM에서만 확인됨
- InfoNCE numerator sum/max/softmax 비교(멘토 피드백 항목) 미구현
- Model-count 스윕 수정판(fixed-probes) 8시드×7지점 본 실행 미완료
- Adaptive Clustering FP의 개선판(모델 분산 반영 클러스터링, LLM 기반 스킬 분류) 미착수
- EmbedLLM 헤드라인 10시드 확장 미실행(현재 3시드)
- Mixed-pool 한계(unseen_only 쿼리 0% 적중)를 발표에 명시할지는 미결정
