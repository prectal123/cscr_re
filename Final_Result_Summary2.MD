# Final Result Summary 2 — 무압축(No PCA-5) + 균등 할당(Uniform Allocation) 파이프라인

`Final_result_Summary.MD`(이하 v1)는 EmbedLLM Ceiling FP를 PCA-5로 압축하고, 카테고리별 probe 예산을 PCA 중요도로 가중 배분하는 파이프라인 기준이었다. 이번 세션에서 **PCA-5 압축이 성능을 깎아먹는 병목**이라는 사실을 발견했고(§1.4~1.5), 이를 All-seen·Unseen 양쪽에서, 그리고 독립적인 재분배(resplit) 테스트로도 재확인했다(§1.8). 이 문서는 **압축 없음 + 균등 할당**을 새 기준으로 삼아 EmbedLLM 결과를 다시 채운 것이다.

**RouterBench·LLMRouterBench는 이번 세션에서 무압축 파이프라인으로 재검증하지 않았다** — 두 벤치마크는 v1의 기존(압축) 수치가 여전히 현재 유효한 참고값이다(§2, §3 참고).

해석은 최소화하고 설정·수치 위주로 정리.

---

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 방법론 이름 | COMPAR (Capability-Oriented Multi-Positive Adaptive Routing) / TAR (Trimmed Advantage Regression, loss 이름) |
| min-loss 공식 | $k_{eff} = \mathrm{clip}(\lceil 0.3 \times n_{pos}\rceil,\ 1,\ 3)$, 상위 $k_{eff}$개 정답 후보의 평균 오차 사용 |
| **FP 압축** | **없음** — 카테고리별 평균 정답률 행렬을 중심화(centering) 후 정규화만, PCA 투영 생략 (v1은 PCA-5) |
| **FP 차원** | **80** (카테고리 수와 동일) — v1은 5(PCA) |
| **Probe 할당 방식** | **균등(uniform)** — 카테고리마다 동일 quota, 카테고리 실제 보유량으로 clip (v1은 PCA 중요도 가중) |
| 인코더 백본 | `sentence-transformers/all-MiniLM-L6-v2` (frozen, CLS 토큰) |
| Projection head | 2-layer MLP (Linear→ReLU→Linear, bias 없음), 출력 차원 = FP 차원(80) |
| Optimizer | Adam, LR=1e-3 |
| Epochs | 10, per-epoch holdout Spearman rho 기준 best checkpoint 선택 |
| Holdout 비율 | 0.15 (Set A 내부, Set B와 별개) |
| Batch size | 64 |
| kNN 평가 K (top-k 후보) | 20 |
| Bandit beta | 0.000001 |
| Cost 정규화 | n_params 기반 |
| AUDC 계산 | λ 로그스케일 스윕(20점, 1e-4~1e2), cost-grid 보간 후 AUDC/QNC/Peak 산출 |
| CSCR 논문값 | All-seen 0.541 / Unseen 0.4848 |

---

## 1. EmbedLLM

### 1.1 All-seen, 공식 1800-probe 기준 (헤드라인)

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-1800` |
| FP 차원 | 80 (무압축) |
| 실제 probe 수 | 1760 (균등 할당) |
| 모델 수 | 111 |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_probe_scale_sweep_uniform_nocompress_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5771 | 0.6147 | 1.3050 |
| 1 | 0.5804 | 0.6213 | 1.2728 |
| 2 | 0.5784 | 0.6217 | 1.4095 |
| **평균** | **0.5787** | 0.6192 | 1.3291 |
| **std** | **0.0014** | 0.0032 | – |

비교: CSCR all-seen = 0.541 → **+7.0% 마진, 3/3 승**. 같은 조건(균등 할당, 1800-probe)의 압축 버전은 0.5473 (§1.4) → 압축만 제거해서 **+0.0314 AUDC 개선**.

### 1.2 Unseen, 공식 1800-probe 기준 (헤드라인)

| 설정 | 값 |
|---|---|
| FP(seen/unseen 공용) | 위와 동일 FP, unseen 모델만 평가 시 선택 |
| FP 차원 | 80 (무압축) |
| 실제 probe 수 | 1760 |
| 모델 수(seen/unseen) | 시드별 상이, 71~73 seen / 35~36 unseen (`newllm_split[_seed{N}].json`) |
| Catfilter | pct=0.3 (seen 모델 기준) |
| 스크립트 | `embedllm_probe_scale_sweep_unseen_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5261 | 0.5620 | 1.2749 |
| 1 | 0.5278 | 0.5553 | 1.1651 |
| 2 | 0.5158 | 0.5433 | 0.9453 |
| **평균** | **0.5232** | 0.5536 | 1.1284 |
| **std** | **0.0053** | 0.0077 | – |

비교: CSCR unseen = 0.4848 → **+7.9% 마진, 3/3 승**. 같은 조건의 압축 버전은 0.5074 (§1.4) → 압축 제거로 **+0.0158 AUDC 개선**.

### 1.3 V2 앵커 (전체 데이터, probe 무제한, 무압축) — 상한 참고값

| 프로토콜 | AUDC(3시드) | std | Peak | CSCR 대비 |
|---|---|---|---|---|
| All-seen | 0.5867 | 0.0030 | 0.6279 | +8.5% |
| Unseen | 0.5299 | 0.0080 | 0.5613 | +9.3% |

FP: `embedllm-ceiling` (80차원, 무압축, probe 제한 없음 — 29,673개 프롬프트 전부 사용).

### 1.4 압축 vs 무압축 Ablation (probe 수·할당 방식 고정, 압축 여부만 변경)

| 프로토콜 | 압축(PCA-5) AUDC | 무압축 AUDC | 차이 | CSCR 대비(압축) | CSCR 대비(무압축) |
|---|---|---|---|---|---|
| All-seen (1800-probe) | 0.5473 (std 0.0011) | 0.5787 (std 0.0014) | **+0.0314** | +1.2% | +7.0% |
| Unseen (1800-probe) | 0.5074 (std 0.0053) | 0.5232 (std 0.0053) | **+0.0158** | +4.7% | +7.9% |

압축·무압축 모두 카테고리별 **균등 할당**, 1800-probe, 3시드 동일 조건. 압축 제거만으로 All-seen +0.031 / Unseen +0.016 AUDC 개선 — 이 세션에서 얻은 단일 변경사항 중 가장 큰 효과.

### 1.5 압축 손실 자체의 격리 (probe 무제한, 전체 데이터 100% 사용)

Probe 수 차이를 완전히 배제하고(양쪽 다 29,673개 프롬프트 전부 사용) 압축 여부만 비교:

| Arm | FP | AUDC(3시드) | std | Peak |
|---|---|---|---|---|
| 무압축 | `embedllm-ceiling` (80차원) | 0.5867 | 0.0030 | 0.6279 |
| 압축(PCA-5) | `embedllm-ceiling-pca5` (5차원) | 0.5558 | 0.0048 | 0.5989 |
| **압축만의 순손실** | | **−0.0309** | | −0.0290 |

스크립트: `embedllm_compression_cost_isolation.py`. §1.4의 1800-probe 압축 손실(0.0314)과 거의 동일한 크기 — **압축 손실은 probe 수와 무관하게 일정**하다는 근거.

### 1.6 Probe-count 스케일링 스윕 — All-seen (균등 할당, 무압축, 3시드)

| target | 실제 probe | AUDC | std | Peak |
|---|---|---|---|---|
| 96 | 80 | 0.5798 | 0.0004 | 0.6244 |
| 300 | 320 | 0.6004†| 0.0006 | 0.6407 |
| 800 | 800 | 0.5909 | 0.0041 | 0.6326 |
| 1200 | 1200 | 0.5833 | 0.0029 | 0.6297 |
| **1800** | **1760** | **0.5787** | 0.0014 | 0.6192 |
| 4000 | 4000 | 0.5852 | 0.0027 | 0.6282 |
| 8000 | 8031 | 0.5793 | 0.0040 | 0.6257 |
| 15000 | 14998 | 0.5847 | 0.0029 | 0.6333 |
| 25000 | 24999 | 0.5867 | 0.0036 | 0.6286 |
| V2(전체) | 29673 | 0.5867 | 0.0030 | 0.6279 |

† **300 지점은 이상치로 확인, §1.8 참고 — HF 공식 train/test split 특유의 우연이며, 재분배 시 소멸함. Scalability 근거로 사용 금지.**

80~29673개(370배) 구간에서 AUDC는 0.579~0.590 사이를 벗어나지 않음 — **단조 증가 없음, probe 수와 사실상 무관.**

### 1.7 Probe-count 스케일링 스윕 — Unseen (균등 할당, 무압축, 3시드)

| target | 실제 probe | AUDC | std | Peak |
|---|---|---|---|---|
| 96 | 80 | 0.5200 | 0.0075 | 0.5474 |
| 300 | 320 | 0.5345†| 0.0048 | 0.5761 |
| **1800** | **1760** | **0.5232** | 0.0053 | 0.5536 |
| 4000 | 4000 | 0.5248 | 0.0072 | 0.5593 |
| 8000 | 8031 | 0.5260 | 0.0057 | 0.5617 |
| 15000 | 14998 | 0.5298 | 0.0049 | 0.5656 |
| 25000 | 24999 | 0.5330 | 0.0055 | 0.5667 |
| V2(전체) | 29673 | 0.5299 | 0.0080 | 0.5613 |

† 300 지점 이상치, §1.8 참고.

300을 제외한 6개 지점(80→24999)은 **완벽한 단조 증가**(Pearson r=0.97, p=0.001, Spearman ρ=1.0) — All-seen과 달리 Unseen에서는 약하지만 통계적으로는 뚜렷한 상승 추세가 존재. 다만 효과 크기는 작음(80개→24999개, 300배 증가에 AUDC +0.013).

### 1.8 방법론 검증: 300-probe 지점 이상치 조사

§1.6/1.7의 300-probe 지점이 이웃 지점보다 뚜렷하게 높게 나와(All-seen +0.02, Unseen +0.013) 과적합/유출 가능성을 의심, 다음을 순서대로 검증:

1. **Set A(train.csv)/Set B(test.csv) 프롬프트 중복** — 0건 (ID·텍스트 모두 대조).
2. **Probe 할당 로직** — 카테고리당 정확히 4개씩(80×4=320), 불균형 없음.
3. **FP의 정답(V2) 근사도** — 300-probe FP는 V2와의 코사인 유사도가 0.550으로, 1800(0.737)·4000(0.819)보다 오히려 낮음. "우연히 정답에 가까워서 좋다"는 가설 기각.
4. **학습 내부 검증 지표(holdout rho)** — 96/300/800/1200 전부 0.40~0.41로 평평, 300에서 특별히 잘 학습된 흔적 없음.
5. **독립적인 재분배(resplit) 테스트** — train.csv+test.csv를 합쳐 완전히 새로운 분할(seed=777, SetA 29,406 / SetB 3,267 프롬프트, probe도 새 Set A 기준으로 재선정)로 96/300/800/1200/1800을 재현.

**재분배 결과 (All-seen, 스크립트 `embedllm_probe300_resplit_diagnostic.py`):**

| target | 원래 split AUDC | 재분배 split AUDC |
|---|---|---|
| 96 | 0.5798 | 0.5734 |
| 300 | **0.6004** | 0.5777 |
| 800 | 0.5909 | 0.5664 |
| 1200 | 0.5833 | 0.5700 |
| 1800 | 0.5787 | 0.5618 |

**재분배 결과 (Unseen, 스크립트 `embedllm_probe300_resplit_unseen_diagnostic.py`):**

| target | 원래 split AUDC | 재분배 split AUDC |
|---|---|---|
| 96 | 0.5200 | 0.5086 |
| 300 | **0.5345** | 0.5128 |
| 800 | – | 0.5085 |
| 1200 | – | 0.5099 |
| 1800 | 0.5232 | 0.5148 |

재분배 후 300은 All-seen·Unseen 양쪽 모두 이웃 지점과 구분되지 않음(std 범위 내). **결론: 300-probe 스파이크는 HF 공식 train/test split과 이 특정 probe 부분집합 사이의 우연한 정렬이며, 재현 가능한 "sweet spot"이 아니다.** §1.6/1.7의 300행은 참고용으로만 표에 남기고 scalability 주장의 근거로 쓰지 않는다.

### 1.9 All-seen Ablation — Catfilter 유무 (무압축+균등 할당으로 재검증 완료)

| probe | Catfilter 있음 | Catfilter 없음 | 차이 |
|---|---|---|---|
| **1800(실제 1760)** | 0.5787 (std 0.0014) | 0.5792 (std 0.0023) | **+0.0005 (거의 없음)** |
| **192(실제 160)** | 0.5883 (std 0.0049) | 0.5696 (std 0.0047) | **+0.0187 (유의미)** |

스크립트: `embedllm_uniform_nocompress_1800_ablation_nocatfilter_allseen_multiseed.py`, `embedllm_uniform_nocompress_192_ablation_nocatfilter_allseen_multiseed.py`. 압축된 5차원 공간에서는 catfilter가 독립적으로 기여했지만, 무압축 80차원·1800-probe에서는 그 효과가 거의 사라짐 — 다만 **probe가 극단적으로 적어지면(192) catfilter가 다시 유의미해짐**. 즉 catfilter의 기여는 probe 희소성에 반비례해서 커지는 것으로 보임.

### 1.10 Fairness — 192-probe (무압축+균등 할당으로 재검증)

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-192` |
| FP 차원 | 80 (무압축) |
| 실제 probe 수 | 160 |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_uniform_nocompress_192_fairness_allseen_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5935 | 0.6453 | 1.3745 |
| 1 | 0.5897 | 0.6367 | 1.1653 |
| 2 | 0.5817 | 0.6370 | 1.1556 |
| **평균** | **0.5883** | 0.6397 | 1.2318 |
| **std** | **0.0049** | 0.0040 | – |

비교: CSCR all-seen = 0.541 → **+8.7% 마진, 3/3 승**. v1의 압축 버전(0.5436, PCA 가중 배분)보다 개선.

### 1.11 방법론 검증 — Random 대조군 (사용자 요청, All-seen 결과의 신뢰성 검증)

All-seen이 극단적으로 적은 probe에서도 높게 나오는 것에 대한 의심에서 시작된 일련의 대조군 실험.

**(a) Probe 선택 방식: top-variance vs 완전 무작위 (target=96, 실제 80개, all-seen)**

| 선택 방식 | AUDC | std |
|---|---|---|
| Top-variance (`nlargest(n,"var")`, 기존 방식) | 0.5798 | 0.0004 |
| 완전 무작위 선택 (seed=42) | **0.5909** | 0.0038 |

무작위 선택이 오히려 더 높거나 동등 — "똑똑한 probe 선택이 적은 데이터로도 잘 나오는 이유"라는 가설 기각.

**(b) FP를 순수 무작위 노이즈 벡터로 교체 (진짜 데이터 정보 전혀 없음, 112개 모델 × 80차원, seed=123)**

| 프로토콜 | 노이즈 FP AUDC | 실제 FP AUDC(1800-probe) | 비고 |
|---|---|---|---|
| All-seen | 0.5769 (std 0.0049) | 0.5787 | **거의 구분 안 됨 — 문제 확인** |
| **Unseen** | **0.3721 (std 0.0329)** | 0.5232 | **완전히 붕괴, CSCR(0.4848)보다도 낮음(3/3)** |

스크립트: `embedllm_randomnoise_fp_allseen_multiseed.py`, `embedllm_randomnoise_fp_unseen_multiseed.py`.

**해석**: All-seen은 "학습 중 라벨을 본 111개의 닫힌 클래스 집합"에 대한 분류 문제로 퇴화하는 경향이 있음 — projection head가 노이즈 벡터라도 "이 카테고리 → 이 모델의 (임의의) 주소"로 암기할 수 있어, FP의 실제 의미 내용과 무관하게 높은 점수가 나올 수 있음. 반면 **Unseen은 학습 중 한 번도 라벨을 본 적 없는 모델을 평가하므로 이 암기 메커니즘이 원천적으로 불가능** — 노이즈로 완전히 붕괴한다는 사실이 실제 FP를 썼을 때의 Unseen 우위(+7.9%)가 진짜 신호에서 온다는 것을 입증함.

**결론**: All-seen 수치는 "COMPAR가 텍스트→능력 매핑을 학습한다"는 핵심 주장의 증거로 쓰기 부적절함(분류 퇴화로 설명 가능한 부분이 큼). **Unseen이 이 방법론의 핵심 주장을 검증하는 실질적 지표.** (CSCR의 셀링 포인트 중 하나가 Unseen 문제 해결이었던 것과 같은 맥락 — Unseen에서 FP 품질이 실제로 요구된다는 뜻이므로, FP를 잘 만드는 것 자체가 어려운 포인트라는 것이 이번 검증으로 재확인됨.)

*(참고: 이 negative control은 EmbedLLM에서만 실행 — RouterBench·LLMRouterBench에도 같은 검증이 필요할 수 있음, §2.3/§3.3 미착수)*

---

## 2. RouterBench (11개 모델) — Pure V2 (전체 데이터, probe 무제한) vs CSCR

| 설정 | 값 |
|---|---|
| FP | `routerbench-ceiling` (전체 Set A, 86차원, 무압축, 원래부터 probe 제한/PCA 압축 없음) |
| 모델 수 | 11 (구조적으로 Unseen 불가) |
| Loss | Combined (Catfilter + min(0.3,3)) |
| 스크립트 | `routerbench_purev2_ablation3way_allseen_multiseed.py` |

| | AUDC(3시드) | std |
|---|---|---|
| **COMPAR (Pure V2)** | **0.7205** | 0.0013 |
| CSCR (논문값) | 0.711 | – |

**+1.3% 마진, 3/3 승.**

### 2.1 Fairness — CSCR에게 probe를 1800개로 늘려줬을 때 (반대 방향 검증, 세션 전반부 기 실행)

CSCR의 Perplexity FP에 COMPAR와 같은 probe 예산(1800)을 주고, CSCR의 실제 loss(`cost_spectrum_info_nce`, 업스트림과 byte-identical 확인됨)로 그대로 학습 — "COMPAR가 유리한 게 아니라 CSCR이 원래 probe가 적어서 불리했던 것 아니냐"는 질문에 대한 반대 방향 검증.

| 버전 | seed0 | seed1 | seed2 | 평균 | std | CSCR 논문값(0.711) 대비 |
|---|---|---|---|---|---|---|
| 차원 고정(32) + 표본만 1800으로 증가 | 0.6382 | 0.6352 | 0.6371 | **0.6368** | 0.0012 | 0/3 승, 오히려 하회 |
| Naive(probe=차원=1800, 참고용) | 0.6352 | 0.6279 | 0.6388 | 0.6340 | 0.0045 | 0/3 승 |

스크립트: `routerbench_fair_probe1800_multiseed.py`(차원고정), `routerbench_naive_dim1800_cscrloss.py`(naive). **CSCR은 probe를 10배 가까이 늘려도(원래 192 → 1800) 성능이 개선되지 않고 오히려 소폭 하락** — COMPAR(0.7205)와의 격차가 데이터 양 차이가 아니라 방법론 차이에서 온다는 근거. (해석 메모: CSCR 논문이 원래 192 정도의 작은 차원을 유지한 것이, 이 소규모 pool에서의 과적합 함정을 피하려는 의도적 설계였을 가능성 — naive하게 차원까지 같이 늘리면 오히려 더 나빠짐.)

### 2.2 역방향 검증 — COMPAR의 loss를 CSCR의 FP(Perplexity)에 그대로 적용하면?

"COMPAR가 이기는 게 loss 때문인지 FP 때문인지"를 가르기 위해, 반대로 COMPAR의 loss(Catfilter+min(0.3,3))를 CSCR의 Perplexity FP(dim=32, `routerbench-perplexity-nprobes1800`)에 그대로 적용.

| | AUDC(3시드) | std |
|---|---|---|
| COMPAR loss + CSCR FP | **0.6170** | 0.0006 |
| (비교) CSCR loss + CSCR FP | 0.6368 | 0.0012 |
| (비교) COMPAR loss + Ceiling FP(정식) | 0.7205 | 0.0013 |

스크립트: `routerbench_cscrfp_comparloss_allseen_multiseed.py`. **COMPAR의 loss를 CSCR FP에 붙이면 CSCR 자체 loss보다도 못함.** 즉 COMPAR의 우위는 loss 하나만의 효과가 아니라 "정밀한 FP(Ceiling) + 그 정밀함을 전제로 설계된 loss(catfilter의 track-record 필터링, min(0.3,3)의 상위 k개 정밀 매칭)"가 함께 있어야 나온다는 근거 — **FP와 loss를 분리해서 아무거나 조합하면 안 됨.**

## 3. LLMRouterBench (33개 모델) — Pure V2 (전체 데이터, probe 무제한) vs CSCR

| 설정 | 값 |
|---|---|
| FP | `llmrouterbench-ceiling-purev2` (전체 Set A 2345행, 8차원, 무압축, 신규 빌드 — 기존 900-probe/192차원 버전과 다름) |
| 모델 수 | 33 |
| Loss | Combined (Catfilter + min(0.3,3)) |
| 스크립트 | `llmrouterbench/purev2_ablation3way_allseen_unseen_multiseed.py` |

| 프로토콜 | COMPAR AUDC(3시드) | std | CSCR 비교 |
|---|---|---|---|
| **All-seen** | **0.7349** | 0.0061 | v1 fairness 테스트의 CSCR 실제 loss 재현치(Perplexity-900) = 0.6712 → 상회 |
| **Unseen** | **0.6719** | 0.0357 | 공식 CSCR 논문값 없음(독자 구축 pool), 기준값 없음 |

### 3.1 Fairness — COMPAR 자체의 900-probe 버전 (균등 할당, 무압축, 신규 실행)

기존엔 LLMRouterBench에서 "900-probe, 균등 할당, 무압축(dim=8)" 조건으로 COMPAR를 돌려본 적이 없었음(있던 건 PCA 가중+192차원 binning 버전, 또는 오늘의 probe 무제한 Pure V2뿐) — CSCR과 정확히 같은 900-probe 예산으로 맞대결시키기 위해 신규 실행.

| 설정 | 값 |
|---|---|
| FP | `llmrouterbench-ceiling-900-uniform-nocompress` (8개 데이터셋에 균등 배분, top-variance 선택, dim=8, 무압축) |
| 실제 probe 수 | 900 (aime 48개·livemathbench 96개는 전량 사용, 나머지는 126개씩) |
| 스크립트 | `llmrouterbench/build_ceiling_900_uniform_nocompress.py`, `fair900_uniform_nocompress_allseen_unseen_multiseed.py` |

| 프로토콜 | seed0 | seed1 | seed2 | 평균 | std |
|---|---|---|---|---|---|
| **All-seen** | 0.7183 | 0.7271 | 0.7103 | **0.7186** | 0.0068 |
| **Unseen** | 0.6043 | 0.6894 | 0.6820 | **0.6586** | 0.0385 |

### 3.2 Fairness — CSCR에게 probe를 900개로 늘려줬을 때 (반대 방향 검증, 세션 전반부 기 실행)

LLMRouterBench는 Set A 전체가 2345행뿐이라 RouterBench처럼 1800을 주면 "거의 전체 데이터"가 되어버려 의미 있는 예산 증가가 아님 — 대신 900(Set A의 ~38%)으로 CSCR의 Perplexity FP를 확장, 실제 loss로 학습.

| 버전 | seed0 | seed1 | seed2 | 평균 | std |
|---|---|---|---|---|---|
| 차원 고정(192) + 표본만 900으로 증가 | 0.6887 | 0.6621 | 0.6629 | **0.6712** | 0.0123 |
| Naive(probe=차원=900, 참고용) | – | – | – | 0.6586 | – |

스크립트: `llmrouterbench/build_perplexity_naive_dim900.py` + `naive_dim900_cscrloss.py`.

### 3.3 900-probe 동일 조건 직접 비교 (§3.1 vs §3.2)

| | COMPAR (§3.1) | CSCR (§3.2) | 마진 |
|---|---|---|---|
| All-seen | **0.7186** | 0.6712 | **+7.1%** |
| Unseen | **0.6586** | 기준값 없음 | – |

같은 900-probe 예산, 같은 무압축(dim=8/192로 조건은 다르지만 둘 다 binning-fixed) 조건에서 COMPAR가 CSCR을 +7.1% 앞섬 — RouterBench(§2.1, +13.1%p 상당의 격차: 0.7205 vs 0.6368)와 같은 방향. RouterBench만큼 격차가 크진 않음(모델 수/probe 비율이 낮아 과적합 압력이 약한 것과 같은 이유로 추정).

*(참고: §3.2는 CSCR 자체 loss를 §3.1과 완전히 동일한 FP에 돌린 게 아니라 세션 전반부의 별도 CSCR용 FP 기준 — 완전 동일 FP 위에서의 직접 대결이 필요하면 추가 실행 가능.)*

### 3.4 역방향 검증 — COMPAR의 loss를 CSCR의 FP(Perplexity)에 그대로 적용하면?

RouterBench(§2.2)와 동일한 검증을 LLMRouterBench에서도 실행 — COMPAR loss를 CSCR의 Perplexity FP(dim=192, `llmrouterbench_v15_900/perplexity`)에 적용. 결정론적 파이프라인이라 재실행 시 완전히 동일한 값 재현 확인(시스템 재부팅 직후 실행이라 결과 신뢰성 의심 → 재실행으로 검증 완료).

| | AUDC(3시드) | std |
|---|---|---|
| COMPAR loss + CSCR FP, All-seen | **0.2594** | 0.0375 |
| COMPAR loss + CSCR FP, Unseen | **0.4183** | 0.0679 |
| (비교) CSCR loss + CSCR FP, All-seen(§3.2) | 0.6712 | 0.0123 |
| (비교) COMPAR loss + Pure V2 FP(§3), All-seen | 0.7349 | 0.0061 |

스크립트: `llmrouterbench/cscrfp_comparloss_allseen_unseen_multiseed.py`. **RouterBench(§2.2, dim=32)에서는 "약간 나쁨" 수준이었던 게, LLMRouterBench(dim=192)에서는 완전 붕괴 수준으로 나타남** — CSCR FP + COMPAR loss 조합의 손해가 FP 차원이 클수록(노이즈 낀 축이 많을수록) 더 커지는 것으로 보임. §2.2의 결론("FP와 loss는 짝을 맞춰야 함")을 더 극단적으로 재확인.

---

## 4. 전체 종합

| 벤치마크 | 프로토콜 | 압축 | 할당 | probe | 평균 AUDC | std | CSCR 대비 |
|---|---|---|---|---|---|---|---|
| EmbedLLM | **All-seen (공식 1800)** | 없음 | 균등 | 1760 | **0.5787** | 0.0014 | **+7.0%** |
| EmbedLLM | **Unseen (공식 1800)** | 없음 | 균등 | 1760 | **0.5232** | 0.0053 | **+7.9%** |
| EmbedLLM | All-seen, V2(전체) | 없음 | – | 29673 | 0.5867 | 0.0030 | +8.5% |
| EmbedLLM | Unseen, V2(전체) | 없음 | – | 29673 | 0.5299 | 0.0080 | +9.3% |
| EmbedLLM | All-seen, 1800(압축 버전, 참고) | PCA-5 | 균등 | 1760 | 0.5473 | 0.0011 | +1.2% |
| EmbedLLM | Unseen, 1800(압축 버전, 참고) | PCA-5 | 균등 | 1760 | 0.5074 | 0.0053 | +4.7% |
| EmbedLLM | All-seen, Fairness(192-probe) | 없음 | 균등 | 160 | 0.5883 | 0.0049 | +8.7% |
| EmbedLLM | All-seen, Ablation(catfilter 없음, 1800-probe) | 없음 | 균등 | 1760 | 0.5792 | 0.0023 | +7.1% |
| EmbedLLM | All-seen, Ablation(catfilter 없음, 192-probe) | 없음 | 균등 | 160 | 0.5696 | 0.0047 | +5.3% |
| EmbedLLM | All-seen, **노이즈 FP 대조군** | – | – | – | 0.5769 | 0.0049 | (+6.6%, 무의미 — §1.11) |
| EmbedLLM | Unseen, **노이즈 FP 대조군** | – | – | – | **0.3721** | 0.0329 | **−23.2%, CSCR 이하** |
| RouterBench | **All-seen, Pure V2** | 없음 | – | 전체 | **0.7205** | 0.0013 | +1.3% (CSCR 0.711) |
| RouterBench | All-seen, CSCR(1800-probe로 확장) | – | 균등 | 1800 | 0.6368 | 0.0012 | 0/3, COMPAR 대비 −13.1% |
| RouterBench | All-seen, **COMPAR loss + CSCR FP** | – | – | 1800 | 0.6170 | 0.0006 | CSCR loss보다도 낮음 |
| LLMRouterBench | **All-seen, Pure V2** | 없음 | – | 전체 | **0.7349** | 0.0061 | CSCR loss 재현치(0.6712) 상회 |
| LLMRouterBench | **Unseen, Pure V2** | 없음 | – | 전체 | **0.6719** | 0.0357 | 독자 pool, 기준값 없음 |
| LLMRouterBench | **All-seen, Fairness(900-probe, 균등·무압축)** | 없음 | 균등 | 900 | **0.7186** | 0.0068 | +7.1% (동일조건 CSCR 0.6712) |
| LLMRouterBench | Unseen, Fairness(900-probe, 균등·무압축) | 없음 | 균등 | 900 | 0.6586 | 0.0385 | 기준값 없음 |
| LLMRouterBench | All-seen, CSCR(900-probe로 확장) | – | 균등 | 900 | 0.6712 | 0.0123 | 0/3, COMPAR(§3.1) 대비 −6.6% |

**RouterBench·LLMRouterBench 참고**: 두 벤치마크 모두 원래부터 무압축. Pure V2(probe 무제한)와, CSCR과 동일 probe 예산을 준 Fairness 버전(RouterBench 1800 / LLMRouterBench 900) 둘 다 확인 — 두 조건 모두 COMPAR가 CSCR을 앞섬(+1.3%~+13.1%). CSCR은 probe를 늘려도 성능이 개선되지 않음(RouterBench 0/3, LLMRouterBench 0/3). Ablation은 두 벤치마크 모두 모델 풀이 작아 각 요소의 개별 기여가 희석되는 것으로 확인되어 생략.

**핵심 주장(EmbedLLM)**: 압축 제거만으로 All-seen +1.2%→+7.0%, Unseen +4.7%→+7.9%로 마진이 크게 확대. Probe 수 자체의 scalability 효과는 미미하며(§1.6/1.7), 1800-probe라는 공식 기준을 벗어나도 큰 이득은 없음(§1.6/1.7 표 참고) — 즉 헤드라인 주장은 "압축을 걷어낸 COMPAR"이지 "probe를 늘린 COMPAR"가 아님.

**방법론 검증(§1.11)**: All-seen은 노이즈 FP로도 재현되는 "닫힌 집합 분류 퇴화" 성분을 포함하고 있어 핵심 주장의 근거로는 부적절함. **Unseen이 노이즈에서 완전히 붕괴한다는 사실이 COMPAR의 핵심 주장(텍스트→능력 매핑 학습)을 뒷받침하는 결정적 증거** — 발표·논문에서는 Unseen을 주 지표로, All-seen은 "닫힌 풀에서도 CSCR을 이긴다"는 보조 주장으로 격을 낮춰 서술할 것.

**남은 과제**:
1. RouterBench·LLMRouterBench에도 §1.11과 같은 노이즈 FP 대조군 검증 필요(아직 EmbedLLM에서만 실행) — 특히 LLMRouterBench Unseen에서 재현되는지가 핵심
2. RouterBench·LLMRouterBench의 probe-budget(fairness) 버전은 이번 라운드에서 보류 — Pure V2만 검증됨, 필요시 추가
3. Unseen 스윕 Deferral Curve 시각화 — 원시 곡선 미저장, 재실행 필요
4. 헤드라인 수치 3시드 → 10시드 확장 여부 미결정
