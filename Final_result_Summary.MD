# Final Result Summary — min(pct=0.3, cap=3) 방법론 (COMPAR / TAR)

이 문서는 min-pos(K=1)를 **min(pct=0.3, K_CAP=3)**로 일반화한 이후 실행된 모든 실험 결과를 기록한다. 해석은 최소화하고 설정·수치 위주로 정리.

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 방법론 이름 | COMPAR (Capability-Oriented Multi-Positive Adaptive Routing) — 방법론 전체 / TAR (Trimmed Advantage Regression) — loss 이름 |
| min-loss 공식 | $k_{eff} = \mathrm{clip}(\lceil 0.3 \times n_{pos}\rceil,\ 1,\ 3)$, 상위 $k_{eff}$개 정답 후보의 평균 오차 사용 |
| 인코더 백본 | `sentence-transformers/all-MiniLM-L6-v2` (frozen, CLS 토큰) |
| Projection head | 2-layer MLP (Linear→ReLU→Linear, bias 없음), 출력 차원 = FP 차원 |
| Optimizer | Adam, LR=1e-3 |
| Epochs | 10, per-epoch holdout Spearman rho 기준 best checkpoint 선택 |
| Holdout 비율 | 0.15 (학습 데이터 내부, Set B와 별개) |
| Batch size | EmbedLLM: 64 / RouterBench·LLMRouterBench: 32 |
| kNN 평가 K (top-k 후보) | EmbedLLM: 20 / RouterBench: 11(전체) / LLMRouterBench: 33(전체) |
| Bandit beta | 0.000001 |
| Cost 정규화 | n_params 기반 min-max 정규화 |
| AUDC 계산 | λ 로그스케일 스윕(20점, 1e-4~1e2), cost-grid 보간 후 AUDC/QNC/Peak 산출 |

---

## 1. EmbedLLM

### 1.1 All-seen, Combined (min(0.3,3) + catfilter pct=0.3)

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling`(V2, 전체 데이터 카테고리 집계) |
| FP 차원 | 80 |
| 모델 수 | 111 |
| Catfilter | pct=0.3 (카테고리 트랙레코드 상위 30%) |
| 스크립트 | `embedllm_minpctcap3_allseen_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5886 | 0.6260 | 1.2542 |
| 1 | 0.5890 | 0.6350 | 1.1837 |
| 2 | 0.5824 | 0.6227 | 1.0982 |
| **평균** | **0.5867** | 0.6279 | 1.1787 |
| **std** | **0.0030** | – | – |

비교: K=1(min-pos)+catfilter = 0.5652 (std 0.0005) / CSCR all-seen 논문값 = 0.541

### 1.1b All-seen, Combined — 진짜 V1.5(1800-probe, floor=15, PCA 가중 배분)

**주의**: 1.1은 probe 제한이 없는 V2(전체 데이터) 버전이다. "1800-probe로 COMPAR를 정규 검증한다"는 목적에는 아래 결과를 사용해야 한다.

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling-pcaweighted-pca5` (V1.5, PCA 가중 배분, floor=15, target=1800) |
| FP 차원 | 5 (PCA) |
| 실제 probe 수 | 1800 |
| 모델 수 | 111 |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_v15_1800_minpctcap3_allseen_multiseed.py` |

**10시드로 확정** (중요 실험이라 사용자 요청으로 시드 0-2에서 3-9까지 확장):

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5511 | 0.5840 | 1.0079 |
| 1 | 0.5547 | 0.5763 | 1.2453 |
| 2 | 0.5549 | 0.5833 | 1.2008 |
| 3 | 0.5419 | 0.5700 | 0.9108 |
| 4 | 0.5505 | 0.5760 | 1.0563 |
| 5 | 0.5487 | 0.5833 | 0.9799 |
| 6 | 0.5511 | 0.5897 | 0.9917 |
| 7 | 0.5566 | 0.5877 | 0.9203 |
| 8 | 0.5429 | 0.5700 | 0.9691 |
| 9 | 0.5466 | 0.5750 | 1.0020 |
| **평균(10시드)** | **0.5499** | 0.5795 | 1.0034 |
| **std** | **0.0047** | – | – |

비교: 구버전 K=1+catfilter V1.5(1800-probe) all-seen ≈ 0.556 (거의 동일, min(0.3,3) 효과 미미) / CSCR all-seen 논문값 = 0.541 → **+1.65% 마진, 10/10 승**

### 1.2 Unseen, Combined (min(0.3,3) + catfilter pct=0.3)

| 설정 | 값 |
|---|---|
| FP(학습, seen) | `embedllm-ceiling-pca5` |
| FP(평가, unseen) | `embedllm-ceiling-pca5-unseen-only[-seed{N}]` |
| FP 차원 | 5 (PCA) |
| 모델 수(seen/unseen) | seed별 상이, 총 112개 중 seen≈71~74 / unseen≈38~41 (`newllm_split[_seed{N}].json`) |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_minpctcap3_unseen_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5367 | 0.5783 | 1.211 |
| 1 | 0.5232 | 0.5427 | 0.825 |
| 2 | 0.5116 | 0.5430 | 0.817 |
| **평균** | **0.5238** | 0.5547 | 0.951 |
| **std** | **0.0103** | – | – |

비교: K=1+catfilter(pct=0.3) unseen = 0.5162 (3시드) / CSCR unseen 논문값 = 0.4848

### 1.2b Unseen, Combined — 진짜 V1.5(1800-probe, floor=15, PCA 가중 배분)

**주의**: 1.2는 probe 제한 없는 버전이다. "1800-probe로 COMPAR를 정규 검증한다"는 목적에는 아래 결과를 사용해야 한다.

| 설정 | 값 |
|---|---|
| FP(seen/unseen 공용) | `embedllm-ceiling-pcaweighted-pca5` (V1.5, 별도 unseen-only 서브셋 없이 같은 디렉토리에서 unseen 모델만 선택) |
| FP 차원 | 5 (PCA) |
| 실제 probe 수 | 1800 |
| 모델 수(seen/unseen) | 시드별 상이(`newllm_split[_seed{N}].json`), seen 71~73 / unseen 38~40 |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_v15_1800_minpctcap3_unseen_multiseed.py` |

**10시드로 확정**:

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5060 | 0.5467 | 0.937 |
| 1 | 0.5173 | 0.5500 | 1.155 |
| 2 | 0.5017 | 0.5400 | 0.920 |
| 3 | 0.5137 | 0.5500 | 1.389 |
| 4 | 0.5054 | 0.5307 | 0.868 |
| 5 | 0.5238 | 0.5620 | 0.866 |
| 6 | 0.5131 | 0.5357 | 0.934 |
| 7 | 0.5146 | 0.5510 | 1.042 |
| 8 | 0.5065 | 0.5500 | 1.008 |
| 9 | 0.5256 | 0.5450 | 0.904 |
| **평균(10시드)** | **0.5128** | 0.5461 | 0.992 |
| **std** | **0.0076** | – | – |

비교: K=1+catfilter V1.5(1800-probe) unseen 기준값 없음(이번이 최초 시도) / CSCR unseen 논문값 = 0.4848 → **+5.77% 마진, 10/10 승** (all-seen +1.65%보다 마진이 큼 — 원인 미분석)

### 1.3 All-seen Ablation — Combined loss만, Catfilter 없음

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling`(V2) |
| FP 차원 | 80 |
| 모델 수 | 111 |
| Catfilter | 없음(PCT_CATFILTER=1.0, 전체 정답 유지) |
| 스크립트 | `embedllm_minpctcap3_nocatfilter_allseen_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5748 | 0.6043 | 1.0805 |
| 1 | 0.5771 | 0.6070 | 1.1537 |
| 2 | 0.5658 | 0.6107 | 0.9735 |
| **평균** | **0.5725** | 0.6073 | 1.0692 |
| **std** | **0.0049** | – | – |

### 1.4 Fairness — Ceiling을 CSCR 원 예산(192-probe)으로 축소

| 설정 | 값 |
|---|---|
| FP | `embedllm-ceiling-pcaweighted-pca5-192`(PCA 중요도 가중 배분, floor=1) |
| FP 차원 | 5 (PCA) |
| 실제 probe 수 | 192 (균등 배분 대비 90.0% 절감) |
| 모델 수 | 111 |
| Catfilter | pct=0.3 |
| 스크립트 | `embedllm_pcaweighted_192_minpctcap3_multiseed.py` |

| seed | AUDC | Peak | QNC |
|---|---|---|---|
| 0 | 0.5490 | 0.5867 | 1.2800 |
| 1 | 0.5291 | 0.5603 | 1.2998 |
| 2 | 0.5526 | 0.5883 | 1.3934 |
| **평균** | **0.5436** | 0.5784 | 1.3244 |
| **std** | **0.0103** | – | – |

비교: K=1 192-probe = 0.5444 (std 0.0025) / CSCR 논문값(all-seen) = 0.541

---

## 2. RouterBench (11개 모델)

### 2.1 All-seen, Combined (min(0.3,3) + catfilter top50pct) — Fairness "CSCR↑" 방향

| 설정 | 값 |
|---|---|
| FP | `routerbench-ceiling-v15` (PCA 가중 배분) |
| FP 차원 | 86 (uncompressed) |
| 실제 probe 수 | 1761 |
| 모델 수 | 11 |
| Catfilter | top50pct (percentile 무효 구간, §2.5 참고) |
| 스크립트 | `routerbench_v15_minpctcap3_multiseed.py` |

| seed | AUDC | Peak | QNC | delta(vs random) | p |
|---|---|---|---|---|---|
| 0 | 0.7288 | 0.7768 | 0.0026 | +0.2271 | 0.000999 |
| 1 | 0.7304 | 0.7759 | 0.0025 | +0.2310 | 0.000999 |
| 2 | 0.7276 | 0.7730 | 0.0025 | +0.2305 | 0.000999 |
| **평균** | **0.7289** | 0.7752 | 0.0025 | +0.2295 | – |
| **std** | **0.0011** | – | – | – | – |

비교: K=1+catfilter = 0.7282 (std 0.0019) / Perplexity+CSCR 실제 loss(1800-probe) = 0.6368 / CSCR 논문값 = 0.711

---

## 3. LLMRouterBench (33개 모델)

### 3.1 All-seen, Combined (min(0.3,3) + catfilter top50pct, 카테고리=데이터셋) — Fairness "CSCR↑" 방향

| 설정 | 값 |
|---|---|
| FP | `llmrouterbench_v15_900/ceiling` (PCA 가중 배분) |
| FP 차원 | 192 (8개 데이터셋×24, 900개 probe를 그룹평균) |
| 실제 probe 수 | 900 |
| 모델 수 | 33 |
| Catfilter | top50pct (데이터셋 단위) |
| 스크립트 | `llmrouterbench/v15_minpctcap3_multiseed.py` |

| seed | AUDC | Peak | QNC | delta(vs random) | p |
|---|---|---|---|---|---|
| 0 | 0.7609 | 0.7895 | 0.0545 | +0.3812 | 0.000999 |
| 1 | 0.7659 | 0.8031 | 0.0505 | +0.4045 | 0.000999 |
| 2 | 0.7687 | 0.7920 | 0.0534 | +0.4214 | 0.000999 |
| **평균** | **0.7652** | 0.7949 | 0.0528 | +0.4024 | – |
| **std** | **0.0032** | – | – | – | – |

비교: K=1+catfilter = 0.7644 (3시드) / Perplexity-900+CSCR 실제 loss = 0.6712 / CSCR 논문값 없음(독자 구축 pool)

### 3.2 Unseen (이 벤치마크 최초 시도), Combined (min(0.3,3) + catfilter top50pct)

| 설정 | 값 |
|---|---|
| FP | `llmrouterbench_v15_900/ceiling` 중 unseen 모델 서브셋(재구축 없이 기존 파일 재사용) |
| FP 차원 | 192 |
| Seen/Unseen 분할 | 시드별 랜덤(RandomState(seed)), 22 seen / 11 unseen |
| Catfilter | top50pct(seen 모델·Set A 기준으로만 계산) |
| 스크립트 | `llmrouterbench/minpctcap3_unseen_multiseed.py` |

| seed | seen/unseen | AUDC | Peak | QNC | delta(vs random) | p |
|---|---|---|---|---|---|---|
| 0 | 22/11 | 0.6927 | 0.7199 | 0.0509 | +0.2151 | 0.000999 |
| 1 | 22/11 | 0.7166 | 0.7615 | 0.0602 | +0.2404 | 0.000999 |
| 2 | 22/11 | 0.7091 | 0.7419 | 0.0794 | +0.3346 | 0.000999 |
| **평균** | | **0.7061** | 0.7411 | 0.0635 | +0.2634 | – |
| **std** | | **0.0100** | – | – | – | – |

비교 기준값 없음(최초 시도).

---

## 4. 전체 종합 (평균 AUDC만)

| 벤치마크 | 프로토콜 | FP/probe | 평균 AUDC | std |
|---|---|---|---|---|
| EmbedLLM | All-seen (V2, 전체 데이터, probe 무제한) | 80차원 | 0.5867 (3시드) | 0.0030 |
| EmbedLLM | **All-seen (V1.5, 진짜 1800-probe, 10시드)** | 5차원(PCA) | **0.5499** | 0.0047 |
| EmbedLLM | Unseen (probe 무제한) | 5차원(PCA) | 0.5238 (3시드) | 0.0103 |
| EmbedLLM | **Unseen (V1.5, 진짜 1800-probe, 10시드)** | 5차원(PCA) | **0.5128** | 0.0076 |
| EmbedLLM | All-seen, ablation(catfilter 없음, probe 무제한) | 80차원 | 0.5725 (3시드) | 0.0049 |
| EmbedLLM | All-seen, fairness(192-probe) | 5차원(PCA) | 0.5436 (3시드) | 0.0103 |
| RouterBench | All-seen, fairness(1761-probe) | 86차원 | 0.7289 (3시드) | 0.0011 |
| LLMRouterBench | All-seen, fairness(900-probe) | 192차원 | 0.7652 (3시드) | 0.0032 |
| LLMRouterBench | Unseen(900-probe) | 192차원 | 0.7061 (3시드) | 0.0100 |

**"1800-probe로 COMPAR 정규 검증" 시 사용해야 할 숫자는 볼드 처리된 두 줄(10시드 확정: All-seen 0.5499, +1.65%, 10/10승 / Unseen 0.5128, +5.77%, 10/10승)** — 나머지 EmbedLLM all-seen/unseen/ablation 행은 probe 무제한(V2급) 버전이므로 구분해서 인용할 것.
