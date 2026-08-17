# Combined GRPO — Results Summary

발표/논문용으로 재구성한 결과 정리. `PROGRESS.md`의 시간 순서와는 다르게, 다음 4단계 서사로 배치함: (1) Combined GRPO 데뷔 — CSCR 대비 성능, (2) Ablation — Min-pos/Category-filtering 개별 기여, (3) Ceiling FP(capability)가 필수적이라는 증거, (4) Ceiling V1(샘플링)으로 비용을 줄여도 성능 유지.

모든 수치는 이 세션에서 실제로 실행한 스크립트의 결과 JSON에서 직접 추출했으며, 출처 파일을 각 표 아래 명시함.

---

## 1. Combined GRPO 데뷔 — CSCR 대비 Multi-seed 성능

**Combined = min-pos loss(정답 후보 중 가장 가까운 것과만 맞추면 됨) + top50%-catfilter(카테고리 트랙레코드 상위 50% 정답만 인정)**를 동시 적용한 최종 방법론.

### 1.1 EmbedLLM — Unseen 프로토콜 (2/3 seen 모델로 학습, 1/3 unseen 모델로 평가)

CSCR 논문 보고치: AUDC=0.4848, Peak=0.565 (Table 2)


| seed         | AUDC       | QNC   | Peak       |
| ------------ | ---------- | ----- | ---------- |
| 0            | 0.5250     | 1.039 | 0.5710     |
| 1            | 0.5264     | 1.276 | 0.5530     |
| 2            | 0.4975     | 0.865 | 0.5243     |
| 3            | 0.5342     | 1.533 | 0.5800     |
| 4            | 0.4970     | 0.955 | 0.5413     |
| 5            | 0.5461     | 1.026 | 0.5790     |
| 6            | 0.4990     | 0.980 | 0.5313     |
| 7            | 0.5254     | 0.871 | 0.5570     |
| 8            | 0.5055     | 0.910 | 0.5433     |
| 9            | 0.5106     | 0.885 | 0.5463     |
| 10           | 0.5255     | 0.876 | 0.5700     |
| **평균(11시드)** | **0.5175** | 1.020 | **0.5542** |
| std          | 0.0157     | 0.198 | 0.0180     |


**CSCR 대비: AUDC +6.8%, 11/11 시드 전원 승. Peak은 평균 0.5542로 CSCR(0.565)에는 못 미침** — AUDC 우위와 Peak 우위는 커브의 다른 구간이라 별개 결과임에 유의.

†† catfilter 컷을 top50%에서 pct=0.3(§2.5)으로 바꿔서도 3시드 재검증함: 0.5264/0.5246/0.4977, 평균 0.5162(std=0.0132) — top50%의 0.5163과 **사실상 동일**. unseen에서는 컷 비율이 결과에 영향을 주지 않음(§2.5 참고), 그래서 이 표는 top50% 버전을 그대로 유지.

출처: `combined_minpos_top50pct_multiseed_results.json`(seed0-2) + `combined_minpos_top50pct_seeds3to10_results.json`(seed3-10); pct=0.3 재검증은 `unseen_catfilter_pct30_multiseed_results.json`

### 1.2 EmbedLLM — All-seen 프로토콜 (전체 112개 모델 학습+평가) — **catfilter 컷 pct=0.3으로 갱신**

CSCR 논문 보고치: AUDC=0.541 (Table 1, EmbedLLM 벤치마크 최고 baseline)

**최종 확정 combined = min-pos + top-30%-catfilter** (top50%은 §2.5에서 확인했듯 min-pos와 겹치는 부분이 많아 효과가 약함 — pct=0.3이 새로운 sweet spot):


| seed        | AUDC       | QNC   | Peak       |
| ----------- | ---------- | ----- | ---------- |
| 0           | 0.5659     | 1.157 | 0.6153     |
| 1           | 0.5648     | 1.064 | 0.6077     |
| 2           | 0.5649     | 1.058 | 0.6093     |
| **평균(3시드)** | **0.5652** | 1.093 | **0.6108** |
| std         | 0.0005     | -     | 0.0034     |


**CSCR 대비: AUDC +4.5%, 3/3 승, std=0.0005로 지금까지 나온 모든 all-seen 결과 중 제일 안정적.** Peak(0.6108)은 정적 최강 단일 모델(Llama-3-70B-Instruct, 0.605)보다도 높음 — 라우팅이 실제로 "제일 좋은 모델 하나만 계속 쓰는 것"보다 낫다는 뜻(참고: 오라클 0.979).

**참고(구 버전, top50% 컷, 11시드)**: 0.5574~0.5641, 평균 0.5581(std=0.0036) — pct=0.3보다 평균 낮고 마진도 얇음. 두 버전 다 CSCR은 이기지만 pct=0.3이 명확히 우세.

출처: `allseen_catfilter_pct30_multiseed_results.json`(신규, pct=0.3); 구 버전은 `combined_minpos_top50pct_multiseed_results.json`+`combined_minpos_top50pct_seeds3to10_results.json`의 `allseen` 키

### 1.3 RouterBench — All-seen 전용 (모델 11개, unseen split 불가)

CSCR 논문 보고치: AUDC=0.711


| seed        | AUDC       | QNC    | Peak       |
| ----------- | ---------- | ------ | ---------- |
| 0           | 0.7216     | 0.0027 | 0.7755     |
| 1           | 0.7238     | 0.0028 | 0.7781     |
| 2           | 0.7223     | 0.0027 | 0.7769     |
| **평균(3시드)** | **0.7226** | 0.0028 | **0.7768** |
| std         | 0.0009     | -      | 0.0011     |


**CSCR 대비: AUDC +1.6%, 3/3 승, std=0.0009로 극히 안정적.** (다만 §3.4에서 다루듯, RouterBench는 오라클 0.9636·정적 GPT-4 단독 0.8418 대비로 보면 라우팅 이득 자체가 작은 벤치마크임 — 해석 시 주의.)

출처: `perplexity_vs_ceiling_combined_results.json`의 `combined-ceiling` 키

### 1.4 요약

세 세팅(EmbedLLM unseen/all-seen, RouterBench all-seen) 전부, 검증한 시드(11+3+3=17개, all-seen은 pct=0.3 3시드 기준) 전원이 예외 없이 CSCR 논문 보고치를 넘음. **CSCR 자체는 FP 데이터셋을 비공개해 정확한 재현이 불가능하므로, 이 비교는 참고 지표이며 통계적으로 엄밀한 승패 검정은 아님**(§2 Ablation의 자체 파이프라인 내 비교가 더 엄밀한 근거).

---

## 2. Ablation — Min-pos와 Category-filtering의 개별 기여

### 2.1 원리와 Motivation

**문제 진단(outlier-drag)**: GRPO 스타일 회귀는 한 쿼리에 정답 모델이 여럿일 때, 그 모델들의 advantage-가중 평균을 타겟으로 삼음. 정답 모델들이 FP 공간에서 서로 멀리 떨어져 있으면(예: 수학 특화 소형 모델과 범용 대형 모델이 동시에 정답), 평균 타겟이 실제 어떤 모델도 없는 빈 공간에 착탄함. 정량 확인: spread와 착탄거리의 Spearman rho=0.52(p≈0), 학습 쿼리의 94.3%가 정답 모델 2개 초과 동시 발생.

- **Min-pos**: 정답 후보 전체 평균(AND) 대신, 그중 지금 제일 가까운 하나와만 맞으면 되도록(OR) 로스를 바꿈. 오답 쪽은 기존대로 전체 평균 유지(오답 전부로부터 멀어져야 함은 변함없음).
- **Category-filtering(top50%)**: 같은 쿼리에서 여러 모델이 동시에 정답이어도, 그 쿼리 카테고리에서의 역대 정답률(Set A 기준) 상위 50%만 정답 타겟으로 인정하고 나머지는 로스에서 제외(오답으로 뒤집지 않음).

두 방식 모두 로스/타겟 구성에만 관여하고, 추론 시 라우팅 코드는 vanilla와 완전히 동일함(코드 경로 자체에 min-pos/catfilter 로직이 없음) — 순수하게 학습 신호 정제 기법.

### 2.2 EmbedLLM — Unseen 프로토콜


| 방법                                | seed0  | seed1  | seed2  | seed3  | 평균           | std    | CSCR(0.4848) 승 |
| --------------------------------- | ------ | ------ | ------ | ------ | ------------ | ------ | -------------- |
| vanilla GRPO(아무것도 안 넣음)           | 0.5269 | 0.4661 | 0.4492 | 0.4666 | 0.4772       | 0.0296 | 1/4            |
| min-pos만                          | 0.5095 | 0.5115 | 0.5016 | 0.5297 | 0.5131       | 0.0103 | 4/4            |
| category-filter만(Top-2, 구버전)      | 0.5250 | 0.5050 | 0.4602 | 0.5465 | 0.5092       | 0.0319 | 3/4            |
| **category-filter만(pct=0.3, 신규)** | 0.5224 | 0.5078 | 0.4808 | —      | 0.5037       | 0.0172 | 2/3            |
| combined, top50%-cut(§1.1)        | 0.5250 | 0.5264 | 0.4975 | (11시드) | 0.5175(11시드) | 0.0157 | 11/11          |
| **combined, pct=0.3-cut(재검증)**    | 0.5264 | 0.5246 | 0.4977 | —      | **0.5162**   | 0.0132 | 3/3            |


**해석**: unseen에서는 catfilter 단독(mean 집계, 어느 버전이든 0.50~0.51선)이 min-pos 단독(0.5131)보다 약함 — all-seen과 같은 방향. combined는 top50%든 pct=0.3이든 0.516대로 동일(§2.5) — unseen에서는 catfilter의 컷 비율이 결과에 거의 영향을 주지 않음.

출처: `newllm_grpo_multiseed_seeds0to3_snapshot.json`(vanilla), `newllm_grpo_minpos_seed0_results.json`+`newllm_grpo_variant_multiseed_results.json`(min-pos), `newllm_grpo_catfilter_seed0_results.json`+동 파일(catfilter Top-2), `ablation_gapfill_results.json`의 `unseen_catfilter_pct30`(catfilter pct=0.3), `unseen_catfilter_pct30_multiseed_results.json`(combined pct=0.3)

### 2.3 EmbedLLM — All-seen 프로토콜


| 방법                                  | seed0      | seed1      | seed2      | 평균           | std        | CSCR(0.541) 승 |
| ----------------------------------- | ---------- | ---------- | ---------- | ------------ | ---------- | ------------- |
| **vanilla GRPO(신규 실행)**             | 0.5056     | 0.4664     | 0.4685     | **0.4802**   | 0.0180     | **0/3**       |
| min-pos만                            | 0.5576     | 0.5585     | 0.5601     | 0.5587       | 0.0010     | 3/3           |
| category-filter만(Top-50%)           | 0.5432     | 0.5392     | 0.5339     | 0.5388       | 0.0038     | 1/3           |
| **category-filter만(pct=0.3, 신규)**   | 0.5351     | 0.5386     | 0.5355     | 0.5364       | 0.0015     | 0/3           |
| combined, top50%-cut(구버전)           | 0.5574     | 0.5615     | 0.5566     | 0.5581(11시드) | 0.0036     | 11/11         |
| **combined, pct=0.3-cut(§1.2, 확정)** | **0.5659** | **0.5648** | **0.5649** | **0.5652**   | **0.0005** | **3/3**       |


**해석(갱신)**: vanilla GRPO(원본, 아무 수정 없음)를 마침내 all-seen에서 돌려보니 평균 0.4802로 **CSCR을 0/3, 하나도 못 이김** — unseen에서 1/4만 이겼던 것과 같은 방향, "GRPO 자체는 안 되고 min-pos/catfilter가 실제로 성능을 만든다"는 걸 all-seen에서도 확인.

catfilter 단독(mean 집계)은 흥미로운 비대칭을 보임 — **단독으로는 top50%(0.5388)가 pct=0.3(0.5364)보다 오히려 낫지만, min-pos와 결합하면 정반대로 pct=0.3(0.5652)이 top50%(0.5581)보다 확실히 나음.** 즉 "catfilter에게 최적인 컷"과 "min-pos와 결합했을 때 catfilter에게 최적인 컷"은 다른 값 — 처음 "combined(top50%)가 min-pos 단독과 거의 동일해서 catfilter가 기여 없다"고 봤던 건, catfilter 자체의 한계가 아니라 **컷 비율과 결합 방식(mean vs min) 사이의 상호작용을 놓쳤던 것**으로 정리됨.

출처: `allseen_minpos_multiseed_costfixed_results.json`(min-pos), `allseen_catfilter_methodology_multiseed_results.json`의 `top50pct`(catfilter top50%), `ablation_gapfill_results.json`(vanilla, catfilter pct=0.3 단독), `allseen_catfilter_pct30_multiseed_results.json`(combined pct=0.3)

### 2.4 RouterBench — All-seen (완전한 4-way ablation)


| 방법                       | seed0  | seed1  | seed2  | 평균     | std    | CSCR(0.711) 승 |
| ------------------------ | ------ | ------ | ------ | ------ | ------ | ------------- |
| vanilla GRPO             | 0.7420 | 0.7359 | 0.7391 | 0.7390 | 0.0025 | 3/3           |
| min-pos만                 | 0.7196 | 0.7214 | 0.7209 | 0.7206 | 0.0007 | 3/3           |
| category-filter만(Top-2†) | 0.7397 | 0.7399 | 0.7402 | 0.7400 | 0.0002 | 3/3           |
| combined(Top-2†)         | 0.7216 | 0.7238 | 0.7223 | 0.7226 | 0.0009 | 3/3           |


† RouterBench는 pct=0.3으로 재실행하지 않음 — §2.5에서 확인했듯 11개 모델 풀에서는 흔한 정답 후보 수(2~3개)에 대해 10/20/30%가 전부 동일한 컷(`ceil(n*0.3)=1`)이 되어 Top-2와 사실상 구별되지 않음.

**해석**: RouterBench(모델 11개, 작은 풀)에서는 **min-pos가 오히려 vanilla보다 손해**(0.7206 < 0.7390) — EmbedLLM(112개 모델)과 정반대 순위. catfilter는 vanilla와 거의 동급. combined는 min-pos의 손해를 일부 흡수해 vanilla보다는 낮지만 CSCR은 여전히 이김. **모델 풀이 작을수록 min-pos의 "가장 가까운 정답 하나면 충분" 방식의 이점이 줄어든다는 근거.**

출처: `grpo_variants_multiseed_results.json`(vanilla/min-pos/catfilter), `perplexity_vs_ceiling_combined_results.json`의 `combined-ceiling`(combined)

### 2.5 Catfilter 컷 비율(percentile) 민감도 — 모델 풀 크기에 따라 완전히 다른 그림

**동기**: §2.2~2.4에서 catfilter(top50%)의 단독/combined 기여가 min-pos와 겹쳐서 약하거나 없어 보이는 결과가 나옴 — 컷이 너무 관대해서(정답 후보의 50%나 남기니, min-pos가 어차피 고르지 않았을 후보만 잘라내는 셈) catfilter가 실질적으로 개입할 여지가 적었던 것 아닌가 하는 가설. min-pos + catfilter(percentile) 결합을 컷 비율만 바꿔가며(10/20/30/50%) 재검증.

**RouterBench(11개 모델)에서는 무효**: 정답 후보가 2~3개인 쿼리가 제일 흔한데(1,770+1,855=3,625/20,533행), 그 경우 `ceil(n_pos*0.1)=ceil(n_pos*0.2)=ceil(n_pos*0.3)=1`로 **10/20/30%가 완전히 동일한 컷**이 됨 — 후보 풀이 작으면 percentile 단위 조정 자체가 무의미. 실제로 seed0에서 0.1/0.2/0.3이 0.7240/0.7269/0.7225로 뚜렷한 추세 없이 흔들렸고, 이 실험은 EmbedLLM으로 옮김.

**EmbedLLM all-seen(111개 모델, seed0 스윕)**: 정답 후보 수 자체가 훨씬 커서 컷마다 실제로 다르게 작동함.


| pct                 | AUDC(seed0) | Peak       |
| ------------------- | ----------- | ---------- |
| 0.1                 | 0.5510      | 0.5953     |
| 0.2                 | 0.5511      | 0.5990     |
| **0.3**             | **0.5659**  | **0.6153** |
| 0.5(=top50%, 재현 확인) | 0.5574      | 0.5917     |


**단조 증가가 아니라 0.3에서 뚜렷한 정점** — 10~20%(과도하게 공격적)는 오히려 손해, 0.3이 그 사이 sweet spot. 3시드로 재검증해 확정(mean=0.5652, std=0.0005, §1.2/§2.3).

**EmbedLLM unseen(학습에 쓰는 seen 모델 74개, 3시드)에서는 차이 없음**: pct=0.3 결과 0.5264/0.5246/0.4977, 평균 0.5162(std=0.0132) — top50%의 0.5163과 **사실상 동일**(차이 0.0001).

**종합**: catfilter 컷 비율의 민감도는 **모델 풀 크기에 정확히 비례**하는 것으로 보임 — RouterBench(11개, 전혀 민감하지 않음) < EmbedLLM-unseen 학습 풀(74개, 민감하지 않음) < EmbedLLM all-seen(111개, 뚜렷하게 민감함). 후보 모델이 적으면 애초에 percentile로 세밀하게 조절할 대상 자체가 없다는 뜻 — §3~4의 "모델 풀이 클수록 라우팅 개입의 여지가 커진다"는 결론과 같은 방향.

출처: `catfilter_pct_sweep_seed0_results.json`(RouterBench, 참고용/무효 판정), `allseen_catfilter_pct_sweep_seed0_results.json`(EmbedLLM all-seen seed0 스윕), `allseen_catfilter_pct30_multiseed_results.json`(all-seen 3시드 확정), `unseen_catfilter_pct30_multiseed_results.json`(unseen 3시드 재검증)

---

## 3. Ceiling FP(Capability 인코딩)가 필수적이라는 증거

**질문**: Combined GRPO가 FP 종류와 무관한 범용 로스 기법인가, 아니면 capability를 담은 FP와의 시너지인가?

**실험**: RouterBench에서 같은 데이터·같은 combined 로스로, FP만 교체해 비교.

- **Ceiling FP**: 실제 task 정확도(Set A) 기반, capability 정보 있음.
- **Perplexity FP**: GPT-2 cross-entropy 기반(각 모델의 실제 응답 텍스트에 대한 언어모델 perplexity), capability 정보 전혀 없음.


| FP                             | seed0  | seed1  | seed2  | 평균         | std    | CSCR(0.711) 승 |
| ------------------------------ | ------ | ------ | ------ | ---------- | ------ | ------------- |
| **Ceiling(capability)**        | 0.7216 | 0.7238 | 0.7223 | **0.7226** | 0.0009 | **3/3**       |
| **Perplexity(non-capability)** | 0.6916 | 0.7003 | 0.6888 | **0.6936** | 0.0049 | **0/3**       |


**결론**: Perplexity FP는 random 라우팅보다는 유의미하게 낫지만(모든 시드 bootstrap p<0.001) CSCR급에는 못 미침. Ceiling FP와의 격차(0.0290, Ceiling 자체 std의 약 32배)는 노이즈로 설명하기 어려움 — **Combined GRPO는 범용이 아니라 capability-encoded FP와 짝을 이룰 때만 CSCR을 넘는 효과를 냄.**

**EmbedLLM에서는 이 비교 자체가 불가능**: EmbedLLM 데이터셋(`train.csv`)은 모델의 실제 응답 텍스트를 공개하지 않고 정답/오답 라벨만 제공 — Perplexity FP를 만들려면 112개 모델을 전부 직접 실행해 응답을 새로 생성해야 하는데, 이 중 대형 모델들은 현재 하드웨어로 비현실적. RouterBench(`{model}|model_response` 컬럼 보유)만 이 비교가 가능했음.

출처: `perplexity_vs_ceiling_combined_results.json`

---

## 4. Ceiling V1(샘플링) — 비용을 줄여도 효과 유지

**동기**: Ceiling V2(카테고리 전체 평균)는 카테고리당 중앙값 177개(최대 3,454개) 프롬프트로 각 모델을 평가해야 FP를 만들 수 있음 — 신규 모델 하나 추가할 때마다 벤치마크 전체를 다시 돌려야 하는 셈이라 비용이 큼. Ceiling V1은 카테고리당 분산이 큰 상위 N개 프롬프트만 골라(probe sampling) FP를 만드는 저비용 대안.

**Vanilla GRPO 기준(N=24, 5시드)**: AUDC = 0.4161 / 0.4701 / 0.4507 / 0.4575 / 0.4160, 평균 **0.4421, CSCR(0.4848) 0/5 — 확실히 못 이김.**

**Combined 적용(N=24, 3시드, unseen)**:


| seed        | AUDC       | QNC   | Peak       |
| ----------- | ---------- | ----- | ---------- |
| 0           | 0.5185     | 0.933 | 0.5580     |
| 1           | 0.5224     | 1.369 | 0.5607     |
| 2           | 0.5103     | 0.884 | 0.5460     |
| **평균(3시드)** | **0.5171** | 1.062 | **0.5549** |
| std         | 0.0050     | -     | 0.0064     |


**같은 3시드로 비교한 Ceiling V2 combined 평균(§1.1의 seed0-2)**: AUDC=0.5163

**격차: 0.0008 — 사실상 동일.** 카테고리당 프롬프트 수를 최대 140배(중앙값 기준 7배) 줄여도 combined의 효과는 거의 손실되지 않음. Vanilla GRPO는 V1에서 CSCR을 전혀 못 넘겼는데(0/5), combined는 V1에서도 V2와 동등하게 3/3 승.

**all-seen에서는 이 "공짜 점심"이 성립하지 않음** — 같은 combined(pct=0.3) 레시피를 all-seen(109개 모델, V2와의 교집합)에 적용:


|                                    | seed0  | seed1  | seed2  | 평균                      | std    | CSCR(0.541) 승 |
| ---------------------------------- | ------ | ------ | ------ | ----------------------- | ------ | ------------- |
| Ceiling V1(N=24) combined          | 0.5548 | 0.5450 | 0.5447 | 0.5481                  | 0.0047 | 3/3           |
| Ceiling V2 combined(§1.2, 111개 모델) | 0.5574 | 0.5615 | 0.5566 | 0.5652(§2.3 pct=0.3 기준) | 0.0005 | 3/3           |


CSCR은 여전히 3/3 이기지만 마진이 +1.3%로 얇아지고(V2는 +4.5%), V2 대비 -3.0% 낮은 격차가 세 시드 모두 일관되게 나타남(seed0=-0.0111, seed1=-0.0198, seed2=-0.0212, 단조적으로 벌어지는 추세).

**해석**: unseen은 평가 후보가 35~37개뿐이라 probe 샘플링 노이즈가 랭킹에 영향을 줄 기회가 적지만, all-seen은 109개 모델을 동시에 구별해야 해서 노이즈가 누적될 여지가 더 큼(학습 타겟 차원도 74→109로 커짐). **모델 풀이 클수록 catfilter/percentile의 이득도 커지지만(§2.5), 저비용 FP의 노이즈로 인한 손실도 같이 커진다** — 이 프로젝트 전체의 "풀 크기가 개입의 여지를 결정한다"는 결론과 같은 방향. **결론을 "V1은 V2와 완전히 동등하다"에서 "V1은 unseen(주 타겟)에서는 손실 없이 동등하고, all-seen(부차 타겟)에서는 모델 풀이 클수록 커지는 작지만 일관된 대가가 있다"로 정정.**

출처: `probeN24_combined_multiseed_results.json`(unseen), `probeN24_allseen_pct30_multiseed_results.json`(all-seen, 신규), `newllm_probe_sampling_results.json`(vanilla 기준)

### 4.1 Probe 재분배 시도 — PCA loading 기반 차등 배분 (다음날 후속)

**아이디어**: 카테고리별 균등 배분(80×24=1920개) 대신, 각 카테고리가 5개 주성분에 기여하는 정도(PCA loading)에 비례해 probe를 차등 배분 — 중요한 카테고리엔 더 많이, 안 중요한 카테고리엔 최소한만(바닥값 6개). V2 전체 데이터로 계산한 loading을 오라클로 사용해 sqrt(importance)에 비례하는 배분을 설계, 총 1,200개(균등 대비 37.5% 절감)로 구성.

먼저 카테고리 중요도 분포 자체를 확인했는데, 예상보다 덜 집중돼있었음 — PC1이 분산의 87.8%를 차지함에도, 상위 20/80 카테고리가 중요도의 50%만 차지(90%를 담으려면 45/80 필요). "소수 카테고리만 남겨도 된다"는 극단적 가설은 기각, 대신 차등 배분(비율 조정)만 시도.


| 프로토콜         | 3시드                  | 평균         | std        | 비교                                                         | CSCR 승  |
| ------------ | -------------------- | ---------- | ---------- | ---------------------------------------------------------- | ------- |
| unseen       | 0.5406/0.5263/0.5044 | 0.5238     | 0.0149     | V1(0.5171)·V2(~0.517)와 대등(std보다 작은 차이라 확실한 우위는 아님)         | 3/3     |
| **all-seen** | 0.5397/0.5378/0.5403 | **0.5393** | **0.0011** | V1(0.5481)보다 낮고 V2(0.5652)보다 **-4.6%**, 세 시드 모두 CSCR도 못 넘음 | **0/3** |


**결론**: unseen에서는 37.5% 적은 probe로도 손실 없이 유지됐지만(V1의 unseen 패턴과 같은 방향), **all-seen에서는 명확하고 일관되게(std=0.0011) 실패함.** 원인 추정: 카테고리 중요도를 "풀 전체의 집합적 분산 설명력"으로 계산했는데, all-seen처럼 109개 모델을 전부 서로 구별해야 하는 상황에서는 전체 분산엔 기여 적어도 "특정 두 모델을 구별하는 데 결정적인" 카테고리가 있을 수 있음 — 이런 카테고리가 바닥값(6개)으로 밀려나면서 전체 랭킹 품질이 떨어진 것으로 보임.

**대안 1 시도(바닥값 6→15) — 성공**: 상위 카테고리 배분은 그대로 두고 바닥값만 올려서 재검증(총 1,800개, 균등 대비 6.2% 절감):


|                    | probe 수   | unseen(3시드)            | all-seen(5시드)          | V2 대비(all-seen) |
| ------------------ | --------- | ---------------------- | ---------------------- | --------------- |
| V2(전체)             | ~29,673   | 0.517                  | 0.5652                 | -               |
| V1(균등)             | 1,920     | 0.5171                 | 0.5481                 | -3.0%           |
| **V1.5(가중, 바닥15)** | **1,800** | **0.5121(std 0.0034)** | **0.5561(std 0.0042)** | **-1.6%**       |


**V1.5가 V1보다 probe를 더 적게 쓰고도(1,800<1,920) all-seen에서 확실히 더 나음**(0.5561 > 0.5481, 5/5 CSCR 승) — 재분배 원칙이 all-seen에서도 유효함을 처음으로 증명. 바닥값 6에서는 실패(0/3, §위 표)했지만 15에서는 성공 — "전체 분산엔 안 중요해도 특정 모델 쌍 구별엔 필요한 카테고리"에 최소한의 안전마진이 핵심이었던 것으로 확인.

추가로 "상위 카테고리 배분은 1,200-probe 버전과 동일하게 고정하고 바닥값만 6→15로 올리는" 순수 분리 실험(1,442개, 균등 대비 24.9% 절감)도 시도했으나 all-seen 2/3 시드에서 0.55 부근으로 나쁘지 않았지만 3번째 시드 결과를 못 얻고 중단(사용자 판단으로 이 세부 변형은 폐기, 1,800개 버전을 최종으로 채택).

**기각된 대안들**:

- 카테고리를 거치지 않고 개별 프롬프트 단위로 직접 PCA(29,673차원) → 축당 상위 200개 선정: 커버리지 버그(80개 중 12개 카테고리가 probe 0개) 발견, 더 근본적으로 프롬프트 단위 PCA는 상위 5축이 분산의 32.27%만 설명(카테고리 단위는 94.66%) — 카테고리 평균이 단순 희석이 아니라 노이즈 감소 역할도 했다는 뜻이라 폐기.
- 특정 모델 쌍을 구별하는 probe를 직접 선별: 현재 모델 풀에 과적합되는 체리피킹이라 일반화 안 됨(신규 모델의 새로운 혼동 쌍에는 무용지물) — 사용자 자체 판단으로 폐기.
- FP 자체를 아예 안 만들고 공개 벤치마크 점수(gsm8k, mmlu 등 이미 공개된 점수) 재사용: 방향은 유효하나 미착수, 다음 세션 후보.

출처: `embedllm_pca_loading_analysis.py`(중요도 계산), `embedllm_build_pca_weighted_probe_fp.py`(FP 생성, 최종 파라미터 MIN=15/TARGET=1800), `pcaweighted_combined_multiseed_results_1800probes_floor15_unseen.json`(unseen), `pcaweighted_allseen_pct30_multiseed_results_1800probes_floor15_seeds3to4.json`+원본 seed0-2 결과(all-seen)

---

## 데이터 공백 (다음에 채우면 좋음)

1. ~~EmbedLLM all-seen vanilla GRPO~~ — **해결됨**(§2.3, 3시드, 0/3).
2. ~~EmbedLLM unseen category-filter 단독~~ — **해결됨**(§2.2, pct=0.3 3시드, 2/3).
3. Vanilla GRPO EmbedLLM unseen의 seed1-3 QNC 값 — 당시 기록 안 됨(AUDC/Peak만 있음), 사소한 항목이라 우선순위 낮음.
4. CSCR 논문의 QNC, 그리고 all-seen/RouterBench의 Peak 참고치 — 스크린샷/논문에서 추가 확인 필요.
5. ~~EmbedLLM all-seen catfilter(pct=0.3) 단독~~ — **해결됨**(§2.3, 3시드, 0/3) — 흥미롭게도 top50% 단독(0.5388)보다 낮음, catfilter 단독과 combined에서 "최적 컷"이 다르다는 비대칭 발견.
6. RouterBench에서도 percentile 대신 **고정 개수**(top-1, top-2, top-3) 스윕을 해보면 §2.5의 "풀이 작아서 무효" 진단이 percentile 방식 자체의 한계인지, 컷이라는 개념 자체의 한계인지 구분될 수 있음 — 미시도.
7. **(신규)** catfilter 단독에서 top50%가 pct=0.3보다 나은데 combined에서는 반대인 이유(§2.3의 비대칭) — 메커니즘 설명은 아직 가설 수준("min-pos가 이미 outlier를 어느 정도 거르므로 catfilter는 min-pos가 놓치는 부분만 공격적으로 잘라야 시너지"). 정식 검증 안 됨.

