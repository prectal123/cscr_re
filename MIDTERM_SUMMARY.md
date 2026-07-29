# CSCR 재구현 — 중간 발표용 정리 (재구현 + Observation only)

**범위 안내**: 이 문서는 순수 재구현과 그 과정에서 나온 관찰(Observation)만 담습니다. 개선 방안(새 FP 방법론 제안 등, `FP_IDEAS.md` 참고)은 의도적으로 제외했습니다 — 중간 발표는 "재구현 + 문제 진단"에 집중하기 위함입니다.

**완료/진행 중 표기 원칙**: 실제로 수치 결과까지 나온 것은 "완료", 환경/스크립트만 준비되고 아직 결과가 없는 것은 "진행 중/예정"으로 명확히 구분했습니다.

---

## 1. 재구현 목적 및 논문 주장

**대상 논문**: CSCR (Cost-Aware Contrastive Routing for LLMs), NeurIPS 2025 spotlight, arXiv:2508.12491
원본 GitHub: https://github.com/rezashkv/cscr

**논문의 핵심 주장 (Section 4.3.1, Table 4)**: MixInstruct 벤치마크에서 logit descriptor(6개 모델)와 perplexity descriptor(5개 모델)를 같은 FAISS pool 안에 섞어도(Mixed row) AUDC가 거의 변하지 않음(Logit 0.0461 → Perp 0.0467 → Mixed 0.0473) → "두 descriptor는 혼합해도 문제없다"는 근거로 사용, "unified metric" 논지를 뒷받침.

**재구현의 목적**: 이 주장의 근거가 되는 파이프라인을 실제로 재현하여, 그 결과가 얼마나 견고한지, 그리고 두 descriptor 및 평가 방법론 자체에 어떤 구조적 문제가 있는지 진단.

---

## 2. 원본 대비 변경한 실험 조건과 그 이유

| 항목 | 원본(논문/리포) | 우리 재구현 | 변경 이유 |
|---|---|---|---|
| MixInstruct 모델 pool 크기 | 11개 | **7개** | 4개는 실제 시도로 확인된 이유(아래 4번 표)로 제외 |
| Probe 개수 (N) | 192 | **192로 통일** | 처음엔 32로 축소 테스트했으나, 논문의 "N=K" 조건(9번 참고)을 지키기 위해 최종적으로 192로 확정 |
| Logit descriptor의 top-k 토큰 수 (K) | 256 | **192로 조정** | 논문 Section 3.1.2가 "N=K"를 unified space의 조건으로 명시했는데, 정작 논문 실험(K=256, N=192)은 이 조건을 안 지킴(9.1 참고) — 우리는 논문이 스스로 말한 조건을 실제로 지켜서 재현 |
| 실행 환경 | 원본 미지정 | Colab(초기) → 로컬 GPU(RTX 5060 Ti 8GB) 인프라로 이전 | Colab 무료 티어 GPU 사용량 제한에 실제로 도달하여 이전 |
| Perplexity descriptor의 결측치 처리 | 원본 없음(암묵적으로 미처리) | 결측 probe를 0-fill 처리 | 원본 코드가 이 경우를 처리 안 해서 크래시 발생(5.2 참고), 논문의 N=K 조건을 지키려면 차원 유지가 필요해 열 삭제 대신 0-fill 선택 |

---

## 3. 재구현 과정에서 발견한 원본 리포의 결함 (코드 버그)

이번 재구현으로 **실제 코드를 실행해봐야만 드러나는** 3가지 결함을 발견함:

### 3.1 Padding 방향 버그
`compute_logit_descriptor()`가 `tokenizer.pad_token`은 설정하지만 `tokenizer.padding_side`는 설정하지 않아 기본값(right-padding)이 적용됨. Causal LM의 배치 생성(`batch_size>1`)에서 오른쪽 패딩은 실제 오류를 유발함 — 배치 내 최장 문장을 제외한 나머지는 실제 마지막 토큰이 아닌 PAD 토큰 위치에서 생성이 시작되어 확률값이 왜곡됨.

### 3.2 Perplexity descriptor의 NaN 전파 버그
짧거나(GPT2 기준 1토큰 이하) 응답이 있으면 `perplexity_fingerprint()`가 `inf`를 반환하고, 이 값이 L2 정규화 과정에서 해당 모델의 descriptor 벡터 전체를 손상시킴(정상 값들은 0으로 소실, 문제 값은 NaN으로 남음). 이 NaN이 `cosine_similarity()`(`--plot` 옵션)를 크래시시키는데, 이 체크가 파일 저장 로직보다 먼저 실행되어 **정상적으로 계산된 다른 모델들의 결과까지 전부 저장되지 않는** 구조.
- 원인 추적 결과, 문제 응답들은 실제로 빈 응답이 아니라 **짧지만 정답인 응답**(`"400"`, `"Yes"`, `"paper"` 등)이었고, 9건 중 6건이 `flan-t5-xxl`(QA에 단답하는 성향)에서 발생 — 무작위 엣지케이스가 아니라 간결하게 답하는 모델일수록 더 자주 걸리는 구조적 패턴.
- Git 히스토리 확인 결과, 이 버그는 파일 최초 커밋부터 있었고 이후 "bug fixes" 커밋들에서도 수정되지 않음. 원 리포의 `end_to_end.sh`도 이 옵션(`--plot`)을 그대로 사용하므로, 저자들의 정식 파이프라인도 동일 상황에서 크래시가 났어야 함 — 공개된 재현 코드가 실제 논문 결과를 생성한 코드와 다를 수 있다는 정황 증거.

### 3.3 GPU 메모리 누적 버그
`compute_logit_descriptor()`의 배치 루프가 `enc/gen/logits/probs` 텐서를 명시적으로 해제하지 않아 CUDA 캐시 파편화가 누적됨. N=32(8배치)에서는 문제없었으나 N=192(48배치)로 확장하자 `flan-t5-xxl`이 Colab T4(14.56GB)에서 OOM 발생. 배치마다 `del` + `torch.cuda.empty_cache()` 추가로 해결.

---

## 4. MixInstruct 모델 Pool — 최종 7개 확정 (원본 11개 대비)

| 상태 | 모델 | 비고 |
|---|---|---|
| ✅ 포함 | vicuna-13b-1.1, alpaca-native, stablelm-tuned-alpha-7b, oasst-sft-4-pythia-12b, koala-7B-HF, flan-t5-xxl, chatglm-6b | chatglm-6b는 구버전 `transformers==4.33.0` 전용 환경 구축으로 살려냄 |
| ❌ 영구 제외 | databricks/dolly-v2-12b, mosaicml/mpt-7b-instruct | HF Hub API로 재확인 — 저장소 자체가 완전히 삭제됨(`401 Repository Not Found`) |
| 🔄 미해결(GPU 서버 재시도 대상) | fnlp/moss-moon-003-sft, mosesjun0h/llama-7b-hf-baize-lora-bf16 | 전자는 순수 VRAM 문제(16B), 후자는 토크나이저 파일 누락 — 최종 재구현 결과에는 미포함 |

**Observation**: 2023년식 개인/소규모 팀 업로드 위주로 구성된 벤치마크가, 시간이 지나며 자연스럽게 마모(model rot)되는 현상을 실측으로 확인함.

---

## 5. 정량적 Observation ① — Logit/Perplexity descriptor 벡터 기하학 분석

`local_descriptors/`의 7개 모델 logit·perplexity descriptor(각 192차원) 기준.

**A. 타입 간 분리도**
- Silhouette score = 0.123
- MMD = 0.289, 정확 순열검정(14개를 7:7로 나누는 전체 3432가지) **p = 0.0012** — 통계적으로 유의미하게 구분됨

**B. 구조 재현도(RSA)** — 더 중요한 결과
- Logit 기준 7×7 유사도 행렬 vs Perplexity 기준 7×7 유사도 행렬의 Spearman 상관 **rho = −0.079**(사실상 0)
- 정확 Mantel 검정(7!=5040가지 순열 전부) **p = 0.762** — 유의미한 상관 없음
- 극적인 예시: `koala-7B↔vicuna-13b`(logit 0.964 vs perp 0.212), `alpaca-native↔oasst-pythia`(logit 0.094 vs perp 0.890, 완전 반전)
- **해석**: rho≈0은 "같은 공간을 다르게 좌표화"(회전/재배열, 이 경우 rho가 1에 가까웠어야 함)가 아니라, **모델 간 유사도 관계 구조 자체가 두 descriptor 사이에 독립적**이라는 뜻 — 논문의 unified metric 주장에 대한 직접 반박 근거.

**C. Capability(BartScore 기반)와의 비교 — 완료** (7개 모델 × 105,000 프롬프트로 구축한 capability vector, 3번 문서 참고)

| 비교 | rho | p (정확 Mantel, 7!=5040) |
|---|---|---|
| Logit vs Perplexity | −0.079 | 0.762 (유의하지 않음) |
| Logit vs Capability | −0.252 | 0.411 (유의하지 않음) |
| Perplexity vs Capability | −0.226 | 0.422 (유의하지 않음) |

**Logit도 Perplexity도 서로뿐 아니라 실제 capability와도 통계적으로 무관함** — "두 descriptor가 서로 다른 걸 잰다"를 넘어서 "둘 중 어느 것도 진짜 능력을 반영하지 못한다"는 더 강한 주장의 근거.

**D. 종합 스칼라 지표** (원값이 아니라 순위 기반 — 시각화 가시성 피드백에 따라 순위 중심으로 재구성)
- **Kendall's W(3개 표현 전체의 종합 일치도) = 0.2095**, Monte Carlo p=0.9346(유의하지 않음, n=10,000 순열)
- **평균 순위 이동량**: 세 비교 전부 21계단 중 평균 7.6~7.7계단 이동 — **완전 무작위로 순서를 섞었을 때 예상되는 이동량(6.99계단)과 통계적으로 구분 안 됨**. 통계 지식 없이도 바로 이해되는 한 줄 요약: "이 세 방식은 사실상 서로 무작위로 섞어놓은 것과 다를 바 없다."

산점도(`rsa_scatter_3way.png`)와 Rank bump chart(`rank_bump_3way.png`)로 시각화 완료 — `local_descriptors/analysis/` 참고.

---

## 6. 정량적 Observation ② — 라벨 구조 및 라우터 붕괴 현상

### 6.1 MixInstructOracle 라벨의 근본적 편향
`MARGIN=0.1` 기준 이진 라벨(0/1) 분포를 7개 pool 기준 확인(train 100,000 + validation 5,000 샘플):
- 7개 모델 전부 **label=1 비율 85~95%** — 대부분의 프롬프트에서 거의 모든 모델이 동시에 "정답"으로 라벨링됨
- **단독 1위(sole winner) 비율은 전부 1% 미만** — 프롬프트별로 "이 모델이 낫다"는 차별적 신호가 라벨에 거의 없음

### 6.2 라우터 붕괴 현상 (예비 관찰, 4회 비공식 sweep 기준 — 엄밀한 다중 시드 검증은 진행 중/예정)
`n_bands`(5→3→2→1)를 바꿔가며 학습한 결과, 매번 **프롬프트와 무관하게 전문가 1명한테 98~99% 쏠림**이 관찰됨:
- n_bands=5: oasst-pythia-12b (72%)
- n_bands=3, 2: flan-t5-xxl (94~99.6%, 2회 독립 발생)
- n_bands=1: alpaca-native (99.6%)

AUDC가 좋게/나쁘게 나오는지는 붕괴 대상이 우연히 성능 좋은 모델이었는지에 좌우되어 재현성이 없었음. flan-t5-xxl(유일한 encoder-decoder 구조, 5번 섹션 RSA 분석에서도 outlier)이 2회 반복 붕괴 대상이 된 것은 우연이 아닐 가능성이 있으나, **표본이 4회뿐이라 통계적으로 확정된 결과는 아님**.

**⚠️ 진행 중/예정**: 이 현상을 엄밀하게 검증하기 위한 다중 시드(10~15회) 재현 실험은 환경 구축까지만 완료되었고, 아직 실행 전임. 논문 명시 하이퍼파라미터(아래 8번)를 그대로 사용한 순수 baseline 재현을 우선 진행하고, 이후 원인 규명용 ablation(batch_size, MARGIN 등)을 별도로 진행할 계획.

---

## 7. 정량적 Observation ③ — Capability Ground Truth(BartScore) 자체의 신뢰도 문제

7개 모델 pool 기준 bartscore(MixInstruct의 reference-based 자동평가 지표)를 직접 조사:

- **응답 길이가 bartscore와 상관관계를 보임**(Pearson r=0.24, Spearman rho=0.31, n=28,000쌍, p≈0) — 길게 답할수록 점수가 유리해지는 경향
- **참고 답변(reference)이 짧을수록 모델 간 점수 편차가 커짐**(Pearson r=−0.45, p≈2.5×10⁻¹⁹⁹, n=4,000 프롬프트) — 짧은 reference를 가진 프롬프트일수록 자동 채점이 불안정해짐

**Observation**: 이는 앞서(3.2) 발견한 "짧은 텍스트에서 LM 기반 자동 채점이 불안정해진다"는 문제와 동일 계열이며, **descriptor 계산 단계뿐 아니라 capability를 재는 ground truth(bartscore) 자체에도 나타나는 반복적 패턴**임을 시사함 — MixInstruct 벤치마크가 기반한 "LM 기반 자동 평가" 방법론 전반의 구조적 약점으로 해석 가능.

---

## 8. 논문에 명시된 학습 하이퍼파라미터 (Appendix D)

> "Training is performed for 10 epochs using the AdamW optimizer with a batch size of 512 and a learning rate of 5×10⁻⁴. For the cost spectrum loss, we set the number of cost bands to K=5 and the negative cost penalty to λ=0.1. The hyperparameters for the linear schedule of band-specific temperatures are set as α=0.25 and τ_min=0.05."

이 값들은 데이터셋/pool 크기별 구분 없이 EmbedLLM·MixInstruct·RouterBench 전체에 동일하게 적용된 것으로 보임(논문 텍스트에 pool 크기별 조정 언급 없음) — 6번 섹션의 붕괴 현상과 관련하여, 큰 배치 크기(512)가 이미 약한 프롬프트별 신호(6.1)를 더 희석시킬 가능성에 대한 배경 정보로 참고.

---

## 9. 논문 자체의 내적 불일치 (코드 버그와는 별개 카테고리)

### 9.1 "N=K" 조건을 논문 스스로 안 지킴
Section 3.1.2가 "N=K로 맞춰야 두 descriptor가 같은 unit sphere에 놓인다"(logit의 top-k 토큰 수 = perplexity의 probe 수)고 명시. 그러나 실제 실험 설정(Section 4.1/Appendix D.1)은 K=256(logit), N=192(probe)로 서로 다름 — Table 4의 "Mixed" 결과가 바로 이 설정에서 나옴.

### 9.2 Cost-Spectrum InfoNCE — 본문과 부록 증명의 공식이 다름
- 본문 Eq.8: 분모가 `exp((q^⊤e_m' − γc_m')/τ_k)` — 유사도와 비용 페널티를 함께 τ_k로 나눔
- Appendix B.3(Lemma 5.3 증명): `Sim = q^⊤e_m/τ_k`로 먼저 정의한 뒤 `exp(Sim_m' − γc_m')` — 비용 페널티는 τ_k로 안 나눔
- 실제 코드는 본문(Eq.8) 버전을 구현 — 즉 코드에 이론적 보장(Lemma 5.3)이 실제로는 증명되지 않은 형태로 적용되고 있을 가능성.

---

## 10. 다음 단계 (재구현/Observation 범위 내에서 아직 안 끝난 것)

1. ✅ **완료**: RSA(Logit, Capability), RSA(Perplexity, Capability) + 종합 스칼라(Kendall's W, 평균 순위 이동량) — 5번 섹션 C·D 참고
2. **다중 시드 baseline 재현** — 논문 명시 하이퍼파라미터(8번) 그대로, 시드만 바꿔 10~15회 반복하여 6.2번 붕괴 현상의 통계적 유의성 확인
3. (2번 완료 후) **원인 규명용 ablation** — batch_size, MARGIN을 baseline과 명확히 분리하여 별도로 변화시켜, 붕괴의 원인을 진단(단, 이는 "개선"이 아니라 "왜 이런 baseline 결과가 나오는지"에 대한 진단 목적임을 발표에서 명시)

---

## 11. 이 문서에서 제외한 것

- 새 FP 방법론 제안(v1.1 Lexical Fingerprint, v1.2 LLM 백본 기반 전문성 임베딩) — `FP_IDEAS.md` 참고, 개선 방안이라 이번 중간 발표 범위 밖으로 명시적으로 분리함
