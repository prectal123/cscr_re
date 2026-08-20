# 발표 자료 구성안 (2026-08-20 정리)

메인 vs Appendix 기준: **이야기 흐름에 필수인가** vs **질문 받으면 방어용으로 꺼낼 근거인가**.
숫자 원본은 [Results_Reference_Table.md](Results_Reference_Table.md), 과정/맥락은 `PROGRESS.md` 참고.

---

## 메인 발표 순서

### 1. 동기
CSCR §4.3.1 재검토 — descriptor mixing(logit/perplexity) "영향 없음" 주장이 단일 trial·얇은 근거(§4.3.1 관련: `project_cscr_reimplementation.md` 상단 참고)에서 출발. "descriptor 구성이 로스 설계보다 중요한 거 아닐까"라는 질문으로 발전.

### 2. 방법론
COMPAR = **Ceiling FP**(capability 중심 — 카테고리별 실제 정답률, 중심화+정규화) + **TAR 로스**(GRPO식 쿼리별 표준화 + catfilter + min(0.3,3)). 간단히, 수식은 `Final_Result_Summary2.MD` §0 참고.

### 3. 헤드라인 결과
3개 벤치마크(EmbedLLM/RouterBench/LLMRouterBench) × All-seen/Unseen, 전부 CSCR 이김.
→ Results_Reference_Table.md §1

### 4. Fairness — CSCR 자기 예산(192)에 맞춰도 이기는가
세 칸(EmbedLLM All-seen/Unseen, RouterBench) 전부 192-probe로도 승리, 일부는 1800보다 오히려 높음.
→ Results_Reference_Table.md §2
**선제 방어**: "COMPAR가 더 많은 자원을 써서 유리했다"는 비판 무력화.

### 5. Probe / Model 스케일링 — 늘려도 무관/포화
Probe 80→30000(EmbedLLM), 64→2345(LLMRouterBench) 늘려도 AUDC 거의 불변. Model 5→111도 로그형으로 오르다 75 근방 포화.
→ Results_Reference_Table.md §3, §8
**선제 방어**: "왜 1800개나 썼냐" 질문 자체를 무력화 — 우리도 그만큼 필요하다고 주장한 적 없고, 오히려 이게 발견.

### 6. 핵심 — FP × Loss 2x2 그리드 (클라이맥스)
로스를 CSCR 것 그대로 둬도 FP만 Ceiling으로 바꾸면 이기고(RouterBench 0.7147, EmbedLLM Unseen 0.5193, 둘 다 기준 이김), FP를 CSCR 것 그대로 두고 로스만 TAR로 바꾸면 오히려 크게 나빠짐(0.6170). **격차의 원인이 FP라는 직접 증거.**
→ Results_Reference_Table.md §6

### 7. 왜 되는가 — FP 분리도(separation geometry) 메커니즘
Perplexity FP는 모델 간 코사인 유사도가 거의 안 갈라짐(0.770/0.725), Ceiling FP는 뚜렷이 갈라짐(0.367/−0.440). All-seen(암기)·Unseen(일반화) 둘 다 좋은 이유가 사실 하나의 메커니즘.
→ Results_Reference_Table.md §7

### 8. TAR 로스의 진짜 역할 — 평균 성능이 아니라 안정성
Vanilla loss와 평균 AUDC는 비등비등(승자가 지점마다 바뀜)하지만, probe-count 구간 표준편차는 TAR가 2.4배 작음(0.0039 vs 0.0096). Collapse 진단에서도 "폭을 줄이는" 게 아니라 "향하는 방향을 교정"(rho 0.447→0.669). 정직한 재정의.
→ Results_Reference_Table.md §5

### 9. 한계와 결론
아래 "정해야 할 것" 반영해서 마무리.

---

## Appendix (질문 나오면 꺼낼 것)

| 소재 | 방어 대상 질문 |
|---|---|
| Negative control 디테일(노이즈 FP, 셔플 FP, random selection) | "진짜 신호가 필요한 이유가 뭔데?" |
| 3-way ablation·vanilla loss 전체 표 | "TAR가 안정성에 기여한다는 근거를 더 보여봐" |
| Collapse 진단 전체(top3_share, rho, 4-way) | "collapse를 줄였다는 거야 방향만 바꿨다는 거야?" |
| 300-probe 이상치 조사, PCA 압축 버그 발견 과정 | "결과가 우연 아니야? 검증 제대로 한 거 맞아?" |
| 논문/repo 원본 대조(AUDC 코드 byte-identical, CSCR 실제 예산 192 확인) | "네 비교 방법 자체가 CSCR이랑 같은 기준이 맞아?" |
| **K-Means Adaptive Clustering FP** (§9 in Results table) | "카테고리 라벨 없으면 이 방법 아예 못 쓰는 거 아니야?" |

---

## 아직 정해야 할 것 (2026-08-20 기준 미결정)

1. **Mixed-pool 한계 공개 여부** — Unseen 프로토콜에서 seen 모델이 후보에 섞이면 unseen_only 정답 쿼리 적중률이 0/206(0%)이라는 결과(Final_Result_Summary2.MD §1.11 근방, PROGRESS.md 26.7). 메인에 "한계"로 명시할지, Appendix에도 안 넣고 넘어갈지 미정 — 정직하게 가면 신뢰도는 오르지만 공격 포인트가 될 수 있음.
2. **Model-count 스윕 포함 여부** — n=20 dip 원인 미해결, fixed-probes 수정판 본 실행도 미완료(§8 in Results table). 이번 발표에서 Appendix로라도 넣을지, 다음 기회로 완전히 미룰지 미정.

이 문서는 초안이니 자료 만들면서 순서·문구는 자유롭게 바꿔도 됨 — 숫자 근거만 Results_Reference_Table.md에서 그대로 가져다 쓰면 됨.
