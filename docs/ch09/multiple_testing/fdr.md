# 거짓발견율 (Benjamini-Hochberg)

## FWER에서 FDR로

[가족단위 오류율](fwer.md)은 잘못된 기각이 단 하나라도 생길 확률을 통제한다. 거짓 양성 하나하나가 심각한 결과를 낳을 때는 적절하지만, 검정 수가 많아지면 FWER 통제는 지나치게 보수적이 된다. 검정이 $m = 500{,}000$개인 전장유전체 연관분석에서 Bonferroni 문턱 $\alpha / m$은 너무 엄격해서 실제 효과를 많이 놓친다. Benjamini와 Hochberg가 1995년에 소개한 **거짓발견율(FDR)**은 덜 보수적인 대안을 제시한다: 거짓 양성을 아예 막는 대신, 기각된 가설 중 거짓 양성의 기대 **비율**을 통제한다.

## 기호와 설정

귀무가설 $H_1, \ldots, H_m$을 동시에 검정한다고 하자. 이 중 $m_0$개가 실제로 귀무이고 $m - m_0$개가 실제로 대립이다. 검정 후 결과는 다음과 같이 분류할 수 있다:

| | 기각 안 함 | 기각 | 합계 |
|---|---|---|---|
| 참인 귀무 | $U$ | $V$ | $m_0$ |
| 거짓인 귀무 | $T$ | $S$ | $m - m_0$ |
| 합계 | $m - R$ | $R$ | $m$ |

여기서 $R$은 전체 기각 수, $V$는 거짓 발견의 수(기각된 참인 귀무가설), $S$는 참 발견의 수이다. $V$, $S$, $U$, $T$, $R$은 모두 자료와 판정 규칙에 따라 달라지는 확률변수이다.

## FDR의 정의

**거짓발견비율(FDP)**은 기각 중 잘못된 것의 비율이다:

$$
\text{FDP} = \frac{V}{R}
$$

$R = 0$이면(아무것도 기각하지 않으면) 관례상 $\text{FDP} = 0$으로 둔다. **거짓발견율**은 이 비율의 기댓값이다:

$$
\text{FDR} = E\!\left[\frac{V}{\max(R, 1)}\right]
$$

분모의 $\max(R, 1)$이 0으로 나누는 것을 막는다. 동등하게 $\text{FDR} = E[V/R \mid R > 0] \cdot P(R > 0)$이다.

??? note "FDR과 FWER"
    FWER = $P(V \geq 1)$은 잘못된 기각이 **하나라도** 있는지 묻는다. FDR = $E[V / \max(R, 1)]$은 평균적으로 기각의 **몇 분의 몇**이 잘못되었는지를 묻는다. 모든 귀무가설이 참이면($m_0 = m$) 모든 기각이 거짓 발견이고 $\text{FDR} = P(R > 0) = \text{FWER}$이므로, FDR을 수준 $\alpha$로 통제하면 FWER도 수준 $\alpha$로 통제된다. 일부 귀무가설이 거짓이면 FDR이 보통 FWER보다 작다.

## Benjamini-Hochberg 절차

Benjamini-Hochberg(BH) 절차는 FDR을 통제하는 **단계적 상승** 방법이다. p-값 $p_1, p_2, \ldots, p_m$이 주어졌을 때:

**1단계.** p-값을 오름차순으로 정렬한다:

$$
p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}
$$

$H_{(i)}$를 $p_{(i)}$에 대응하는 귀무가설이라 하자.

**2단계.** 다음을 만족하는 가장 큰 지표 $k$를 찾는다:

$$
p_{(k)} \leq \frac{k}{m} \alpha
$$

**3단계.** 가설 $H_{(1)}, H_{(2)}, \ldots, H_{(k)}$를 모두 기각한다.

그런 $k$가 없으면 아무것도 기각하지 않는다.

문턱 $k\alpha / m$은 $k$에 따라 증가하며 $(1, \alpha/m)$에서 $(m, \alpha)$로 이어지는 직선을 이룬다. 이 절차는 그 직선 아래로 떨어지는 가장 오른쪽 p-값을 찾아 그 왼쪽의 모든 것을 기각한다.

### BH 절차가 통하는 이유 (직관)

핵심은 단계적 상승 문턱이 기각 전체 중 거짓 발견의 비율이 평균적으로 $\alpha$를 넘지 않도록 조정되어 있다는 점이다. 기각의 대부분이 참 발견이면(실제 효과에서 나온 작은 p-값들이면) 추가 발견에 넉넉한 문턱을 허용한다. 기각되는 가설이 적으면 문턱이 자동으로 더 엄격해져 Bonferroni와 비슷한 보호를 흉내 낸다.

## BH 정리

!!! info "Benjamini-Hochberg 정리 (1995)"
    $m$개 검정통계량이 **독립**이면 BH 절차는 FDR을 다음 수준으로 통제한다:

    $$
    \text{FDR} \leq \frac{m_0}{m} \alpha \leq \alpha
    $$

    여기서 $m_0$은 참인 귀무가설의 개수이다.

인자 $m_0 / m \leq 1$은 BH 절차가 실제로는 약간 보수적임을 뜻한다: 참 FDR은 최대 $\alpha$에 참인 귀무가설의 비율을 곱한 값이다.

Benjamini와 Yekutieli(2001)는 이후 BH 절차가 **부분집합 각각에 대한 양의 회귀 종속성(PRDS)** 아래에서도 FDR을 통제함을 보였다. 이 조건은 비음의 상관을 갖는 다변량 정규를 비롯한 여러 흔한 다변량 분포가 만족한다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 약물 선별. 한 제약회사가 표적에 대한 활성을 보려고 화합물 $m = 10$개를 선별하며, 화합물마다 $\alpha = 0.10$에서 가설검정을 한 번씩 한다. 얻은 p-값은:

| 화합물 | $p$-값 |
|---|---|
| A | 0.005 |
| B | 0.011 |
| C | 0.032 |
| D | 0.048 |
| E | 0.065 |
| F | 0.100 |
| G | 0.180 |
| H | 0.350 |
| I | 0.560 |
| J | 0.920 |

</div>

??? success "풀이"
    p-값은 이미 정렬되어 있다. 각 순위에 대해 BH 문턱 $k\alpha / m$을 계산한다:

    | 순위 $k$ | $p_{(k)}$ | BH 문턱 $k \times 0.10 / 10$ | $p_{(k)} \leq$ 문턱? |
    |---|---|---|---|
    | 1 | 0.005 | 0.010 | 예 |
    | 2 | 0.011 | 0.020 | 예 |
    | 3 | 0.032 | 0.030 | 아니오 |
    | 4 | 0.048 | 0.040 | 아니오 |
    | 5 | 0.065 | 0.050 | 아니오 |
    | 6 | 0.100 | 0.060 | 아니오 |
    | 7 | 0.180 | 0.070 | 아니오 |
    | 8 | 0.350 | 0.080 | 아니오 |
    | 9 | 0.560 | 0.090 | 아니오 |
    | 10 | 0.920 | 0.100 | 아니오 |

    $p_{(k)} \leq k\alpha / m$을 만족하는 가장 큰 $k$는 $k = 2$이다. BH 절차는 $H_{(1)}$과 $H_{(2)}$를 기각하여 화합물 A와 B를 활성으로 선언한다.

    비교하면 수준 $\alpha = 0.10$의 [Bonferroni 보정](bonferroni_holm.md)은 문턱 $0.10 / 10 = 0.01$을 쓴다. 화합물 A($p = 0.005$)만 이 문턱을 통과한다. BH 절차는 기대 거짓발견비율을 10% 이하로 유지하면서 발견을 하나(화합물 B) 더 찾아낸다.

## 보정된 p-값

각 p-값을 그에 해당하는 BH 문턱과 비교하는 대신 **BH 보정 p-값**(q-값이라고도 한다)을 계산하는 편이 흔히 더 편하다. $i$번째로 정렬된 가설의 보정 p-값은:

$$
\tilde{p}_{(i)} = \min_{j \geq i} \left\{\frac{m}{j} \, p_{(j)}\right\}
$$

가장 큰 순위에서 아래로 내려오며 계산하고, 보정 p-값이 비감소가 되도록 한다. 어떤 가설이 수준 $\alpha$의 BH 절차에서 기각될 필요충분조건은 보정 p-값이 $\tilde{p}_{(i)} \leq \alpha$인 것이다.

## FDR과 FWER 중 무엇을 쓸 것인가

| 기준 | FWER 통제 | FDR 통제 |
|---|---|---|
| 오류 보장 | (높은 확률로) 거짓 양성 없음 | 거짓 양성의 비율을 통제 |
| 보수성 | 높음 (특히 $m$이 클 때) | 중간 |
| 검정력 | 낮음 | 높음 |
| 알맞은 상황 | 확증적 연구, 작은 $m$ | 탐색적 연구, 큰 $m$ |
| 예 | 임상시험 평가변수, 규제 제출 | 유전체학, 단백질체학, 뇌영상 |

FDR 통제는 후속 조사를 할 유망한 후보 집합을 찾는 것이 목표이고 잘못된 단서가 조금 섞여도 괜찮은 고차원 선별 문제의 표준 접근이다.

<div class="codebox" markdown>

### 예제 1. 벤자미니-호크버그 절차 구현 { .eg }

```python
import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """벤자미니-호크버그 절차를 적용한다.

        p-값을 작은 것부터 늘어놓고 k번째를 k/m*alpha 와 견준다. 조건을 만족하는
        가장 큰 k 를 찾아 그 아래를 모두 기각한다. FWER 대신 FDR 을 통제하므로
        본페로니보다 훨씬 덜 보수적이다.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from m hypothesis tests.
    alpha : float
        Target FDR level.

    Returns
    -------
    rejected : ndarray of bool
        True for hypotheses that are rejected.
    adjusted : ndarray of float
        BH-adjusted p-values.
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    # BH 문턱값: k번째로 작은 p-값을 k/m*alpha 와 견준다.
    thresholds = np.arange(1, m + 1) / m * alpha

    # 조건을 만족하는 가장 큰 k 를 찾는다. 그 아래는 모두 기각한다.
    below = sorted_p <= thresholds
    if not below.any():
        k = 0
    else:
        k = np.max(np.where(below)[0]) + 1

    # 기각 여부를 표시한다.
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k]] = True

    # 보정 p-값. 큰 것에서 작은 쪽으로 훑으며 단조성을 강제한다.
    # 이 되짚기가 없으면 원래 p-값의 순서가 뒤집히는 일이 생겨
    # "더 작은 p-값이 더 큰 보정 p-값을 갖는" 이상한 결과가 나온다.
    adjusted_sorted = np.minimum(1, sorted_p * m / np.arange(1, m + 1))
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted      # 원래 순서로 되돌린다

    return rejected, adjusted


# 아래 연습문제 1의 p-값으로 확인한다.
pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.23, 0.76])
rej, adj = benjamini_hochberg(pvals, alpha=0.10)
print("rejected:", rej)
print("adjusted:", np.round(adj, 4))

# statsmodels와 대조
from statsmodels.stats.multitest import multipletests
rej_sm, adj_sm, _, _ = multipletests(pvals, alpha=0.10, method="fdr_bh")
print("statsmodels adjusted:", np.round(adj_sm, 4))
```

출력:

```
rejected: [ True  True  True  True False False]
adjusted: [0.006  0.024  0.0615 0.0615 0.276  0.76  ]
statsmodels adjusted: [0.006  0.024  0.0615 0.0615 0.276  0.76  ]
```

앞의 네 개가 기각된다. Bonferroni였다면 문턱이 $0.10/6 = 0.0167$이라 처음 두 개만 기각되었을 것이다.

보정 p-값에서 셋째와 넷째가 0.0615로 같아진 것이 위에서 말한 단조성 강제의 결과다. 곧이곧대로 계산하면 셋째가 $0.039 \times 6/3 = 0.078$, 넷째가 $0.041 \times 6/4 = 0.0615$로 원래 p-값의 순서와 어긋난다. 뒤에서부터 훑으며 최솟값을 취해 셋째를 0.0615로 끌어내린다.

</div>

## 다른 주제와의 연결

- FWER의 정의와 보정하지 않은 검정이 왜 문제인지는 [가족단위 오류율](fwer.md)을 보라.
- FWER을 통제하는 절차(Bonferroni와 Holm)는 [Bonferroni와 Holm 보정](bonferroni_holm.md)을 보라.
- BH 절차는 `scipy.stats.false_discovery_control`(SciPy 1.11 이상)과 `statsmodels.stats.multitest.multipletests`에 구현되어 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
p-값 6개 0.001, 0.008, 0.039, 0.041, 0.23, 0.76에 FDR 수준 $q = 0.10$에서 Benjamini-Hochberg(BH) 절차를 적용하라. 어느 가설이 기각되는가?

</div>

??? success "풀이"
    p-값을 정렬하고 BH 문턱 $q \cdot j/m$을 계산한다:

    | 순위 $j$ | $p_{(j)}$ | 문턱 $q \cdot j/m = 0.10 \cdot j/6$ | $p_{(j)} \leq$ 문턱? |
    |---|---|---|---|
    | 1 | 0.001 | 0.0167 | 예 |
    | 2 | 0.008 | 0.0333 | 예 |
    | 3 | 0.039 | 0.0500 | 예 |
    | 4 | 0.041 | 0.0667 | 예 |
    | 5 | 0.230 | 0.0833 | 아니오 |
    | 6 | 0.760 | 0.1000 | 아니오 |

    $p_{(j)} \leq q \cdot j/m$을 만족하는 가장 큰 $j$는 $j = 4$이다. $p_{(1)}$부터 $p_{(4)}$에 대응하는 가설 4개를 기각한다. $q = 0.10$의 BH에서 이 4개 기각 중 거짓발견비율의 기댓값은 최대 10%이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
거짓발견율(FDR)을 정의하고 가족단위 오류율(FWER)과 어떻게 다른지 설명하라.

</div>

??? success "풀이"
    **FDR**은 기각된 가설 전체 중 거짓 발견의 기대 비율이다:

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right]
    $$

    여기서 $V$는 잘못된 기각의 수, $R$은 전체 기각 수이며 $R \vee 1 = \max(R, 1)$이 0으로 나누는 것을 막는다.

    **FWER**은 잘못된 기각을 적어도 한 번 할 확률이다: $\text{FWER} = P(V \geq 1)$.

    주요 차이:

    - FWER이 더 엄격하다: *어떤* 거짓 양성이든 그 확률을 통제하는 반면, FDR은 발견 전체에서 작은 비율이기만 하면 거짓 양성을 어느 정도 허용한다.
    - FDR이 더 강력하다: 발견 중 일정 비율의 오류를 감수하므로 더 많은 가설을 기각한다.
    - 모든 귀무가설이 참이면($m_0 = m$) 모든 기각이 거짓이므로 FDR = FWER이다. 대립가설이 많이 참이면 FDR $\ll$ FWER이다.
    - 수천 개의 검정을 수행하고 거짓 발견을 어느 정도 감수할 수 있는 대규모 검정(유전체학, 뇌영상)에서는 FDR을 선호한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
검정통계량이 독립일 때 BH 절차가 FDR을 수준 $q$로 통제함을 증명하라.

</div>

??? success "풀이"
    독립일 때 BH 절차는 FDR을 정확히 $q \cdot m_0/m \leq q$로 통제한다. 여기서 $m_0$은 참인 귀무가설의 개수이다.

    핵심 통찰: 참인 각 귀무가설 $H_i$에 대해 p-값 $p_i$는 Uniform$(0,1)$이고 다른 것들과 독립이다. BH 절차는 자료에 따라 정해지는 순위 $R_i$에 대해 $p_i \leq q \cdot R_i/m$일 때 $H_i$를 기각한다.

    형식적인 증명(Benjamini & Hochberg, 1995)은 다음을 보이는 방식으로 진행된다:

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right] = \sum_{i \in \mathcal{H}_0} E\!\left[\frac{1}{R \vee 1} \cdot \mathbf{1}(H_i \text{ rejected})\right] = \frac{m_0}{m} \cdot q \leq q
    $$

    마지막 등식은 참인 귀무가설의 p-값이 거짓인 귀무가설의 p-값과 독립이라는 데 기댄다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
어떤 유전체 연구가 유전자 10,000개를 검정한다. $q = 0.05$의 BH로 가설 500개를 기각했다. 이 기각 중 거짓 발견은 몇 개로 기대되는가? FWER 통제(Bonferroni)와 비교하면 어떤가?

</div>

??? success "풀이"
    $q = 0.05$의 BH에서 거짓 발견 수의 기댓값은 최대 $q \times R = 0.05 \times 500 = 25$개이다. 즉 500개 발견 중 약 475개가 실제일 것으로 기대된다.

    $\alpha = 0.05$의 Bonferroni에서는 검정당 문턱이 $0.05/10000 = 5 \times 10^{-6}$이다. 이 극도로 엄격한 문턱 아래의 p-값만 기각된다. 실제로 Bonferroni는 (500개보다 훨씬 적은) 20~50개 유전자만 기각하여 많은 참 발견을 놓칠 수 있다.

    FDR 통제의 검정력 이점을 보여준다: BH는 (약 25개의 거짓을 감수하며) 유전자 500개를 발견하는 반면 Bonferroni는 (거짓을 거의 0으로 하면서) 훨씬 적게 발견한다. 선택은 거짓 발견의 대가와 놓친 발견의 대가를 견주어 정한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
BH 절차의 **실제 FDR과 검정력**을 $\pi_1$(참 대립가설의 비율)에 따라 모의실험으로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(31)
    M, m = 400, 1_000

    def bh_reject(p, q):
        m = len(p)
        o = np.argsort(p)
        ps = p[o]
        ok = np.where(ps <= q * np.arange(1, m + 1) / m)[0]
        r = np.zeros(m, bool)
        if len(ok):
            r[o[:ok.max() + 1]] = True
        return r

    print(f"{'π1':>6s} {'실제 FDR':>10s} {'q·π0':>8s} {'검정력':>8s} "
          f"{'평균 기각 수':>12s}")
    for pi1 in [0.01, 0.05, 0.10, 0.30]:
        fdrs, pows, nrej = [], [], []
        for _ in range(M):
            m1 = int(m * pi1)
            z = rng.standard_normal(m)
            z[:m1] += 3.5                        # 참 대립가설
            p = 2 * stats.norm.sf(np.abs(z))
            r = bh_reject(p, 0.05)
            R, V = r.sum(), r[m1:].sum()
            fdrs.append(V / max(R, 1))
            pows.append(r[:m1].sum() / m1)
            nrej.append(R)
        print(f"{pi1:6.2f} {np.mean(fdrs):10.4f} {0.05 * (1 - pi1):8.4f} "
              f"{np.mean(pows):8.4f} {np.mean(nrej):12.1f}")
    ```

    ```text
        π1    실제 FDR     q·π0     검정력      평균 기각 수
      0.01     0.0501   0.0495   0.4325          4.6
      0.05     0.0486   0.0475   0.6409         33.7
      0.10     0.0476   0.0450   0.7320         76.9
      0.30     0.0341   0.0350   0.8449        262.5
    ```

    **실제 FDR이 $q\pi_0$와 거의 같다.** 이것이 BH 정리의 핵심이다.

    $$
    \text{FDR}\le\frac{m_0}{m}q=\pi_0q\le q
    $$

    **따라서 BH는 $\pi_0$가 1에 가까울 때만 문턱을 온전히 쓴다.** $\pi_1=0.30$이면 실제 FDR이 0.034로 목표 0.05의 3분의 2다. **보수적**이다.

    **개선 — 적응적 BH.** $\pi_0$를 자료에서 추정해 $q/\hat\pi_0$를 쓰면 이 보수성이 사라진다. 스토리의 방법이 대표적이다.

    ```python
    def pi0_storey(p, lam=0.5):
        return min((p > lam).sum() / ((1 - lam) * len(p)), 1.0)

    m1 = 300
    z = rng.standard_normal(m)
    z[:m1] += 3.5
    p = 2 * stats.norm.sf(np.abs(z))
    print(f"참 π0 = {1 - m1 / m:.3f},  스토리 추정 = {pi0_storey(p):.3f}")
    ```

    ```text
    참 π0 = 0.700,  스토리 추정 = 0.660
    ```

    **검정력이 $\pi_1$과 함께 오른다.** $\pi_1=0.01$에서 0.43, $\pi_1=0.30$에서 0.84다. **BH의 문턱이 기각 수에 따라 느슨해지기** 때문이다. 참 신호가 많으면 문턱이 올라가 더 많이 잡아낸다.

    **이것이 본페로니와의 근본적 차이다.** 본페로니의 문턱 $\alpha/m$은 자료와 무관하게 고정이지만, BH는 **자료에 적응**한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
BH와 본페로니의 **검정력**을 같은 조건에서 비교하고, 차이가 어디서 오는지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(31)
    M, m = 400, 1_000

    def bh_reject(p, q):
        mm = len(p)
        o = np.argsort(p)
        ok = np.where(p[o] <= q * np.arange(1, mm + 1) / mm)[0]
        r = np.zeros(mm, bool)
        if len(ok):
            r[o[:ok.max() + 1]] = True
        return r

    print(f"{'π1':>6s} {'본페로니 검정력':>15s} {'BH 검정력':>11s} {'비':>6s}")
    for pi1 in [0.01, 0.05, 0.10]:
        pb, pbh = [], []
        for _ in range(M):
            m1 = int(m * pi1)
            z = rng.standard_normal(m)
            z[:m1] += 3.5
            p = 2 * stats.norm.sf(np.abs(z))
            pb.append((p[:m1] < 0.05 / m).sum() / m1)
            pbh.append(bh_reject(p, 0.05)[:m1].sum() / m1)
        print(f"{pi1:6.2f} {np.mean(pb):15.4f} {np.mean(pbh):11.4f} "
              f"{np.mean(pbh) / np.mean(pb):6.2f}")
    ```

    ```text
        π1    본페로니 검정력    BH 검정력      비
      0.01          0.2925      0.4455    1.52
      0.05          0.2868      0.6419    2.24
      0.10          0.2858      0.7252    2.54
    ```

    **BH가 1.5~2.5배 강력하다.** $\pi_1$이 클수록 차이가 커진다.

    **본페로니의 검정력은 $\pi_1$과 무관하다**(0.286~0.293). 문턱이 $\alpha/m$으로 고정이기 때문이다.

    **BH의 검정력은 $\pi_1$과 함께 오른다.** 신호가 많으면 문턱이 느슨해져 **선순환**이 생긴다.

    **차이의 원천 — 통제하는 양이 다르다.**

    | | 본페로니(FWER) | BH(FDR) |
    |---|---|---|
    | 통제 | $P(V\ge1)\le\alpha$ | $E[V/R]\le q$ |
    | 뜻 | **거짓발견이 하나라도** 있을 확률 | 발견 중 거짓의 **비율** |
    | $m$이 클 때 | 극도로 엄격 | 합리적 |

    **$m=1000$에서 FWER 통제는 "1000개 중 거짓 하나도 없기"** 를 요구한다. 지나치게 엄격하다. FDR은 "200개를 발견했다면 그중 10개쯤은 거짓일 수 있다"를 허용한다.

    **어느 것이 맞는가 — 후속 절차에 달렸다.**

    - **발견 하나하나를 확진 실험으로 검증**할 것이라면 FDR로 충분하다. 몇 개의 거짓 후보를 검증 단계에서 걸러 내면 된다.
    - **발견을 곧바로 결론으로 삼는다면** FWER이 필요하다. 규제 제출, 최종 결론.

    **유전체·뇌영상의 관행.** 탐색 단계에서는 FDR, 확증 단계에서는 FWER이나 매우 엄격한 문턱을 쓴다. **2단계 구조가 FDR을 정당화**한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
**$q$-값**을 정의하고 $p$-값과의 차이를 설명하라. 스토리의 $\pi_0$ 추정이 왜 필요한가?

</div>

??? success "풀이"
    **정의.**

    - **$p$-값**: 이 가설을 기각하는 문턱을 관측값에 맞췄을 때의 **제1종 오류율**.
    - **$q$-값**: 이 가설을 기각하는 문턱을 관측값에 맞췄을 때의 **FDR**.

    형식적으로

    $$
    q(p_i)=\min_{t\ge p_i}\text{FDR}(t),
    \qquad
    \text{FDR}(t)\approx\frac{\pi_0\,m\,t}{\#\{p_j\le t\}}
    $$

    **해석.** "$q=0.02$인 유전자를 기각 목록에 넣으면, 그 목록의 2%가 거짓발견으로 예상된다."

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5)
    m, m1 = 2_000, 200
    z = rng.standard_normal(m)
    z[:m1] += 3.0
    p = 2 * stats.norm.sf(np.abs(z))

    def pi0_storey(p, lam=0.5):
        return min((p > lam).sum() / ((1 - lam) * len(p)), 1.0)

    def qvalues(p, pi0=1.0):
        mm = len(p)
        o = np.argsort(p)
        ps = p[o]
        q = pi0 * mm * ps / np.arange(1, mm + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(mm)
        out[o] = np.minimum(q, 1)
        return out

    pi0 = pi0_storey(p)
    q_cons = qvalues(p)                 # π0 = 1 (BH 와 동일)
    q_ada = qvalues(p, pi0)             # π0 추정 반영
    print(f"참 π0 = {1 - m1 / m:.3f},  스토리 추정 π0 = {pi0:.3f}")
    for thr in [0.01, 0.05, 0.10]:
        print(f"q ≤ {thr:.2f}:  BH {(q_cons <= thr).sum():4d}개   "
              f"적응형 {(q_ada <= thr).sum():4d}개   "
              f"(참 신호 중 발견: {(q_ada[:m1] <= thr).sum():3d}/{m1})")
    ```

    ```text
    참 π0 = 0.900,  스토리 추정 π0 = 0.916
    q ≤ 0.01:  BH   43개   적응형   43개   (참 신호 중 발견:  43/200)
    q ≤ 0.05:  BH   83개   적응형   90개   (참 신호 중 발견:  87/200)
    q ≤ 0.10:  BH  115개   적응형  122개   (참 신호 중 발견: 111/200)
    ```

    **$\pi_0$ 추정의 이득은 $\pi_0$가 작을 때 크다.** 여기서는 $\pi_0=0.9$라 0~7개 차이뿐이다. $\pi_0=0.5$라면 기각 수가 두 배 가까이 늘어난다. 여기서는 추정값 0.916이 참값 0.900보다 조금 커서 이득이 더 줄었다.

    **$\pi_0$를 어떻게 추정하는가.** 큰 $p$-값은 거의 모두 참인 귀무가설에서 나온다. $H_0$ 아래 $p$가 균등분포이므로

    $$
    \hat\pi_0(\lambda)=\frac{\#\{p_j>\lambda\}}{(1-\lambda)m}
    $$

    **$\lambda$의 선택이 관건이다.** 크면 편향은 작지만 분산이 크고, 작으면 반대다. 스플라인으로 $\lambda\to1$의 극한을 외삽하는 방법이 표준이다.

    **$p$-값과 $q$-값의 대비.**

    | | $p$-값 | $q$-값 |
    |---|---|---|
    | 단위 | 개별 가설 | 기각 목록 전체 |
    | 해석 | $H_0$ 아래 자료의 희귀성 | 목록의 거짓 비율 |
    | 다른 검정에 의존 | 하지 않음 | **의존한다** |
    | 단조성 | — | $p$의 단조 함수 |

    **셋째 줄이 중요하다.** 같은 유전자의 $q$-값이 **함께 검정한 유전자 집합에 따라 달라진다.** 이것이 직관에 어긋나지만, FDR이 목록 전체의 성질이므로 당연하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
BH 절차가 **종속 자료**에서 어떻게 작동하는지 확인하고, 언제 BY로 바꿔야 하는지 판단하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(77)
    M, m, m1 = 300, 500, 50

    def bh_reject(p, q):
        mm = len(p)
        o = np.argsort(p)
        ok = np.where(p[o] <= q * np.arange(1, mm + 1) / mm)[0]
        r = np.zeros(mm, bool)
        if len(ok):
            r[o[:ok.max() + 1]] = True
        return r

    print(f"{'상관 구조':>14s} {'BH FDR':>9s} {'BY FDR':>9s} "
          f"{'BH 검정력':>11s} {'BY 검정력':>11s}")
    c_m = np.sum(1 / np.arange(1, m + 1))
    for name, rho in [("독립", 0.0), ("양의 등상관 0.5", 0.5),
                      ("양의 등상관 0.9", 0.9)]:
        f1, f2, p1, p2 = [], [], [], []
        for _ in range(M):
            common = rng.standard_normal(1)
            z = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.standard_normal(m)
            z[:m1] += 3.5
            p = 2 * stats.norm.sf(np.abs(z))
            for q, fl, pl in [(0.05, f1, p1), (0.05 / c_m, f2, p2)]:
                r = bh_reject(p, q)
                fl.append(r[m1:].sum() / max(r.sum(), 1))
                pl.append(r[:m1].sum() / m1)
        print(f"{name:>14s} {np.mean(f1):9.4f} {np.mean(f2):9.4f} "
              f"{np.mean(p1):11.4f} {np.mean(p2):11.4f}")
    ```

    ```text
             상관 구조    BH FDR    BY FDR    BH 검정력    BY 검정력
             독립    0.0475    0.0075      0.7312      0.4803
      양의 등상관 0.5    0.0396    0.0054      0.6847      0.4330
      양의 등상관 0.9    0.0185    0.0016      0.6963      0.4837
    ```

    **BH가 양의 등상관에서도 FDR을 잘 통제한다.** 0.019~0.048로 모두 0.05 이하이며, 상관이 커질수록 오히려 보수적이 된다.

    **이론적 근거.** 벤야미니-예쿠티엘리가 **PRDS(양의 회귀 종속)** 조건 아래 BH의 FDR 통제를 증명했다. 등상관 정규는 이 조건을 만족한다.

    **BY는 지나치게 보수적이다.** FDR이 0.002~0.008로 목표의 10분의 1이고, 검정력을 20~25%포인트 잃는다. $c(m)=\sum_{i=1}^{500}1/i=6.79$배로 문턱을 낮추기 때문이다.

    **언제 BY가 필요한가.**

    | 상황 | 권장 |
    |---|---|
    | 독립 또는 양의 상관 | **BH** |
    | 상관 구조를 모르지만 양의 상관이 그럴듯 | **BH**(대부분의 실무) |
    | **음의 상관이 섞여 있음** | BY 또는 순열 |
    | 최악의 경우 보장이 필수 | BY |
    | 교환가능성이 성립 | **순열 기반 FDR**(가장 정확) |

    **실무 판단.** 대부분의 응용에서 검정통계량은 **양의 상관**을 갖는다(같은 시료, 같은 피험자, 인접한 유전자·복셀). 따라서 **BH를 그대로 쓰는 것이 표준**이다.

    **음의 상관이 생기는 경우.** 총합이 고정된 조성 자료(비율, 상대풍부도), 대비 간의 구조적 제약. 이런 경우에만 BY나 순열을 고려한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
FDR 통제 결과를 **어떻게 해석하고 보고해야 하는지** 정리하라. 흔한 오해는 무엇인가?

</div>

??? success "풀이"
    **흔한 오해 다섯.**

    **1 — "$q=0.05$이면 이 발견이 참일 확률이 95%다."**
    틀렸다. $q$-값은 **개별 가설의 확률이 아니라 목록의 기대 비율**이다. 개별 확률은 지역 FDR(local FDR)이라는 다른 양이며, 대체로 $q$-값보다 크다.

    **2 — "FDR을 통제했으니 각 발견이 검증되었다."**
    아니다. 200개를 발견했다면 그중 10개는 거짓으로 **예상**된다. 어느 10개인지는 모른다.

    **3 — "$V/R$의 기댓값이 $q$ 이하"와 "$V/R$이 항상 $q$ 이하"를 혼동.**
    기댓값 통제이므로, 특정 실험에서는 실제 비율이 훨씬 클 수 있다. 분산이 크다. **FDX(초과확률 통제)** 라는 대안이 있다.

    **4 — "$R=0$일 때의 처리."**
    기각이 없으면 $V/R$이 정의되지 않는다. 관례적으로 0으로 둔다. 따라서 FDR은 $E[V/R\mid R>0]P(R>0)$와 다르다(후자는 pFDR).

    **5 — "$q$-값이 개별 가설의 성질이다."**
    앞서 본 대로 **함께 검정한 집합에 의존**한다. 유전자 하나를 다른 논문에서 검정하면 $q$-값이 달라진다.

    **보고할 것.**

    1. **$m$(검정 수)과 가족의 정의.**
    2. **$q$ 수준과 방법**(BH/BY/스토리/순열).
    3. **기각 수 $R$과 예상 거짓발견 수 $\approx qR$.**
    4. **$\pi_0$ 추정값**(적응형을 썼다면).
    5. **원 $p$-값과 $q$-값을 모두** 담은 표나 부록.
    6. **종속 구조에 대한 논의.**

    **좋은 보고의 예.**

    > 유전자 20,531개에 대해 두 군의 발현 차이를 검정하고 BH 절차로 FDR을 0.05 수준에서 통제했다. 그 결과 342개가 유의했으며, 이 중 약 17개는 거짓발견으로 예상된다. 스토리의 방법으로 추정한 $\pi_0$는 0.83이었다. 검정통계량이 같은 시료에서 나와 양의 상관이 예상되므로 BH의 PRDS 조건이 충족된다고 보았다. 전체 $p$-값과 $q$-값은 보충자료 표 S1에 있다.

    **후속 절차의 명시.** FDR을 쓴다는 것은 **거짓발견을 어느 정도 허용**한다는 뜻이므로, 그것을 어떻게 처리할지 밝혀야 한다.

    > 상위 20개 유전자에 대해 독립 코호트에서 검증 실험을 수행할 계획이다.

    이 한 문장이 FDR 선택을 정당화한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
FDR 이외의 **다른 오류 측도**들을 소개하고, 각각이 어떤 상황에 맞는지 정리하라.

</div>

??? success "풀이"

    | 측도 | 정의 | 통제하는 것 | 언제 |
    |---|---|---|---|
    | **FWER** | $P(V\ge1)$ | 거짓발견이 하나라도 | 확증, 규제 |
    | **FDR** | $E[V/R]$ | 거짓의 기대 비율 | 탐색 + 후속 검증 |
    | **pFDR** | $E[V/R\mid R>0]$ | 기각이 있을 때의 비율 | $q$-값의 바탕 |
    | **FDX** | $P(V/R>\gamma)$ | 비율이 $\gamma$를 넘을 확률 | **분산이 걱정될 때** |
    | **$k$-FWER** | $P(V\ge k)$ | 거짓이 $k$개 이상일 확률 | 소수의 거짓은 허용 |
    | **FNR** | $E[T/(m-R)]$ | 놓친 참의 비율 | 민감도가 중요할 때 |
    | **per-comparison** | $E[V]/m$ | 개별 오류율 | 사실상 무보정 |

    **FDR의 약점 — 분산.** $E[V/R]\le q$는 평균의 진술이다. 실제 $V/R$이 $q$를 크게 넘는 실험이 있을 수 있다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(13)
    M, m, m1 = 2_000, 500, 25

    def bh_reject(p, q):
        mm = len(p)
        o = np.argsort(p)
        ok = np.where(p[o] <= q * np.arange(1, mm + 1) / mm)[0]
        r = np.zeros(mm, bool)
        if len(ok):
            r[o[:ok.max() + 1]] = True
        return r

    fdps = []
    for _ in range(M):
        z = rng.standard_normal(m)
        z[:m1] += 3.0
        p = 2 * stats.norm.sf(np.abs(z))
        r = bh_reject(p, 0.05)
        fdps.append(r[m1:].sum() / max(r.sum(), 1))
    fdps = np.array(fdps)
    print(f"FDR(평균) = {fdps.mean():.4f}")
    print(f"백분위 50/75/90/95/99: "
          + " ".join(f"{v:.3f}" for v in np.percentile(fdps, [50, 75, 90, 95, 99])))
    print(f"실제 비율이 0.10 을 넘은 비율 = {np.mean(fdps > 0.10):.4f}")
    ```

    ```text
    FDR(평균) = 0.0478
    백분위 50/75/90/95/99: 0.000 0.091 0.143 0.167 0.250
    실제 비율이 0.10 을 넘은 비율 = 0.1910
    ```

    **평균은 0.048로 잘 통제되지만, 19%의 실험에서 실제 비율이 0.10을 넘는다.** 99 백분위는 0.25다.

    **FDX가 이를 다룬다.** "$P(V/R>0.10)\le0.05$"처럼 **초과확률을 직접 통제**한다. 기각 수가 적을 때 특히 유용하다.

    **$k$-FWER의 쓰임.** "거짓발견이 3개 이하면 괜찮다"는 상황. FWER보다 강력하고 FDR보다 엄격하다. 중간 규모($m$이 수십~수백)의 검정에 적합하다.

    **선택 지침.**

    - **$m$이 작고(<20) 확증적**: FWER(홀름).
    - **$m$이 크고 탐색적, 후속 검증 있음**: FDR(BH).
    - **$m$이 크지만 기각이 적을 것으로 예상**: FDX나 $k$-FWER. FDR의 분산이 크기 때문이다.
    - **$m$이 중간, 소수의 거짓은 허용**: $k$-FWER.
    - **베이즈 틀이 자연스러움**: 지역 FDR이나 계층모형의 사후확률.

    **한 문장.** **"어떤 오류를 얼마나 허용할 것인가"는 통계적 질문이 아니라 그 연구의 후속 절차와 비용이 답하는 질문**이다.

---

## 정리하며

FDR 은 **기각한 것 중 거짓의 비율**을 통제한다. FWER 과 통제 대상이 다르다.

| | 통제하는 것 |
|---|---|
| FWER | 거짓 양성이 **하나라도** 생길 확률 |
| FDR | 기각한 것 중 거짓의 **기대 비율** $\mathbb{E}[V/R]$ |

- **발상의 전환이다.** 발견 1000개 중 50개가 거짓이어도 받아들일 만하다면, 거짓을 하나도 허용하지 않는 것보다 훨씬 많은 실제 효과를 건질 수 있다.
- **탐색적 연구에 맞는 기준이다.** 유전체학·뇌영상처럼 후보를 추려 뒤에 검증하는 상황이 전형적이며, **거짓 양성 하나가 치명적인 규제 심사에서는 여전히 FWER 이 옳다.**
- **벤저미니–호흐베르그 절차**가 표준이다. $p$ 값을 정렬해 $p_{(i)}\le\frac{i}{m}q$ 를 만족하는 가장 큰 $i$ 를 찾고 그 이하를 모두 기각한다.
- **$R=0$ 일 때의 정의에 주의한다.** 아무것도 기각하지 않으면 $V/R$ 이 정의되지 않으므로 그때 $0$ 으로 둔다.
- **독립 또는 양의 의존에서 보장된다.** 일반적인 의존 구조에서는 벤저미니–예쿠티엘리 보정이 필요하며, 더 보수적이다.

다음 절 **다중검정 보정**에서 방법들을 한자리에 모은다.
