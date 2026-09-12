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

## Python 구현

```python
import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """Apply the Benjamini-Hochberg procedure.

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

    # BH thresholds
    thresholds = np.arange(1, m + 1) / m * alpha

    # Find largest k where p_(k) <= k/m * alpha
    below = sorted_p <= thresholds
    if not below.any():
        k = 0
    else:
        k = np.max(np.where(below)[0]) + 1

    # Rejection decisions
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
