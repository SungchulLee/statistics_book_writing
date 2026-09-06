# Fisher 방법을 이용한 분산분석 수동 계산

## 개요

분산분석을 깊이 이해하려면 적어도 한 번은 모든 양을 손으로 계산해 보아야 한다. 이 페이지에서는 일원배치 분산분석의 분해를 처음부터 유도하고, 모의생성한 키 자료에 대해 SST, SSE, MST, MSE와 F-통계량을 수동으로 계산하며, 결과를 `scipy.stats.f_oneway`와 대조해 확인한 뒤, Fisher의 최소유의차(LSD) 사후 절차로 어느 집단 쌍이 다른지 찾는다.

## 일원배치 분산분석의 분해

표본크기가 $n_1, \dots, n_k$이고 전체 표본크기가 $N = \sum_{i=1}^{k} n_i$인 $k$개 집단을 관측한다고 하자. $\bar{y}$를 전체 평균, $\bar{y}_i$를 집단 $i$의 평균이라 하면 전체 변동은 다음과 같이 분해된다:

$$
\underbrace{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij} - \bar{y})^2}_{\text{SS}_{\text{total}}} = \underbrace{\sum_{i=1}^{k} n_i (\bar{y}_i - \bar{y})^2}_{\text{SST (between)}} + \underbrace{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij} - \bar{y}_i)^2}_{\text{SSE (within)}}
$$

평균제곱과 F-통계량은

$$
\text{MST} = \frac{\text{SST}}{k - 1}, \qquad \text{MSE} = \frac{\text{SSE}}{N - k}, \qquad F = \frac{\text{MST}}{\text{MSE}}
$$

이다. $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$ 아래에서 $F \sim F(k-1,\, N-k)$이다.

## Python으로 수동 계산

다음 함수는 분산분석의 모든 양을 처음부터 계산한다:

```python
import numpy as np
from scipy import stats

def manual_anova(groups):
    all_data = np.concatenate(list(groups.values()))
    grand_mean = all_data.mean()
    N = len(all_data)
    k = len(groups)

    SST = sum(len(g) * (g.mean() - grand_mean) ** 2
              for g in groups.values())
    SSE = sum(np.sum((g - g.mean()) ** 2)
              for g in groups.values())

    MST = SST / (k - 1)
    MSE = SSE / (N - k)
    F = MST / MSE
    p_value = 1 - stats.f.cdf(F, k - 1, N - k)
    return SST, SSE, MST, MSE, F, p_value
```

scipy로 확인하는 것은 한 줄이면 된다:

```python
F_scipy, p_scipy = stats.f_oneway(*groups.values())
```

두 방식이 동일한 $F$와 $p$-값을 주어 수동 계산이 맞음을 확인해 준다.

## Fisher LSD 사후비교

전역 귀무가설을 기각한 뒤 Fisher의 최소유의차로 어느 평균 쌍이 다른지 찾는다. 집단 $i$와 $j$에 대한 LSD 문턱은

$$
\text{LSD} = t_{\alpha/2,\, N-k} \sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

이다. $|\bar{y}_i - \bar{y}_j| > \text{LSD}$이면 그 쌍을 수준 $\alpha$에서 유의하게 다르다고 선언한다.

```python
from itertools import combinations

def fisher_lsd(groups, MSE, alpha=0.05):
    names = list(groups.keys())
    N_total = sum(len(g) for g in groups.values())
    k = len(groups)
    df_within = N_total - k
    results = []
    for (n1, g1), (n2, g2) in combinations(groups.items(), 2):
        t_crit = stats.t.ppf(1 - alpha / 2, df_within)
        lsd_val = t_crit * np.sqrt(MSE * (1/len(groups[n1]) + 1/len(groups[n2])))
        diff = abs(groups[n1].mean() - groups[n2].mean())
        results.append({"pair": f"{n1} vs {n2}",
                        "diff": diff, "LSD": lsd_val,
                        "significant": diff > lsd_val})
    return results
```

## 해석

함께 제공되는 스크립트에서 모의생성한 키 자료 세 집단(네덜란드 $\mu = 183$, 일본 $\mu = 172$, 덴마크 $\mu = 181$, 각각 $n = 30$)은 큰 F-통계량과 $p \approx 0$을 주어 $H_0$을 단호히 기각한다.

이어지는 Fisher LSD는 대체로 다음을 찾아낸다:

- **네덜란드 대 일본:** 유의함(평균 차이가 큼).
- **네덜란드 대 덴마크:** 유의하지 않음(평균이 비슷함).
- **일본 대 덴마크:** 유의함(평균 차이가 큼).

흔한 패턴을 잘 보여준다. 전역 분산분석은 기각하지만 모든 쌍별 비교가 유의하지는 않다. 어느 집단이 전체 효과를 이끄는지 알려면 사후 방법이 꼭 필요하다.

## 연습문제

**연습문제 1.**
평균이 $\bar{y}_1 = 10$, $\bar{y}_2 = 14$, $\bar{y}_3 = 12$이고 각 크기가 $n = 20$이며 전체 평균이 $\bar{y} = 12$인 세 집단에서 SST를 계산하라.

??? success "연습문제 1 풀이"
    $\text{SST} = \sum_{i=1}^{k} n_i (\bar{y}_i - \bar{y})^2$을 쓰면

    $$
    \text{SST} = 20(10 - 12)^2 + 20(14 - 12)^2 + 20(12 - 12)^2 = 20(4) + 20(4) + 20(0) = 160
    $$

---

**연습문제 2.**
집단 수 $k$가 클 때 Fisher LSD 절차가 가족단위 오류율을 통제하지 못하는 이유를 설명하라. 어떤 대안을 권하겠는가?

??? success "연습문제 2 풀이"
    Fisher LSD는 각 쌍별 비교를 조정 없이 수준 $\alpha$에서 수행한다. 비교가 $\binom{k}{2}$개면 거짓 기각이 적어도 하나 나올 확률이 빠르게 커진다. $k = 5$이면 쌍별 검정이 10개이고, 전역 귀무가설 아래에서 가족단위 오류율이 $1 - (1 - \alpha)^{10} \approx 0.40$에 이를 수 있다.

    표준적인 대안은 Tukey의 정직유의차(HSD) 방법이다. $t$-분포 대신 스튜던트화 범위 분포를 써서 모든 쌍별 비교에 대해 가족단위 오류율을 $\alpha$로 동시에 통제한다.

---

**연습문제 3.**
항등식 $y_{ij} - \bar{y} = (\bar{y}_i - \bar{y}) + (y_{ij} - \bar{y}_i)$을 전개하여 $\text{SS}_{\text{total}} = \text{SST} + \text{SSE}$임을 보여라.

??? success "연습문제 3 풀이"
    양변을 제곱하여 합하면

    $$
    \sum_{i}\sum_{j}(y_{ij} - \bar{y})^2 = \sum_{i}\sum_{j}(\bar{y}_i - \bar{y})^2 + 2\sum_{i}\sum_{j}(\bar{y}_i - \bar{y})(y_{ij} - \bar{y}_i) + \sum_{i}\sum_{j}(y_{ij} - \bar{y}_i)^2
    $$

    이다. 각 집단 $i$에서

    $$
    \sum_{j=1}^{n_i}(y_{ij} - \bar{y}_i) = 0
    $$

    이므로 교차항이 사라진다. 따라서 모든 $i$에서 $(\bar{y}_i - \bar{y})\sum_j (y_{ij} - \bar{y}_i) = 0$이다. 남은 두 항은 각각 정확히 $\text{SST}$($\sum_j (\bar{y}_i - \bar{y})^2 = n_i(\bar{y}_i - \bar{y})^2$임에 유의)와 $\text{SSE}$이다. $\square$

---

**연습문제 4.**
키 예제에서 덴마크 집단의 크기가 $n = 30$이 아니라 $n = 5$라고 하자. 균형인 경우와 비교해 네덜란드 대 덴마크의 LSD 문턱은 어떻게 달라지는가?

??? success "연습문제 4 풀이"
    LSD 문턱은

    $$
    \text{LSD} = t_{\alpha/2,\, N-k}\sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
    $$

    이다. $n_{\text{덴마크}}$가 30에서 5가 되면 $1/n_j$가 $1/30 \approx 0.033$에서 $1/5 = 0.2$로 커진다. 합 $1/n_i + 1/n_j$는 약 $0.067$에서 $0.233$으로 늘어 제곱근 안의 값이 거의 네 배가 된다. 그 결과 LSD 문턱이 크게 커져 네덜란드–덴마크 차이를 유의하다고 선언하기 어려워진다. 또한 전체 $N$이 줄고 $\text{MSE}$도 달라질 수 있어 문턱이 더 넓어질 수 있다.

---

**연습문제 5.**
$\text{MST}$가 $\sigma^2$의 불편추정값이 되는 조건은 무엇인가? $H_0$이 거짓일 때 $\text{MST}$는 무엇을 추정하는가?

??? success "연습문제 5 풀이"
    $H_0: \mu_1 = \cdots = \mu_k$ 아래에서 각 집단 평균 $\bar{Y}_i$가 공통 평균 $\mu$를 추정하며

    $$
    E[\text{MST}] = \sigma^2
    $$

    이므로 MST는 공통 분산의 불편추정량이다. $H_0$이 거짓이면

    $$
    E[\text{MST}] = \sigma^2 + \frac{\sum_{i=1}^{k} n_i (\mu_i - \bar{\mu})^2}{k - 1}
    $$

    이며 $\bar{\mu} = \sum n_i \mu_i / N$이다. 집단 평균이 모두 같지 않으면 둘째 항이 양수이므로 $E[\text{MST}] > \sigma^2$이다. $H_0$과 무관하게 $E[\text{MSE}] = \sigma^2$이므로 대립가설 아래에서 비 $F = \text{MST}/\text{MSE}$가 1보다 커지는 경향이 있고, 이것이 F-검정이 차이를 탐지할 검정력을 갖는 이유이다.
