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

<div class="codebox" markdown>

### 예제 1. 분산분석표 직접 계산하기 { .eg }

```python
import numpy as np
from scipy import stats

def manual_anova(groups):
    all_data = np.concatenate(list(groups.values()))
    grand_mean = all_data.mean()
    N = len(all_data)
    k = len(groups)

    # SST: 집단평균이 전체평균에서 얼마나 떨어져 있는가. n_i로 가중한다.
    # 큰 집단의 평균이 어긋나는 것이 더 무겁게 세어져야 하기 때문이다.
    SST = sum(len(g) * (g.mean() - grand_mean) ** 2
              for g in groups.values())
    # SSE: 각 관측값이 **자기 집단의** 평균에서 얼마나 떨어져 있는가.
    SSE = sum(np.sum((g - g.mean()) ** 2)
              for g in groups.values())

    MST = SST / (k - 1)
    MSE = SSE / (N - k)
    F = MST / MSE                    # 신호 대 잡음
    p_value = 1 - stats.f.cdf(F, k - 1, N - k)
    return SST, SSE, MST, MSE, F, p_value


# R의 PlantGrowth 자료 (대조군과 두 처리, 각 10개)
groups = {
    "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
    "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
    "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
}

SST, SSE, MST, MSE, F, p = manual_anova(groups)
print(f"SST = {SST:.4f}, SSE = {SSE:.4f}")
print(f"MST = {MST:.4f}, MSE = {MSE:.4f}")
print(f"F   = {F:.4f}, p = {p:.4f}")
```

출력:

```
SST = 3.7663, SSE = 10.4921
MST = 1.8832, MSE = 0.3886
F   = 4.8461, p = 0.0159
```

SSE가 SST의 세 배 가까이 크지만 자유도로 나누고 나면(2 대 27) MST가 MSE의 다섯 배가 된다. 분산분석에서 제곱합 자체가 아니라 **자유도로 나눈 평균제곱**을 비교하는 이유다.

</div>

scipy로 확인하는 것은 한 줄이면 된다:

<div class="codebox" markdown>

### 예제 2. scipy 결과와 맞춰 보기 { .eg }

```python
# 손으로 구한 값과 맞는지 확인한다. 한 줄이면 되는 계산을 굳이 풀어 쓴 까닭은
# 제곱합이 어떻게 갈라지는지를 보이기 위해서다.
F_scipy, p_scipy = stats.f_oneway(*groups.values())
print(f"scipy: F = {F_scipy:.4f}, p = {p_scipy:.4f}")
```

출력:

```
scipy: F = 4.8461, p = 0.0159
```

두 방식이 동일한 $F$와 $p$-값을 주어 수동 계산이 맞음을 확인해 준다.

</div>

## Fisher LSD 사후비교

전역 귀무가설을 기각한 뒤 Fisher의 최소유의차로 어느 평균 쌍이 다른지 찾는다. 집단 $i$와 $j$에 대한 LSD 문턱은

$$
\text{LSD} = t_{\alpha/2,\, N-k} \sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

이다. $|\bar{y}_i - \bar{y}_j| > \text{LSD}$이면 그 쌍을 수준 $\alpha$에서 유의하게 다르다고 선언한다.

<div class="codebox" markdown>

### 예제 3. Fisher LSD 사후비교 { .eg }

```python
from itertools import combinations

def fisher_lsd(groups, MSE, alpha=0.05):
    """Fisher 의 최소유의차로 쌍별 비교를 한다.

    쌍마다 t-검정을 하되 표준오차를 그 두 집단이 아니라 전체 MSE 로 만든다.
    모든 집단의 정보를 쓰므로 자유도가 커지는 것이 이점이다.
    다만 다중비교를 보정하지 않으므로, 분산분석이 유의할 때만 쓴다.
    """
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


for r in fisher_lsd(groups, MSE):
    print(f"{r['pair']:<14} diff = {r['diff']:.4f}  LSD = {r['LSD']:.4f}  {r['significant']}")
```

출력:

```
ctrl vs trt1   diff = 0.3710  LSD = 0.5720  False
ctrl vs trt2   diff = 0.4940  LSD = 0.5720  False
trt1 vs trt2   diff = 0.8650  LSD = 0.5720  True
```

전역 검정은 $p = 0.0159$로 기각했는데 쌍별로 보면 trt1 대 trt2 하나만 유의하다. 대조군은 두 처리 어느 쪽과도 유의하게 다르지 않다. 두 처리가 대조군을 사이에 두고 반대 방향으로 벌어져 있어, 서로 간의 차이가 각각과 대조군의 차이보다 큰 것이다.

집단 크기가 모두 10으로 같아 LSD 문턱도 0.5720 하나로 같다. 크기가 다르면 쌍마다 문턱이 달라진다.

</div>

## 해석

위 PlantGrowth 자료에서 전역 F-검정은 $F = 4.85$, $p = 0.0159$로 $\alpha = 0.05$에서 $H_0$을 기각한다. 세 집단의 평균이 모두 같지는 않다는 뜻이다.

이어지는 Fisher LSD는 다음을 찾아낸다:

- **ctrl 대 trt1:** 유의하지 않음(차이 0.371 < LSD 0.572).
- **ctrl 대 trt2:** 유의하지 않음(차이 0.494 < LSD 0.572).
- **trt1 대 trt2:** 유의함(차이 0.865 > LSD 0.572).

흔한 패턴을 잘 보여준다. 전역 분산분석은 기각하지만 모든 쌍별 비교가 유의하지는 않다. 어느 집단이 전체 효과를 이끄는지 알려면 사후 방법이 꼭 필요하다.

한 가지 덧붙이면, Fisher LSD는 보정을 하지 않으므로 여기서 유의하다고 나온 trt1 대 trt2도 Tukey HSD로 다시 보면 $p_{\text{adj}} = 0.012$로 유의성이 약해진다(같은 자료를 다룬 [분산분석 파이프라인](oneway_pipeline.md) 참조). 집단이 셋일 때는 차이가 크지 않지만 집단이 많아지면 벌어진다(연습문제 2).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
평균이 $\bar{y}_1 = 10$, $\bar{y}_2 = 14$, $\bar{y}_3 = 12$이고 각 크기가 $n = 20$이며 전체 평균이 $\bar{y} = 12$인 세 집단에서 SST를 계산하라.

</div>

??? success "풀이"
    $\text{SST} = \sum_{i=1}^{k} n_i (\bar{y}_i - \bar{y})^2$을 쓰면

    $$
    \text{SST} = 20(10 - 12)^2 + 20(14 - 12)^2 + 20(12 - 12)^2 = 20(4) + 20(4) + 20(0) = 160
    $$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
집단 수 $k$가 클 때 Fisher LSD 절차가 가족단위 오류율을 통제하지 못하는 이유를 설명하라. 어떤 대안을 권하겠는가?

</div>

??? success "풀이"
    Fisher LSD는 각 쌍별 비교를 조정 없이 수준 $\alpha$에서 수행한다. 비교가 $\binom{k}{2}$개면 거짓 기각이 적어도 하나 나올 확률이 빠르게 커진다. $k = 5$이면 쌍별 검정이 10개이고, 전역 귀무가설 아래에서 가족단위 오류율이 $1 - (1 - \alpha)^{10} \approx 0.40$에 이를 수 있다.

    표준적인 대안은 Tukey의 정직유의차(HSD) 방법이다. $t$-분포 대신 스튜던트화 범위 분포를 써서 모든 쌍별 비교에 대해 가족단위 오류율을 $\alpha$로 동시에 통제한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
항등식 $y_{ij} - \bar{y} = (\bar{y}_i - \bar{y}) + (y_{ij} - \bar{y}_i)$을 전개하여 $\text{SS}_{\text{total}} = \text{SST} + \text{SSE}$임을 보여라.

</div>

??? success "풀이"
    양변을 제곱하여 합하면

    $$
    \sum_{i}\sum_{j}(y_{ij} - \bar{y})^2 = \sum_{i}\sum_{j}(\bar{y}_i - \bar{y})^2 + 2\sum_{i}\sum_{j}(\bar{y}_i - \bar{y})(y_{ij} - \bar{y}_i) + \sum_{i}\sum_{j}(y_{ij} - \bar{y}_i)^2
    $$

    이다. 각 집단 $i$에서

    $$
    \sum_{j=1}^{n_i}(y_{ij} - \bar{y}_i) = 0
    $$

    이므로 교차항이 사라진다. 따라서 모든 $i$에서 $(\bar{y}_i - \bar{y})\sum_j (y_{ij} - \bar{y}_i) = 0$이다. 남은 두 항은 각각 정확히 $\text{SST}$($\sum_j (\bar{y}_i - \bar{y})^2 = n_i(\bar{y}_i - \bar{y})^2$임에 유의)와 $\text{SSE}$이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
키 예제에서 덴마크 집단의 크기가 $n = 30$이 아니라 $n = 5$라고 하자. 균형인 경우와 비교해 네덜란드 대 덴마크의 LSD 문턱은 어떻게 달라지는가?

</div>

??? success "풀이"
    LSD 문턱은

    $$
    \text{LSD} = t_{\alpha/2,\, N-k}\sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
    $$

    이다. $n_{\text{덴마크}}$가 30에서 5가 되면 $1/n_j$가 $1/30 \approx 0.033$에서 $1/5 = 0.2$로 커진다. 합 $1/n_i + 1/n_j$는 약 $0.067$에서 $0.233$으로 늘어 제곱근 안의 값이 거의 네 배가 된다. 그 결과 LSD 문턱이 크게 커져 네덜란드–덴마크 차이를 유의하다고 선언하기 어려워진다. 또한 전체 $N$이 줄고 $\text{MSE}$도 달라질 수 있어 문턱이 더 넓어질 수 있다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$\text{MST}$가 $\sigma^2$의 불편추정값이 되는 조건은 무엇인가? $H_0$이 거짓일 때 $\text{MST}$는 무엇을 추정하는가?

</div>

??? success "풀이"
    $H_0: \mu_1 = \cdots = \mu_k$ 아래에서 각 집단 평균 $\bar{Y}_i$가 공통 평균 $\mu$를 추정하며

    $$
    E[\text{MST}] = \sigma^2
    $$

    이므로 MST는 공통 분산의 불편추정량이다. $H_0$이 거짓이면

    $$
    E[\text{MST}] = \sigma^2 + \frac{\sum_{i=1}^{k} n_i (\mu_i - \bar{\mu})^2}{k - 1}
    $$

    이며 $\bar{\mu} = \sum n_i \mu_i / N$이다. 집단 평균이 모두 같지 않으면 둘째 항이 양수이므로 $E[\text{MST}] > \sigma^2$이다. $H_0$과 무관하게 $E[\text{MSE}] = \sigma^2$이므로 대립가설 아래에서 비 $F = \text{MST}/\text{MSE}$가 1보다 커지는 경향이 있고, 이것이 F-검정이 차이를 탐지할 검정력을 갖는 이유이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
예제의 PlantGrowth 자료에 **피셔 LSD**를 실제로 적용하고, 다중비교 보정을 한 결과와 비교하라.

</div>

??? success "풀이"
    **LSD의 정의.** 합동 $\text{MSE}$를 쓴 $t$ 검정이다.

    $$
    \text{LSD}_{ij}=t_{\alpha/2,\,N-k}\sqrt{\text{MSE}\Bigl(\frac1{n_i}+\frac1{n_j}\Bigr)}
    $$

    ```python
    import numpy as np
    from scipy import stats
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    groups = {
        "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
        "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
        "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
    }
    names = list(groups)
    values = np.concatenate([groups[n] for n in names])
    N, k = len(values), len(names)
    SSE = sum(((g - g.mean())**2).sum() for g in groups.values())
    MSE = SSE / (N - k)
    t_crit = stats.t.ppf(0.975, N - k)
    print(f"MSE = {MSE:.4f},  df = {N - k},  t_crit = {t_crit:.4f}\n")

    print("피셔 LSD")
    pvals, labels = [], []
    for a, b in combinations(range(k), 2):
        ga, gb = groups[names[a]], groups[names[b]]
        diff = ga.mean() - gb.mean()
        se = np.sqrt(MSE * (1 / len(ga) + 1 / len(gb)))
        t = diff / se
        p = 2 * stats.t.sf(abs(t), N - k)
        pvals.append(p)
        labels.append(f"{names[a]}-{names[b]}")
        print(f"  {names[a]}-{names[b]}:  차이 {diff:+.4f},  LSD {t_crit * se:.4f},"
              f"  t = {t:+.4f},  p = {p:.4f}"
              f"   {'유의' if abs(diff) > t_crit * se else ''}")

    for method, name in [("bonferroni", "본페로니"), ("holm", "홀름  ")]:
        adj = multipletests(pvals, method=method)[1]
        print(f"  {name}: " + "   ".join(f"{labels[i]} {adj[i]:.4f}"
                                         for i in range(3)))

    lab = np.repeat(names, [len(groups[n]) for n in names])
    print("\n투키 HSD")
    print(pairwise_tukeyhsd(values, lab, alpha=0.05))
    ```

    ```text
    MSE = 0.3886,  df = 27,  t_crit = 2.0518

    피셔 LSD
      ctrl-trt1:  차이 +0.3710,  LSD 0.5720,  t = +1.3308,  p = 0.1944   
      ctrl-trt2:  차이 -0.4940,  LSD 0.5720,  t = -1.7720,  p = 0.0877   
      trt1-trt2:  차이 -0.8650,  LSD 0.5720,  t = -3.1028,  p = 0.0045   유의
      본페로니: ctrl-trt1 0.5832   ctrl-trt2 0.2630   trt1-trt2 0.0134
      홀름  : ctrl-trt1 0.1944   ctrl-trt2 0.1754   trt1-trt2 0.0134

    투키 HSD
    Multiple Comparison of Means - Tukey HSD, FWER=0.05
    ===================================================
    group1 group2 meandiff p-adj   lower  upper  reject
    ---------------------------------------------------
      ctrl   trt1   -0.371 0.3909 -1.0622 0.3202  False
      ctrl   trt2    0.494  0.198 -0.1972 1.1852  False
      trt1   trt2    0.865  0.012  0.1738 1.5562   True
    ---------------------------------------------------
    ```

    **네 방법이 같은 결론에 이른다.** trt1과 trt2만 유의하다.

    **LSD가 세 쌍에 대해 같은 문턱(0.5720)을 쓴다.** 표본크기가 모두 10으로 같기 때문이다. **불균형이면 쌍마다 문턱이 달라진다**(연습문제 4).

    **보정 후 $p$ 값의 차이가 보인다.**

    | 쌍 | LSD $p$ | 본페로니 | 홀름 | 투키 |
    |---|---|---|---|---|
    | ctrl-trt1 | 0.194 | 0.583 | 0.194 | 0.391 |
    | ctrl-trt2 | 0.088 | 0.263 | 0.175 | 0.198 |
    | **trt1-trt2** | **0.0045** | **0.0134** | **0.0134** | **0.012** |

    **투키가 본페로니보다 덜 보수적**이다(0.198 대 0.263). 스튜던트화 범위분포를 쓰므로 **쌍별 비교에 특화**되어 있기 때문이다.

    **홀름은 가장 큰 $p$를 보정하지 않는다**(0.194 그대로). 세 방법 중 구조가 다르다.

    **ctrl-trt2가 경계에 있다.** LSD로 0.088, 투키로 0.198이다. **보정 여부가 결론을 바꿀 수 있는 자리**인데, 여기서는 어느 쪽으로도 유의하지 않다.

    **옴니버스 $F$ 검정이 $p=0.0159$로 유의했으므로** 사후비교로 넘어가는 것이 정당하다(보호된 절차).

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
연습문제 2가 지적한 LSD의 문제를 **모의실험으로 정량화**하라. "보호된" LSD는 정말 안전한가?

</div>

??? success "풀이"
    **두 가지 LSD를 구분해야 한다.**

    | 방식 | 절차 |
    |---|---|
    | 무보호 LSD | 옴니버스 $F$ 없이 바로 쌍별 $t$ 검정 |
    | **보호된 LSD** | $F$가 유의할 때만 쌍별 비교 |

    ```python
    import numpy as np
    from scipy import stats
    from itertools import combinations
    from statsmodels.stats.libqsturng import qsturng

    rng = np.random.default_rng(1212)
    M, n = 5_000, 10

    print("① 완전 귀무: 모든 평균이 같다")
    print(f"{'k':>3s} {'쌍':>4s} {'보호 LSD':>10s} {'무보호 LSD':>11s} "
          f"{'본페로니':>10s} {'투키':>8s}")
    for k in [3, 4, 6, 10]:
        n_pair = k * (k - 1) // 2
        a = b = c = d = 0
        for _ in range(M):
            g = [rng.standard_normal(n) for _ in range(k)]
            N = k * n
            MSE = sum(((x - x.mean())**2).sum() for x in g) / (N - k)
            se = np.sqrt(MSE * 2 / n)
            diffs = [abs(g[i].mean() - g[j].mean())
                     for i, j in combinations(range(k), 2)]
            lsd_hit = any(dd > stats.t.ppf(0.975, N - k) * se for dd in diffs)
            b += lsd_hit
            a += (stats.f_oneway(*g).pvalue < 0.05) and lsd_hit
            c += any(dd > stats.t.ppf(1 - 0.025 / n_pair, N - k) * se
                     for dd in diffs)
            d += any(dd > qsturng(0.95, k, N - k) * np.sqrt(MSE / n)
                     for dd in diffs)
        print(f"{k:3d} {n_pair:4d} {a / M:10.4f} {b / M:11.4f} "
              f"{c / M:10.4f} {d / M:8.4f}")
    ```

    ```text
    ① 완전 귀무: 모든 평균이 같다
      k    쌍     보호 LSD     무보호 LSD       본페로니       투키
      3    3     0.0520      0.1224     0.0442   0.0516
      4    6     0.0470      0.1954     0.0384   0.0476
      6   15     0.0468      0.3550     0.0362   0.0524
     10   45     0.0498      0.6098     0.0362   0.0536
    ```

    **무보호 LSD는 재앙이다.** $k=10$이면 모든 평균이 같은데도 **61%**가 뭔가를 발견한다.

    **보호된 LSD는 완전 귀무에서 잘 작동한다**(0.047~0.052). 옴니버스 $F$가 문지기 역할을 제대로 한다.

    **그런데 이것이 전부가 아니다.** 실제 상황에서는 **일부 평균만 다른** 경우가 흔하다.

    ```python
    rng = np.random.default_rng(3434)
    print("② 부분 귀무: 한 집단만 멀리 떨어지고 나머지는 모두 같다")
    print(f"{'k':>3s} {'F 기각률':>10s} {'보호 LSD':>10s} {'본페로니':>10s} "
          f"{'투키':>8s}")
    for k in [3, 4, 6, 10]:
        mu = np.zeros(k)
        mu[0] = 4.0
        n_pair = k * (k - 1) // 2
        f_rej = a = c = d = 0
        for _ in range(M):
            g = [rng.normal(m, 1, n) for m in mu]
            N = k * n
            MSE = sum(((x - x.mean())**2).sum() for x in g) / (N - k)
            se = np.sqrt(MSE * 2 / n)
            p_f = stats.f_oneway(*g).pvalue
            f_rej += p_f < 0.05
            # 참으로 같은 집단들(1..k-1) 사이에서 거짓 발견이 있는가
            idx = list(range(1, k))
            eq = [abs(g[i].mean() - g[j].mean())
                  for i, j in combinations(idx, 2)]
            if p_f < 0.05:
                a += any(dd > stats.t.ppf(0.975, N - k) * se for dd in eq)
            c += any(dd > stats.t.ppf(1 - 0.025 / n_pair, N - k) * se
                     for dd in eq)
            d += any(dd > qsturng(0.95, k, N - k) * np.sqrt(MSE / n)
                     for dd in eq)
        print(f"{k:3d} {f_rej / M:10.4f} {a / M:10.4f} {c / M:10.4f} "
              f"{d / M:8.4f}")
    ```

    ```text
    ② 부분 귀무: 한 집단만 멀리 떨어지고 나머지는 모두 같다
      k      F 기각률     보호 LSD       본페로니       투키
      3     1.0000     0.0540     0.0192   0.0222
      4     1.0000     0.1204     0.0246   0.0290
      6     1.0000     0.2864     0.0272   0.0352
     10     1.0000     0.5598     0.0280   0.0406
    ```

    **보호가 무너진다.** $k=10$에서 **참으로 같은 45쌍 중 적어도 하나를 거짓 발견할 확률이 0.560**이다.

    **왜 그런가.** 집단 1이 멀리 떨어져 있으므로 $F$ 검정이 **언제나 기각**한다(기각률 1.0000). 문지기가 문을 항상 열어 주므로 **보호가 사라진다.** 그 뒤의 쌍별 비교는 사실상 무보호 LSD다.

    **$k=3$에서만 보호가 유효하다**(0.054). 집단이 셋이면 "한 쌍만 다르다"는 상황에서 남는 비교가 하나뿐이라, $F$가 기각한 조건 아래에서도 그 하나의 수준이 $\alpha$를 넘지 않는다. 이것이 **피셔 LSD가 $k=3$에서만 권장되는 이유**다.

    **본페로니와 투키는 어느 경우에도 안전하다**(0.019~0.041). 오히려 보수적이다.

    **결론.**

    | $k$ | 권장 |
    |---|---|
    | 3 | 보호된 LSD도 무방 |
    | **4 이상** | **투키(모든 쌍) 또는 더넷(대조군 대비)** |
    | 어느 경우든 | 무보호 LSD는 쓰지 않는다 |

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
일원배치 분산분석이 **더미변수 회귀와 같다**는 것을 예제 자료로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    groups = {
        "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
        "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
        "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
    }
    names = list(groups)
    y = np.concatenate([groups[n] for n in names])
    N, k = len(y), len(names)

    # 처리(treatment) 코딩: 절편 + 두 더미
    X = np.zeros((N, k))
    X[:, 0] = 1
    for i in range(1, k):
        X[i * 10:(i + 1) * 10, i] = 1

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    SSE = resid @ resid
    SS_total = ((y - y.mean())**2).sum()
    SSR = SS_total - SSE
    F = (SSR / (k - 1)) / (SSE / (N - k))
    MSE = SSE / (N - k)
    se = np.sqrt(np.diag(MSE * np.linalg.inv(X.T @ X)))

    print(f"회귀계수 {np.round(beta, 4).tolist()}")
    print(f"  절편   = ctrl 평균        {groups['ctrl'].mean():.4f}")
    print(f"  β1     = trt1 − ctrl     "
          f"{groups['trt1'].mean() - groups['ctrl'].mean():+.4f}")
    print(f"  β2     = trt2 − ctrl     "
          f"{groups['trt2'].mean() - groups['ctrl'].mean():+.4f}")
    print(f"\nSSR = {SSR:.4f}  (= SST),  SSE = {SSE:.4f}")
    print(f"F = {F:.4f},  p = {stats.f.sf(F, k - 1, N - k):.4f}")
    print(f"scipy F = {stats.f_oneway(*groups.values()).statistic:.4f}")
    print(f"\nR² = η² = {SSR / SS_total:.4f}")
    omega2 = (SSR - (k - 1) * MSE) / (SS_total + MSE)
    print(f"ω²      = {omega2:.4f}   (편향 보정한 효과크기)")

    print(f"\n계수별 검정 (= ctrl 대비 비교)")
    for i, label in enumerate(["절편(ctrl)", "trt1−ctrl ", "trt2−ctrl "]):
        t = beta[i] / se[i]
        print(f"  {label}  β = {beta[i]:+.4f},  SE = {se[i]:.4f},  "
              f"t = {t:+.4f},  p = {2 * stats.t.sf(abs(t), N - k):.4f}")
    ```

    ```text
    회귀계수 [5.032, -0.371, 0.494]
      절편   = ctrl 평균        5.0320
      β1     = trt1 − ctrl     -0.3710
      β2     = trt2 − ctrl     +0.4940

    SSR = 3.7663  (= SST),  SSE = 10.4921
    F = 4.8461,  p = 0.0159
    scipy F = 4.8461

    R² = η² = 0.2641
    ω²      = 0.2041   (편향 보정한 효과크기)

    계수별 검정 (= ctrl 대비 비교)
      절편(ctrl)  β = +5.0320,  SE = 0.1971,  t = +25.5265,  p = 0.0000
      trt1−ctrl   β = -0.3710,  SE = 0.2788,  t = -1.3308,  p = 0.1944
      trt2−ctrl   β = +0.4940,  SE = 0.2788,  t = +1.7720,  p = 0.0877
    ```

    **완전히 일치한다.** $F=4.8461$, $SSR=SST=3.7663$이다.

    **계수의 해석이 직관적이다.**

    | 계수 | 뜻 |
    |---|---|
    | 절편 | **기준 집단(ctrl)의 평균** |
    | $\beta_1$ | trt1이 ctrl보다 얼마나 높은가 |
    | $\beta_2$ | trt2가 ctrl보다 얼마나 높은가 |

    **계수별 $t$ 검정이 곧 LSD**다. $p=0.1944$와 $0.0877$이 연습문제 6의 LSD $p$ 값과 정확히 같다. **보정이 전혀 없다**는 뜻이기도 하다.

    **$R^2=\eta^2=0.2641$.** 분산분석에서 $\eta^2$(에타제곱)이라 부르는 효과크기가 회귀의 결정계수와 같은 양이다.

    **$\omega^2=0.2041$이 더 작다.** $\eta^2$은 위로 편향되어 있고, $\omega^2$은 그것을 보정한다.

    $$
    \omega^2=\frac{\text{SST}-(k-1)\text{MSE}}{\text{SS}_{\text{total}}+\text{MSE}}
    $$

    **$H_0$가 참이어도 $\eta^2$의 기댓값이 $(k-1)/(N-1)=2/29=0.069$**이므로, $\eta^2$을 액면대로 읽으면 없는 효과를 만들어 낸다. 10장의 크라메르 $V$와 같은 문제다.

    **이 관점이 열어 주는 것 넷.**

    1. **공변량을 넣을 수 있다** — 공분산분석(ANCOVA).
    2. **요인을 여러 개** 넣으면 이원배치 분산분석이다.
    3. **코딩 방식을 바꾸면** 다른 대비를 검정한다(효과 코딩, 다항 대비 등).
    4. **이분산 로버스트 표준오차**를 쓰면 웰치 분산분석에 가까워진다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
연습문제 5가 묻는 $E[\text{MST}]$를 **모의실험으로 확인**하고, 연습문제 4가 다룬 불균형 설계의 대가를 재어라.

</div>

??? success "풀이"
    **이론.**

    $$
    E[\text{MSE}]=\sigma^2,\qquad
    E[\text{MST}]=\sigma^2+\frac{\sum_i n_i(\mu_i-\bar\mu)^2}{k-1}
    $$

    $H_0$가 참이면 둘 다 $\sigma^2$이라 $F$의 기댓값이 1 근처가 된다. $H_0$가 거짓이면 분자만 커진다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5656)
    M, sigma2 = 20_000, 4.0

    print(f"{'μ':>18s} {'n':>4s} {'E[MSE]':>9s} {'σ²':>6s} "
          f"{'E[MST]':>10s} {'이론':>10s}")
    for mu, n in [([0, 0, 0], 10), ([0, 1, 2], 10),
                  ([0, 3, 6], 10), ([0, 1, 2], 30)]:
        mu = np.array(mu, float)
        k, N = len(mu), len(mu) * n
        mst, mse = [], []
        for _ in range(M):
            g = [rng.normal(m, np.sqrt(sigma2), n) for m in mu]
            allv = np.concatenate(g)
            gm = allv.mean()
            mst.append(sum(len(x) * (x.mean() - gm)**2 for x in g) / (k - 1))
            mse.append(sum(((x - x.mean())**2).sum() for x in g) / (N - k))
        theory = sigma2 + n * ((mu - mu.mean())**2).sum() / (k - 1)
        print(f"{str(mu.tolist()):>18s} {n:4d} {np.mean(mse):9.4f} "
              f"{sigma2:6.1f} {np.mean(mst):10.4f} {theory:10.4f}")
    ```

    ```text
                     μ    n    E[MSE]     σ²     E[MST]         이론
       [0.0, 0.0, 0.0]   10    4.0092    4.0     3.9900     4.0000
       [0.0, 1.0, 2.0]   10    3.9855    4.0    13.9298    14.0000
       [0.0, 3.0, 6.0]   10    4.0061    4.0    94.0162    94.0000
       [0.0, 1.0, 2.0]   30    4.0061    4.0    34.0595    34.0000
    ```

    **$E[\text{MSE}]=\sigma^2$가 네 경우 모두에서 성립**한다(3.99~4.01). 평균이 다르든 말든 **MSE는 언제나 $\sigma^2$의 불편추정값**이다.

    **$E[\text{MST}]$는 이론값과 정확히 일치**한다(13.93 대 14, 94.02 대 94).

    **연습문제 5의 답이 여기서 확인된다.**

    - $\text{MST}$가 $\sigma^2$의 불편추정값인 것은 **$H_0$가 참일 때뿐**이다.
    - $H_0$가 거짓이면 $\sigma^2+\sum n_i(\mu_i-\bar\mu)^2/(k-1)$을 추정한다.

    **불균형 설계의 대가.**

    ```python
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    sigma, k = 1.0, 3
    mu = np.array([0, 0.5, 1.0])
    print(f"\n총 N = 60 고정,  μ = {mu.tolist()},  σ = {sigma}")
    print(f"{'배분':>16s} {'λ':>8s} {'검정력':>8s} {'1-3 쌍의 LSD 문턱':>18s}")
    for ns in [(20, 20, 20), (30, 25, 5), (10, 20, 30), (40, 15, 5), (26, 26, 8)]:
        ns = np.array(ns)
        N = ns.sum()
        m_bar = (ns * mu).sum() / N
        lam = (ns * (mu - m_bar)**2).sum() / sigma**2
        crit = stats.f.ppf(0.95, k - 1, N - k)
        power = stats.ncf.sf(crit, k - 1, N - k, lam)
        lsd = stats.t.ppf(0.975, N - k) * sigma * np.sqrt(1 / ns[0] + 1 / ns[2])
        print(f"{str(ns.tolist()):>16s} {lam:8.4f} {power:8.4f} {lsd:18.4f}")
    ```

    ```text

    총 N = 60 고정,  μ = [0.0, 0.5, 1.0],  σ = 1.0
                  배분        λ      검정력      1-3 쌍의 LSD 문턱
        [20, 20, 20]  10.0000   0.7933             0.6332
         [30, 25, 5]   6.1458   0.5710             0.9673
        [10, 20, 30]   8.3333   0.7120             0.7312
         [40, 15, 5]   6.1458   0.5710             0.9499
         [26, 26, 8]   7.1500   0.6407             0.8096
    ```

    **균형 설계가 가장 강력하다**(검정력 0.793).

    **한 집단만 작으면 두 배로 손해다.**

    | | 균형 (20,20,20) | 불균형 (30,25,5) |
    |---|---|---|
    | $\lambda$ | 10.00 | 6.15 |
    | 검정력 | **0.793** | 0.571 |
    | LSD 문턱 | **0.633** | 0.967 |

    **옴니버스 검정력이 22%포인트 떨어지고, 사후비교의 문턱이 53% 높아진다.** 연습문제 4의 직관이 수치로 확인된다.

    **왜 균형이 유리한가.** $\lambda=\sum n_i(\mu_i-\bar\mu)^2/\sigma^2$에서 **$\bar\mu$가 가중평균**이라, 한쪽에 표본이 몰리면 그 집단 쪽으로 중심이 끌려가 편차제곱합이 줄어든다.

    **다만 예외가 있다.** 분산이 다르면 균형이 최적이 아니다. 9장에서 본 네이만 배분처럼 **표준편차에 비례**해 배분하는 것이 낫다. 여기서는 등분산을 가정했다.

    **실무 권고 셋.**

    1. **등분산이 예상되면 균형 설계**로 간다.
    2. **탈락을 예상해 조금 여유 있게** 모집한다. 한 집단만 작아지는 것이 가장 나쁘다.
    3. **대조군 대비 비교가 주 관심**이면 대조군을 $\sqrt{k-1}$배로 키우는 배분이 유리하다(더넷 설계).

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
일원배치 분산분석의 **손계산 절차와 점검 목록**을 정리하라.

</div>

??? success "풀이"

    **계산 순서.**

    ```text
    ① 전체 평균 ȳ 와 집단 평균 ȳ_i 를 구한다
              ↓
    ② SST = Σ n_i (ȳ_i − ȳ)²        ← 집단 간
       SSE = Σ Σ (y_ij − ȳ_i)²       ← 집단 내
              ↓ 검산
       SS_total = Σ (y_ij − ȳ)²  =  SST + SSE
              ↓
    ③ MST = SST/(k−1),  MSE = SSE/(N−k)
              ↓
    ④ F = MST / MSE,  p = P(F_{k−1,N−k} > F)   ← 오른쪽 꼬리
              ↓
    ⑤ 효과크기 η² = SST/SS_total,  ω² (편향 보정)
              ↓
    ⑥ 유의하면 사후비교 (k≥4 이면 투키)
    ```

    **분산분석표의 형태.**

    | 요인 | SS | df | MS | F |
    |---|---|---|---|---|
    | 집단 간 | SST | $k-1$ | MST | MST/MSE |
    | 집단 내 | SSE | $N-k$ | MSE | |
    | 전체 | SS$_{\text{total}}$ | $N-1$ | | |

    **자유도의 합이 맞는지 확인**한다. $(k-1)+(N-k)=N-1$이다.

    **검산 셋.**

    1. **$\text{SST}+\text{SSE}=\text{SS}_{\text{total}}$** — 가장 중요
    2. **자유도의 합** — $(k-1)+(N-k)=N-1$
    3. **$F$가 음수가 아닌가** — 제곱합은 모두 0 이상

    **점검 목록.**

    - [ ] 관측이 **독립**인가(반복측정이면 다른 방법)
    - [ ] 각 집단이 근사적으로 **정규**인가(또는 $n$이 충분한가)
    - [ ] **분산이 비슷**한가 — 아니면 웰치 분산분석
    - [ ] 표본크기가 **균형**인가
    - [ ] **효과크기**($\eta^2$ 또는 $\omega^2$)를 보고했는가
    - [ ] 사후비교에 **보정**을 했는가
    - [ ] $k\ge4$인데 LSD를 쓰지 않았는가

    **자주 하는 실수 다섯.**

    | 실수 | 대가 |
    |---|---|
    | SST에 $n_i$ 가중치를 빠뜨림 | 불균형 설계에서 틀린 값 |
    | 자유도를 $k$와 $N$으로 잘못 씀 | $p$ 값이 틀림 |
    | 왼쪽 꼬리를 봄 | $p$ 값이 $1-p$ |
    | $k\ge4$에서 보호된 LSD 사용 | FWER이 0.56까지(연습문제 7) |
    | $\eta^2$을 액면대로 해석 | 위로 편향 |

    **첫째가 손계산에서 가장 흔하다.** $\text{SST}=\sum_i n_i(\bar y_i-\bar y)^2$에서 $n_i$를 빠뜨리면, 표본크기가 큰 집단의 이탈이 과소평가된다. 균형 설계에서는 전체가 상수배로 어긋나 $F$가 크게 달라진다.

    **왜 손으로 해 보는가.** `scipy.stats.f_oneway`가 한 줄이지만, 손으로 분해해 보면

    1. **$F$가 "신호 대 잡음"**임이 눈에 들어온다.
    2. **자유도가 어디서 오는지** 이해된다.
    3. **불균형이 왜 손해인지**(연습문제 9) 식으로 보인다.
    4. **회귀와 같은 것**임을 알아볼 수 있다(연습문제 8).

    **한 문장.** 분산분석표는 **전체 변동을 두 조각으로 나눈 회계장부**이고, $F$는 그 두 조각의 크기를 자유도로 정규화해 비교한 값이다.

---

## 정리하며

분산분석의 모든 양을 **손으로 계산**해 보았다.

$$
\underbrace{\sum_{ij}(y_{ij}-\bar y)^2}_{\text{SS}_{\text{total}}}
=\underbrace{\sum_i n_i(\bar y_i-\bar y)^2}_{\text{SST}}
+\underbrace{\sum_{ij}(y_{ij}-\bar y_i)^2}_{\text{SSE}}
$$

- **분해가 정확한 항등식이다.** 교차항이 $\sum_j(y_{ij}-\bar y_i)=0$ 때문에 사라지며, 0장의 직교사영과 같은 구조다. **근사가 아니다.**
- **자유도도 함께 분해된다.** $(N-1)=(k-1)+(N-k)$ 이며, 제곱합과 자유도가 나란히 쪼개지는 것이 분산분석표의 뼈대다.
- **`f_oneway` 와 대조해 검산한다.** 손 계산이 라이브러리와 맞으면 이해가 확인되고, 틀리면 대개 평균을 잘못 잡았거나 자유도를 착각한 것이다.
- **피셔의 LSD 는 보정이 없다.** 전체 $F$ 가 유의할 때만 쓰는 것이 원칙이며, 그 조건 아래에서도 $k$ 가 크면 FWER 이 통제되지 않는다. **보수적인 대안이 다음 절들의 주제다.**
- **한 번은 손으로 해 보는 것이 값어치가 있다.** 뒤에 나올 이원배치와 사후검정이 모두 이 분해의 확장이다.

다음 절부터 **이원배치 분산분석**으로 넘어간다. 요인이 둘이 되면 교호작용이 등장한다.
