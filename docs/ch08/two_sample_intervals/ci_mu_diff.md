# μ₁ − μ₂의 신뢰구간

## 평균 차이에 대한 이표본 신뢰구간

두 모집단을 비교할 때 우리는 흔히 두 평균의 차이에 관심을 둔다. $\mu_1 - \mu_2$의 신뢰구간은 표본변동을 감안하여 이 차이에 대해 그럴듯한 값들의 범위를 준다.

---

## 상황별 공식

### 분산을 아는 경우 (z-구간)

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \times \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

여기서 $\sigma_1^2$과 $\sigma_2^2$은 알려진 모분산이고, $z_{\alpha/2}$는 $P(Z > z_{\alpha/2}) = \alpha/2$를 만족하는 임계값이다.

### 분산을 모르고 서로 다른 경우 — Welch의 t-구간

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, \text{df}} \times \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

여기서 자유도는 **Welch–Satterthwaite 식**으로 계산한다:

$$
\text{df} = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{1}{n_1 - 1}\left(\frac{s_1^2}{n_1}\right)^2 + \frac{1}{n_2 - 1}\left(\frac{s_2^2}{n_2}\right)^2}
$$

!!! tip "기본 선택"
    분산이 같다는 강한 근거가 없다면 Welch의 t-구간을 택하라.

### 분산을 모르고 서로 같은 경우 — 합동 t-구간

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, n_1+n_2-2} \times \sqrt{s_p^2\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

여기서 **합동분산**은

$$
s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}
$$

이고 $\text{df} = n_1 + n_2 - 2$이다.

### 큰 표본 (표본분산을 쓰는 z-구간)

$n_1 \ge 30$이고 $n_2 \ge 30$이면 분산을 모르고 서로 달라도 정규근사를 쓸 수 있다:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \times \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

<div class="codebox" markdown>

#### 예제 1. 두 평균 차이의 신뢰구간 계산 { .eg }

```python
import numpy as np
import scipy.stats as stats

n1, n2 = 30, 25
mean1, mean2 = 100, 90
s1, s2 = 15, 20
confidence_level = 0.95

# 두 표본이 독립이므로 분산이 더해진다.
# 표준편차가 아니라 **분산**을 더한 뒤 제곱근이라는 점에 주의.
standard_error = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# Welch-Satterthwaite 자유도. 정수가 아니어도 scipy는 받아 준다.
# 두 표본의 정보량을 하나로 합치는 근사이며, 한쪽 분산이 압도하면
# 그쪽 표본의 자유도 쪽으로 끌려간다.
df = ((s1**2 / n1) + (s2**2 / n2))**2 / (
    ((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1))
)

t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
margin_of_error = t_critical * standard_error

confidence_interval = (
    (mean1 - mean2) - margin_of_error,
    (mean1 - mean2) + margin_of_error,
)
print(f"{confidence_interval = }")
```

출력:

```
confidence_interval = (0.22892977648461788, 19.77107022351538)
```

구간이 0을 아슬아슬하게 벗어난다. 점추정값은 차이가 10이라고 말하지만, 구간은 0.23만큼 작을 수도 19.77만큼 클 수도 있다고 말한다. "차이가 있다"까지는 말할 수 있어도 "얼마나 있다"는 거의 말하지 못하는 자료다.

</div>

---

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 평균 차이의 95% 신뢰구간 (Welch). 독립인 두 표본: 표본 1은 $n_1 = 30$, $\bar{X}_1 = 100$, $s_1 = 15$이고, 표본 2는 $n_2 = 25$, $\bar{X}_2 = 90$, $s_2 = 20$이다.

</div>

??? success "풀이"

    $$
    \text{SE} = \sqrt{\frac{225}{30} + \frac{400}{25}} = \sqrt{7.5 + 16} = \sqrt{23.5} \approx 4.847
    $$

    Welch–Satterthwaite 자유도:

    $$
    \text{df} = \frac{(7.5 + 16)^2}{\frac{7.5^2}{29} + \frac{16^2}{24}} = \frac{552.25}{12.606} \approx 43.8
    $$

    $t_{0.025, 43} \approx 2.017$이므로:

    $$
    \text{ME} = 2.017 \times 4.847 \approx 9.78
    $$

    $$
    \boxed{(0.22,\ 19.78)}
    $$

    두 모평균의 참 차이가 0.22와 19.78 사이에 있다고 95% 신뢰한다.
---

## 모의실험: 이표본 평균 신뢰구간의 포함확률

<div class="codebox" markdown>

### 예제 2. 네 방법의 포함확률 비교 { .eg }

```python
#!/usr/bin/env python3
"""두 평균의 차이에 대한 신뢰구간을 네 방법으로 만들어 포함확률을 비교한다.

Welch 는 두 분산이 다를 수 있다고 보고, 합동(pooled)은 같다고 본다.
분산이 실제로 다를 때 합동 방법의 포함확률이 어떻게 무너지는지가 요점이다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

rng_seed = 42        # 아래 그림을 재현하려면 고정한다
n_simulations = 100
n1, n2 = 12, 10
mu1, mu2 = 0.0, 0.5
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05
method = "welch"  # 'welch' | 'pooled' | 'z_known' | 'z_plugin'


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    delta_true = mu1 - mu2
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu1, scale=sigma1, size=n1)
        y = np.random.normal(loc=mu2, scale=sigma2, size=n2)
        xbar, ybar = x.mean(), y.mean()
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        diff_hat = xbar - ybar
        centers[i] = diff_hat

        if method == "welch":
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
            df = num / den
            crit = t.ppf(1 - alpha / 2.0, df=df)
        elif method == "pooled":
            df = n1 + n2 - 2
            sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
            se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
            crit = t.ppf(1 - alpha / 2.0, df=df)
        elif method == "z_known":
            se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
            crit = norm.ppf(1 - alpha / 2.0)
        else:  # z_plugin
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            crit = norm.ppf(1 - alpha / 2.0)

        lowers[i] = diff_hat - crit * se
        uppers[i] = diff_hat + crit * se

    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)
    ax.axvline(delta_true, linestyle="--", linewidth=1.5)
    ax.set_title(
        f"{n_simulations} Two-Sample Mean CIs ({method}) | n1={n1}, n2={n2}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Δ = μ₁ − μ₂")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

![100 Two-Sample Mean CIs (welch) | n1=12, n2=10, CL=95%](./img/ci_mu_diff_123.png)

포함확률 97.0%로 명목값을 달성한다(100회 모의실험의 표준오차가 2.2%p이므로 95%와 구별되지 않는다). `method`를 `"pooled"`나 `"z_plugin"`으로 바꿔 같은 자료에 다시 돌려 보면 방법마다 어디서 무너지는지 볼 수 있다.

</div>

---

## 핵심 정리

- 두 모평균을 비교할 때는 $\mu_1 - \mu_2$의 신뢰구간을 구성한다.
- 모분산을 모르고 서로 다르면 **Welch의 t-구간**을 쓴다(기본 선택).
- 분산이 같다고 가정하면 **합동 t-구간**이 결합된 분산추정값을 쓴다.
- 신뢰구간의 너비는 표본크기, 표본분산, 신뢰수준에 달려 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
독립인 두 표본: $\sigma_A = 15, n_A = 36$; $\sigma_B = 20, n_B = 49$. $\mathrm{SE}(\bar X_A - \bar X_B)$를 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE} = \sqrt{15^2/36 + 20^2/49} = \sqrt{6.25 + 8.16} = \sqrt{14.41} \approx 3.80$.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
표본 1: $\bar X_1 = 55, s_1 = 8, n_1 = 30$. 표본 2: $\bar X_2 = 50, s_2 = 10, n_2 = 35$. $\mu_1 - \mu_2$의 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\mathrm{SE} = \sqrt{64/30 + 100/35} \approx 2.233$. $n$이 크므로 $z$를 쓴다: 오차한계 = $1.96 \cdot 2.233 \approx 4.38$.

    신뢰구간: $(5 - 4.38, 5 + 4.38) = (0.62, 9.38)$. 0을 포함하지 *않는다* — 차이가 있다는 증거이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
두 교수법에 대한 Welch 신뢰구간: A ($n = 30, \bar X = 78, s = 8$), B ($n = 35, \bar X = 82, s = 10$). $\mu_A - \mu_B$의 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\mathrm{SE} = \sqrt{64/30 + 100/35} \approx 2.234$.

    Welch 자유도: $\nu = (2.133 + 2.857)^2/[2.133^2/29 + 2.857^2/34] = 24.9/0.397 \approx 62.7$. $t_{62}$를 쓴다.

    $t_{0.975, 62} \approx 2.00$. 신뢰구간: $(78 - 82) \pm 2.00 \cdot 2.234 = -4 \pm 4.47 = (-8.47, 0.47)$.

    0을 포함한다 — 5% 수준에서 유의한 차이를 결론지을 수 없다. 대부분 음수여서 방법 B가 더 나을 수 있음을 시사하지만 증거가 결정적이지 않다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**합동 대 Welch.** 각각 언제 쓰며, 둘을 가르는 가정은 무엇인가?

</div>

??? success "풀이"
    **합동 $t$-검정:** $\sigma_1 = \sigma_2$를 가정한다. 두 표본을 합쳐 공통 $\sigma$를 추정한다. 자유도 $= n_1 + n_2 - 2$.

    **Welch $t$-검정:** $\sigma_1 \ne \sigma_2$를 허용한다. 각각의 분산을 따로 쓴다. Welch-Satterthwaite 공식으로 근사 자유도를 구한다.

    **합동을 쓸 때:** (예컨대 실험설계에 의해) 분산이 같다고 *알려져* 있을 때. 효율 이득이 크지 않다.

    **Welch를 쓸 때:** 기본. 분산이 같을 필요가 없고, 실제로 분산이 같을 때도 합동에 거의 맞먹는 효율을 낸다.

    현대의 권고: 등분산이 구조적으로 보장되지 않는 한 **항상 Welch를 쓰라**. R의 `t.test`는 기본이 Welch이다. 분산이 같은데 Welch를 "잘못" 쓰는 대가는 작지만, 다른데 "잘못" 합동하는 대가는 클 수 있다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**대응 대 독립.** 피험자 $n = 50$명을 치료 전후로 측정했다. 이 연습문제의 이표본 신뢰구간을 적용해야 하는가?

</div>

??? success "풀이"
    **아니다.** 전후 측정은 같은 피험자를 두 번 잰 것이므로 *대응*되어 있다. 서로 독립이 *아니다*. 이표본 신뢰구간을 쓰면 이들을 독립으로 취급하여 피험자 내 상관을 무시하게 된다.

    **올바른 접근:** 피험자마다 차이 $D_i = \mathrm{post}_i - \mathrm{pre}_i$를 계산하고, $\bar D, s_D$와 자유도 $n - 1$로 $D$에 대한 일표본 신뢰구간을 구한다.

    **왜 중요한가:** 전후가 (흔히 그렇듯) 양의 상관을 가지면 $\mathrm{Var}(D) < \mathrm{Var}(\mathrm{pre}) + \mathrm{Var}(\mathrm{post})$이다. 대응 분석은 표준오차가 작고 신뢰구간이 좁으며 검정력이 크다.

    이표본 신뢰구간은 대응에 관한 정보를 잃고 필요 이상으로 넓은 구간을 준다. 분석은 항상 설계에 맞추어야 한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**효과크기.** $\mu_1 - \mu_2$의 신뢰구간과 함께 **Cohen의 $d$** = (효과)/(합동 표준편차)를 보고하라. 해석하라.

</div>

??? success "풀이"
    연습문제 3에서 합동 표준편차 $= \sqrt{((29 \cdot 64) + (34 \cdot 100))/63} = \sqrt{(1856 + 3400)/63} = \sqrt{83.4} \approx 9.13$.

    Cohen의 $d = (78 - 82)/9.13 \approx -0.44$.

    **해석:**

    - $|d| = 0.2$: 작은 효과.
    - $|d| = 0.5$: 중간 효과.
    - $|d| = 0.8$: 큰 효과.

    $d = -0.44$는 "중간에 조금 못 미치는" 효과이다 — 방법 B의 평균이 A의 평균보다 약 0.44 표준편차 위에 있다. 신뢰구간은 유의성에 살짝 못 미쳤지만 효과크기는 무시할 수준이 아니다.

    항상 둘 다 보고하라:

    - **통계적 유의성** (신뢰구간이 0을 배제, $p < \alpha$).
    - **실질적 유의성** (효과크기가 의미 있을 만큼 큰가).

    효과크기는 무차원이어서 연구와 분야를 넘어 비교할 수 있으므로 메타분석에 유용하다. 유의성 검정은 표본크기에 좌우되지만 효과크기는 그렇지 않다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**새터스웨이트 자유도**가 언제 극단으로 가는지 조사하고, 실무적 의미를 설명하라.

</div>

??? success "풀이"
    **공식.**

    $$
    \nu=\frac{\left(\dfrac{s_1^2}{n_1}+\dfrac{s_2^2}{n_2}\right)^2}
    {\dfrac{(s_1^2/n_1)^2}{n_1-1}+\dfrac{(s_2^2/n_2)^2}{n_2-1}}
    $$

    ```python
    import numpy as np

    def satt(n1, n2, s1, s2):
        a, b = s1**2 / n1, s2**2 / n2
        return (a + b)**2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))

    print(f"{'n1':>4s} {'n2':>4s} {'s1':>5s} {'s2':>5s} {'ν':>8s} "
          f"{'하한':>6s} {'상한':>6s}")
    for n1, n2, s1, s2 in [(10, 10, 1, 1), (10, 10, 1, 5),
                           (10, 50, 1, 1), (10, 50, 5, 1),
                           (10, 50, 1, 5), (5, 100, 1, 10)]:
        v = satt(n1, n2, s1, s2)
        print(f"{n1:4d} {n2:4d} {s1:5.1f} {s2:5.1f} {v:8.2f} "
              f"{min(n1, n2) - 1:6d} {n1 + n2 - 2:6d}")
    ```

    ```text
      n1   n2    s1    s2        ν     하한     상한
      10   10   1.0   1.0    18.00      9     18
      10   10   1.0   5.0     9.72      9     18
      10   50   1.0   1.0    12.87      9     58
      10   50   5.0   1.0     9.14      9     58
      10   50   1.0   5.0    57.94      9     58
       5  100   1.0  10.0    71.64      4    103
    ```

    **$\nu$는 항상 $\min(n_1,n_2)-1$과 $n_1+n_2-2$ 사이에 있다.** 이것이 정리로 증명되어 있다.

    **두 극단.**

    1. **한쪽 항이 지배하면** $\nu$가 그 집단의 자유도에 가까워진다. $(10,50,5,1)$에서 $s_1^2/n_1=2.5$가 $s_2^2/n_2=0.02$를 압도해 $\nu=9.14\approx n_1-1=9$다. **작은 표본에 큰 분산이면 그 집단의 자유도만 쓰는 셈**이다.

    2. **다른 쪽이 지배하면** 큰 집단의 자유도 쪽으로 간다. $(5,100,1,10)$에서 $\nu=71.6$으로 $n_2-1=99$에 가까워진다.

    3. **두 항이 비슷하면** 합쳐진 자유도에 가까워진다. $(10,10,1,1)$에서 $\nu=18=n_1+n_2-2$로 최댓값이다.

    **실무적 의미.**

    - **표본을 어디에 투자할지 알려 준다.** 분산이 큰 집단의 표본을 늘려야 $\nu$가 오르고 구간이 좁아진다. **분산이 작은 집단을 늘리는 것은 거의 소용이 없다.**

    - **$\nu$가 작으면 결과를 조심한다.** $\nu<10$이면 $t$ 임계값이 2.2를 넘어 구간이 크게 넓어진다. 이런 상황이면 표본 설계를 재검토할 필요가 있다.

    - **$\nu$는 확률변수다.** $s_1$, $s_2$에 의존하므로 표본마다 달라진다. 보고할 때 소수점 한 자리까지 적는 것이 관례다(정수로 반올림하면 정보가 사라진다).

    **최적 배분과의 연결.** 앞서 본 대로 분산이 다르면 $n_2/n_1=\sigma_2/\sigma_1$이 최적이다. 그때 $s_1^2/n_1\approx s_2^2/n_2$가 되어 **$\nu$가 최대에 가까워진다.** 최적 배분이 자유도도 최대화한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
두 집단 비교에서 **층화**나 **공변량 보정**이 구간을 얼마나 좁히는지 계산하고, 언제 쓸 가치가 있는지 논하라.

</div>

??? success "풀이"
    **원리.** 결과와 상관된 공변량 $X$를 모형에 넣으면 잔차분산이 $\sigma^2(1-R^2)$로 줄어든다.

    $$
    \frac{\text{보정 후 폭}}{\text{보정 전 폭}}\approx\sqrt{1-R^2}
    $$

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(55)
    n, M, tau = 40, 5_000, 1.0                  # 집단당 40명
    print(f"{'R²':>6s} {'단순 폭':>9s} {'보정 폭':>9s} {'비':>7s} "
          f"{'유효 n 배율':>12s}")
    for R2 in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        w_raw = w_adj = 0.0
        for _ in range(M // 10):
            g = np.repeat([0, 1], n)
            x = rng.normal(0, 1, 2 * n)
            y = tau * g + np.sqrt(R2) * x + np.sqrt(1 - R2) * rng.normal(0, 1, 2 * n)

            a, b = y[g == 0], y[g == 1]
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            w_raw += 2 * stats.t.ppf(0.975, 2 * n - 2) * sp * np.sqrt(2 / n)

            X = np.column_stack([np.ones(2 * n), g, x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            df = 2 * n - 3
            se = np.sqrt((r @ r / df) * np.linalg.inv(X.T @ X)[1, 1])
            w_adj += 2 * stats.t.ppf(0.975, df) * se
        w_raw /= M // 10
        w_adj /= M // 10
        print(f"{R2:6.1f} {w_raw:9.4f} {w_adj:9.4f} {w_adj / w_raw:7.4f} "
              f"{(w_raw / w_adj)**2:12.2f}")
    ```

    ```text
        R²    단순 폭    보정 폭       비     유효 n 배율
       0.0    0.8916    0.8978   1.0069         0.99
       0.1    0.8823    0.8411   0.9533         1.10
       0.3    0.8904    0.7456   0.8373         1.43
       0.5    0.8886    0.6293   0.7082         1.99
       0.7    0.8896    0.4906   0.5514         3.29
       0.9    0.8861    0.2848   0.3214         9.68
    ```

    **$R^2=0.5$면 폭이 29% 줄고, 표본을 두 배로 늘린 것과 같다.** $R^2=0.9$면 표본 10배에 가까운 효과다.

    **$R^2=0$이면 손해다.** 폭이 0.7% 늘어난다. 자유도를 하나 잃기 때문이다. **무관한 공변량을 넣는 비용은 매우 작지만 이득도 없다.**

    **언제 쓸 가치가 있는가.**

    | $R^2$ | 판단 |
    |---|---|
    | $<0.05$ | 넣을 이유 없음 |
    | 0.1~0.3 | 넣으면 이득(표본 10~40% 절약) |
    | $>0.5$ | **반드시 넣는다** |

    **무작위 실험에서의 주의점.**

    1. **공변량은 배정 전에 측정된 것이어야 한다.** 사후 변수를 넣으면 효과의 일부를 지운다.
    2. **공변량 목록을 사전에 고정**한다. 자료를 보고 고르면 선택 편향이 생긴다.
    3. **무작위 배정이면 편향은 없다.** 공변량을 넣든 안 넣든 추정량은 불편이며, 정밀도만 달라진다. 이 점에서 관측연구와 근본적으로 다르다.

    **층화 대 보정.** 층화(블록화)는 설계 단계에서, 보정은 분석 단계에서 같은 일을 한다.

    - **층화**는 균형을 보장하고 해석이 쉽다. 다만 변수가 많으면 층이 폭발한다.
    - **보정**은 연속변수를 그대로 쓸 수 있고 여러 변수를 다루기 쉽다.
    - **층화했다면 분석에서도 층을 반영**해야 한다. 안 하면 보수적이 된다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
두 집단의 **분포 전체**를 비교하는 구간을 만들고 싶다면 어떤 방법이 있는가?

</div>

??? success "풀이"
    **평균 차이의 한계.** 두 분포가 같은 평균을 가지면서 전혀 다를 수 있다. 처리가 **일부에게만** 효과가 있거나 **분산을 바꾸는** 경우 평균 하나로는 잡히지 않는다.

    **방법 1 — 분위수 차이.** 여러 분위수에서 차이를 재고 각각 구간을 만든다.

    ```python
    import numpy as np

    rng = np.random.default_rng(19)
    x = rng.normal(10, 2, 60)                   # 대조군
    y = rng.normal(10, 4, 60)                   # 처리군: 평균 같고 분산만 큼
    B = 20_000
    bx = x[rng.integers(0, len(x), (B, len(x)))]
    by = y[rng.integers(0, len(y), (B, len(y)))]

    print(f"{'분위수':>7s} {'차이':>8s} {'95% 구간':>20s}")
    for q in [10, 25, 50, 75, 90]:
        d = np.percentile(y, q) - np.percentile(x, q)
        bd = np.percentile(by, q, axis=1) - np.percentile(bx, q, axis=1)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        print(f"{q:6d}% {d:8.3f}   ({lo:7.3f}, {hi:7.3f})")

    d0 = y.mean() - x.mean()
    bd0 = by.mean(1) - bx.mean(1)
    print(f"\n평균 차이 {d0:.3f}   "
          f"({np.percentile(bd0, 2.5):.3f}, {np.percentile(bd0, 97.5):.3f})")
    ```

    ```text
      분위수       차이             95% 구간
        10%   -3.179   ( -5.003,  -1.031)
        25%   -1.515   ( -2.904,  -0.058)
        50%    0.978   ( -1.710,   2.846)
        75%    2.723   (  1.396,   3.443)
        90%    3.411   (  1.747,   5.185)

    평균 차이 0.431   (-0.805,  1.676)
    ```

    **평균 차이는 0에 가깝고 유의하지 않은데, 분위수를 보면 분포가 확연히 다르다.** 아래 꼬리는 낮아지고 위 꼬리는 높아졌다 — 정확히 분산이 커진 모습이다. **분위수 분석이 평균이 놓친 것을 잡아낸다.**

    **방법 2 — 이동-척도 그림(Q-Q plot).** 두 표본의 분위수를 서로 대응시켜 그린다. 직선이면 이동만 있고, 기울기가 1이 아니면 척도가 다르며, 휘어 있으면 모양이 다르다.

    **방법 3 — 콜모고로프-스미르노프 통계량.** 두 경험분포함수의 최대 차이에 대한 구간. 분포가 같은지에 대한 전체적 판단을 준다. 다만 **어디가 다른지**는 말해 주지 않는다.

    **방법 4 — 확률적 우세.** $P(Y>X)$를 추정한다. 만-휘트니 통계량을 $n_1n_2$로 나눈 것이 불편추정량이다.

    ```python
    from scipy import stats
    u = stats.mannwhitneyu(y, x, alternative="two-sided")
    print(f"P(Y > X) 추정 {u.statistic / (len(x) * len(y)):.4f}")
    ```

    ```text
    P(Y > X) 추정 0.5494
    ```

    **0.55로 0.5에 가깝다.** 확률적 우세로도 두 분포의 차이가 잡히지 않는다. **이 지표는 위치 차이에는 민감하지만 척도 차이에는 무디다.**

    **방법 5 — 분위수회귀.** 공변량이 있는 경우로 확장한다. 각 분위수에서 처리효과를 추정하고 구간을 준다.

    **권고.** **그림을 먼저 그린다.** 두 집단의 밀도나 상자그림을 나란히 놓으면 어떤 요약이 필요한지 보인다. 평균 차이 하나만 보고하는 관행은 **분포가 이동만 하는 경우**에나 충분하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
관측연구에서 두 집단의 평균 차이 구간을 **인과적으로 해석할 수 있는 조건**을 정리하라.

</div>

??? success "풀이"
    **구간 자체는 인과적 주장을 하지 않는다.** $\bar X_1-\bar X_2$의 구간은 **두 모집단 평균의 차이**를 담을 뿐이다. 그것을 "처리의 효과"로 읽으려면 추가 조건이 필요하다.

    **필요한 가정 — 넷.**

    1. **교환가능성(무교란).** 측정한 공변량 $L$을 고정하면 처리 배정이 잠재결과와 독립이다.

       $$
       Y^{(a)}\perp A\mid L
       $$

       **검증할 수 없다.** 무작위 배정에서는 설계로 보장되지만, 관측연구에서는 영역 지식에 기댄 가정이다.

    2. **양(positivity).** 모든 $L$ 값에서 두 처리가 모두 일어날 확률이 양수다. 특정 집단이 언제나 한 처리만 받으면 그 부분에서는 비교가 불가능하다.

    3. **일관성.** 관측된 처리에 대응하는 잠재결과가 관측값과 같다. "처리"가 잘 정의되어야 한다 — "운동"처럼 모호한 노출은 여러 버전이 섞여 있어 위험하다.

    4. **간섭 없음(SUTVA).** 한 개체의 처리가 다른 개체의 결과에 영향을 주지 않는다. 감염병, 교육, 네트워크 효과에서 자주 깨진다.

    **무엇을 해야 하는가.**

    | 단계 | 내용 |
    |---|---|
    | 설계 | 인과그래프를 그려 **조정해야 할 변수**를 정한다 |
    | 보정 | 회귀, 성향점수, 가중, 짝짓기 |
    | 진단 | 보정 후 공변량 균형 확인 |
    | 민감도 | 측정되지 않은 교란이 얼마나 강해야 결론이 뒤집히는가 |
    | 보고 | 가정을 **명시적으로** 나열하고, 어느 것이 취약한지 논의 |

    **흔한 실수 — 넷.**

    1. **중간변수를 보정.** 처리의 결과인 변수를 넣으면 효과의 일부가 지워진다. 앞서 본 과잉짝짓기와 같은 문제다.

    2. **충돌자를 보정.** 처리와 결과의 공통 결과를 넣으면 **없던 연관이 생긴다.** "모든 변수를 넣으면 안전하다"는 직관이 틀린 이유다.

    3. **"보정했으니 인과"라고 말하기.** 보정은 **관측된** 교란만 처리한다. 측정되지 않은 교란은 그대로다.

    4. **구간의 폭을 불확실성 전부로 착각.** 구간은 **표집 불확실성**만 담는다. 인과 가정의 불확실성은 그 안에 없다. 실제 불확실성은 훨씬 크다.

    **정직한 보고의 예.**

    > 보정 후 두 군의 평균 차이는 2.3단위였다(95% 신뢰구간 0.8~3.8). 이를 인과효과로 해석하려면 나이·성별·기저질환으로 조정한 뒤 잔여 교란이 없다는 가정이 필요하다. 로젠바움 민감도 분석에서, 측정되지 않은 요인이 노출 오즈를 1.4배 이상 바꾸면 결론이 유지되지 않는다. 따라서 이 결과는 **인과관계를 시사하되 확정하지 못한다.**

    **한 문장 요약.** **신뢰구간은 "얼마나 다른가"에 답하고, "왜 다른가"에는 답하지 않는다.** 후자는 설계와 가정의 몫이다.

---

## 정리하며

두 평균 차의 구간은 **표준오차를 어떻게 합치느냐**로 갈린다.

$$
(\bar X_1-\bar X_2)\pm(\text{임계값})\times\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}
$$

- **분산이 더해진다는 것이 핵심이다.** 두 표본이 독립이므로 차의 분산이 분산의 합이며(3장 정리 3), 표준오차는 그 제곱근이다. **표준오차끼리 더하는 것이 아니다.**
- **웰치가 기본값이다.** 등분산을 가정하지 않고 새터스웨이트 자유도를 쓴다. 분산이 실제로 같아도 손해가 거의 없고, 다르면 합동 방법보다 훨씬 낫다.
- **합동 $t$ 구간은 등분산을 가정한다.** 두 표본분산을 자유도로 가중평균해 하나의 $s_p^2$ 을 쓰며, 자유도가 $n_1+n_2-2$ 로 커지는 것이 이득이다. **가정이 맞을 때만 그렇다.**
- **"먼저 등분산 검정을 하고 고른다"는 나쁜 관행이다.** 2단계 절차가 전체 오류율을 왜곡하며, 그냥 웰치를 쓰는 편이 낫다.
- **구간이 $0$ 을 포함하는지가 검정과 이어진다.** 포함하면 수준 $\alpha$ 양측검정에서 기각하지 못한다는 뜻이며, 9장의 쌍대성이 그것이다.

다음 절 **$p_1-p_2$ 의 신뢰구간**으로 넘어간다.
