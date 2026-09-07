# p₁ − p₂의 신뢰구간

## 이표본 비율 신뢰구간

실무의 여러 상황에서 우리는 두 모집단의 비율을 비교한다 — 예를 들어 서로 다른 두 정책을 지지하는 사람의 비율이나 두 생산라인의 불량률 같은 것이다.

### 공식 (Wald)

독립인 두 집단의 모비율을 $p_1$과 $p_2$라 하자. $p_1 - p_2$의 신뢰구간은

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}_1(1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2(1 - \hat{p}_2)}{n_2}}
$$

여기서 $\hat{p}_1 = x_1/n_1$과 $\hat{p}_2 = x_2/n_2$는 표본비율이다.

### 타당성 조건

정규근사가 성립하려면:

- $n_1\hat{p}_1 \ge 5$이고 $n_1(1 - \hat{p}_1) \ge 5$,
- $n_2\hat{p}_2 \ge 5$이고 $n_2(1 - \hat{p}_2) \ge 5$.

### 대안적인 방법

| 방법 | 설명 | 언제 쓰는가 |
|---|---|---|
| **Wald** | $\Delta \pm z \cdot \text{SE}$ | $n$이 크고 0이나 1에 가깝지 않을 때 |
| **Newcombe (Wilson 기반)** | 집단마다 Wilson 신뢰구간을 구한 뒤 결합: $[L_1 - U_2,\; U_1 - L_2]$ | **권장되는 기본값** |
| **Clopper–Pearson 결합** | 집단마다 정확한 신뢰구간을 구한 뒤 결합 | $n$이 작을 때, 규제 상황 |

### Python 코드

```python
import numpy as np
import scipy.stats as stats

n1, n2 = 200, 250
x1, x2 = 120, 130
confidence_level = 0.95

p1 = x1 / n1
p2 = x2 / n2

standard_error = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
margin_of_error = z_critical * standard_error

confidence_interval = ((p1 - p2) - margin_of_error, (p1 - p2) + margin_of_error)
print(f"{confidence_interval = }")
```

---

## 예제

### 예제 1: 비율 차이의 95% 신뢰구간

표본 1: $n_1 = 200$, 성공 $x_1 = 120$회. 표본 2: $n_2 = 250$, 성공 $x_2 = 130$회.

**풀이.**

$$
\hat{p}_1 = 0.60, \qquad \hat{p}_2 = 0.52
$$

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{200} + \frac{0.52 \times 0.48}{250}} = \sqrt{0.0012 + 0.001} = \sqrt{0.0022} \approx 0.0469
$$

$$
\text{ME} = 1.96 \times 0.0469 \approx 0.0919
$$

$$
\boxed{(-0.0119,\ 0.1719)}
$$

참 차이가 $-0.0119$와 $0.1719$ 사이에 있다고 95% 신뢰한다. 구간이 0을 포함하므로 95% 수준에서 통계적으로 유의한 차이는 없다.

### 예제 2: 새 고등학교 건립

Duncan은 도시의 북부와 남부에서 새 고등학교에 대한 지지를 비교한다.

| 지지하는가? | 북부 | 남부 |
|---|---|---|
| 예 | 54 | 77 |
| 아니오 | 66 | 63 |
| 합계 | 120 | 140 |

$p_N - p_S$의 90% 신뢰구간을 구성하라.

**풀이.**

```python
import numpy as np
from scipy import stats

n_1 = 120  # north
n_2 = 140  # south
p_1_hat = 54 / n_1
p_2_hat = 77 / n_2

confidence_level = 0.90
alpha = 1 - confidence_level
z_star = -stats.norm().ppf(alpha / 2)
margin_of_error = z_star * np.sqrt(
    p_1_hat * (1 - p_1_hat) / n_1 + p_2_hat * (1 - p_2_hat) / n_2
)
print(f"90% CI: {p_1_hat - p_2_hat:.4f} ± {margin_of_error:.4f}")
```

---

## 모의실험: 두 비율 차이 신뢰구간의 포함확률

```python
#!/usr/bin/env python3
"""
Difference of two proportions CI simulation: Newcombe, Wald, Clopper-Pearson.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

rng_seed = None
n_simulations = 100
n1, n2 = 50, 40
p1_true, p2_true = 0.60, 0.50
alpha = 0.05
method = "newcombe"  # 'newcombe' | 'wald' | 'cp'


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    delta_true = p1_true - p2_true
    z = norm.ppf(1 - alpha / 2.0)
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    for i in range(n_simulations):
        k1 = np.random.binomial(n1, p1_true)
        k2 = np.random.binomial(n2, p2_true)
        p1hat, p2hat = k1 / n1, k2 / n2
        centers[i] = p1hat - p2hat

        if method == "wald":
            se = np.sqrt(p1hat * (1 - p1hat) / n1 + p2hat * (1 - p2hat) / n2)
            lo, hi = centers[i] - z * se, centers[i] + z * se
        elif method == "newcombe":
            denom1 = 1 + z**2 / n1
            center1 = (p1hat + z**2 / (2 * n1)) / denom1
            half1 = z * np.sqrt(p1hat * (1 - p1hat) / n1 + z**2 / (4 * n1**2)) / denom1
            L1, U1 = center1 - half1, center1 + half1
            denom2 = 1 + z**2 / n2
            center2 = (p2hat + z**2 / (2 * n2)) / denom2
            half2 = z * np.sqrt(p2hat * (1 - p2hat) / n2 + z**2 / (4 * n2**2)) / denom2
            L2, U2 = center2 - half2, center2 + half2
            lo, hi = L1 - U2, U1 - L2
        elif method == "cp":
            L1 = 0.0 if k1 == 0 else beta.ppf(alpha / 2.0, k1, n1 - k1 + 1)
            U1 = 1.0 if k1 == n1 else beta.ppf(1 - alpha / 2.0, k1 + 1, n1 - k1)
            L2 = 0.0 if k2 == 0 else beta.ppf(alpha / 2.0, k2, n2 - k2 + 1)
            U2 = 1.0 if k2 == n2 else beta.ppf(1 - alpha / 2.0, k2 + 1, n2 - k2)
            lo, hi = L1 - U2, U1 - L2

        lowers[i] = max(-1.0, lo)
        uppers[i] = min(1.0, hi)

    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)
    ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Δ=p1−p2 CIs ({method.title()}) | n1={n1}, n2={n2}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Δ = p1 − p2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

---

## 핵심 정리

- $p_1 - p_2$의 신뢰구간은 표본이 크다는 가정 아래 이항분포에 대한 정규근사를 쓴다.
- 너비는 표본비율, 표본크기, 신뢰수준에 달려 있다.
- 신뢰구간이 0을 포함하면 주어진 신뢰수준에서 두 비율 사이에 통계적으로 유의한 차이가 없다.
- 포함확률이 더 좋으므로, 특히 표본크기가 중간 정도일 때 **Newcombe(Wilson 기반)** 방법을 기본으로 권한다.

---

## 연습문제

**연습문제 1.**
독립인 두 표본에서 표본 1은 200명 중 150명이, 표본 2는 180명 중 120명이 같은 브랜드를 선호한다. 비율 차이의 95% 신뢰구간을 구성하라.

??? success "풀이"
    $$
    \hat{p}_1 = \frac{150}{200} = 0.75, \qquad \hat{p}_2 = \frac{120}{180} \approx 0.667
    $$

    $$
    \text{SE} = \sqrt{\frac{0.75 \times 0.25}{200} + \frac{0.667 \times 0.333}{180}} \approx \sqrt{0.000938 + 0.001234} \approx 0.0466
    $$

    $$
    \text{ME} = 1.96 \times 0.0466 \approx 0.091
    $$

    $$
    (0.083 - 0.091,\ 0.083 + 0.091) = (-0.008,\ 0.175)
    $$

    구간이 0을 포함하므로 5% 수준에서 차이는 통계적으로 유의하지 않다.

---

**연습문제 2.**
연습문제 1의 자료에서 정규근사의 타당성 조건이 만족되는지 확인하라.

??? success "풀이"
    조건은 $n_i\hat{p}_i \geq 5$이고 $n_i(1-\hat{p}_i) \geq 5$이다:

    - 표본 1: $200 \times 0.75 = 150 \geq 5$이고 $200 \times 0.25 = 50 \geq 5$
    - 표본 2: $180 \times 0.667 = 120 \geq 5$이고 $180 \times 0.333 = 60 \geq 5$

    모든 조건이 만족된다.

---

**연습문제 3.**
비율 차이의 신뢰구간에서 Wald 방법보다 Newcombe(Wilson 기반) 방법을 권하는 이유를 설명하라.

??? success "풀이"
    Wald 방법은 표준오차 공식에 표본비율을 그대로 쓰는데, 참 비율이 0이나 1에 가깝거나 표본크기가 중간 정도이면 포함확률이 나빠질 수 있다. Wald 구간은 심지어 $[-1, 1]$ 밖으로 벗어나는 구간을 낼 수도 있다.

    Newcombe 방법은 각 비율에 대한 Wilson 신뢰구간을 결합하여 차이의 신뢰구간을 만든다. Wilson 구간은 보정항 $z^2/(2n)$을 더해 극단적인 비율을 0.5 쪽으로 "축소"하므로 더 안정적인 구간이 나온다. 모의실험 연구들은 Newcombe 방법이 더 넓은 범위의 모수값과 표본크기에서 명목 수준에 더 가까운 포함확률을 달성함을 일관되게 보여준다.

---

**연습문제 4.**
$p_1 - p_2$의 신뢰구간이 $(0.03, 0.15)$이라면 이 결과를 맥락에 맞게 해석하고 유의한 차이의 증거가 있는지 말하라.

??? success "풀이"
    95% 신뢰구간 $(0.03, 0.15)$은 참 차이 $p_1 - p_2$가 0.03과 0.15 사이에 있다고 95% 신뢰한다는 뜻이다. 구간 전체가 양수이므로(0을 포함하지 않으므로) 5% 수준에서 $p_1$이 $p_2$보다 유의하게 크다고 결론짓는다. 추정된 차이는 집단 1에 유리하게 3~15 퍼센트포인트이다.
