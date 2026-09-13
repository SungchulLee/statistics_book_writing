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
| **Newcombe (Wilson 기반)** | 집단마다 Wilson 신뢰구간 $[L_i, U_i]$를 구한 뒤 제곱합으로 결합 | **권장되는 기본값** |
| **Clopper–Pearson 결합** | 집단마다 정확한 신뢰구간을 구한 뒤 양끝을 빼서 결합 | $n$이 작을 때, 규제 상황 |

#### Newcombe 구간의 결합 방식

집단별 Wilson 구간을 $[L_1, U_1]$과 $[L_2, U_2]$라 할 때 Newcombe 구간은

$$
\left(
(\hat p_1 - \hat p_2) - \sqrt{(\hat p_1 - L_1)^2 + (U_2 - \hat p_2)^2},\;\;
(\hat p_1 - \hat p_2) + \sqrt{(U_1 - \hat p_1)^2 + (\hat p_2 - L_2)^2}
\right)
$$

이다. 여기서 **제곱합**을 쓰는 것이 핵심이다. $[L_1 - U_2,\; U_1 - L_2]$처럼 양끝을 그대로 빼면 두 집단이 동시에 최악으로 어긋나는 상황을 가정하는 셈이 되어 구간이 지나치게 넓어진다. 두 표본이 독립이므로 오차는 함께 커지는 것이 아니라 피타고라스식으로 합쳐진다.

<div class="codebox" markdown>

##### 예제 1. 두 비율 차이의 신뢰구간 계산 { .eg }

```python
import numpy as np
import scipy.stats as stats

n1, n2 = 200, 250
x1, x2 = 120, 130
confidence_level = 0.95

p1 = x1 / n1
p2 = x2 / n2

# 두 표본이 독립이므로 분산이 더해진다.
# 여기서는 두 비율을 **따로** 추정해 넣는다. 검정에서 쓰는 합동비율은
# "두 비율이 같다"는 귀무가설 아래의 계산이라 신뢰구간에는 맞지 않는다.
standard_error = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
margin_of_error = z_critical * standard_error

confidence_interval = ((p1 - p2) - margin_of_error, (p1 - p2) + margin_of_error)
print(f"{confidence_interval = }")
```

출력:

```
confidence_interval = (-0.011897024279429055, 0.171897024279429)
```

</div>

---

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 비율 차이의 95% 신뢰구간. 표본 1: $n_1 = 200$, 성공 $x_1 = 120$회. 표본 2: $n_2 = 250$, 성공 $x_2 = 130$회.

</div>

??? success "풀이"

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
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 새 고등학교 건립. Duncan은 도시의 북부와 남부에서 새 고등학교에 대한 지지를 비교한다.

| 지지하는가? | 북부 | 남부 |
|---|---|---|
| 예 | 54 | 77 |
| 아니오 | 66 | 63 |
| 합계 | 120 | 140 |

$p_N - p_S$의 90% 신뢰구간을 구성하라.


</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    n_1 = 120  # north
    n_2 = 140  # south
    p_1_hat = 54 / n_1
    p_2_hat = 77 / n_2

    confidence_level = 0.90
    alpha = 1 - confidence_level
    # 앞에서는 ppf(1 - alpha/2)를 썼고 여기서는 -ppf(alpha/2)를 썼다.
    # 표준정규가 0을 중심으로 대칭이라 두 값이 같다.
    z_star = -stats.norm().ppf(alpha / 2)
    margin_of_error = z_star * np.sqrt(
        p_1_hat * (1 - p_1_hat) / n_1 + p_2_hat * (1 - p_2_hat) / n_2
    )
    print(f"90% CI: {p_1_hat - p_2_hat:.4f} ± {margin_of_error:.4f}")
    ```

    출력:

    ```
    90% CI: -0.1000 ± 0.1018
    ```

    $\hat p_N = 0.450$, $\hat p_S = 0.550$으로 차이가 정확히 $-0.10$이다. 90% 구간은 $(-0.202, 0.002)$로 0을 아슬아슬하게 담는다. 표본 260개로는 10%p 차이도 잡아내기 어렵다는 뜻이다. 비율의 차이는 평균의 차이보다 훨씬 큰 표본을 요구한다.
---

## 모의실험: 두 비율 차이 신뢰구간의 포함확률

<div class="codebox" markdown>

### 예제 2. 세 방법의 포함확률 비교 { .eg }

```python
#!/usr/bin/env python3
"""두 비율 차이의 신뢰구간을 세 방법으로 만들어 포함확률을 비교한다.

Wald 는 식이 가장 간단하지만 비율이 0 이나 1 에 가까우면 포함확률이
명목수준에 한참 못 미친다. Newcombe 는 그 약점을 고친 방법이다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

rng_seed = 42        # 아래 그림을 재현하려면 고정한다
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
            # 집단마다 Wilson 구간을 구한다.
            denom1 = 1 + z**2 / n1
            center1 = (p1hat + z**2 / (2 * n1)) / denom1
            half1 = z * np.sqrt(p1hat * (1 - p1hat) / n1 + z**2 / (4 * n1**2)) / denom1
            L1, U1 = center1 - half1, center1 + half1
            denom2 = 1 + z**2 / n2
            center2 = (p2hat + z**2 / (2 * n2)) / denom2
            half2 = z * np.sqrt(p2hat * (1 - p2hat) / n2 + z**2 / (4 * n2**2)) / denom2
            L2, U2 = center2 - half2, center2 + half2
            # 두 구간을 **제곱합**으로 합친다. L1 - U2 처럼 양끝을 그냥 빼면
            # 두 집단이 동시에 최악으로 어긋나는 경우를 가정하는 셈이라
            # 구간이 지나치게 넓어진다(아래 설명 참조).
            lo = (p1hat - p2hat) - np.sqrt((p1hat - L1)**2 + (U2 - p2hat)**2)
            hi = (p1hat - p2hat) + np.sqrt((U1 - p1hat)**2 + (p2hat - L2)**2)
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

![100 Δ=p1−p2 CIs (Newcombe) | n1=50, n2=40, CL=95%](./img/ci_p_diff_149.png)

100회 중 실패 1회다. 100회짜리 모의실험으로는 방법을 가릴 수 없으니 반복을 20,000회로 늘려 실제 포함확률을 재면 Newcombe 94.8%, Wald 94.4%가 나온다. 둘 다 명목값을 조금 밑돌지만 Newcombe가 낫다.

여기서 양끝을 그냥 빼는 결합, 즉 $[L_1 - U_2,\; U_1 - L_2]$을 쓰면 어떻게 되는지 비교해 볼 만하다. 같은 조건에서 포함확률이 **99.5%**로 뛰고 구간의 평균 너비가 0.391에서 0.552로 41% 늘어난다. 명목보다 높은 포함확률은 공짜가 아니다. 그만큼 결론이 무뎌진다.

또 하나 눈에 띄는 것은 100개 중 81개가 0을 담고 있다는 점이다. 참 차이가 0.10인데 $n_1 = 50$, $n_2 = 40$으로는 다섯 번에 네 번 "차이가 없을 수도 있다"고 말하게 된다.

</div>

---

## 핵심 정리

- $p_1 - p_2$의 신뢰구간은 표본이 크다는 가정 아래 이항분포에 대한 정규근사를 쓴다.
- 너비는 표본비율, 표본크기, 신뢰수준에 달려 있다.
- 신뢰구간이 0을 포함하면 주어진 신뢰수준에서 두 비율 사이에 통계적으로 유의한 차이가 없다.
- 포함확률이 더 좋으므로, 특히 표본크기가 중간 정도일 때 **Newcombe(Wilson 기반)** 방법을 기본으로 권한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
독립인 두 표본에서 표본 1은 200명 중 150명이, 표본 2는 180명 중 120명이 같은 브랜드를 선호한다. 비율 차이의 95% 신뢰구간을 구성하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
연습문제 1의 자료에서 정규근사의 타당성 조건이 만족되는지 확인하라.

</div>

??? success "풀이"
    조건은 $n_i\hat{p}_i \geq 5$이고 $n_i(1-\hat{p}_i) \geq 5$이다:

    - 표본 1: $200 \times 0.75 = 150 \geq 5$이고 $200 \times 0.25 = 50 \geq 5$
    - 표본 2: $180 \times 0.667 = 120 \geq 5$이고 $180 \times 0.333 = 60 \geq 5$

    모든 조건이 만족된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
비율 차이의 신뢰구간에서 Wald 방법보다 Newcombe(Wilson 기반) 방법을 권하는 이유를 설명하라.

</div>

??? success "풀이"
    Wald 방법은 표준오차 공식에 표본비율을 그대로 쓰는데, 참 비율이 0이나 1에 가깝거나 표본크기가 중간 정도이면 포함확률이 나빠질 수 있다. Wald 구간은 심지어 $[-1, 1]$ 밖으로 벗어나는 구간을 낼 수도 있다.

    Newcombe 방법은 각 비율에 대한 Wilson 신뢰구간을 결합하여 차이의 신뢰구간을 만든다. Wilson 구간은 보정항 $z^2/(2n)$을 더해 극단적인 비율을 0.5 쪽으로 "축소"하므로 더 안정적인 구간이 나온다. 모의실험 연구들은 Newcombe 방법이 더 넓은 범위의 모수값과 표본크기에서 명목 수준에 더 가까운 포함확률을 달성함을 일관되게 보여준다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
$p_1 - p_2$의 신뢰구간이 $(0.03, 0.15)$이라면 이 결과를 맥락에 맞게 해석하고 유의한 차이의 증거가 있는지 말하라.

</div>

??? success "풀이"
    95% 신뢰구간 $(0.03, 0.15)$은 참 차이 $p_1 - p_2$가 0.03과 0.15 사이에 있다고 95% 신뢰한다는 뜻이다. 구간 전체가 양수이므로(0을 포함하지 않으므로) 5% 수준에서 $p_1$이 $p_2$보다 유의하게 크다고 결론짓는다. 추정된 차이는 집단 1에 유리하게 3~15 퍼센트포인트이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
뉴콤의 방법을 구현하여 여러 자료에서 왈드 구간과 비교하라. 특히 **$k=0$이나 $k=n$** 인 경우를 확인하라.

</div>

??? success "풀이"
    **뉴콤의 구성.** 각 비율에 윌슨 구간 $(l_j,u_j)$를 만들고

    $$
    L=\hat p_1-\hat p_2-\sqrt{(\hat p_1-l_1)^2+(u_2-\hat p_2)^2}
    $$

    $$
    U=\hat p_1-\hat p_2+\sqrt{(u_1-\hat p_1)^2+(\hat p_2-l_2)^2}
    $$

    **각 끝에서 "가장 불리한" 조합**을 취하는 구조다.

    ```python
    import numpy as np
    from scipy import stats

    z = stats.norm.ppf(0.975)

    def wilson(k, n):
        ph = k / n
        d = 1 + z**2 / n
        c = (ph + z**2 / (2 * n)) / d
        h = z / d * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2))
        return c - h, c + h

    def newcombe(k1, n1, k2, n2):
        p1, p2 = k1 / n1, k2 / n2
        l1, u1 = wilson(k1, n1)
        l2, u2 = wilson(k2, n2)
        D = p1 - p2
        return (D - np.sqrt((p1 - l1)**2 + (u2 - p2)**2),
                D + np.sqrt((u1 - p1)**2 + (p2 - l2)**2))

    cases = [(150, 200, 120, 180), (5, 50, 0, 50), (0, 20, 0, 20),
             (20, 20, 15, 20), (3, 100, 1, 100)]
    for k1, n1, k2, n2 in cases:
        p1, p2 = k1 / n1, k2 / n2
        D = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        lo, hi = newcombe(k1, n1, k2, n2)
        print(f"{k1:3d}/{n1:3d} vs {k2:3d}/{n2:3d}: "
              f"왈드 ({D - z * se:7.4f}, {D + z * se:7.4f})  "
              f"뉴콤 ({lo:7.4f}, {hi:7.4f})")
    ```

    ```text
    150/200 vs 120/180: 왈드 (-0.0080,  0.1747)  뉴콤 (-0.0079,  0.1737)
      5/ 50 vs   0/ 50: 왈드 ( 0.0168,  0.1832)  뉴콤 ( 0.0090,  0.2136)
      0/ 20 vs   0/ 20: 왈드 ( 0.0000,  0.0000)  뉴콤 (-0.1611,  0.1611)
     20/ 20 vs  15/ 20: 왈드 ( 0.0602,  0.4398)  뉴콤 ( 0.0378,  0.4687)
      3/100 vs   1/100: 왈드 (-0.0187,  0.0587)  뉴콤 (-0.0287,  0.0751)
    ```

    **네 가지를 읽는다.**

    1. **표본이 크고 비율이 중간이면 둘이 거의 같다.** 첫 줄에서 소수 셋째 자리까지 일치한다.

    2. **$k_1=k_2=0$이면 왈드가 완전히 무너진다.** 두 추정값이 모두 0이라 표준오차도 0이고, **구간이 점 $\{0\}$** 이 된다. "두 집단의 비율이 정확히 같다"고 단언하는 셈이다. $n=20$씩밖에 없는데 터무니없다. 뉴콤은 $(-0.161,\ 0.161)$로 합리적이다.

    3. **한쪽이 경계면 왈드가 좁다.** $5/50$ 대 $0/50$에서 왈드 하한 0.017이 뉴콤 0.009보다 높다. 왈드가 두 번째 집단의 불확실성을 0으로 보기 때문이다. **과신이다.**

    4. **양쪽이 경계여도 뉴콤은 작동한다.** $20/20$ 대 $15/20$에서 뉴콤이 양방향으로 더 넓다.

    **구현 주의.** $k=0$이나 $k=n$에서 윌슨 구간은 여전히 잘 정의된다. 이것이 뉴콤이 작동하는 이유다. 왈드를 바탕으로 같은 구조를 만들면 같은 문제가 남는다.

    **더 나아간 판.** 뉴콤의 원 논문은 연속성 수정을 넣은 판도 제시하며, 대응 자료에는 앞서 본 대로 상관항 $\phi$를 더한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
비율 차이 구간 세 가지(왈드, 뉴콤, 애그레스티-카페)의 **포함확률**을 여러 $(p_1,p_2,n)$에서 비교하라.

</div>

??? success "풀이"
    **애그레스티-카페.** 각 집단에 성공과 실패를 **1개씩** 더하고 왈드 공식을 쓴다($\tilde p_j=(k_j+1)/(n_j+2)$). 단일 비율의 애그레스티-콜이 2개씩 더하는 것과 다르다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(31)
    M = 20_000
    z = stats.norm.ppf(0.975)
    print(f"{'p1':>5s} {'p2':>5s} {'n':>5s} {'왈드':>8s} {'뉴콤':>8s} "
          f"{'AC':>8s}")
    for p1, p2, n in [(0.50, 0.40, 50), (0.10, 0.05, 50), (0.05, 0.02, 100),
                      (0.02, 0.01, 50), (0.90, 0.80, 30)]:
        k1, k2 = rng.binomial(n, p1, M), rng.binomial(n, p2, M)
        ph1, ph2 = k1 / n, k2 / n
        D, true = ph1 - ph2, p1 - p2

        se = np.sqrt(ph1 * (1 - ph1) / n + ph2 * (1 - ph2) / n)
        cw = np.mean((D - z * se <= true) & (true <= D + z * se))

        d = 1 + z**2 / n
        def wil(k):
            ph = k / n
            c = (ph + z**2 / (2 * n)) / d
            h = z / d * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2))
            return c - h, c + h
        l1, u1 = wil(k1)
        l2, u2 = wil(k2)
        lo = D - np.sqrt((ph1 - l1)**2 + (u2 - ph2)**2)
        hi = D + np.sqrt((u1 - ph1)**2 + (ph2 - l2)**2)
        cn = np.mean((lo <= true) & (true <= hi))

        nt = n + 2
        pt1, pt2 = (k1 + 1) / nt, (k2 + 1) / nt
        sea = np.sqrt(pt1 * (1 - pt1) / nt + pt2 * (1 - pt2) / nt)
        ca = np.mean((pt1 - pt2 - z * sea <= true) & (true <= pt1 - pt2 + z * sea))

        print(f"{p1:5.2f} {p2:5.2f} {n:5d} {cw:8.4f} {cn:8.4f} {ca:8.4f}")
    ```

    ```text
       p1    p2     n       왈드       뉴콤       AC
     0.50  0.40    50   0.9469   0.9476   0.9476
     0.10  0.05    50   0.9478   0.9702   0.9692
     0.05  0.02   100   0.9396   0.9694   0.9737
     0.02  0.01    50   0.7785   0.9991   0.9998
     0.90  0.80    30   0.9425   0.9648   0.9629
    ```

    **결정적인 행은 $(0.02,\ 0.01,\ 50)$이다.** 왈드의 포함확률이 **0.779**로 붕괴한다.

    **왜 그런가.** $np_1=1$, $np_2=0.5$이므로 $k_1=k_2=0$일 확률이 $0.98^{50}\times0.99^{50}=0.364\times0.605=0.220$이다. 그때 왈드 구간이 점 $\{0\}$이 되고, 참값 0.01을 담지 못한다. **다섯 번에 한 번은 구간이 퇴화한다.**

    **뉴콤과 AC는 보수적이다.** 같은 행에서 0.999로, 명목을 크게 넘는다. **이산성이 극심한 영역에서는 보수성을 피할 수 없다.**

    **중간 영역에서는 셋이 비슷하다.** $(0.50,0.40)$에서 0.947~0.948로 사실상 동일하다.

    **권고.**

    | 상황 | 권장 |
    |---|---|
    | $\min(k_j,\ n_j-k_j)\ge10$ | 아무거나(왈드도 무방) |
    | 경계 근처, 희귀사건 | **뉴콤** 또는 AC |
    | 정확한 보장이 필요 | 정확 조건부 방법(계산이 무겁다) |
    | 의심스러우면 | **뉴콤** |

    **AC 대 뉴콤.** 두 방법이 비슷하다. AC는 계산이 훨씬 단순하고, 뉴콤은 경계를 절대 벗어나지 않는다는 구조적 보장이 있다. **AC 구간은 $[-1,1]$을 벗어날 수 있다**는 점만 기억하면 된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
비율 차이 대신 **위험비**와 **오즈비**의 구간을 구하고, 세 지표를 언제 쓰는지 정리하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    a, n1 = 30, 200      # 처리군: 사건 30건
    b, n2 = 50, 200      # 대조군: 사건 50건
    z = stats.norm.ppf(0.975)
    p1, p2 = a / n1, b / n2

    # 위험차
    se_d = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    print(f"위험차 {p1 - p2:7.4f}  ({p1 - p2 - z * se_d:7.4f}, "
          f"{p1 - p2 + z * se_d:7.4f})")

    # 위험비 (로그 척도)
    rr = p1 / p2
    se_lr = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n2)
    print(f"위험비 {rr:7.4f}  ({rr * np.exp(-z * se_lr):7.4f}, "
          f"{rr * np.exp(z * se_lr):7.4f})")

    # 오즈비 (로그 척도, 울프의 방법)
    c, d = n1 - a, n2 - b
    orr = (a * d) / (b * c)
    se_lo = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    print(f"오즈비 {orr:7.4f}  ({orr * np.exp(-z * se_lo):7.4f}, "
          f"{orr * np.exp(z * se_lo):7.4f})")

    lo_d, hi_d = p1 - p2 - z * se_d, p1 - p2 + z * se_d
    print(f"\n치료필요수 NNT = 1/|위험차| = {1 / abs(p1 - p2):.1f}")
    print(f"  95% 구간 {1 / abs(lo_d):.1f} ~ {1 / abs(hi_d):.1f}")
    ```

    ```text
    위험차 -0.1000  (-0.1778, -0.0222)
    위험비  0.6000  ( 0.3990,  0.9023)
    오즈비  0.5294  ( 0.3201,  0.8755)

    치료필요수 NNT = 1/|위험차| = 10.0
      95% 구간 5.6 ~ 45.0
    ```

    **세 구간이 모두 "효과 없음"을 배제한다**(0, 1, 1을 담지 않음).

    **세 지표의 성격.**

    | 지표 | 값 | 해석 | 척도 |
    |---|---|---|---|
    | 위험차 | $-0.10$ | 100명당 10명 감소 | 절대 |
    | 위험비 | 0.60 | 위험이 40% 감소 | 상대 |
    | 오즈비 | 0.53 | 오즈가 47% 감소 | 상대 |

    **오즈비가 위험비보다 극단적이다.** 사건이 흔할수록($p>0.1$) 차이가 커진다. **오즈비를 "위험이 절반"으로 읽으면 과장**이다.

    **언제 무엇을 쓰는가.**

    - **위험차와 NNT**: 임상적·정책적 판단. "몇 명을 치료해야 한 명이 이득을 보는가"가 직관적이다. **기저 위험에 의존**하므로 다른 집단에 옮길 때 주의한다.
    - **위험비**: 기전 연구, 서로 다른 기저 위험을 가진 집단 간 비교. 코호트 연구에서 자연스럽다.
    - **오즈비**: **환자-대조군 연구에서는 이것만 추정 가능**하다(위험을 알 수 없으므로). 로지스틱회귀의 자연스러운 모수이기도 하다.

    **NNT 구간의 함정.** 위 예에서 $(5.6,\ 45.0)$인데, 위험차 구간이 0을 담으면 NNT 구간이 **$(-\infty,\ a)\cup(b,\ \infty)$** 로 끊어진다. 역수 변환이 0에서 발산하기 때문이다. **NNT는 유의한 경우에만 보고**하는 것이 안전하다.

    **권고.** **위험차와 상대지표를 함께** 보고한다. 상대지표만 적으면 실무적 중요성이 가려지고, 절대지표만 적으면 다른 집단으로의 일반화가 어렵다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
여러 **층**에서 비율 차이를 추정했다면 어떻게 결합하는가? 맨텔-헨첼 방법을 적용하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    # 층별 (a, n1, b, n2) : 처리군 사건/총, 대조군 사건/총
    strata = [(20, 100, 30, 100),      # 저위험군
              (40, 120, 55, 110),      # 중위험군
              (35,  60, 42,  55)]      # 고위험군
    z = stats.norm.ppf(0.975)

    print(f"{'층':>3s} {'p1':>7s} {'p2':>7s} {'위험차':>8s} {'위험비':>7s}")
    for i, (a, n1, b, n2) in enumerate(strata, 1):
        print(f"{i:3d} {a / n1:7.4f} {b / n2:7.4f} "
              f"{a / n1 - b / n2:8.4f} {(a / n1) / (b / n2):7.4f}")

    # 역분산 가중 (위험차)
    ws, ds = [], []
    for a, n1, b, n2 in strata:
        p1, p2 = a / n1, b / n2
        v = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        ws.append(1 / v)
        ds.append(p1 - p2)
    ws, ds = np.array(ws), np.array(ds)
    d_iv = (ws * ds).sum() / ws.sum()
    se_iv = np.sqrt(1 / ws.sum())
    print(f"\n역분산 가중 위험차 {d_iv:.4f}  "
          f"({d_iv - z * se_iv:.4f}, {d_iv + z * se_iv:.4f})")

    # 이질성 검정 (코크런 Q)
    Q = (ws * (ds - d_iv)**2).sum()
    print(f"코크런 Q = {Q:.3f}  df = {len(strata) - 1}  "
          f"p = {stats.chi2.sf(Q, len(strata) - 1):.4f}")
    print(f"I² = {max(0, (Q - (len(strata) - 1)) / Q) * 100:.1f}%")

    # 뭉뚱그린 분석 (층 무시)
    A = sum(a for a, _, _, _ in strata)
    N1 = sum(n1 for _, n1, _, _ in strata)
    B = sum(b for _, _, b, _ in strata)
    N2 = sum(n2 for _, _, _, n2 in strata)
    p1, p2 = A / N1, B / N2
    se = np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
    print(f"\n층 무시 위험차 {p1 - p2:.4f}  "
          f"({p1 - p2 - z * se:.4f}, {p1 - p2 + z * se:.4f})")
    ```

    ```text
      층      p1      p2     위험차    위험비
      1  0.2000  0.3000  -0.1000  0.6667
      2  0.3333  0.5000  -0.1667  0.6667
      3  0.5833  0.7636  -0.1803  0.7639

    역분산 가중 위험차 -0.1418  (-0.2187, -0.0648)
    코크런 Q = 0.824  df = 2  p = 0.6622
    I² = 0.0%

    층 무시 위험차 -0.1400  (-0.2218, -0.0581)
    ```

    **이질성이 없다**($Q=0.82$, $p=0.66$, $I^2=0\%$). 세 층의 위험차가 $-0.10\sim-0.18$로 비슷하다.

    **이 자료에서는 층을 무시해도 결과가 거의 같다**($-0.140$ 대 $-0.142$). 층마다 두 군의 표본이 대체로 균형($100{:}100$, $120{:}110$, $60{:}55$)이기 때문이다.

    **배분이 불균형하면 달라진다.** 같은 층별 비율을 유지하되 표본 배분만 바꿔 보자.

    ```python
    strata2 = [(20, 100, 6, 20), (40, 120, 55, 110), (9, 20, 42, 55)]
    ws2, ds2 = [], []
    for a, n1, b, n2 in strata2:
        p1, p2 = a / n1, b / n2
        ws2.append(1 / (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
        ds2.append(p1 - p2)
    ws2, ds2 = np.array(ws2), np.array(ds2)
    d2 = (ws2 * ds2).sum() / ws2.sum()
    se2 = np.sqrt(1 / ws2.sum())
    A = sum(x[0] for x in strata2); N1 = sum(x[1] for x in strata2)
    B = sum(x[2] for x in strata2); N2 = sum(x[3] for x in strata2)
    q1, q2 = A / N1, B / N2
    sep = np.sqrt(q1 * (1 - q1) / N1 + q2 * (1 - q2) / N2)
    print(f"역분산 {d2:.4f}  ({d2 - z * se2:.4f}, {d2 + z * se2:.4f})")
    print(f"층 무시 {q1 - q2:.4f}  ({q1 - q2 - z * sep:.4f}, {q1 - q2 + z * sep:.4f})")
    ```

    ```text
    역분산 -0.1766  (-0.2760, -0.0773)
    층 무시 -0.2693  (-0.3609, -0.1776)
    ```

    **층 무시가 $-0.27$로 층화 추정의 1.5배를 말한다.** 위험차가 가장 큰 3층에서 대조군만 많이 뽑혔기 때문에, 뭉뚱그리면 그 층이 과대 반영된다. **심프슨의 역설과 같은 구조**다.

    **결합 전에 반드시 확인할 것.**

    1. **이질성.** $Q$ 검정과 $I^2$. 이질성이 크면 **단일한 요약값이 의미가 없다.** 층별로 보고하거나 임의효과 모형을 쓴다.

    2. **어느 척도에서 동질적인가.** 위 예에서 **위험비**는 세 층에서 0.67, 0.67, 0.76으로 더 일정하다. 위험차는 $-0.10\sim-0.18$로 층의 기저위험을 따라 변한다. **위험비 척도에서 결합하는 것이 더 자연스러운 자료**다. 이는 흔한 패턴이다.

    3. **층이 왜 층인가.** 교란요인이라면 조정해야 하고, 효과수정자라면 층별로 보고해야 한다.

    **맨텔-헨첼 대 역분산.** 사건이 드물어 $1/v$가 불안정하면 맨텔-헨첼 가중이 더 안정적이다. 사건이 충분하면 둘이 거의 같다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
비율 차이의 구간이 **$[-1,1]$을 벗어나는** 경우를 만들고, 어떤 방법이 이를 막는지 정리하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    z = stats.norm.ppf(0.975)

    def wilson(k, n):
        ph = k / n
        d = 1 + z**2 / n
        c = (ph + z**2 / (2 * n)) / d
        h = z / d * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2))
        return c - h, c + h

    for k1, n1, k2, n2 in [(9, 10, 1, 10), (10, 12, 0, 8)]:
        p1, p2 = k1 / n1, k2 / n2
        D = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        nt1, nt2 = n1 + 2, n2 + 2
        pt1, pt2 = (k1 + 1) / nt1, (k2 + 1) / nt2
        sea = np.sqrt(pt1 * (1 - pt1) / nt1 + pt2 * (1 - pt2) / nt2)
        l1, u1 = wilson(k1, n1)
        l2, u2 = wilson(k2, n2)
        lo = D - np.sqrt((p1 - l1)**2 + (u2 - p2)**2)
        hi = D + np.sqrt((u1 - p1)**2 + (p2 - l2)**2)
        print(f"{k1}/{n1} vs {k2}/{n2}")
        print(f"  왈드 ({D - z * se:7.4f}, {D + z * se:7.4f})")
        print(f"  AC   ({pt1 - pt2 - z * sea:7.4f}, {pt1 - pt2 + z * sea:7.4f})")
        print(f"  뉴콤 ({lo:7.4f}, {hi:7.4f})")
    ```

    ```text
    9/10 vs 1/10
      왈드 ( 0.5370,  1.0630)
      AC   ( 0.3685,  0.9649)
      뉴콤 ( 0.3699,  0.9161)
    10/12 vs 0/8
      왈드 ( 0.6225,  1.0442)
      AC   ( 0.4015,  0.9699)
      뉴콤 ( 0.4039,  0.9530)
    ```

    **왈드의 상한이 1.06, 1.04로 1을 넘는다.** 두 비율의 차이가 1을 넘을 수 없으므로 명백히 불가능한 값이다.

    **AC는 이 자료에서 경계 안에 머문다**(0.965, 0.970). 가상의 성공·실패를 더해 추정값을 중앙으로 끌어당긴 덕이다. 다만 **구조적 보장은 없다** — 여전히 왈드 형태이므로 더 극단적인 자료에서는 넘을 수 있다.

    **뉴콤만이 구조적으로 안전하다.** 각 윌슨 구간이 $[0,1]$ 안에 있으므로

    $$
    L\ge l_1-u_2\ge0-1=-1,\qquad U\le u_1-l_2\le1-0=1
    $$

    이 자동으로 성립한다. 위 예에서 상한이 0.92, 0.95다.

    **왜 이것이 중요한가.**

    1. **신뢰를 잃는다.** 불가능한 값을 담은 구간은 방법이 부적절함을 스스로 드러낸다.
    2. **잘라내면 더 나빠진다.** $[-1,1]$로 자르는 임시방편은 포함확률을 더 떨어뜨린다. 구간이 좁아지기 때문이다.
    3. **다른 곳에서도 문제가 있다는 신호.** 경계를 넘을 정도면 그 근처의 포함확률도 나쁘다.

    **같은 문제가 있는 다른 상황.**

    | 모수 | 경계 | 왈드가 넘는 예 |
    |---|---|---|
    | 비율 | $[0,1]$ | $\hat p$가 0이나 1 근처 |
    | 분산 | $[0,\infty)$ | 하한이 음수 |
    | 상관계수 | $[-1,1]$ | $\hat\rho$가 $\pm1$ 근처(피셔 변환으로 해결) |
    | 생존확률 | $[0,1]$ | 꼬리에서(로그-로그 변환으로 해결) |

    **일반적 해법.** **모수공간 전체를 실수로 펴는 변환**을 찾아 그 척도에서 구간을 만들고 되돌린다. 로짓, 로그, 피셔의 $z$가 모두 그 예다. 뉴콤은 조금 다른 길(윌슨 구간의 결합)을 택했지만 목적은 같다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
비율 차이를 보고할 때 **어떤 형태로 제시해야 하는지** 정리하고, 흔한 오해를 지적하라.

</div>

??? success "풀이"
    **반드시 함께 적을 것.**

    1. **두 집단의 원 자료.** $k_1/n_1$과 $k_2/n_2$. 요약만 적으면 독자가 재계산할 수 없다.
    2. **두 비율 각각.** 그리고 각각의 구간.
    3. **차이와 그 구간.** 이것이 주 결과다.
    4. **사용한 방법.** 왈드인지 뉴콤인지.
    5. **절대 지표와 상대 지표를 모두.**

    **좋은 보고의 예.**

    > 처리군에서 30/200(15.0%, 95% CI 10.7~20.6), 대조군에서 50/200(25.0%, 95% CI 19.5~31.4)에 사건이 발생했다. 위험차는 $-10.0$%포인트(95% CI $-18.1$~$-1.9$, 뉴콤 방법), 위험비는 0.60(95% CI 0.40~0.90)이었다. 치료필요수는 10명(95% CI 6~52)이다.

    **흔한 오해 — 여섯.**

    1. **"20%에서 15%로 줄었으니 5% 감소."** **5%포인트** 감소이고, **상대적으로는 25%** 감소다. **%와 %포인트를 구별**하지 않으면 5배의 차이가 생긴다.

    2. **상대 감소만 보고.** "위험이 절반으로 줄었다"가 0.002에서 0.001로 준 것일 수 있다. **절대 차이를 반드시 함께** 적는다. 이것이 의약품 광고에서 가장 흔한 왜곡이다.

    3. **두 구간의 겹침으로 판단.** 앞서 본 대로 틀렸다. **차이의 구간**을 계산한다.

    4. **오즈비를 위험비로 읽기.** 사건이 흔하면 오즈비가 위험비를 크게 과장한다. $p_1=0.30$, $p_2=0.50$이면 RR$=0.60$, OR$=0.43$이다.

    5. **NNT를 구간 없이 보고.** "10명을 치료하면 1명이 이득"은 점추정값일 뿐이고, 위 예에서 구간은 6~52명이다.

    6. **기저위험을 빼고 상대지표를 옮기기.** 위험비 0.6은 다른 기저위험을 가진 집단에 적용할 때 **다른 절대 이득**을 준다. 기저위험 0.5인 집단에서는 20%포인트 감소, 0.02인 집단에서는 0.8%포인트다.

    **그림으로.** 두 비율을 점과 구간으로 나란히 그리고, **차이의 구간을 별도 축에** 표시한다. 숲그림 형태가 층별 결과를 함께 보이기에 좋다.

    **보고하지 말아야 할 것.** $p$-값만. 비율 차이는 실무적 크기가 결정적으로 중요한 양이며, $p$-값은 그것을 전혀 말해 주지 않는다.

---

## 정리하며

두 비율 차의 왈드 구간도 **분산을 더해** 만든다.

$$
(\hat p_1-\hat p_2)\pm z_{\alpha/2}\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}
$$

- **일표본 왈드 구간의 약점을 그대로 물려받는다.** 어느 한쪽 비율이 $0$ 이나 $1$ 에 가까우면 포함확률이 명목값에 못 미친다.
- **타당성 조건을 확인해야 한다.** 각 집단에서 성공과 실패가 모두 충분히 있어야 정규근사가 성립하며, 흔히 $n\hat p\ge5$, $n(1-\hat p)\ge5$ 를 본다.
- **개선책도 일표본과 같은 발상이다.** 각 집단의 성공·실패에 조금씩 더하는 **아그레스티–카포**(각 칸에 1 씩) 보정이 간단하면서 포함확률을 크게 개선한다.
- **검정과 구간의 표준오차가 다르다는 점에 주의하라.** 검정에서는 $H_0: p_1=p_2$ 아래 합동비율 $\hat p$ 로 표준오차를 만들고, 구간에서는 각자의 $\hat p_i$ 를 쓴다. **그래서 $p$ 값과 구간이 경계에서 어긋날 수 있다.**
- **차이 대신 비나 오즈비가 더 적절할 때가 있다.** 비율이 아주 작으면 절대차가 작아 보여도 상대적으로는 큰 차이일 수 있다.

다음 절 **$\sigma_1^2/\sigma_2^2$ 의 신뢰구간**으로 넘어간다. 차이가 아니라 **비**를 다루는 이유가 거기 있다.
