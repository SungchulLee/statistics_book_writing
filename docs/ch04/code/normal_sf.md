# 정규분포의 생존함수

## 개요

**생존함수**(SF)는 CDF의 여집합에 해당한다:

$$
S(x) = P(X > x) = 1 - F(x)
$$

$X$가 주어진 문턱값을 넘을 확률을 준다. 생존함수는 신뢰성 공학, 보험계리학, 임상시험처럼 사건 발생까지의 시간이나 초과 확률이 관심 대상인 분야에서 널리 쓰인다.

---

## 코드

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 0, 1
dist = stats.norm(loc=mu, scale=sigma)

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
cdf = dist.cdf(x)
sf = dist.sf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, cdf, lw=2, label='CDF  P(X ≤ x)')
ax.plot(x, sf, lw=2, label='SF   P(X > x)')
ax.axvline(0, ls=':', color='gray', alpha=0.6)
ax.axhline(0.5, ls=':', color='gray', alpha=0.6)
ax.annotate("CDF + SF = 1", xy=(1.2, 0.5), fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
ax.set_xlabel('x')
ax.set_ylabel('Probability')
ax.set_ylim(-0.03, 1.03)
ax.legend(loc='center left', frameon=False)
ax.set_title(f"Normal({mu}, {sigma}) — CDF vs Survival Function")
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## 생존함수를 쓰는 이유

상단꼬리 확률이 극단적으로 작을 때 $1 - F(x)$를 직접 계산하면 $F(x)$가 1에 매우 가까워 부동소수점 상쇄가 일어날 수 있다. 전용 메서드 `sf()`는 꼬리 확률을 직접 계산하여 이 문제를 피한다.

```python
# Poor: floating-point cancellation
p_bad = 1 - stats.norm.cdf(8)    # may give 0.0

# Good: numerically stable
p_good = stats.norm.sf(8)         # gives ~6.22e-16
```

---

## 연습문제

**연습문제 1.**
$Z \sim N(0,1)$에 대해 CDF와 SF를 각각 사용하여 $P(Z > 2)$를 계산하고 두 결과가 일치함을 확인하라.

??? success "연습문제 1 풀이"
    CDF로: $P(Z > 2) = 1 - \mathcal{N}(2) = 1 - 0.9772 = 0.0228$.

    SF로: `stats.norm.sf(2) = 0.0228`.

    둘 다 같은 결과를 준다. $P(Z > 8)$처럼 극단적인 값에서는 SF 방식이 수치적으로 더 낫다.

---

**연습문제 2.**
임의의 연속확률변수에 대해 모든 $x$에서 $S(x) + F(x) = 1$임을 보여라.

??? success "연습문제 2 풀이"
    정의에 의해:

    $$
    F(x) + S(x) = P(X \le x) + P(X > x) = P(X \in (-\infty, x]) + P(X \in (x, \infty))
    $$

    $(-\infty, x]$와 $(x, \infty)$는 $\mathbb{R}$의 분할이므로 두 확률의 합은 1이다. $\square$

---

**연습문제 3.**
어떤 부품은 응력 $X \sim N(500, 2500)$이 문턱값 600을 넘으면 고장 난다. 고장 확률은 얼마인가?

??? success "연습문제 3 풀이"
    표준화하면 $Z = (600 - 500)/50 = 2$이다.

    $$
    P(X > 600) = P(Z > 2) = S(2) \approx 0.0228
    $$

    부품의 약 2.3%가 고장 난다.

---

**연습문제 4.**
**위험함수**는 $h(x) = f(x)/S(x)$로 정의된다. 표준정규분포에 대해 $h(0)$을 계산하고 $x > 0$에서 $h(x)$가 증가하는 이유를 설명하라.

??? success "연습문제 4 풀이"
    $x = 0$에서 $f(0) = 1/\sqrt{2\pi} \approx 0.3989$이고 $S(0) = 0.5$이다.

    $$
    h(0) = \frac{0.3989}{0.5} \approx 0.7979
    $$

    $x > 0$에서는 $S(x)$가 $f(x)$보다 빠르게 감소한다. 분모는 ($x$ 위에 남은 값이 줄어들어) 작아지는 반면, 분자인 밀도도 감소하지만 상대적으로는 더 천천히 줄어들기 때문이다. 그래서 $h(x)$가 증가한다. $x$까지 생존했다는 조건 아래 $x$에서 "고장"이 날 조건부 확률이 $x$와 함께 커지는 것이다. 정규분포는 **증가하는 고장률**을 갖는다.
