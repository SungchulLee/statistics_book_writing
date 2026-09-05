# Weibull 밀도함수와 위험함수

## 개요

**Weibull 분포**는 수명, 고장 시간, 생존 자료를 다루는 유연한 모형이다. 형상 모수 $k$가 위험률이 증가하는지, 감소하는지, 일정한지를 결정한다:

$$
f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad x \ge 0
$$

| $k$ | 위험률 거동 | 특수한 경우 |
|---|---|---|
| $k < 1$ | 감소하는 위험률 (초기 고장) | — |
| $k = 1$ | 일정한 위험률 | Exponential($\lambda$) |
| $k > 1$ | 증가하는 위험률 (노화/마모) | — |
| $k \approx 3.6$ | 근사적으로 정규분포 모양 | — |

---

## 생존함수와 위험함수

$$
S(x) = \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
$$

위험함수 $h(x) = f(x)/S(x)$는 시각 $x$까지 생존했다는 조건 아래 그 시점의 순간 고장률을 준다.

---

## 코드

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.linspace(0.01, 3.0, 500)
lam = 1.0

params = [
    (0.5, "k=0.5 (decreasing hazard)"),
    (1.0, "k=1.0 (exponential)"),
    (1.5, "k=1.5 (increasing hazard)"),
    (3.0, "k=3.0 (near-normal shape)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for k, desc in params:
    pdf = (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)
    sf = np.exp(-(x / lam) ** k)
    haz = (k / lam) * (x / lam) ** (k - 1)

    axes[0].plot(x, pdf, lw=2, label=f'k = {k}')
    axes[1].plot(x, sf, lw=2, label=f'k = {k}')
    axes[2].plot(x, haz, lw=2, label=f'k = {k}')

axes[0].set_title('Weibull PDF')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[1].set_title('Survival Function S(x)')
axes[1].set_xlabel('x')
axes[2].set_title('Hazard Function h(x)')
axes[2].set_xlabel('x')
axes[2].set_ylim(0, 5)

for ax in axes:
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

---

## 연습문제

**연습문제 1.**
$k = 1$인 Weibull 분포가 비율 $1/\lambda$인 Exponential 분포로 환원됨을 보여라.

??? success "연습문제 1 풀이"
    $k = 1$로 두면:

    $$
    f(x) = \frac{1}{\lambda}\exp\!\left(-\frac{x}{\lambda}\right)
    $$

    이는 평균이 $\lambda$인 $\text{Exponential}(\text{rate} = 1/\lambda)$의 PDF이다. 위험률은 $h(x) = 1/\lambda$로 상수가 되며, 이는 무기억성과 일관된다. $\square$

---

**연습문제 2.**
Weibull 분포의 CDF와 중앙값을 유도하라.

??? success "연습문제 2 풀이"
    **CDF:**

    $$
    F(x) = 1 - S(x) = 1 - \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right)
    $$

    **중앙값:** $F(m) = 0.5$를 풀면:

    $$
    \exp\!\left(-\left(\frac{m}{\lambda}\right)^k\right) = 0.5 \implies \left(\frac{m}{\lambda}\right)^k = \ln 2 \implies m = \lambda(\ln 2)^{1/k}
    $$

---

**연습문제 3.**
어떤 부품의 수명이 $k = 2$, $\lambda = 1000$시간인 Weibull 분포를 따른다. 500시간을 넘겨 생존할 확률은? 1500시간은?

??? success "연습문제 3 풀이"
    $$
    S(500) = \exp\!\left(-\left(\frac{500}{1000}\right)^2\right) = e^{-0.25} \approx 0.779
    $$

    $$
    S(1500) = \exp\!\left(-\left(\frac{1500}{1000}\right)^2\right) = e^{-2.25} \approx 0.105
    $$

    이 부품이 500시간을 견딜 확률은 77.9%이지만 1500시간을 견딜 확률은 10.5%에 불과하다. 위험률이 증가하므로($k = 2 > 1$) 부품이 나이를 먹을수록 고장이 더 잘 난다.

---

**연습문제 4.**
더 유연한 분포들이 있는데도 Weibull 분포가 신뢰성 공학에서 널리 쓰이는 이유를 설명하라.

??? success "연습문제 4 풀이"
    Weibull 분포가 널리 쓰이는 데에는 몇 가지 실용적인 장점이 있다:

    1. **해석 가능한 위험률:** 모수 $k$ 하나가 고장률이 증가할지, 감소할지, 일정할지를 직접 결정한다.
    2. **닫힌 형태의 함수:** PDF, CDF, 생존함수, 위험함수가 모두 간단한 닫힌 형태로 주어져 수치적분이 필요 없다.
    3. **선형화 가능:** $\ln(-\ln(S(x)))$를 $\ln(x)$에 대해 그리면 직선이 되어 그래프로 모수를 추정할 수 있다(Weibull 확률지).
    4. **Exponential 분포를 포함:** $k=1$로 두면 가장 단순한 수명 모형이 복원되므로 Weibull은 그 자연스러운 일반화가 된다.
