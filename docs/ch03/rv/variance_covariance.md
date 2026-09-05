# 분산과 공분산

## 개요

기댓값이 분포의 중심을 요약한다면 **분산**은 분포가 평균 주위로 얼마나 퍼져 있는지를 잰다. **공분산**과 **상관**은 두 확률변수가 함께 움직이는 정도를 포착한다. 이 개념들은 위험 측정, 포트폴리오 이론, 통계적 추론에 필수적이다.

---

## 분산

### 정의

확률변수 $X$의 **분산**은 평균으로부터의 기대 제곱편차다.

$$
\text{Var}(X) = E\left[(X - \mu)^2\right] = E[X^2] - (E[X])^2
$$

여기서 $\mu = E[X]$이다. 두 번째 형태 $E[X^2] - (E[X])^2$이 계산에는 흔히 더 편하다.

### 표준편차

**표준편차**는 분산의 제곱근으로, 퍼짐을 원래 단위로 되돌린다.

$$
\sigma_X = \text{SD}(X) = \sqrt{\text{Var}(X)}
$$

---

## 분산의 성질

1. **비음성:** $\text{Var}(X) \geq 0$이며, $X$가 상수일 때에 한해 등호가 성립한다.
2. **상수:** $\text{Var}(c) = 0$
3. **척도:** $\text{Var}(aX) = a^2 \text{Var}(X)$
4. **평행이동 불변:** $\text{Var}(X + c) = \text{Var}(X)$
5. **합(독립):** $X \perp\!\!\!\perp Y$이면 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

일반적인(종속일 수도 있는) 확률변수에 대해서는

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)
$$

이다.

---

## 공분산

### 정의

$X$와 $Y$의 **공분산**은 둘이 함께 변하는 정도를 잰다.

$$
\text{Cov}(X, Y) = E\left[(X - \mu_X)(Y - \mu_Y)\right] = E[XY] - E[X] \cdot E[Y]
$$

- $\text{Cov}(X, Y) > 0$: $X$와 $Y$가 같은 방향으로 움직이는 경향.
- $\text{Cov}(X, Y) < 0$: $X$와 $Y$가 반대 방향으로 움직이는 경향.
- $\text{Cov}(X, Y) = 0$: 선형관계가 없음(그래도 종속일 수는 있다).

### 공분산의 성질

1. **자기공분산:** $\text{Cov}(X, X) = \text{Var}(X)$
2. **대칭성:** $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
3. **쌍선형성:** $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
4. **독립이면 0:** $X \perp\!\!\!\perp Y$이면 $\text{Cov}(X, Y) = 0$(그러나 역은 거짓)

---

## 상관

**피어슨 상관계수**는 공분산을 $[-1, 1]$ 안에 놓이도록 정규화한다.

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

- $\rho = 1$: 완벽한 양의 선형관계
- $\rho = -1$: 완벽한 음의 선형관계
- $\rho = 0$: 선형관계 없음(무상관)

**중요:** 무상관($\rho = 0$)이 독립을 함의하지는 않는다. 예를 들어 $X \sim N(0,1)$이고 $Y = X^2$이면 $\text{Cov}(X, Y) = E[X^3] = 0$이지만 $X$와 $Y$는 분명히 종속이다.

---

## 합의 분산 (일반적인 경우)

임의의 확률변수 $X_1, \ldots, X_n$에 대해

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)
$$

이다. 모든 $X_i$가 쌍별로 무상관이면 교차항이 사라져

$$
\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i)
$$

이 된다.

---

## 예제

### 예: 공정한 주사위의 분산

$$
E[X] = 3.5, \quad E[X^2] = \frac{1^2 + 2^2 + \cdots + 6^2}{6} = \frac{91}{6}
$$

$$
\text{Var}(X) = \frac{91}{6} - 3.5^2 = \frac{91}{6} - \frac{49}{4} = \frac{35}{12} \approx 2.917
$$

### 예: 베르누이 확률변수

$X \sim \text{Bernoulli}(p)$에 대해

$$
E[X] = p, \quad E[X^2] = p, \quad \text{Var}(X) = p - p^2 = p(1-p)
$$

이다. 분산은 $p = 0.5$에서 최대가 되고(불확실성이 최대), $p = 0$ 또는 $p = 1$에서 0이다(확실함).

### 예: 포트폴리오 분산

수익률이 $R_1$과 $R_2$이고 가중치가 $w_1$, $w_2$($w_1 + w_2 = 1$)인 두 자산이 있다. 포트폴리오 수익률은 $R_p = w_1 R_1 + w_2 R_2$이므로

$$
\text{Var}(R_p) = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2w_1 w_2 \text{Cov}(R_1, R_2)
$$

이다. $\rho < 1$이면 분산투자가 포트폴리오 분산을 개별 분산들의 가중평균 아래로 낮춘다.

---

## 파이썬으로 살펴보기

```python
import numpy as np

# Variance of a fair die
values = np.arange(1, 7)
probs = np.ones(6) / 6
E_X = np.sum(values * probs)
E_X2 = np.sum(values**2 * probs)
var_X = E_X2 - E_X**2
print(f"E[X] = {E_X:.4f}")
print(f"E[X²] = {E_X2:.4f}")
print(f"Var(X) = {var_X:.4f}")
print(f"SD(X) = {np.sqrt(var_X):.4f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_correlation():
    """Show uncorrelated does not imply independent."""
    np.random.seed(42)
    n = 10_000

    X = np.random.randn(n)
    Y = X ** 2  # deterministically dependent on X

    cov_XY = np.cov(X, Y)[0, 1]
    corr_XY = np.corrcoef(X, Y)[0, 1]

    print(f"Cov(X, X²) = {cov_XY:.4f} (theoretically 0)")
    print(f"Corr(X, X²) = {corr_XY:.4f}")
    print(f"Yet X and X² are clearly dependent!")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(X[:500], Y[:500], alpha=0.3, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y = X²')
    ax.set_title(f'Uncorrelated but Dependent (ρ = {corr_XY:.3f})')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

demonstrate_correlation()
```

```python
import numpy as np
import matplotlib.pyplot as plt

def portfolio_variance_demo():
    """Demonstrate diversification benefit."""
    sigma1, sigma2 = 0.20, 0.30
    correlations = [-0.5, 0.0, 0.5, 1.0]

    fig, ax = plt.subplots(figsize=(12, 4))
    weights = np.linspace(0, 1, 100)

    for rho in correlations:
        cov_12 = rho * sigma1 * sigma2
        port_var = (weights**2 * sigma1**2
                    + (1 - weights)**2 * sigma2**2
                    + 2 * weights * (1 - weights) * cov_12)
        port_sd = np.sqrt(port_var)
        ax.plot(weights, port_sd, label=f'ρ = {rho}')

    ax.set_xlabel('Weight in Asset 1')
    ax.set_ylabel('Portfolio Std Dev')
    ax.set_title('Diversification: Portfolio Risk vs. Allocation')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

portfolio_variance_demo()
```

---

## 핵심 요약

- **분산**은 흩어짐을 잰다: $\text{Var}(X) = E[X^2] - (E[X])^2$.
- **공분산**은 선형적인 동반 움직임을 재고, **상관**은 이를 $[-1, 1]$로 정규화한다.
- 무상관($\rho = 0$)은 독립을 함의하지 **않는다**.
- 독립인 변수에서는 합의 분산이 분산의 합과 같고, 종속인 변수에서는 공분산 항을 포함해야 한다.
- 금융에서는 자산 수익률의 공분산 구조가 포트폴리오 분산투자의 이득을 결정한다.

## 연습문제

**연습문제 1.**
확률질량함수가 $P(X = 1, 2, 3, 4) = 0.1, 0.3, 0.4, 0.2$이다. (a) $\mathbb{E}[X]$를 구하라. (b) $\mathbb{E}[X^2]$와 $\mathrm{Var}(X)$를 구하라. (c) $Y = 3X + 5$일 때 $\mathbb{E}[Y]$와 $\mathrm{Var}(Y)$를 구하라.

??? success "연습문제 1 풀이"
    (a) $\mathbb{E}[X] = 1(0.1) + 2(0.3) + 3(0.4) + 4(0.2) = 2.7$.

    (b) $\mathbb{E}[X^2] = 1(0.1) + 4(0.3) + 9(0.4) + 16(0.2) = 8.1$. $\mathrm{Var}(X) = 8.1 - 7.29 = 0.81$.

    (c) $\mathbb{E}[Y] = 3 \cdot 2.7 + 5 = 13.1$. $\mathrm{Var}(Y) = 9 \cdot 0.81 = 7.29$. 상수를 더하는 것은 분산에 영향을 주지 않고, 곱하는 것은 분산을 그 제곱만큼 키운다.

---

**연습문제 2.**
**합의 분산 공식을 증명하라:** $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y)$.

??? success "연습문제 2 풀이"
    $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$라 하자. 그러면 $\mathbb{E}[X + Y] = \mu_X + \mu_Y$이고

    $$
    \mathrm{Var}(X + Y) = \mathbb{E}[(X + Y - \mu_X - \mu_Y)^2] = \mathbb{E}[((X - \mu_X) + (Y - \mu_Y))^2]
    $$

    이다. 제곱을 전개하면

    $$
    = \mathbb{E}[(X - \mu_X)^2] + 2\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] + \mathbb{E}[(Y - \mu_Y)^2]
    $$

    $$
    = \mathrm{Var}(X) + 2\mathrm{Cov}(X, Y) + \mathrm{Var}(Y)
    $$

    이다. $\square$

    **일반화:** $\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i) + 2\sum_{i < j} \mathrm{Cov}(X_i, X_j)$. 이 이중합 구조가 금융에서 상관이 포트폴리오 분산에 영향을 주는 이유다. 자산이 $N$개면 분산 항 $N$개에 더해 상관 항이 $N(N-1)/2$개 기여한다.

---

**연습문제 3.**
**무상관 $\ne$ 독립.** $X \sim \mathrm{Uniform}(-1, 1)$이고 $Y = X^2$이라 하자. $\mathrm{Cov}(X, Y) = 0$이지만 $X$와 $Y$가 종속임을 보여라.

??? success "연습문제 3 풀이"
    균등분포가 0에 대해 대칭이므로 $\mathbb{E}[X] = 0$이다. 또한 $\mathbb{E}[X^3] = 0$이다(대칭 정의역 위의 홀함수).

    $\mathrm{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X \cdot X^2] - 0 = \mathbb{E}[X^3] = 0$.

    따라서 $X$와 $Y$는 *무상관*이다. 그러나 $X = 0.5$임을 알면 $Y = 0.25$임이 정확히 결정된다. 둘은 *결정론적으로 종속*이다. 상관계수는 선형 연관만 재므로 이차 구조를 놓친다.

    **교훈:** 상관이 0인 것은 독립의 *필요*조건이지 *충분*조건이 아니다. 다변량 정규분포(와 몇몇 특수한 분포)에서는 무상관이 독립을 함의하지만, 이는 예외이지 일반 규칙이 아니다. 언제나 자료를 그려 보고 상관에만 의존하지 마라.

---

**연습문제 4.**
**두 자산의 포트폴리오 분산.** 두 자산이 $\sigma_1 = 0.20$, $\sigma_2 = 0.30$, $\rho = 0.30$이다. 포트폴리오 분산을 최소화하는 가중치 $w_1, w_2$($w_1 + w_2 = 1$, 둘 다 음이 아님)를 구하라.

??? success "연습문제 4 풀이"
    포트폴리오 분산은

    $$
    \sigma_p^2(w_1) = w_1^2 \sigma_1^2 + (1 - w_1)^2 \sigma_2^2 + 2 w_1(1 - w_1)\rho \sigma_1 \sigma_2
    $$

    이다. $w_1$에 대해 미분해 0으로 두면

    $$
    \frac{d\sigma_p^2}{dw_1} = 2 w_1 \sigma_1^2 - 2(1 - w_1)\sigma_2^2 + 2(1 - 2w_1)\rho \sigma_1 \sigma_2 = 0
    $$

    이고, 풀면 $w_1^* = (\sigma_2^2 - \rho\sigma_1\sigma_2)/(\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2)$이다.

    $\sigma_1 = 0.20$, $\sigma_2 = 0.30$, $\rho = 0.30$을 넣으면

    분자: $0.09 - 0.30 \cdot 0.20 \cdot 0.30 = 0.09 - 0.018 = 0.072$.

    분모: $0.04 + 0.09 - 2 \cdot 0.30 \cdot 0.20 \cdot 0.30 = 0.13 - 0.036 = 0.094$.

    $w_1^* = 0.072/0.094 \approx 0.766$, $w_2^* \approx 0.234$이다.

    예상대로 최소분산 포트폴리오는 변동성이 낮은 자산에 더 큰 가중치를 둔다. 최소 분산은 $\sigma_p^2 = 0.766^2 \cdot 0.04 + 0.234^2 \cdot 0.09 + 2 \cdot 0.766 \cdot 0.234 \cdot 0.018 \approx 0.0354$로, 어느 개별 자산의 분산보다도 작다.

---

**연습문제 5.**
**공분산행렬.** $X \sim N(0, 1)$이고 $Z \sim N(0, \sigma_Z^2)$이 $X$와 독립일 때 $Y = aX + Z$인 $(X, Y)$의 $2 \times 2$ 공분산행렬을 계산하라. $\rho(X, Y)$는 얼마인가?

??? success "연습문제 5 풀이"
    주변분산:

    $\mathrm{Var}(X) = 1$, $\mathrm{Var}(Y) = a^2 \cdot 1 + \sigma_Z^2 = a^2 + \sigma_Z^2$.

    공분산:

    $\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX + Z) = a \mathrm{Var}(X) + \mathrm{Cov}(X, Z) = a + 0 = a$.

    공분산행렬:

    $$
    \boldsymbol{\Sigma} = \begin{pmatrix} 1 & a \\ a & a^2 + \sigma_Z^2 \end{pmatrix}
    $$

    상관:

    $$
    \rho(X, Y) = \frac{a}{\sqrt{1 \cdot (a^2 + \sigma_Z^2)}} = \frac{a}{\sqrt{a^2 + \sigma_Z^2}}
    $$

    특수한 경우:

    - $\sigma_Z = 0$: $\rho = a/|a| = \pm 1$. $Y$가 $X$의 결정론적 함수다.
    - $\sigma_Z \to \infty$: $\rho \to 0$. 잡음이 지배하여 $X$와 $Y$가 사실상 독립이 된다.

    이 분해 $Y = aX + Z$가 선형회귀의 바탕이다. 계수 $a$가 회귀 기울기이고, 상관 $\rho$는 $Y$의 변동 중 얼마를 $X$가 설명하는지를 잰다.

---

**연습문제 6.**
**표본에서의 분산 추정.** i.i.d. 표본 $X_1, \ldots, X_n$에 대해 **표본공분산** $\hat{\mathrm{Cov}}(X, Y) = \frac{1}{n-1}\sum_i (X_i - \bar X)(Y_i - \bar Y)$이 $\mathrm{Cov}(X, Y)$의 불편추정량임을 보여라.

??? success "연습문제 6 풀이"
    전개하면 $\sum_i (X_i - \bar X)(Y_i - \bar Y) = \sum_i X_i Y_i - n \bar X \bar Y$이다.

    기댓값을 취하면

    $\mathbb{E}\sum X_i Y_i = n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y)$이다(각 항이 $\mathbb{E}[X_i Y_i] = \mathrm{Cov}(X, Y) + \mu_X \mu_Y$를 기여한다).

    $\mathbb{E}[n \bar X \bar Y]$의 경우, 관측값 사이의 독립성을 이용하면

    $$
    \mathbb{E}[\bar X \bar Y] = \frac{1}{n^2}\sum_{i, j} \mathbb{E}[X_i Y_j] = \frac{1}{n^2}\left[n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y) + n(n - 1)\mu_X \mu_Y\right]
    $$

    $= (\mathrm{Cov}(X, Y) + \mu_X \mu_Y)/n + (n - 1)\mu_X \mu_Y / n = \mathrm{Cov}(X, Y)/n + \mu_X \mu_Y$이다.

    따라서 $\mathbb{E}[n \bar X \bar Y] = \mathrm{Cov}(X, Y) + n\mu_X \mu_Y$이다.

    빼면 $\mathbb{E}[\sum_i (X_i - \bar X)(Y_i - \bar Y)] = n(\mathrm{Cov}(X, Y) + \mu_X \mu_Y) - \mathrm{Cov}(X, Y) - n\mu_X \mu_Y = (n - 1)\mathrm{Cov}(X, Y)$이다.

    $n - 1$로 나누면 $\mathbb{E}[\hat{\mathrm{Cov}}] = \mathrm{Cov}(X, Y)$이다. $\square$

    분모 $n - 1$은 공분산에 대한 **베셀 보정**이며 표본분산의 보정과 동일하다. 자료에서 $\mu_X$와 $\mu_Y$를 추정하면서 자유도 하나를 "써버리기" 때문이다.
