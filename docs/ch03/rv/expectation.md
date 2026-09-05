# 기댓값과 선형성

## 개요

확률변수의 **기댓값**은 실험을 여러 번 반복했을 때의 장기적 평균값이다. 분포의 "중심"을 하나의 수로 요약해 준다. **기댓값의 선형성**은 확률론 전체에서 가장 강력하고 널리 쓰이는 성질 중 하나다.

---

## 정의

### 이산확률변수

확률질량함수가 $p_{x_i}$인 이산확률변수 $X$에 대해

$$
E[X] = \sum_i x_i \cdot P(X = x_i) = \sum_i x_i \cdot p_{x_i}
$$

이다. 벽돌 비유로 말하면 $E[X]$는 실직선을 따라 놓인 벽돌들의 **무게중심**이다.

### 연속확률변수

확률밀도함수가 $f(x)$인 연속확률변수 $X$에 대해

$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

이다.

---

## 무의식적 통계학자의 법칙 (LOTUS)

$g(X)$의 분포를 먼저 구하지 않고 $g(X)$의 기댓값을 계산하려면

$$
E[g(X)] =
\begin{cases}
\displaystyle\sum_i g(x_i) \cdot P(X = x_i), & \text{discrete} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx, & \text{continuous}
\end{cases}
$$

를 쓴다. 이렇게 하면 $g(X)$의 분포를 유도하는 흔히 번거로운 단계를 피할 수 있다.

---

## 기댓값의 선형성

임의의 확률변수 $X$와 $Y$(독립일 필요가 없다)와 상수 $a, b, c$에 대해

$$
E[aX + bY + c] = aE[X] + bE[Y] + c
$$

이다. 이는 임의의 유한합으로 확장된다.

$$
E\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

**핵심 통찰:** 선형성은 **확률변수들이 독립이든 종속이든 관계없이** 성립한다. 그래서 유난히 강력한 도구가 된다.

---

## 기댓값의 성질

1. **상수:** $E[c] = c$
2. **척도:** $E[aX] = aE[X]$
3. **가법성:** $E[X + Y] = E[X] + E[Y]$
4. **단조성:** 항상 $X \leq Y$이면 $E[X] \leq E[Y]$
5. **곱(독립일 때만):** $X \perp\!\!\!\perp Y$이면 $E[XY] = E[X] \cdot E[Y]$

성질 5는 독립성을 요구하지만 성질 1–4는 그렇지 않다.

---

## 예제

### 예: 공정한 주사위의 기댓값

$$
E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = 3.5
$$

### 예: 동전 n번 던지기에서 앞면의 기대 개수

$i$번째 던지기가 앞면이면 $X_i = 1$, 아니면 0이라 하자. 그러면 $X = \sum_{i=1}^n X_i$가 전체 앞면 수를 센다. 선형성에 의해

$$
E[X] = \sum_{i=1}^n E[X_i] = \sum_{i=1}^n p = np
$$

이다. 공정한 동전을 $n = 100$번 던지면 $E[X] = 50$이다.

### 예: 쿠폰 수집가 문제

서로 다른 쿠폰이 $n$종류 있다. 구매할 때마다 균등하게 무작위한 쿠폰 하나를 받는다. $T$를 $n$종류를 모두 모으는 데 필요한 총 구매 횟수라 하자.

과정을 단계로 나눈다. $i$번째 단계는 서로 다른 쿠폰 $i-1$종류를 가진 상태에서 시작해 $i$번째 새 쿠폰을 얻을 때 끝난다. $i$번째 단계에서 한 번 구매가 새 쿠폰일 확률은 $\frac{n - i + 1}{n}$이므로, 그 단계의 구매 횟수는 평균이 $\frac{n}{n - i + 1}$인 기하분포를 따른다.

선형성에 의해

$$
E[T] = \sum_{i=1}^{n} \frac{n}{n - i + 1} = n \sum_{k=1}^{n} \frac{1}{k} = nH_n \approx n \ln n
$$

이다. $n = 50$종류면 $E[T] \approx 50 \times \ln(50) \approx 225$번 구매해야 한다.

### 예: 연속 — 지수분포

$x \geq 0$에서 확률밀도함수가 $f(x) = \lambda e^{-\lambda x}$인 $X \sim \text{Exponential}(\lambda)$에 대해

$$
E[X] = \int_0^{\infty} x \cdot \lambda e^{-\lambda x} \, dx = \frac{1}{\lambda}
$$

이다.

---

## 파이썬으로 살펴보기

```python
import numpy as np

# Expected value of a fair die
values = np.arange(1, 7)
probs = np.ones(6) / 6
expected = np.sum(values * probs)
print(f"E[fair die] = {expected:.4f}")

# Simulation
np.random.seed(42)
rolls = np.random.randint(1, 7, size=100_000)
print(f"Simulated mean = {rolls.mean():.4f}")
```

```python
import numpy as np

def coupon_collector_simulation(n_coupons, n_trials=10_000):
    """Simulate the coupon collector problem."""
    np.random.seed(42)
    totals = []
    for _ in range(n_trials):
        collected = set()
        count = 0
        while len(collected) < n_coupons:
            collected.add(np.random.randint(0, n_coupons))
            count += 1
        totals.append(count)

    simulated = np.mean(totals)
    H_n = sum(1/k for k in range(1, n_coupons + 1))
    theoretical = n_coupons * H_n

    print(f"n = {n_coupons}")
    print(f"Simulated E[T] = {simulated:.1f}")
    print(f"Theoretical E[T] = n·Hₙ = {theoretical:.1f}")

coupon_collector_simulation(50)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def linearity_demonstration():
    """Demonstrate linearity of expectation with dependent variables."""
    np.random.seed(42)
    n_sim = 100_000

    # X ~ Uniform(0,1), Y = X^2 (clearly dependent on X)
    X = np.random.rand(n_sim)
    Y = X ** 2

    print("X and Y = X² are dependent, but linearity still holds:")
    print(f"E[X] = {X.mean():.4f} (theoretical: 0.5)")
    print(f"E[Y] = {Y.mean():.4f} (theoretical: 0.3333)")
    print(f"E[X + Y] = {(X + Y).mean():.4f}")
    print(f"E[X] + E[Y] = {X.mean() + Y.mean():.4f}")

linearity_demonstration()
```

---

## 핵심 요약

- 기댓값 $E[X]$는 가능한 모든 값의 확률가중 평균이다.
- **LOTUS** 덕분에 $X$의 분포에서 곧바로 $E[g(X)]$를 계산할 수 있다.
- **기댓값의 선형성**은 종속인 변수에 대해서도 언제나 성립하며, 확률론에서 가장 유용한 도구 중 하나다.
- 곱 규칙 $E[XY] = E[X]E[Y]$는 독립성을 요구하지만 선형성은 그렇지 않다.

## 연습문제

**연습문제 1.**
이산확률변수 $X$의 분포가 $P(X=-1) = 0.3$, $P(X=0) = 0.4$, $P(X=2) = 0.3$이다. $E[X]$와 $E[X^2]$를 계산하라.

??? success "연습문제 1 풀이"
    $$
    E[X] = (-1)(0.3) + (0)(0.4) + (2)(0.3) = -0.3 + 0 + 0.6 = 0.3
    $$

    $g(X) = X^2$에 대해 LOTUS를 쓰면

    $$
    E[X^2] = (-1)^2(0.3) + (0)^2(0.4) + (2)^2(0.3) = 0.3 + 0 + 1.2 = 1.5
    $$

    이다.

---

**연습문제 2.**
$X_1, X_2, \ldots, X_{100}$을 독립인 동전 던지기 100번의 지시변수라 하자. $i$번째 던지기가 앞면(확률 0.5)이면 $X_i = 1$, 아니면 $X_i = 0$이다. 기댓값의 선형성을 이용해 $E\!\left[\sum_{i=1}^{100} X_i\right]$를 구하라.

??? success "연습문제 2 풀이"
    기댓값의 선형성에 의해

    $$
    E\!\left[\sum_{i=1}^{100} X_i\right] = \sum_{i=1}^{100} E[X_i]
    $$

    이다. 각 $X_i$는 $E[X_i] = P(X_i = 1) = 0.5$인 베르누이 확률변수이므로

    $$
    E\!\left[\sum_{i=1}^{100} X_i\right] = 100 \times 0.5 = 50
    $$

    이다. 100번 던지면 앞면이 50번 나올 것으로 기대한다. 중요한 점은 선형성이 던지기의 독립 여부와 무관하게 성립한다는 것이다. 던지기가 종속이더라도 같은 답이 나온다.

---

**연습문제 3.**
공정한 육면체 주사위를 굴린다. $X$를 나온 수라 하고 $Y = (X - 3.5)^2$이라 하자. LOTUS를 써서 $E[Y]$를 계산하라.

??? success "연습문제 3 풀이"
    LOTUS에 의해 $E[Y] = E[(X-3.5)^2] = \sum_{x=1}^{6} (x - 3.5)^2 \cdot P(X=x)$이다. 각 값에 대해 $P(X=x) = 1/6$이므로

    $$
    E[Y] = \frac{1}{6}\left[(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2\right]
    $$

    $$
    = \frac{1}{6}\left[6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25\right] = \frac{17.5}{6} \approx 2.917
    $$

    이다. 참고로 $E[X] = 3.5$이므로 이는 정확히 공정한 주사위의 $\text{Var}(X)$다.

---

**연습문제 4.**
$X$와 $Y$가 독립이고 $E[X] = 2$, $E[Y] = 3$, $E[X^2] = 5$, $E[Y^2] = 11$이다. $E[XY]$와 $E[(X+Y)^2]$를 계산하라.

??? success "연습문제 4 풀이"
    $X$와 $Y$가 독립이므로

    $$
    E[XY] = E[X] \cdot E[Y] = 2 \times 3 = 6
    $$

    이다. $E[(X+Y)^2]$의 경우 제곱을 전개하면

    $$
    E[(X+Y)^2] = E[X^2 + 2XY + Y^2] = E[X^2] + 2E[XY] + E[Y^2]
    $$

    $$
    = 5 + 2(6) + 11 = 5 + 12 + 11 = 28
    $$

    이다.

---

**연습문제 5.**
**기댓값의 꼬리합 공식.** 음이 아닌 확률변수 $X$에 대해 $\mathbb{E}[X] = \int_0^\infty P(X > t) dt$(연속) 또는 $\sum_{n=0}^\infty P(X > n)$(정숫값)임을 증명하라.

??? success "연습문제 5 풀이"
    **연속인 경우:** 푸비니 정리에 의해

    $$
    \int_0^\infty P(X > t) dt = \int_0^\infty \int_t^\infty f(x) dx \, dt = \int_0^\infty f(x) \int_0^x dt \, dx = \int_0^\infty x f(x) dx = \mathbb{E}[X]
    $$

    이며, 순서 교환은 비음성에 의해 정당화된다.

    **정숫값인 경우:**

    $$
    \sum_{n=0}^\infty P(X > n) = \sum_{n=0}^\infty \sum_{k=n+1}^\infty P(X = k) = \sum_{k=1}^\infty P(X = k) \sum_{n=0}^{k-1} 1 = \sum_{k=1}^\infty k P(X = k) = \mathbb{E}[X]
    $$

    **용도:** 꼬리합 공식을 쓰면 생존확률로부터 기댓값을 계산할 수 있다(표준적인 확률밀도함수 적분보다 쉬울 때가 있다). 예: 첫 성공까지의 시행 횟수를 세는 기하확률변수에서 $P(X > n) = (1 - p)^n$이므로 $\mathbb{E}[X] = \sum_{n=0}^\infty (1 - p)^n = 1/p$이다.

---

**연습문제 6.**
**확률변수로서의 조건부 기댓값.** $X, Y$가 결합분포를 갖는다고 하자. $g(y) = \mathbb{E}[X \mid Y = y]$와 확률변수 $\mathbb{E}[X \mid Y] = g(Y)$를 정의한다. **전기댓값의 법칙** $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$를 증명하라.

??? success "연습문제 6 풀이"
    정의에 의해 $g(y) = \mathbb{E}[X \mid Y = y] = \int x f_{X \mid Y}(x \mid y) dx$이다.

    $\mathbb{E}[g(Y)] = \int g(y) f_Y(y) dy = \int \int x f_{X \mid Y}(x \mid y) f_Y(y) dx \, dy = \int \int x f_{X, Y}(x, y) dx \, dy = \int x f_X(x) dx = \mathbb{E}[X]$.

    $\square$

    확률, 통계, 그리고 기댓값에 대한 동적계획법 접근 전반에서 쓰인다.

    - **최적 예측자로서의 조건부 기댓값:** $\mathbb{E}[X \mid Y]$는 모든 함수 $g$에 대해 $\mathbb{E}[(X - g(Y))^2]$을 최소화한다.
    - **탑 성질**(반복 기댓값): $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[\mathbb{E}[\mathbb{E}[X \mid Y, Z] \mid Y]]$ 등.
    - **MCMC / 분산 감소**: (가능할 때) $X$를 $\mathbb{E}[X \mid Y]$로 대체하면 라오–블랙웰 정리를 통해 추정량의 분산이 줄어든다.
    - **강화학습**: 벨만 방정식 $V(s) = \mathbb{E}[R + \gamma V(s') \mid s]$이 반복된 조건부 기댓값이다.
