# 기댓값과 선형성

분포 전체를 적는 대신 몇 개의 수로 요약할 수 있다면 편할 것이다. 주사위 눈의 분포를 여섯 줄로 적는 대신 "평균 3.5"라고 말하는 식이다.

**기댓값**은 그 첫 번째 요약값이다. 실험을 여러 번 반복했을 때의 장기적 평균이며, 벽돌 비유로는 실직선 위에 놓인 벽돌들의 **무게중심**이다.

기댓값이 특별한 것은 계산이 쉬워서가 아니다. **선형성** 때문이다. 아무리 복잡하게 얽힌 확률변수들이라도 합의 기댓값은 기댓값의 합이며, 여기에는 독립성이 전혀 필요하지 않다. 이 성질 하나로 손대기 어려워 보이던 문제들이 풀린다.

이 절은 세 개의 정리로 이루어진다. 기댓값의 정의(정리 1), 함수의 기댓값을 분포 유도 없이 구하는 방법(정리 2), 그리고 선형성(정리 3)이다.

## 1. 무게중심을 구한다

정의는 "값 곱하기 무게를 모두 더한다"이다. 이산이면 더하고 연속이면 적분한다 — 앞 절의 대응이 그대로 적용된다.

### 정리 1. 기댓값의 정의 — 값의 확률가중 평균

이산확률변수 $X$에 대해

$$
E[X] = \sum_i x_i \, P(X = x_i) = \sum_i x_i \, p_{x_i}
$$

연속확률변수 $X$에 대해

$$
E[X] = \int_{-\infty}^{\infty} x \, f(x)\,dx
$$

이다.

물리의 무게중심 공식과 같은 식이라는 점에 주목하라. 각 위치에 놓인 질량에 위치를 곱해 더한 것이 무게중심이다. 확률분포를 실직선 위의 질량 분포로 보면 기댓값은 정확히 그 균형점이다.

**예: 공정한 주사위.**

$$
E[X] = \sum_{x=1}^{6} x \cdot \tfrac{1}{6} = \frac{1+2+3+4+5+6}{6} = 3.5
$$

$3.5$는 주사위가 결코 낼 수 없는 값이다. 기댓값은 "기대되는 값"이 아니라 **평균**이라는 점을 여기서 확인하고 넘어가는 것이 좋다.

**예: 지수분포.** $x \ge 0$에서 $f(x) = \lambda e^{-\lambda x}$인 $X \sim \text{Exponential}(\lambda)$에 대해

$$
E[X] = \int_0^{\infty} x \, \lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}
$$

이다. 도착률이 $\lambda$면 평균 대기시간이 $1/\lambda$라는 익숙한 관계다.

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

출력:

```
E[fair die] = 3.5000
Simulated mean = 3.5031
```

## 2. 함수의 기댓값은 분포를 몰라도 된다

$X$의 분포는 아는데 $X^2$이나 $e^X$의 기댓값이 필요한 경우가 자주 있다. 원칙대로라면 $Y = g(X)$의 분포를 먼저 유도해야 하는데, 그 과정이 대개 번거롭다. 다행히 건너뛸 수 있다.

### 정리 2. 무의식적 통계학자의 법칙(LOTUS) — 분포 유도를 건너뛴다

$$
E[g(X)] =
\begin{cases}
\displaystyle\sum_i g(x_i)\, P(X = x_i), & \text{이산} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} g(x)\, f(x)\,dx, & \text{연속}
\end{cases}
$$

**$g(X)$의 분포를 구할 필요가 없다.** $X$의 분포에 $g$를 씌워 곧바로 더하거나 적분하면 된다.

이름이 재미있다. 이 공식을 아무 생각 없이 써도 맞기 때문에 "무의식적 통계학자의 법칙(Law of the Unconscious Statistician)"이라 부른다. 실제로는 증명이 필요한 정리이며, 그 증명이 "여러 $x$가 같은 $g(x)$로 옮겨질 때 무게가 합쳐진다"는 3.3절 정리 2의 관찰을 형식화한 것이다.

LOTUS는 곧바로 쓰인다. 다음 절의 분산 $\text{Var}(X) = E[(X - \mu)^2]$이 $g(x) = (x-\mu)^2$인 경우이고, 적률생성함수 $E[e^{tX}]$가 $g(x) = e^{tx}$인 경우다.

## 3. 합의 기댓값은 언제나 기댓값의 합이다

기댓값을 확률론의 주력 도구로 만드는 것이 이 성질이다.

### 정리 3. 기댓값의 선형성 — 독립성이 필요 없다

임의의 확률변수 $X, Y$와 상수 $a, b, c$에 대해

$$
E[aX + bY + c] = a\,E[X] + b\,E[Y] + c
$$

이며, 유한합으로 확장된다.

$$
E\!\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

**$X$와 $Y$가 독립일 필요가 전혀 없다.** 아무리 강하게 얽혀 있어도 성립한다.

기댓값의 다른 성질들과 나란히 놓으면 무엇이 특별한지 분명해진다.

| 성질 | 식 | 독립성 필요? |
|:---|:---|:---:|
| 상수 | $E[c] = c$ | 아니오 |
| 척도 | $E[aX] = a\,E[X]$ | 아니오 |
| 가법성 | $E[X + Y] = E[X] + E[Y]$ | **아니오** |
| 단조성 | $X \le Y \Rightarrow E[X] \le E[Y]$ | 아니오 |
| 곱 | $E[XY] = E[X]\,E[Y]$ | **예** |

곱만 독립성을 요구한다. 다음 절에서 볼 공분산이 정확히 이 곱 규칙이 깨지는 정도를 재는 양이다.

!!! tip "선형성의 사용법: 지시변수로 쪼개기"
    선형성이 강력한 이유는 **어려운 확률변수를 쉬운 것들의 합으로 쪼갤 수 있기** 때문이다. 조각들이 서로 종속이어도 상관없다는 점이 결정적이다.

    **동전 $n$번에서 앞면의 개수.** $i$번째가 앞면이면 $X_i = 1$, 아니면 0으로 두면 $X = \sum_i X_i$다. 각 $E[X_i] = p$이므로

    $$
    E[X] = \sum_{i=1}^{n} E[X_i] = np
    $$

    이항분포의 확률질량함수를 꺼낼 필요도 없이 두 줄로 끝난다.

    **쿠폰 수집가 문제.** 쿠폰이 $n$종류 있고 살 때마다 균등하게 하나를 받는다. 전부 모으는 데 필요한 구매 횟수 $T$의 기댓값은?

    $i$번째 단계를 "서로 다른 쿠폰 $i-1$종을 가진 상태에서 $i$번째 새 쿠폰을 얻을 때까지"로 두자. 그 단계에서 한 번의 구매가 새 쿠폰일 확률은 $\frac{n-i+1}{n}$이므로 단계의 길이는 평균 $\frac{n}{n-i+1}$인 기하분포다. 선형성에 의해

    $$
    E[T] = \sum_{i=1}^{n} \frac{n}{n-i+1} = n\sum_{k=1}^{n}\frac{1}{k} = n H_n \approx n \ln n
    $$

    이다. $n = 50$이면 약 225번 사야 한다. 단계들의 길이는 서로 종속이지만 선형성은 개의치 않는다.

```python
import numpy as np

def coupon_collector_simulation(n_coupons, n_trials=10_000):
    """쿠폰 수집가 문제: n종을 모두 모으려면 몇 개를 사야 하는가."""
    np.random.seed(42)
    totals = []
    for _ in range(n_trials):
        collected = set()      # set이라 중복은 저절로 걸러진다
        count = 0
        # 종류를 다 모을 때까지 무작위로 하나씩 뽑는다
        while len(collected) < n_coupons:
            collected.add(np.random.randint(0, n_coupons))
            count += 1
        totals.append(count)

    simulated = np.mean(totals)

    # 이론값 유도: 이미 k종을 모았을 때 새 종이 나올 확률은 (n-k)/n 이므로
    # 새 종 하나를 더 얻기까지 기대 횟수는 n/(n-k) 다.
    # 이를 k = 0..n-1 로 모두 더하면
    #   n/n + n/(n-1) + ... + n/1 = n * (1 + 1/2 + ... + 1/n) = n * H_n
    # 기댓값의 선형성 덕분에 각 단계가 독립이 아니어도 그냥 더할 수 있다.
    H_n = sum(1/k for k in range(1, n_coupons + 1))      # 조화수 H_n
    theoretical = n_coupons * H_n

    print(f"n = {n_coupons}")
    print(f"Simulated E[T] = {simulated:.1f}")
    print(f"Theoretical E[T] = n·Hₙ = {theoretical:.1f}")

coupon_collector_simulation(50)
```

출력:

```
n = 50
Simulated E[T] = 225.5
Theoretical E[T] = n·Hₙ = 225.0
```

종속인 변수에서도 선형성이 성립함을 직접 확인해 보자. $Y = X^2$은 $X$에 완전히 종속이다.

```python
import numpy as np
import matplotlib.pyplot as plt

def linearity_demonstration():
    """종속인 두 변수에서도 기댓값의 선형성이 성립함을 확인한다.

    E[X + Y] = E[X] + E[Y] 는 X와 Y가 **독립이 아니어도** 성립한다.
    독립이 필요한 것은 곱의 기댓값 E[XY] = E[X]E[Y] 이나
    분산의 덧셈 Var(X+Y) = Var(X) + Var(Y) 쪽이다.
    이 구분이 확률론에서 가장 자주 헷갈리는 지점 중 하나다.
    """
    np.random.seed(42)
    n_sim = 100_000

    # X ~ Uniform(0,1), Y = X^2. X를 알면 Y가 완전히 결정되므로 극단적으로 종속이다.
    X = np.random.rand(n_sim)
    Y = X ** 2

    print("X and Y = X² are dependent, but linearity still holds:")
    print(f"E[X] = {X.mean():.4f} (theoretical: 0.5)")
    print(f"E[Y] = {Y.mean():.4f} (theoretical: 0.3333)")
    print(f"E[X + Y] = {(X + Y).mean():.4f}")
    print(f"E[X] + E[Y] = {X.mean() + Y.mean():.4f}")

linearity_demonstration()
```

출력:

```
X and Y = X² are dependent, but linearity still holds:
E[X] = 0.4995 (theoretical: 0.5)
E[Y] = 0.3326 (theoretical: 0.3333)
E[X + Y] = 0.8321
E[X] + E[Y] = 0.8321
```

## 연습문제

**연습문제 1.**
이산확률변수 $X$의 분포가 $P(X=-1) = 0.3$, $P(X=0) = 0.4$, $P(X=2) = 0.3$이다. $E[X]$와 $E[X^2]$를 계산하라.

??? success "풀이"
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

??? success "풀이"
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

??? success "풀이"
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

??? success "풀이"
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

??? success "풀이"
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

??? success "풀이"
    정의에 의해 $g(y) = \mathbb{E}[X \mid Y = y] = \int x f_{X \mid Y}(x \mid y) dx$이다.

    $\mathbb{E}[g(Y)] = \int g(y) f_Y(y) dy = \int \int x f_{X \mid Y}(x \mid y) f_Y(y) dx \, dy = \int \int x f_{X, Y}(x, y) dx \, dy = \int x f_X(x) dx = \mathbb{E}[X]$.

    $\square$

    확률, 통계, 그리고 기댓값에 대한 동적계획법 접근 전반에서 쓰인다.

    - **최적 예측자로서의 조건부 기댓값:** $\mathbb{E}[X \mid Y]$는 모든 함수 $g$에 대해 $\mathbb{E}[(X - g(Y))^2]$을 최소화한다.
    - **탑 성질**(반복 기댓값): $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[\mathbb{E}[\mathbb{E}[X \mid Y, Z] \mid Y]]$ 등.
    - **MCMC / 분산 감소**: (가능할 때) $X$를 $\mathbb{E}[X \mid Y]$로 대체하면 라오–블랙웰 정리를 통해 추정량의 분산이 줄어든다.
    - **강화학습**: 벨만 방정식 $V(s) = \mathbb{E}[R + \gamma V(s') \mid s]$이 반복된 조건부 기댓값이다.

## 정리하며

기댓값은 분포를 하나의 수로 줄이는 첫 번째 요약이다.

- **정리 1**은 그것을 무게중심으로 정의했다. 값에 확률을 곱해 더한다.
- **정리 2**(LOTUS)는 함수의 기댓값을 분포 유도 없이 계산하게 해 준다. 다음 절의 분산과 적률생성함수가 모두 이 법칙의 적용이다.
- **정리 3**(선형성)은 합의 기댓값이 언제나 기댓값의 합임을 보였다. **독립성이 필요 없다**는 것이 핵심이며, 어려운 확률변수를 쉬운 조각의 합으로 쪼개는 전략이 여기서 나온다.

기댓값만으로는 부족하다. 평균이 같아도 분포는 전혀 다를 수 있기 때문이다. 언제나 정확히 3.5가 나오는 가짜 주사위와 공정한 주사위는 기댓값이 같지만 성격이 완전히 다르다.

빠진 것은 **퍼짐**이다. 값들이 중심에서 얼마나 흩어져 있는가. 다음 절의 **분산**이 그 두 번째 요약값이며, 두 변수가 함께 움직이는 정도를 재는 **공분산**이 뒤따른다. 그리고 공분산은 이 절에서 유일하게 독립성을 요구했던 곱 규칙 $E[XY] = E[X]E[Y]$가 깨지는 정도를 재는 양이다.
