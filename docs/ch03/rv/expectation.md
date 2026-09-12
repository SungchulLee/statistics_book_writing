# 기댓값과 선형성

분포 전체를 적는 대신 몇 개의 수로 요약할 수 있다면 편할 것이다. 주사위 눈의 분포를 여섯 줄로 적는 대신 "평균 3.5"라고 말하는 식이다.

**기댓값**은 그 첫 번째 요약값이다. 실험을 여러 번 반복했을 때의 장기적 평균이며, 벽돌 비유로는 실직선 위에 놓인 벽돌들의 **무게중심**이다.

기댓값이 특별한 것은 계산이 쉬워서가 아니다. **선형성** 때문이다. 아무리 복잡하게 얽힌 확률변수들이라도 합의 기댓값은 기댓값의 합이며, 여기에는 독립성이 전혀 필요하지 않다. 이 성질 하나로 손대기 어려워 보이던 문제들이 풀린다.

이 절은 세 개의 정리로 이루어진다. 기댓값의 정의(정리 1), 함수의 기댓값을 분포 유도 없이 구하는 방법(정리 2), 그리고 선형성(정리 3)이다.

## 1. 무게중심을 구한다

정의는 "값 곱하기 무게를 모두 더한다"이다. 이산이면 더하고 연속이면 적분한다 — 앞 절의 대응이 그대로 적용된다.

<div class="thmbox" markdown>

### 정리 1. 기댓값의 정의 — 값의 확률가중 평균 { .thm }

이산확률변수 $X$에 대해

$$
E[X] = \sum_i x_i \, P(X = x_i) = \sum_i x_i \, p_{x_i}
$$

연속확률변수 $X$에 대해

$$
E[X] = \int_{-\infty}^{\infty} x \, f(x)\,dx
$$

이다.

</div>

물리의 무게중심 공식과 같은 식이라는 점에 주목하라. 각 위치에 놓인 질량에 위치를 곱해 더한 것이 무게중심이다. 확률분포를 실직선 위의 질량 분포로 보면 기댓값은 정확히 그 균형점이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 공정한 주사위의 기댓값. 공정한 주사위의 눈 $X$에 대해

$$
E[X] = \sum_{x=1}^{6} x \cdot \tfrac{1}{6} = \frac{1+2+3+4+5+6}{6} = 3.5
$$

$3.5$는 주사위가 결코 낼 수 없는 값이다. 기댓값은 "기대되는 값"이 아니라 **평균**이라는 점을 여기서 확인하고 넘어가는 것이 좋다.

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

</div>

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 지수분포의 기댓값. $x \ge 0$에서 $f(x) = \lambda e^{-\lambda x}$인 $X \sim \text{Exponential}(\lambda)$에 대해

$$
E[X] = \int_0^{\infty} x \, \lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}
$$

이다. 도착률이 $\lambda$면 평균 대기시간이 $1/\lambda$라는 익숙한 관계다.

</div>

## 2. 함수의 기댓값은 분포를 몰라도 된다

$X$의 분포는 아는데 $X^2$이나 $e^X$의 기댓값이 필요한 경우가 자주 있다. 원칙대로라면 $Y = g(X)$의 분포를 먼저 유도해야 하는데, 그 과정이 대개 번거롭다. 다행히 건너뛸 수 있다.

<div class="thmbox" markdown>

### 정리 2. 무의식적 통계학자의 법칙(LOTUS) — 분포 유도를 건너뛴다 { .thm }

$$
E[g(X)] =
\begin{cases}
\displaystyle\sum_i g(x_i)\, P(X = x_i), & \text{이산} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} g(x)\, f(x)\,dx, & \text{연속}
\end{cases}
$$

</div>

**$g(X)$의 분포를 구할 필요가 없다.** $X$의 분포에 $g$를 씌워 곧바로 더하거나 적분하면 된다.

이름이 재미있다. 이 공식을 아무 생각 없이 써도 맞기 때문에 "무의식적 통계학자의 법칙(Law of the Unconscious Statistician)"이라 부른다. 실제로는 증명이 필요한 정리이며, 그 증명이 "여러 $x$가 같은 $g(x)$로 옮겨질 때 무게가 합쳐진다"는 3.3절 정리 2의 관찰을 형식화한 것이다.

LOTUS는 곧바로 쓰인다. 다음 절의 분산 $\text{Var}(X) = E[(X - \mu)^2]$이 $g(x) = (x-\mu)^2$인 경우이고, 적률생성함수 $E[e^{tX}]$가 $g(x) = e^{tx}$인 경우다.

## 3. 합의 기댓값은 언제나 기댓값의 합이다

기댓값을 확률론의 주력 도구로 만드는 것이 이 성질이다.

<div class="thmbox" markdown>

### 정리 3. 기댓값의 선형성 — 독립성이 필요 없다 { .thm }

임의의 확률변수 $X, Y$와 상수 $a, b, c$에 대해

$$
E[aX + bY + c] = a\,E[X] + b\,E[Y] + c
$$

이며, 유한합으로 확장된다.

$$
E\!\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

</div>

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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이산확률변수 $X$의 분포가 $P(X=-1) = 0.3$, $P(X=0) = 0.4$, $P(X=2) = 0.3$이다. $E[X]$와 $E[X^2]$를 계산하라.

</div>

??? success "풀이"
    $$
    E[X] = (-1)(0.3) + (0)(0.4) + (2)(0.3) = -0.3 + 0 + 0.6 = 0.3
    $$

    $g(X) = X^2$에 대해 LOTUS를 쓰면

    $$
    E[X^2] = (-1)^2(0.3) + (0)^2(0.4) + (2)^2(0.3) = 0.3 + 0 + 1.2 = 1.5
    $$

    이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$X_1, X_2, \ldots, X_{100}$을 독립인 동전 던지기 100번의 지시변수라 하자. $i$번째 던지기가 앞면(확률 0.5)이면 $X_i = 1$, 아니면 $X_i = 0$이다. 기댓값의 선형성을 이용해 $E\!\left[\sum_{i=1}^{100} X_i\right]$를 구하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
공정한 육면체 주사위를 굴린다. $X$를 나온 수라 하고 $Y = (X - 3.5)^2$이라 하자. LOTUS를 써서 $E[Y]$를 계산하라.

</div>

??? success "풀이"
    LOTUS에 의해 $E[Y] = E[(X-3.5)^2] = \sum_{x=1}^{6} (x - 3.5)^2 \cdot P(X=x)$이다. 각 값에 대해 $P(X=x) = 1/6$이므로

    $$
    E[Y] = \frac{1}{6}\left[(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2\right]
    $$

    $$
    = \frac{1}{6}\left[6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25\right] = \frac{17.5}{6} \approx 2.917
    $$

    이다. 참고로 $E[X] = 3.5$이므로 이는 정확히 공정한 주사위의 $\text{Var}(X)$다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
$X$와 $Y$가 독립이고 $E[X] = 2$, $E[Y] = 3$, $E[X^2] = 5$, $E[Y^2] = 11$이다. $E[XY]$와 $E[(X+Y)^2]$를 계산하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**기댓값의 꼬리합 공식.** 음이 아닌 확률변수 $X$에 대해 $\mathbb{E}[X] = \int_0^\infty P(X > t) dt$(연속) 또는 $\sum_{n=0}^\infty P(X > n)$(정숫값)임을 증명하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**확률변수로서의 조건부 기댓값.** $X, Y$가 결합분포를 갖는다고 하자. $g(y) = \mathbb{E}[X \mid Y = y]$와 확률변수 $\mathbb{E}[X \mid Y] = g(Y)$를 정의한다. **전기댓값의 법칙** $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$를 증명하라.

</div>

??? success "풀이"
    정의에 의해 $g(y) = \mathbb{E}[X \mid Y = y] = \int x f_{X \mid Y}(x \mid y) dx$이다.

    $\mathbb{E}[g(Y)] = \int g(y) f_Y(y) dy = \int \int x f_{X \mid Y}(x \mid y) f_Y(y) dx \, dy = \int \int x f_{X, Y}(x, y) dx \, dy = \int x f_X(x) dx = \mathbb{E}[X]$.

    $\square$

    확률, 통계, 그리고 기댓값에 대한 동적계획법 접근 전반에서 쓰인다.

    - **최적 예측자로서의 조건부 기댓값:** $\mathbb{E}[X \mid Y]$는 모든 함수 $g$에 대해 $\mathbb{E}[(X - g(Y))^2]$을 최소화한다.
    - **탑 성질**(반복 기댓값): $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[\mathbb{E}[\mathbb{E}[X \mid Y, Z] \mid Y]]$ 등.
    - **MCMC / 분산 감소**: (가능할 때) $X$를 $\mathbb{E}[X \mid Y]$로 대체하면 라오–블랙웰 정리를 통해 추정량의 분산이 줄어든다.
    - **강화학습**: 벨만 방정식 $V(s) = \mathbb{E}[R + \gamma V(s') \mid s]$이 반복된 조건부 기댓값이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
기댓값이 **존재하지 않을** 수 있다. "무한하다"와 "정의되지 않는다"를 구분하고, 코시분포로 확인하라.

</div>

??? success "풀이"
    $\mathbb{E}[X]$는 $\mathbb{E}[X^+]$와 $\mathbb{E}[X^-]$로 나누어 정의한다($X^\pm$는 양·음 부분).

    | 상황 | $\mathbb{E}[X]$ |
    |---|---|
    | 둘 다 유한 | 유한한 값 |
    | $\mathbb{E}[X^+]=\infty$, $\mathbb{E}[X^-]<\infty$ | $+\infty$ |
    | 둘 다 $\infty$ | **정의되지 않음** |

    **코시분포가 세 번째 경우다.** 밀도가 $f(x) = 1/(\pi(1+x^2))$이라 양쪽 꼬리 모두 $x f(x) \sim 1/(\pi x)$로 발산한다.

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    print("코시 표본평균의 궤적 — 세 번 반복")
    for trial in range(3):
        x = rng.standard_cauchy(1_000_000)
        run = np.cumsum(x) / np.arange(1, len(x) + 1)
        print(f"  시행 {trial+1}: n=10^3 {run[999]:>9.3f}   10^4 {run[9999]:>9.3f}"
              f"   10^5 {run[99999]:>9.3f}   10^6 {run[-1]:>9.3f}")
    ```

    출력:

    ```
    코시 표본평균의 궤적 — 세 번 반복
      시행 1: n=10^3    -0.934   10^4    -0.144   10^5     0.297   10^6     0.471
      시행 2: n=10^3     0.093   10^4    -1.466   10^5    -0.554   10^6    -0.884
      시행 3: n=10^3    -1.187   10^4     0.217   10^5    -0.966   10^6    -2.630
    ```

    **표본평균이 정착하지 않는다.** $n$이 $100$만이어도 시행마다 $0.471$, $-0.884$, $-2.630$으로 흩어진다. 큰수의 법칙이 적용되지 않기 때문이다.

    **더 놀라운 사실.** 코시분포의 표본평균 $\bar{X}_n$은 **$n$과 무관하게 원래 코시분포와 정확히 같은 분포를 갖는다.** 관측을 $100$만 개 모으는 것이 하나만 보는 것과 똑같다.

    **"무한"과 "정의되지 않음"의 실무적 차이.**

    - **$\mathbb{E}[X] = +\infty$**(예: 파레토 $\alpha \le 1$, 상트페테르부르크 게임)이면 표본평균이 **$\infty$로 발산한다.** 방향이 정해져 있다.
    - **정의되지 않으면**(코시) 표본평균이 **어느 방향으로도 가지 않고 헤맨다.** 어떤 값에도 수렴하지 않는다.

    **어디서 코시를 만나는가.** 두 독립 정규의 비 $Z_1/Z_2$가 코시이며, 그래서 **비를 다룰 때 조심해야 한다.** 회귀계수의 비, 두 추정값의 비, 각도 측정에서 코시가 자연스럽게 나타난다.

    **처방.** 평균이 없으면 **중앙값을 쓴다.** 코시분포의 중앙값은 잘 정의되고, 표본중앙값이 그것으로 수렴한다(2장 평균·중앙값 문서 연습문제 8에서 본 대로 코시에서 중앙값이 압도적으로 낫다). $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 $6$의 조건부기댓값이 왜 **"최적 예측"** 인지 보여라. 어떤 의미에서 최적인가?

</div>

??? success "풀이"
    **정리.** 모든 (가측) 함수 $g$에 대해

    $$
    \mathbb{E}\!\left[(Y - \mathbb{E}[Y\mid X])^2\right] \le \mathbb{E}\!\left[(Y-g(X))^2\right]
    $$

    이다. 즉 $\mathbb{E}[Y\mid X]$가 **제곱오차를 최소화하는 $X$의 함수**다.

    **증명.** $m(X) = \mathbb{E}[Y\mid X]$라 두고 전개하면

    $$
    \mathbb{E}[(Y-g)^2] = \mathbb{E}[(Y-m)^2] + 2\mathbb{E}[(Y-m)(m-g)] + \mathbb{E}[(m-g)^2]
    $$

    이다. 가운데 항은 $X$로 조건을 걸어 계산하면 $\mathbb{E}[Y-m \mid X] = 0$이므로 사라진다. 남는 것이 $\mathbb{E}[(m-g)^2] \ge 0$이다. $\square$

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400_000
    x = rng.normal(0, 1, n)
    y = x ** 2 + rng.normal(0, 0.5, n)             # 참 관계는 이차

    predictors = {
        "E[Y|X] = X^2": x ** 2,
        "최적 선형 예측": np.polyval(np.polyfit(x, y, 1), x),
        "상수 (전체 평균)": np.full(n, y.mean()),
    }
    for label, pred in predictors.items():
        print(f"  {label:>16}: MSE {np.mean((y - pred) ** 2):.5f}")

    print(f"\n잡음의 분산 = {0.5 ** 2:.5f}  ← E[Y|X] 의 MSE 가 여기에 닿는다")
    print(f"Cov(X, X^2) = {np.cov(x, x ** 2)[0,1]:+.5f}  ← 그래서 선형 예측이 상수와 같다")
    ```

    출력:

    ```
    E[Y|X] = X^2: MSE 0.24974
              최적 선형 예측: MSE 2.25532
            상수 (전체 평균): MSE 2.25536

    잡음의 분산 = 0.25000  ← E[Y|X] 의 MSE 가 여기에 닿는다
    Cov(X, X^2) = -0.00444  ← 그래서 선형 예측이 상수와 같다
    ```

    **$\mathbb{E}[Y\mid X]$의 MSE가 $0.2496$으로 잡음 분산 $0.25$에 닿는다.** 더 줄일 수 없는 하한이다.

    **최적 선형 예측이 상수 예측과 똑같다**($2.238$). $\operatorname{Cov}(X, X^2) = 0$이므로 선형 회귀의 기울기가 $0$이 되기 때문이다. 앞 절 독립 문서 연습문제 7의 "무상관이지만 종속"이 예측의 언어로 나타난 것이다.

    **기하학적 해석.** $L^2$ 공간에서 $\mathbb{E}[Y\mid X]$는 $Y$를 **"$X$의 함수들이 이루는 부분공간" 위로 직교사영**한 것이다. 잔차 $Y - \mathbb{E}[Y\mid X]$가 그 부분공간의 모든 원소와 직교하며, 그것이 위 증명의 가운데 항이 사라지는 이유다.

    **0장의 최소제곱과 같은 구조다.** 거기서는 $\mathbf{y}$를 $\mathbf{X}$의 **열공간**에 사영했고, 여기서는 $Y$를 $X$의 **함수공간**에 사영한다. 선형회귀는 함수공간을 선형함수로 제한한 특수한 경우이며, 그래서 위 예에서 참 관계를 놓친다.

    **함의.** 회귀분석의 목표가 $\mathbb{E}[Y\mid X]$를 추정하는 것이며, 모형의 유연성이 곧 **함수공간을 얼마나 넓게 잡느냐**다. 1장에서 본 편향–분산 절충이 이 선택의 다른 이름이다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
항의 **개수 자체가 확률변수**인 합의 기댓값은 어떻게 되는가? **왈드 항등식**을 진술하고 확인하라.

</div>

??? success "풀이"
    $N$이 확률변수이고 $X_1, X_2, \ldots$가 i.i.d.이며 $N$과 독립일 때

    $$
    \mathbb{E}\!\left[\sum_{i=1}^{N}X_i\right] = \mathbb{E}[N]\,\mathbb{E}[X]
    $$

    이다. 분산은 조금 더 복잡하다.

    $$
    \operatorname{Var}\!\left(\sum_{i=1}^{N}X_i\right) = \mathbb{E}[N]\operatorname{Var}(X) + \operatorname{Var}(N)\,\mathbb{E}[X]^2
    $$

    **증명은 전기댓값의 법칙이다.** $N$으로 조건을 걸면 $\mathbb{E}[S \mid N=n] = n\mathbb{E}[X]$이므로

    $$
    \mathbb{E}[S] = \mathbb{E}\!\left[\mathbb{E}[S\mid N]\right] = \mathbb{E}[N\,\mathbb{E}[X]] = \mathbb{E}[N]\mathbb{E}[X]
    $$

    이다. 분산도 총분산의 법칙으로 같은 방식으로 나온다. $\square$

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    lam, mu, sigma = 4.0, 3.0, 1.0

    N = rng.poisson(lam, 50_000)
    S = np.array([rng.normal(mu, sigma, k).sum() if k else 0.0 for k in N])

    print(f"E[S]   모의 {S.mean():.4f}   이론 E[N]E[X] = {lam * mu:.4f}")
    print(f"Var(S) 모의 {S.var():.4f}   이론 E[N]Var(X) + Var(N)E[X]^2 = "
          f"{lam * sigma ** 2 + lam * mu ** 2:.4f}")
    ```

    출력:

    ```
    E[S]   모의 12.0062   이론 E[N]E[X] = 12.0000
    Var(S) 모의 40.1686   이론 E[N]Var(X) + Var(N)E[X]^2 = 40.0000
    ```

    **두 공식이 모두 맞는다.**

    **분산 공식의 두 항이 각각 무엇인가.**

    - $\mathbb{E}[N]\operatorname{Var}(X)$: 항의 개수가 고정이었다면 있었을 변동
    - $\operatorname{Var}(N)\mathbb{E}[X]^2$: **개수가 흔들려서 추가로 생기는 변동**

    포아송이면 $\operatorname{Var}(N) = \mathbb{E}[N]$이라 둘째 항이 $\lambda\mu^2$로 커진다. 위 예에서 $4 + 36 = 40$ 중 $36$이 개수의 변동에서 온다. **개수의 불확실성이 지배적**이다.

    **어디에 쓰이는가.**

    | 응용 | $N$ | $X$ |
    |---|---|---|
    | 보험 총 청구액 | 청구 건수 | 건당 금액 |
    | 웹사이트 총 매출 | 방문자 수 | 방문당 지출 |
    | 대기행렬의 총 서비스 시간 | 도착 수 | 서비스 시간 |
    | 순차 검정의 총 표본 | 정지 시각 | 관측 |

    **복합 포아송 모형**이 이 구조의 표준 이름이며, 보험 수리의 기본 도구다.

    **주의: $N$과 $X_i$가 독립이어야 한다.** 독립이 아니면 공식이 깨진다. 순차 검정처럼 $N$이 관측에 의존해 정해지는 경우에는 **정지시각**이라는 조건이 필요하며, 그것이 왈드 항등식의 원래 형태다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
연습문제 $2$의 지시변수 기법을 더 밀고 나가라. **쿠폰 수집가 문제**와 **매칭 문제**를 풀어라.

</div>

??? success "풀이"
    **쿠폰 수집가.** $n$종의 쿠폰을 모두 모으려면 몇 번 뽑아야 하는가?

    $T_k$를 "$k-1$종을 모은 뒤 새 종류가 나올 때까지의 뽑기 수"라 하면 $T_k \sim \text{Geometric}(p_k)$이고 $p_k = (n-k+1)/n$이다. 선형성으로

    $$
    \mathbb{E}[T] = \sum_{k=1}^{n}\frac{n}{n-k+1} = n\sum_{j=1}^{n}\frac{1}{j} = n H_n \approx n(\ln n + \gamma)
    $$

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    print("쿠폰 수집가")
    print(f"{'n':>5}{'모의':>11}{'이론 n H_n':>13}{'근사 n(ln n + 0.577)':>22}")
    for n in (5, 10, 50):
        H = sum(1 / i for i in range(1, n + 1))
        trials = []
        for _ in range(20_000):
            seen, count = set(), 0
            while len(seen) < n:
                seen.add(rng.integers(n)); count += 1
            trials.append(count)
        print(f"{n:>5}{np.mean(trials):>11.3f}{n * H:>13.3f}"
              f"{n * (np.log(n) + 0.5772):>22.3f}")
    ```

    출력:

    ```
    쿠폰 수집가
        n         모의     이론 n H_n    근사 n(ln n + 0.577)
        5     11.377       11.417                10.933
       10     29.244       29.290                28.798
       50    225.207      224.960               224.461
    ```

    **모의와 이론이 맞는다.** 마지막 한 종류를 얻는 데만 평균 $n$번이 걸리므로, 전체의 상당 부분이 막바지에 소요된다.

    **매칭 문제.** $n$명이 모자를 무작위로 다시 가져갈 때 자기 모자를 받는 사람의 기대 수는?

    $I_i$를 "$i$번이 자기 모자를 받음"의 지시변수라 하면 $\mathbb{E}[I_i] = 1/n$이므로

    $$
    \mathbb{E}\!\left[\sum_i I_i\right] = n \cdot \frac{1}{n} = 1
    $$

    **$n$과 무관하게 항상 $1$이다.**

    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    print("\n매칭 문제 (자기 모자를 받는 사람 수)")
    print(f"{'n':>5}{'평균':>10}{'분산':>10}{'0명일 확률':>13}{'1/e':>9}")
    for n in (5, 20, 100):
        counts = np.array([np.sum(rng.permutation(n) == np.arange(n))
                           for _ in range(100_000)])
        print(f"{n:>5}{counts.mean():>10.4f}{counts.var():>10.4f}"
              f"{np.mean(counts == 0):>13.4f}{1 / np.e:>9.4f}")
    ```

    출력:

    ```
    매칭 문제 (자기 모자를 받는 사람 수)
        n        평균        분산       0명일 확률      1/e
        5    1.0005    1.0008       0.3667   0.3679
       20    1.0055    1.0046       0.3653   0.3679
      100    1.0012    1.0016       0.3679   0.3679
    ```

    **평균과 분산이 모두 $1$에 가깝고 $n$과 무관하다.** 그리고 아무도 자기 모자를 받지 못할 확률이 $1/e \approx 0.368$로 수렴한다.

    **지시변수 기법의 힘은 독립을 요구하지 않는다는 것이다.** 매칭 문제에서 $I_i$들은 **명백히 종속**이다($n-1$명이 자기 모자를 받으면 나머지 한 명도 반드시 받는다). 그런데 **기댓값의 선형성은 독립과 무관하게 성립하므로** 합의 기댓값을 그대로 계산할 수 있다.

    **이것이 이 절의 핵심 도구다.** 복잡한 확률변수를 지시변수의 합으로 쪼개면, 각 지시변수의 기댓값은 단순한 확률 하나이고 선형성이 나머지를 해 준다. 분산은 종속성 때문에 더 어렵지만(공분산 항이 필요하다), 기댓값만큼은 언제나 쉽다. $\square$


## 정리하며

기댓값은 분포를 하나의 수로 줄이는 첫 번째 요약이다.

- **정리 1**은 그것을 무게중심으로 정의했다. 값에 확률을 곱해 더한다.
- **정리 2**(LOTUS)는 함수의 기댓값을 분포 유도 없이 계산하게 해 준다. 다음 절의 분산과 적률생성함수가 모두 이 법칙의 적용이다.
- **정리 3**(선형성)은 합의 기댓값이 언제나 기댓값의 합임을 보였다. **독립성이 필요 없다**는 것이 핵심이며, 어려운 확률변수를 쉬운 조각의 합으로 쪼개는 전략이 여기서 나온다.

기댓값만으로는 부족하다. 평균이 같아도 분포는 전혀 다를 수 있기 때문이다. 언제나 정확히 3.5가 나오는 가짜 주사위와 공정한 주사위는 기댓값이 같지만 성격이 완전히 다르다.

빠진 것은 **퍼짐**이다. 값들이 중심에서 얼마나 흩어져 있는가. 다음 절의 **분산**이 그 두 번째 요약값이며, 두 변수가 함께 움직이는 정도를 재는 **공분산**이 뒤따른다. 그리고 공분산은 이 절에서 유일하게 독립성을 요구했던 곱 규칙 $E[XY] = E[X]E[Y]$가 깨지는 정도를 재는 양이다.
