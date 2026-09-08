# 적률생성함수

평균과 분산은 분포의 두 요약값이지만 분포를 결정하지는 못한다. 평균과 분산이 같으면서 모양이 전혀 다른 분포는 얼마든지 있다.

$E[X], E[X^2], E[X^3], \ldots$을 **적률**이라 하는데, 이것들을 다 모으면 대개 분포가 결정된다. 그런데 적률을 하나씩 계산하는 것은 번거롭다. **적률생성함수**는 이 무한히 많은 적률을 **함수 하나**에 담고, 미분만으로 꺼내 쓸 수 있게 한다.

이 도구가 특별한 이유는 세 가지다. 적률을 자동으로 뽑아 주고, 분포를 유일하게 결정하며, 무엇보다 **독립인 변수의 합을 곱셈으로 바꾼다**. 중심극한정리의 증명이 이 마지막 성질 위에 서 있다.

이 절은 세 개의 정리로 이루어진다. 미분으로 적률을 뽑는 성질(정리 1), 분포를 유일하게 결정한다는 성질(정리 2), 그리고 합을 곱으로 바꾸는 성질(정리 3)이다.

## 1. 적률을 한 함수에 담아 미분으로 꺼낸다

정의는 뜻밖으로 단순하다. $e^{tX}$의 기댓값을 $t$의 함수로 본다. LOTUS(3.4절 정리 2)를 $g(x) = e^{tx}$에 적용한 것이다.

### 정리 1. 적률생성함수 — 0에서의 $n$계 도함수가 $n$번째 적률

확률변수 $X$의 **적률생성함수(MGF)** 는

$$
M_X(t) = E\big[e^{tX}\big] =
\begin{cases}
\displaystyle\sum_x e^{tx}\, P(X = x), & \text{이산} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} e^{tx} f(x)\,dx, & \text{연속}
\end{cases}
$$

이며, 0을 포함하는 어떤 열린구간에서 유한하면 **존재한다**고 한다. 이때

$$
M_X^{(n)}(0) = \frac{d^n}{dt^n} M_X(t)\bigg|_{t=0} = E[X^n]
$$

이 성립한다.

**왜 그런가.** $e^{tX}$를 테일러 전개하면 적률이 계수로 줄줄이 나온다.

$$
M_X(t) = E\!\left[\sum_{k=0}^{\infty} \frac{(tX)^k}{k!}\right] = \sum_{k=0}^{\infty} \frac{t^k}{k!}\,E[X^k]
$$

$t^n$의 계수가 $E[X^n]/n!$이므로, $n$번 미분하고 $t = 0$을 넣으면 $E[X^n]$만 남는다. "적률을 생성한다"는 이름이 여기서 왔다.

실무에서 가장 자주 쓰는 것은 처음 둘이다.

$$
E[X] = M_X'(0), \qquad
E[X^2] = M_X''(0), \qquad
\text{Var}(X) = M_X''(0) - \big[M_X'(0)\big]^2
$$

**예: 정규분포.** $X \sim N(\mu, \sigma^2)$의 적률생성함수는

$$
M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
$$

이다. 두 번 미분해 보면

$$
\begin{aligned}
M_X'(t) &= (\mu + \sigma^2 t)\,M_X(t), &\quad M_X'(0) &= \mu = E[X] \\[4pt]
M_X''(t) &= \big(\sigma^2 + (\mu + \sigma^2 t)^2\big)M_X(t), &\quad M_X''(0) &= \sigma^2 + \mu^2 = E[X^2]
\end{aligned}
$$

이고 $\text{Var}(X) = (\sigma^2 + \mu^2) - \mu^2 = \sigma^2$이다. 적분 한 번 하지 않고 평균과 분산을 얻었다.

주요 분포의 적률생성함수를 모아 두면 계산이 빨라진다.

| 분포 | $M_X(t)$ | 조건 |
|:---|:---|:---|
| 베르누이분포 Bernoulli$(p)$ | $1 - p + pe^t$ | |
| 이항분포 B$(n, p)$ | $(1 - p + pe^t)^n$ | |
| 포아송분포 Poisson$(\lambda)$ | $\exp\!\big(\lambda(e^t - 1)\big)$ | |
| 기하분포 Geom$(p)$ | $\dfrac{pe^t}{1 - (1-p)e^t}$ | $t < -\ln(1-p)$ |
| 지수분포 Exp$(\lambda)$ | $\dfrac{\lambda}{\lambda - t}$ | $t < \lambda$ |
| 정규분포 N$(\mu, \sigma^2)$ | $\exp\!\left(\mu t + \dfrac{\sigma^2 t^2}{2}\right)$ | |

```python
import numpy as np
from scipy.misc import derivative

def mgf_normal(t, mu, sigma2):
    """MGF of Normal(mu, sigma2)."""
    return np.exp(mu * t + sigma2 * t**2 / 2)

# Extract moments via numerical differentiation
mu, sigma2 = 3.0, 4.0

E_X = derivative(lambda t: mgf_normal(t, mu, sigma2), 0, n=1, dx=1e-6)
E_X2 = derivative(lambda t: mgf_normal(t, mu, sigma2), 0, n=2, dx=1e-6)
Var_X = E_X2 - E_X**2

print(f"E[X] = {E_X:.4f} (theoretical: {mu})")
print(f"E[X²] = {E_X2:.4f} (theoretical: {sigma2 + mu**2})")
print(f"Var(X) = {Var_X:.4f} (theoretical: {sigma2})")
```

## 2. 적률생성함수가 같으면 분포가 같다

적률생성함수는 요약이 아니라 **분포 전체를 담은 다른 표현**이다. 누적분포함수와 마찬가지로 분포를 완전히 결정한다.

### 정리 2. 유일성 — 적률생성함수는 분포를 식별한다

$X$와 $Y$의 적률생성함수가 존재하고 0을 포함하는 어떤 열린구간에서 일치하면

$$
M_X(t) = M_Y(t) \quad \Longrightarrow \quad X \stackrel{d}{=} Y
$$

이다.

증명 전략이 여기서 나온다. **어떤 확률변수의 분포를 알고 싶으면 적률생성함수를 계산해 표에서 찾으면 된다.** 밀도를 직접 유도하는 것보다 훨씬 쉬운 경우가 많다.

선형변환에 대한 규칙도 함께 알아 두면 유용하다. $Y = aX + b$이면

$$
M_Y(t) = e^{bt}\,M_X(at)
$$

이다. 표준화 $Z = (X - \mu)/\sigma$가 이 규칙의 전형적인 쓰임이다.

## 3. 합을 곱으로 바꾼다

적률생성함수가 확률론의 핵심 도구가 된 진짜 이유가 이것이다.

### 정리 3. 독립인 합 — 적률생성함수의 곱

$X \perp\!\!\!\perp Y$이면

$$
M_{X+Y}(t) = M_X(t)\,M_Y(t)
$$

이며, 독립인 $n$개로 확장된다.

$$
M_{S_n}(t) = \prod_{i=1}^{n} M_{X_i}(t), \qquad S_n = \sum_{i=1}^n X_i
$$

**독립인 변수의 합의 분포를 구하는 일은 원래 매우 어렵다.** 밀도끼리 합성곱(convolution)을 해야 하는데 적분이 지저분하다. 적률생성함수로 옮기면 그 합성곱이 **단순한 곱셈**이 된다. $e^{t(X+Y)} = e^{tX}e^{tY}$이고 독립이면 곱의 기댓값이 기댓값의 곱이기 때문이다(3.4절 정리 3의 곱 규칙).

**예: 독립인 정규분포의 합.** $X_1 \sim N(\mu_1, \sigma_1^2)$, $X_2 \sim N(\mu_2, \sigma_2^2)$이 독립이면

$$
M_{X_1+X_2}(t) = \exp\!\left((\mu_1+\mu_2)t + \frac{(\sigma_1^2+\sigma_2^2)t^2}{2}\right)
$$

이다. 이것은 $N(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$의 적률생성함수이므로, 유일성(정리 2)에 의해

$$
X_1 + X_2 \sim N(\mu_1 + \mu_2,\ \sigma_1^2 + \sigma_2^2)
$$

이다. **정규분포의 합이 다시 정규분포**라는 중요한 사실이 세 줄로 증명된다. 밀도로 직접 하려면 합성곱 적분을 계산해야 한다.

!!! note "중심극한정리 증명의 뼈대"
    평균 $\mu$, 분산 $\sigma^2$인 i.i.d. $X_i$에 대해 표준화된 평균을 $Z_n = \dfrac{\bar X - \mu}{\sigma/\sqrt n}$이라 하자. 정리 3에 의해

    $$
    M_{Z_n}(t) = \left[M_{\frac{X_i-\mu}{\sigma}}\!\left(\frac{t}{\sqrt n}\right)\right]^{n}
    $$

    이다. 안쪽 적률생성함수를 $t/\sqrt n$ 근방에서 2차까지 전개하면 $1 + \dfrac{t^2}{2n} + o(1/n)$이 되고, $n$제곱하면

    $$
    \left(1 + \frac{t^2}{2n} + o(1/n)\right)^{n} \longrightarrow e^{t^2/2}
    $$

    이다. 극한이 $N(0,1)$의 적률생성함수이므로 유일성에 의해 $Z_n \xrightarrow{d} N(0,1)$이다.

    증명의 세 재료가 모두 이 절에 있다. **합을 곱으로 바꾸고**(정리 3), **테일러 전개로 앞 두 적률만 남기고**(정리 1), **극한을 분포로 되돌린다**(정리 2). 3.5절에서 이 정리를 정면으로 다룬다.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_mgf_comparison():
    """Plot MGFs of several distributions."""
    t = np.linspace(-1.5, 1.5, 300)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Normal(0, 1)
    ax.plot(t, np.exp(t**2 / 2), label='N(0, 1)', lw=2)

    # Exponential(1)
    t_exp = t[t < 1]
    ax.plot(t_exp, 1 / (1 - t_exp), label='Exp(1)', lw=2)

    # Poisson(3)
    lam = 3
    ax.plot(t, np.exp(lam * (np.exp(t) - 1)), label='Poisson(3)', lw=2)

    # Bernoulli(0.5)
    p = 0.5
    ax.plot(t, 1 - p + p * np.exp(t), label='Bernoulli(0.5)', lw=2)

    ax.set_xlabel('t')
    ax.set_ylabel('M_X(t)')
    ax.set_title('Moment Generating Functions')
    ax.set_ylim(0, 15)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

plot_mgf_comparison()
```

모든 곡선이 $t = 0$에서 값 1을 지난다는 점에 주목하라. $M_X(0) = E[e^0] = 1$이므로 언제나 그렇다. 그 점에서의 기울기가 평균이다.

```python
import numpy as np

def verify_sum_of_normals(n_simulations=100_000):
    """Verify that sum of independent normals is normal via simulation."""
    np.random.seed(42)
    mu1, sigma1 = 2, 3
    mu2, sigma2 = 5, 4

    X1 = np.random.normal(mu1, sigma1, n_simulations)
    X2 = np.random.normal(mu2, sigma2, n_simulations)
    S = X1 + X2

    print(f"E[X1+X2] = {S.mean():.4f} (theoretical: {mu1 + mu2})")
    print(f"Var(X1+X2) = {S.var():.4f} (theoretical: {sigma1**2 + sigma2**2})")

verify_sum_of_normals()
```

## 연습문제

**연습문제 1.**
$X \sim \mathrm{Exp}(\lambda)$이다. (a) $M_X(t)$와 그 정의역을 유도하라. (b) $\mathbb{E}[X], \mathbb{E}[X^2]$를 계산하라. (c) $\mathrm{Var}(X) = 1/\lambda^2$임을 확인하라.

??? success "풀이"
    (a) $t < \lambda$에 대해 $M_X(t) = \int_0^\infty e^{tx} \lambda e^{-\lambda x} dx = \lambda \int_0^\infty e^{-(\lambda - t) x} dx = \lambda/(\lambda - t)$이다.

    (b) $M_X'(t) = \lambda/(\lambda - t)^2$이므로 $M_X'(0) = 1/\lambda = \mathbb{E}[X]$이다. $M_X''(t) = 2\lambda/(\lambda - t)^3$이므로 $M_X''(0) = 2/\lambda^2 = \mathbb{E}[X^2]$이다.

    (c) $\mathrm{Var}(X) = 2/\lambda^2 - (1/\lambda)^2 = 1/\lambda^2$이다. 평균과 표준편차가 모두 $1/\lambda$로 같다.

---

**연습문제 2.**
$M_{aX + b}(t) = e^{bt} M_X(at)$를 증명하고, $X$와 $Y$가 독립일 때 $M_{X + Y}(t) = M_X(t) M_Y(t)$임을 증명하라.

??? success "풀이"
    **선형변환:**

    $$
    M_{aX + b}(t) = \mathbb{E}[e^{t(aX + b)}] = e^{bt} \mathbb{E}[e^{(at)X}] = e^{bt} M_X(at)
    $$

    **독립인 변수의 합:** $X$와 $Y$가 독립이면 $e^{tX}$와 $e^{tY}$도 독립이므로(독립인 변수의 함수이므로)

    $$
    M_{X + Y}(t) = \mathbb{E}[e^{t(X + Y)}] = \mathbb{E}[e^{tX} \cdot e^{tY}] = \mathbb{E}[e^{tX}] \mathbb{E}[e^{tY}] = M_X(t) M_Y(t)
    $$

    이다.

    두 번째 공식은 일반화된다. 상호독립인 $S_n = X_1 + \cdots + X_n$에 대해 $M_{S_n}(t) = \prod_i M_{X_i}(t)$이다.

    이 두 성질 덕분에 적률생성함수가 확률변수의 변환과 합을 분석하는 데 그토록 유용하다.

---

**연습문제 3.**
적률생성함수 방법으로 **독립인 포아송 확률변수 합의 분포**를 유도하라. $X_i \sim \mathrm{Poisson}(\lambda_i)$이 서로 독립일 때 $\sum_i X_i$의 분포는 무엇인가?

??? success "풀이"
    포아송의 적률생성함수는 $M_X(t) = e^{\lambda (e^t - 1)}$이다.

    합: $M_{\sum X_i}(t) = \prod_i M_{X_i}(t) = \prod_i e^{\lambda_i (e^t - 1)} = e^{(\sum \lambda_i)(e^t - 1)}$.

    이것이 $\mathrm{Poisson}(\sum \lambda_i)$의 적률생성함수다. 유일성에 의해 $\sum X_i \sim \mathrm{Poisson}(\sum \lambda_i)$이다.

    **함의:** 포아송은 덧셈에 대해 "닫혀" 있다. 독립인 포아송들의 합은 비율을 합한 포아송이다. 이것이 계수 자료 모형의 토대다. 독립인 여러 원천에서 나온 총 사건 수가 결합된 비율의 포아송을 따르므로 모듈식 분석이 가능하다.

---

**연습문제 4.**
**적률생성함수로 구하는 왜도와 첨도.** 3차 및 4차 표준화 누율(왜도와 초과첨도)이 $M_X(t)$ 자체가 아니라 $\ln M_X(t)$의 도함수에서 나옴을 보여라.

??? success "풀이"
    **누율생성함수**는 $K_X(t) = \ln M_X(t)$이며 그 테일러 전개는

    $$
    K_X(t) = \kappa_1 t + \kappa_2 \frac{t^2}{2} + \kappa_3 \frac{t^3}{6} + \kappa_4 \frac{t^4}{24} + \cdots
    $$

    이고 $\kappa_n$이 $n$차 누율이다. 처음 몇 개는 다음과 같다.

    - $\kappa_1 = \mathbb{E}[X] = \mu$ (평균)
    - $\kappa_2 = \mathrm{Var}(X) = \sigma^2$
    - $\kappa_3 = \mathbb{E}[(X - \mu)^3]$ (3차 중심적률)
    - $\kappa_4 = \mathbb{E}[(X - \mu)^4] - 3\sigma^4$ (4차 중심적률에서 $3\sigma^4$을 뺀 값)

    그러면 **왜도** $= \kappa_3/\kappa_2^{3/2}$이고 **초과첨도** $= \kappa_4/\kappa_2^2$이다. 둘 다 누율에서 나오며, 누율은 $M_X(t)$가 아니라 $K_X(t)$의 도함수다.

    **누율이 적률보다 나은 이유:** 독립인 합의 누율은 더해진다. 독립인 $X, Y$에 대해 $\kappa_n(X + Y) = \kappa_n(X) + \kappa_n(Y)$이지만 적률은 그렇지 않다. 예를 들어 $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$가 $\kappa_2$의 덧셈이고, 더 높은 누율에서도 마찬가지다. 그래서 누율이 중심극한정리와 에지워스 전개의 자연스러운 언어가 된다.

---

**연습문제 5.**
**적률생성함수의 비존재.** 코시분포가 적률생성함수를 갖지 않음을 보여라. 대안인 **특성함수**는 무엇이며 왜 언제나 존재하는가?

??? success "풀이"
    코시 밀도는 $f(x) = 1/(\pi(1 + x^2))$이다.

    $M(t) = \int_{-\infty}^\infty e^{tx} / (\pi (1 + x^2)) dx$인데, $t \ne 0$이면 $e^{tx}$가 한쪽 방향으로 지수적으로 커져 피적분함수가 적분 가능하지 않다. 적률생성함수는 $t = 0$(자명하게 1)을 제외하면 정의되지 않는다.

    **특성함수(CF):** $\phi_X(t) = \mathbb{E}[e^{itX}]$. $|e^{itX}| = 1$이므로 $\mathbb{E}[|e^{itX}|] = 1 < \infty$가 되어 언제나 존재한다.

    코시분포의 경우 적률생성함수가 없는데도 $\phi(t) = e^{-|t|}$라는 깔끔한 닫힌 형태를 갖는다.

    **실무적 함의:** 꼬리가 두꺼운 분포를 다룰 때는 적률생성함수에서 특성함수로 갈아탄다. 대부분의 이론적 결과(레비 연속성 정리, 역변환 공식)는 더 넓은 부류를 포괄하는 특성함수로 진술된다. 적률생성함수는 모든 적률이 존재하는 분포, 즉 응용통계의 "얌전한" 다수에는 여전히 유용하다.

---

**연습문제 6.**
**적률생성함수가 분포를 결정한다**(유일성 정리) — 다만 0의 근방에서 존재할 때만 그렇다. 모든 차수의 적률이 일치하는 서로 다른 두 분포를 구성하라. (이것이 **적률 문제**의 실패다.)

??? success "풀이"
    $x > 0$에서 확률밀도함수가 $f_1(x) = \frac{1}{x\sqrt{2\pi}} e^{-(\ln x)^2/2}$인 **로그정규분포**와 그것을 변형한

    $$
    f_2(x) = f_1(x) \cdot (1 + \sin(2\pi \ln x))
    $$

    을 생각하자. 둘 다 타당한 확률밀도함수다(변형의 진폭을 음이 아니도록 잡았다).

    적률: $\mathbb{E}_2[X^k] = \int_0^\infty x^k f_1(x)(1 + \sin(2\pi \ln x)) dx = \mathbb{E}_1[X^k] + \int x^k f_1(x) \sin(2\pi \ln x) dx$.

    $u = \ln x$로 치환하면 두 번째 적분이 $\int e^{(k+1)u - u^2/2}\sin(2\pi u)/\sqrt{2\pi}\, du$가 된다. 이 적분은 *모든* 정수 $k$에 대해 0이다. 피적분함수가 가우시안 인자와 적분값이 0이 되는 사인 변조를 결합하기 때문이다. 따라서 모든 적률이 일치하지만 $f_1 \ne f_2$이다.

    **왜 이렇게 되는가:** 로그정규분포의 적률생성함수는 0의 어떤 근방에서도 수렴하지 *않는다*($t > 0$이면 적분 $\int e^{tx} f_1(x) dx$가 발산한다). 그래서 적률생성함수가 존재하지 않고 적률의 열이 분포를 결정하지 못한다.

    **교훈:** 적률생성함수가 열린구간에서 존재하면 적률이 분포를 유일하게 결정한다. 적률생성함수가 존재하지 않으면(두꺼운 꼬리, 로그정규) 적률만으로는 부족할 수 있으며 여러 분포가 모든 적률을 공유할 수 있다. 이것이 **스틸체스 적률 문제**다. 카를레만형 조건이 성립할 때에 한해 적률이 분포를 유일하게 결정한다.

## 정리하며

적률생성함수는 분포를 다른 언어로 옮겨 적은 것이다. 그 언어에서는 어려운 연산이 쉬워진다.

- **정리 1**은 적률을 미분으로 꺼냈다. $M_X^{(n)}(0) = E[X^n]$이며, 적분 없이 평균과 분산을 얻는다.
- **정리 2**는 적률생성함수가 분포를 **유일하게 결정**함을 밝혔다. 그래서 "적률생성함수를 계산해 표에서 찾는" 증명 전략이 가능해진다.
- **정리 3**은 독립인 합을 **곱**으로 바꾸었다. 밀도의 합성곱이라는 지저분한 적분이 단순한 곱셈이 된다.

세 성질이 함께 쓰이면 정규분포의 합이 정규분포임이 세 줄로 증명되고, 중심극한정리의 뼈대가 세워진다.

이것으로 3.4절이 끝난다. 분포를 요약하는 세 층위를 갖추었다. **평균**(중심), **분산**(퍼짐), 그리고 **적률생성함수**(전부).

이제 도구가 갖추어졌으니 확률론에서 가장 중요한 두 결과로 갈 차례다. 관측을 많이 모으면 무슨 일이 벌어지는가?

두 가지 일이 벌어진다. 표본평균이 참값으로 **모여들고**(큰수의 법칙), 그 모여드는 모양이 **정규분포**가 된다(중심극한정리). 3.5절의 주제이며, 5장 이후 이 책의 추론 전체가 이 두 정리 위에 서 있다.
