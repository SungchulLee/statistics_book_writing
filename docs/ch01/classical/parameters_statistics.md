# 모수 대 통계량

## 개요

통계적 추론에서 가장 중요한 구분은 **모수(parameter)** — 모집단을 기술하는, 고정되어 있지만 알려지지 않은 양 — 와 **통계량(statistic)** — 표본에서 계산되어 그 모수의 추정값 역할을 하는 양 — 사이의 구분이다. 이 책의 모든 신뢰구간, 가설검정, 회귀계수가 이 구분 위에 서 있다. 이를 잊으면 응용 실무에서 가장 흔한 두 가지 오류를 범하게 된다. 하나는 표본에서 얻은 양을 마치 정확히 아는 값인 것처럼 다루는 것(추정 불확실성을 무시하는 것)이고, 다른 하나는 모수가 확률분포를 갖는 것처럼 다루는 것(베이즈 방법으로만 해소되는 빈도주의적 오류)이다.

## 정의

| 용어 | 범위 | 표기(대표적) | 알려져 있는가? |
|---|---|---|---|
| **모수** | 모집단 | $\mu,\; \sigma^2,\; p,\; \beta,\; \rho$ | 대개 미지 |
| **통계량** | 표본 | $\bar{x},\; s^2,\; \hat{p},\; \hat{\beta},\; r$ | 자료로부터 계산 가능 |

**모수**는 모집단의 고정된 수치적 특성이다. 예를 들어 NYSE 상장 주식 전체의 진짜 연평균 수익률이 그렇다. **통계량**은 표본에서 계산한 대응되는 양이다. 예를 들어 무작위로 고른 NYSE 주식 50종목의 평균 수익률이 그렇다. 모수는 알고 싶은 것이고, 통계량은 손에 쥔 것이다.

표준 관례는 모수에는 그리스 문자를, 그 추정량에는 로마자(흔히 모자 $\hat{}$ 를 씌워서)를 쓰는 것이다. 베이즈 방법은 모수를 사전분포를 갖는 확률변수로 다루어 이 관례를 완화하지만, 이 책의 대부분을 차지하는 빈도주의 관점에서 모수는 우리가 알아내려 하는 고정된 상수다.

## 이 구분이 중요한 이유

우리는 모집단 전체를 관측하는 일이 거의 없으므로, 모집단 모수를 **추정**하기 위해 표본 통계량에 의존한다. 그 추정의 품질은 다음으로 판단한다.

- **편향**: $\mathrm{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$. 0이 가장 좋다.
- **분산**: $\mathrm{Var}(\hat{\theta})$. 작을수록 좋으며, $n$과 모집단 분산이 이를 좌우한다.
- **평균제곱오차**: $\mathrm{MSE}(\hat{\theta}) = \mathrm{Var}(\hat{\theta}) + \mathrm{bias}(\hat{\theta})^2$ — 표준적인 스칼라 요약값.
- **일치성**: $n \to \infty$일 때 $\hat{\theta}_n \to \theta$ (확률수렴). 웬만한 추정량이라면 갖추어야 할 최소 요건이다.

이 성질들은 제6장에서 자세히 다룬다.

## 흔한 모수–통계량 짝

### 평균

$$
\text{Parameter: } \mu = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad\longleftrightarrow\qquad \text{Statistic: } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

표본이 모집단에서 i.i.d.로 뽑혔을 때 $\bar{x}$는 불편이고($\mathbb{E}[\bar{x}] = \mu$) 분산은 $\mathrm{Var}(\bar{x}) = \sigma^2 / n$이다. 표준오차 $\sigma / \sqrt{n}$은 초급 추론에서 가장 많이 쓰이는 양이다.

### 분산

$$
\text{Parameter: } \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad\longleftrightarrow\qquad \text{Statistic: } s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

표본분산이 $n - 1$로 나누는 것(베셀 보정)은 바로 $\mathbb{E}[s^2] = \sigma^2$가 되도록 하기 위해서다. $n$으로 나누면 $\sigma^2$을 $(n - 1)/n$배만큼 체계적으로 과소추정하게 되는데, 참값 $\mu$ 대신 (제곱편차의 합을 최소화하는 값인) $\bar{x}$를 대입함으로써 자유도 하나를 "써버렸기" 때문이다.

### 비율

$$
\text{Parameter: } p \qquad\longleftrightarrow\qquad \text{Statistic: } \hat{p} = \frac{\#\text{ successes in sample}}{n}
$$

이항 자료에서 $\hat{p}$는 불편이고 $\mathrm{Var}(\hat{p}) = p(1-p)/n$이며, $p = 1/2$에서 최대가 된다(비율 추정이 가장 어려운 경우).

### 회귀계수

$$
\text{Parameter: } \beta_1 \qquad\longleftrightarrow\qquad \text{Statistic: } \hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

표준 가정 아래에서 $\hat{\beta}_1$은 불편이고 분산은 $\sigma^2 / \sum(x_i - \bar{x})^2$이다. 분모를 보면 $x$가 더 넓게 퍼져 있을수록 기울기 추정이 더 정밀해지는 이유를 알 수 있는데, 이것이 실험 설계의 밑바탕에 있는 원리다.

## 표집 변동성

통계량은 무작위 표본에서 계산되므로 그 자체가 확률변수다. 다시 표집하면 다른 값이 나온다. 크기 $n$인 가능한 모든 표본에 걸친 통계량의 분포를 그 통계량의 **표본분포(sampling distribution)** 라 한다. 우리가 끊임없이 인용하게 될 두 가지 특징은 다음과 같다.

- **표준오차** — 표본분포의 표준편차. 평균의 경우 $\mathrm{SE}(\bar{x}) = \sigma/\sqrt{n}$.
- **표집편향** — 모수로부터의 체계적 어긋남으로, 흔히 비무작위 표집에서 생긴다.

중심극한정리에 따르면 $n$이 클 때 $\bar{x}$의 표본분포는 모집단의 모양과 무관하게 근사적으로 $N(\mu, \sigma^2/n)$이다. 이 하나의 사실이 대부분의 대표본 신뢰구간을 뒷받침한다.

## 파이썬 예제

```python
"""Illustrate the sampling distribution of the mean."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# === True population parameters ===
mu_true = 100
sigma_true = 15
population = rng.normal(mu_true, sigma_true, size=500_000)

# === Repeated sampling: compute sample means ===
n = 50
num_samples = 5_000
sample_means = np.array([
    rng.choice(population, n, replace=False).mean()
    for _ in range(num_samples)
])

# === The sampling distribution of x-bar ===
print(f"True μ:                {mu_true}")
print(f"Mean of sample means:  {sample_means.mean():.3f}")
print(f"Theoretical SE:        {sigma_true / np.sqrt(n):.3f}")
print(f"Observed SE:           {sample_means.std(ddof=1):.3f}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(sample_means, bins=40, density=True, alpha=0.7, edgecolor="black")
ax.axvline(mu_true, color="red", lw=2, label=fr"$\mu = {mu_true}$")
ax.set_xlabel("Sample mean")
ax.set_ylabel("Density")
ax.set_title(rf"Sampling distribution of $\bar{{x}}$ (n={n})")
ax.legend()
fig.tight_layout()
plt.show()
```

## 핵심 요약

- 모수는 모집단을 기술하고, 통계량은 표본을 기술한다.
- 통계량은 확률변수이며, 그 표본분포가 자료에서 추론으로 건너가는 다리다.
- 추정량의 유용성은 편향, 분산, 일치성으로 판단한다.
- 표준오차는 $1/\sqrt{n}$으로 줄어든다 — 표본 크기를 네 배로 늘려야 표준오차가 절반이 된다.

## 연습문제

**연습문제 1.**
다음 각 양을 **모수**인지 **통계량**인지 분류하라.

**(a)** 어떤 나라 모든 성인의 평균 키.
**(b)** 2,000가구를 조사해서 계산한 소득의 중앙값.
**(c)** 어느 공장 생산분 전체에서 불량품의 비율.
**(d)** 표본으로 뽑은 학생 50명의 시험점수 표준편차.
**(e)** 공정한 동전이 앞면이 나올 확률.
**(f)** 거래 1,000건의 자료로 적합한 회귀의 기울기 계수.

??? success "연습문제 1 풀이"
    (a) 모수 — 성인 모집단 전체를 기술한다.
    (b) 통계량 — 2,000가구 표본에서 계산했다.
    (c) 모수 — 생산분 전체를 기술한다.
    (d) 통계량 — 학생 50명에서 계산했다.
    (e) 모수 — 동전의 성질이다(개념적 반복의 "모집단").
    (f) 통계량 — $\hat{\beta}_1$은 모집단 모수 $\beta_1$의 추정량이다.

---

**연습문제 2.**
베셀 보정을 적용한 표본분산이 불편임을 보여라. 즉 $s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2$이고 $X_1, \dots, X_n$이 평균 $\mu$, 분산 $\sigma^2$의 i.i.d.일 때 $\mathbb{E}[s^2] = \sigma^2$임을 보여라.

??? success "연습문제 2 풀이"
    항등식 $\sum_i (X_i - \bar{X})^2 = \sum_i X_i^2 - n\bar{X}^2$을 이용한다. 기댓값을 취하면

    $$
    \mathbb{E}\!\left[\sum_i X_i^2\right] = n(\sigma^2 + \mu^2), \qquad \mathbb{E}[n\bar{X}^2] = n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2
    $$

    빼면

    $$
    \mathbb{E}\!\left[\sum_i (X_i - \bar{X})^2\right] = n\sigma^2 + n\mu^2 - \sigma^2 - n\mu^2 = (n-1)\sigma^2
    $$

    따라서 $\mathbb{E}[s^2] = \frac{(n-1)\sigma^2}{n-1} = \sigma^2$이다. $\square$

    대신 $n$으로 나누면 $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$가 되어 체계적으로 과소추정하는 추정량이 되며, 특히 $n$이 작을 때 해롭다.

---

**연습문제 3.**
분산이 $\sigma^2 = 100$인 모집단에서 크기 $n$인 i.i.d. 표본을 뽑을 때 $n = 4, 25, 100, 400$에 대해 $\bar X$의 표준오차를 계산하라. $n$이 네 배가 되면 표준오차는 어떻게 변하는가?

??? success "연습문제 3 풀이"
    $\mathrm{SE}(\bar X) = \sigma / \sqrt{n} = 10/\sqrt{n}$:

    | $n$ | $\mathrm{SE}(\bar X)$ |
    |---|---|
    | 4 | 5.000 |
    | 25 | 2.000 |
    | 100 | 1.000 |
    | 400 | 0.500 |

    $n$이 네 배가 될 때마다 표준오차는 절반이 된다 — 표준적인 $\sqrt{n}$ 속도다. 정밀도를 크게 높이려면 표본 크기를 *자릿수 단위로* 늘려야 하며, 이것이 통계적 표본 크기 계획의 기본 경제학이다.

---

**연습문제 4.**
어떤 조사에서 응답자 $n = 100$명으로부터 $\hat p = 0.40$을 얻었다. $\hat p$가 근사적으로 $N(p, p(1-p)/n)$을 따른다고 볼 때, "모수 $p$가 $(0.30, 0.50)$ 안에 있을 확률이 95%다"라는 진술이 빈도주의적 의미에서 올바른 해석인지 설명하라.

??? success "연습문제 4 풀이"
    빈도주의적 의미에서는 옳지 않다. 빈도주의 추론에서 $p$는 (미지이지만) 고정된 상수이지 확률변수가 아니므로, "$p$가 확률 0.95로 $(a, b)$ 안에 있다" 같은 확률 진술은 적용되지 않는다.

    95% 신뢰구간의 올바른 해석은 *절차*에 대한 진술이다. 표집과 신뢰구간 구성을 여러 번 반복하면 그렇게 얻은 구간의 95%가 참값 $p$를 포함한다는 뜻이다. 특정 구간 $(0.30, 0.50)$은 $p$를 포함하거나 포함하지 않거나 둘 중 하나이며, $p$ 자체에 대한 확률 진술은 없다.

    베이즈 추론에서는 모수에 대한 확률 진술을 할 수 *있지만*, 사전분포를 지정하고 사후분포를 보고한 뒤에야 가능하며 이는 다른 방식의 추론이다. 두 해석은 대중 보도에서 자주 혼동된다.

---

**연습문제 5.**
$\mathrm{Bernoulli}(p)$에서 뽑은 크기 $n$의 i.i.d. 표본에 대해 표본비율은 $\hat p = (1/n)\sum X_i$이다. $\mathbb{E}[\hat p] = p$이고 $\mathrm{Var}(\hat p) = p(1-p)/n$임을 보여라. 분산은 ($p$에 대해) 어디에서 최대가 되며, 이것이 표본 크기 계획에서 왜 중요한가?

??? success "연습문제 5 풀이"
    기댓값의 선형성: $\mathbb{E}[\hat p] = (1/n)\sum \mathbb{E}[X_i] = p$.

    독립성: $\mathrm{Var}(\hat p) = (1/n^2)\sum \mathrm{Var}(X_i) = (1/n^2) \cdot n\, p(1-p) = p(1-p)/n$.

    함수 $p \mapsto p(1-p)$는 위로 볼록한 포물선으로 $p = 1/2$에서 최대가 된다($p(1-p) = 0.25$). 표본 크기를 계획할 때 $p$를 모르면 보통 최악의 경우인 $p(1-p) = 0.25$를 택하며, 95% 신뢰수준에서 목표 오차한계 ME에 대해 익숙한 공식 $n \approx 1/(4 \mathrm{ME}^2)$이 나온다.

---

**연습문제 6.**
**평균제곱오차**는 $\mathrm{MSE}(\hat \theta) = \mathrm{Var}(\hat \theta) + [\mathrm{bias}(\hat \theta)]^2$로 분해된다. 불편추정량보다 MSE가 작은 편향추정량의 예를 들고, 이런 일이 왜 가능한지 설명하라.

??? success "연습문제 6 풀이"
    정규 i.i.d. 표본에서 $\sigma^2$을 추정하는 경우를 생각하자. 최대가능도추정량(MLE)은 $n - 1$이 아니라 $n$으로 나눈다.

    $$
    \tilde s^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
    $$

    이는 편향되어 있다: $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$. 그러나 분산은 불편추정량 $s^2$보다 작다. 계산하면

    $$
    \mathrm{MSE}(\tilde s^2) = \mathrm{Var}(\tilde s^2) + \mathrm{bias}(\tilde s^2)^2 = \frac{2(n-1)\sigma^4}{n^2} + \left(\frac{\sigma^2}{n}\right)^{\!2} = \frac{(2n-1)\sigma^4}{n^2}
    $$

    이고,

    $$
    \mathrm{MSE}(s^2) = \mathrm{Var}(s^2) = \frac{2\sigma^4}{n-1} = \frac{2\sigma^4 n}{n(n-1)}
    $$

    이다. 유한한 $n$에 대해 $\mathrm{MSE}(\tilde s^2) < \mathrm{MSE}(s^2)$ — MLE는 편향되어 있음에도 MSE가 더 작다.

    일반적인 교훈은 이것이다: **편향은 분산과 맞바꿔진다.** 축소추정량(제임스–스타인, 능형회귀)은 약간의 편향을 감수하고 분산을 크게 줄임으로써 이 사실을 활용한다. 편향이 언제나 나쁜 것은 아니며, 2차원 정확도 예산에서 조절할 수 있는 하나의 손잡이일 뿐이다.
