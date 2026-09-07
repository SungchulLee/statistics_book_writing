# 추정 방법 비교

## 개요

이 페이지에서는 점추정의 주요 접근인 **적률법(MoM)**과 **최대가능도추정(MLE)**을 비교한다. 해석적 유도와 Monte Carlo 모의실험을 함께 사용하여 여러 분포에 대해 각 방법의 편향, 분산, 평균제곱오차를 살펴본다. MLE가 언제 그리고 왜 적률법을 능가하는지, 그리고 그 이점의 계산 비용이 얼마인지 이해하는 것이 응용통계 실무의 중심이다.

## 적률법

적률법은 모집단 적률을 표본 적률과 같다고 두고 미지 모수를 푼다. 모수가 $\theta_1, \ldots, \theta_k$인 분포에 대해:

$$
\mu_r'(\theta_1, \ldots, \theta_k) = \frac{1}{n}\sum_{i=1}^n X_i^r, \quad r = 1, \ldots, k
$$

!!! info "장점과 단점"
    **장점:** 닫힌 형태의 단순한 표현. 최적화가 필요 없음. 온건한 조건 아래에서 언제나 일치함.

    **단점:** 모수공간 밖의 추정값을 낼 수 있음. 일반적으로 MLE보다 효율이 낮음. 가능도 정보 전체를 사용하지 않음.

## 최대가능도추정

MLE는 관측된 자료의 가능도를 최대화하는 모수값을 찾는다:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n f(x_i; \theta)
$$

실무에서는 로그가능도를 최대화한다:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log f(x_i; \theta)
$$

## 분산추정량의 비교

기초적인 비교로 크기 $n$인 정규 표본에서 모분산 $\sigma^2$을 추정하는 세 가지 추정량을 살펴본다:

| 추정량 | 분모 | 편향 | MSE |
|-----------|---------|------|-----|
| MLE $\hat{\sigma}^2_n$ | $n$ | $-\sigma^2/n$ | $\frac{2n-1}{n^2}\sigma^4$ |
| Bessel $S^2_{n-1}$ | $n-1$ | $0$ | $\frac{2}{n-1}\sigma^4$ |
| MSE 최적 $\hat{\sigma}^2_{n+1}$ | $n+1$ | $-\frac{2\sigma^2}{n+1}$ | $\frac{2}{n+1}\sigma^4$ |

다음 모의실험이 이 결과들을 경험적으로 확인해 준다.

```python
import numpy as np

def compare_variance_estimators(mu=5, sigma2=4, n=10, n_sim=50_000):
    sigma = np.sqrt(sigma2)
    rng = np.random.default_rng(42)

    results = {}
    samples = rng.normal(mu, sigma, (n_sim, n))
    ss = np.sum((samples - samples.mean(axis=1, keepdims=True)) ** 2, axis=1)

    estimators = {
        "MLE (n)":      ss / n,
        "Bessel (n-1)": ss / (n - 1),
        "MSE-opt (n+1)": ss / (n + 1),
    }

    for name, vals in estimators.items():
        bias = vals.mean() - sigma2
        var = vals.var()
        mse = np.mean((vals - sigma2) ** 2)
        results[name] = {"bias": bias, "var": var, "mse": mse}
        print(f"{name:18s}  bias={bias:+.4f}  var={var:.4f}  MSE={mse:.4f}")

    return results

compare_variance_estimators()
```

!!! note "핵심 관찰"
    불편추정량 $S^2_{n-1}$의 평균제곱오차가 셋 중 **가장 크다**. $n+1$로 나누면 편향이 생기지만 평균제곱오차가 최소가 되며, 편향–분산 맞바꿈을 잘 보여 준다.

## 축소추정량 시연

축소추정량 $\hat{\mu}_\lambda = \lambda \bar{X}$는 편향을 대가로 분산을 줄인다. 평균제곱오차는 다음과 같이 분해된다:

$$
\text{MSE}(\hat{\mu}_\lambda) = \lambda^2 \frac{\sigma^2}{n} + (1 - \lambda)^2 \mu^2
$$

평균제곱오차가 최적인 축소계수는:

$$
\lambda^* = \frac{\mu^2}{\mu^2 + \sigma^2/n}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def shrinkage_mse(mu_true=3, sigma2=4, n=20):
    lambdas = np.linspace(0.01, 1.5, 200)
    bias_sq = (lambdas - 1) ** 2 * mu_true ** 2
    variance = lambdas ** 2 * sigma2 / n
    mse = bias_sq + variance

    lambda_opt = mu_true ** 2 / (mu_true ** 2 + sigma2 / n)
    print(f"Optimal lambda = {lambda_opt:.4f}")
    print(f"MSE at lambda=1 (unbiased): {sigma2 / n:.4f}")
    print(f"MSE at lambda*:             {lambda_opt**2 * sigma2/n + (1 - lambda_opt)**2 * mu_true**2:.4f}")

shrinkage_mse()
```

## Normal 분포의 MLE

$X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에 대해 MLE는 닫힌 형태의 해를 갖는다:

$$
\hat{\mu}_{\text{MLE}} = \bar{X}, \qquad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

음의 로그가능도를 수치적으로 최적화하여 확인할 수도 있다:

$$
-\ell(\mu, \sigma^2) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

```python
import numpy as np
from scipy import optimize

def mle_normal_demo(n=100):
    rng = np.random.default_rng(42)
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)

    # Closed-form MLE
    mu_hat = data.mean()
    sigma2_hat = np.mean((data - mu_hat) ** 2)

    # Numerical MLE via optimization
    def neg_log_lik(params, x):
        mu, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)
        n = len(x)
        return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((x - mu) ** 2) / (2 * sigma2)

    result = optimize.minimize(neg_log_lik, x0=[0, 0], args=(data,), method="Nelder-Mead")
    mu_num, sigma2_num = result.x[0], np.exp(result.x[1])

    print(f"True:        mu = {mu_true:.4f}, sigma^2 = {sigma_true**2:.4f}")
    print(f"Closed-form: mu = {mu_hat:.4f}, sigma^2 = {sigma2_hat:.4f}")
    print(f"Numerical:   mu = {mu_num:.4f}, sigma^2 = {sigma2_num:.4f}")

mle_normal_demo()
```

## Gamma 분포에서 MLE와 적률법

$E[X] = \alpha\beta$이고 $\text{Var}(X) = \alpha\beta^2$인 $X \sim \text{Gamma}(\alpha, \beta)$에 대해 적률법 추정량은:

$$
\hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}, \qquad \hat{\beta}_{\text{MoM}} = \frac{S^2}{\bar{X}}
$$

MLE는 닫힌 형태가 없어 수치 최적화가 필요하다.

```python
import numpy as np
from scipy import stats

def mle_vs_mom_gamma(alpha_true=3, beta_true=2, n=200, n_sim=5000):
    rng = np.random.default_rng(42)
    mle_alpha, mom_alpha = [], []

    for _ in range(n_sim):
        data = rng.gamma(alpha_true, beta_true, n)

        # Method of Moments
        m1 = data.mean()
        v = data.var(ddof=0)
        mom_alpha.append(m1 ** 2 / v)

        # MLE (scipy)
        a_mle, _, _ = stats.gamma.fit(data, floc=0)
        mle_alpha.append(a_mle)

    mle_alpha = np.array(mle_alpha)
    mom_alpha = np.array(mom_alpha)

    for name, vals in [("MLE", mle_alpha), ("MoM", mom_alpha)]:
        bias = vals.mean() - alpha_true
        mse = np.mean((vals - alpha_true) ** 2)
        print(f"{name}: bias={bias:+.4f}, MSE={mse:.6f}")

mle_vs_mom_gamma()
```

!!! success "평균제곱오차에서 MLE의 승리"
    Gamma 분포에서 $\alpha$와 $\beta$ 모두에 대해 MLE의 평균제곱오차가 적률법보다 작다. 점근이론과 일치하는 결과이다. MLE는 (Cramér–Rao 한계를 달성하여) 효율적인 반면 적률법은 일반적으로 그렇지 않다.

## Cramér-Rao 하한 확인

Cramér-Rao 부등식은 임의의 불편추정량 $\hat{\theta}$에 대해 다음을 말한다:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

여기서 $I(\theta)$는 Fisher 정보량이다. $N(\mu, \sigma^2)$의 평균을 추정할 때 CRLB는 $\sigma^2/n$이고 표본평균이 이 한계를 정확히 달성한다.

```python
import numpy as np

def cramer_rao_demo(n=50, n_sim=20_000):
    rng = np.random.default_rng(42)
    mu_true, sigma = 5.0, 2.0
    crlb = sigma ** 2 / n

    means = np.array([rng.normal(mu_true, sigma, n).mean() for _ in range(n_sim)])
    medians = np.array([np.median(rng.normal(mu_true, sigma, n)) for _ in range(n_sim)])

    print(f"CRLB = sigma^2/n = {crlb:.6f}")
    print(f"Var(X_bar)       = {means.var():.6f}  (ratio to CRLB: {means.var()/crlb:.4f})")
    print(f"Var(median)      = {medians.var():.6f}  (ratio to CRLB: {medians.var()/crlb:.4f})")

cramer_rao_demo()
```

## 해석

모의실험은 여러 이론적 결과를 확인해 준다:

1. **편향–분산 맞바꿈은 실재한다**: 평균제곱오차가 최적인 분산추정량은 편향되어 있음에도 $n-1$이 아니라 $n+1$로 나눈다.
2. **MLE는 점근적으로 효율적이다**: Gamma 분포에서 이론이 예측한 대로 MLE가 적률법보다 낮은 평균제곱오차를 달성한다.
3. **축소가 도움이 될 수 있다**: 신호 대 잡음비 $\mu/(\sigma/\sqrt{n})$이 중간 정도일 때 표본평균을 0 쪽으로 당기면 평균제곱오차가 줄어든다.
4. **표본평균은 CRLB를 달성한다**: 정규분포 평균에 대해 그 분산이 Cramér-Rao 하한과 정확히 일치한다.

## 연습문제

**연습문제 1.** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$에 대해 $\lambda$의 적률법 추정량과 MLE를 유도하라. 둘은 같은가?

??? success "풀이"
    Exponential 분포는 $E[X] = 1/\lambda$이므로 적률법은 $\bar{X} = 1/\hat{\lambda}$로 두어 $\hat{\lambda}_{\text{MoM}} = 1/\bar{X}$를 준다.

    로그가능도는 $\ell(\lambda) = n\log\lambda - \lambda \sum x_i$이다. $\ell'(\lambda) = n/\lambda - \sum x_i = 0$으로 두면 $\hat{\lambda}_{\text{MLE}} = n/\sum x_i = 1/\bar{X}$이다.

    Exponential 분포에서 두 추정량은 동일하다. 모수가 하나이고 그것이 하나의 적률로 결정되기 때문이다. $\square$

---

**연습문제 2.** 정규모집단에서 $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n(X_i - \bar{X})^2$ 계열 중 평균제곱오차가 최적인 추정량이 $c^* = n+1$임을 보여라.

??? success "풀이"
    $Q = \sum(X_i - \bar{X})^2$이라 하자. $X_i \sim N(\mu, \sigma^2)$에서 $Q/\sigma^2 \sim \chi^2_{n-1}$이므로 $E[Q] = (n-1)\sigma^2$이고 $\text{Var}(Q) = 2(n-1)\sigma^4$이다.

    $Q/c$의 평균제곱오차는:

    $$
    \text{MSE}(Q/c) = \text{Var}(Q/c) + [\text{Bias}(Q/c)]^2 = \frac{2(n-1)\sigma^4}{c^2} + \left(\frac{n-1}{c} - 1\right)^2\sigma^4
    $$

    $c$에 대해 미분하여 0으로 두면:

    $$
    \frac{d}{dc}\text{MSE} = -\frac{4(n-1)\sigma^4}{c^3} - \frac{2(n-1)\sigma^4}{c^2}\left(\frac{n-1}{c} - 1\right) = 0
    $$

    정리하면 $-4(n-1)/c^3 + 2(n-1)(c - n + 1)/c^3 = 0$이므로 $2(c - n + 1) = 4$, 즉 $c = n + 1$이다. $\square$

---

**연습문제 3.** $\alpha = 2, \beta = 5$인 Beta 분포 $\text{Beta}(\alpha, \beta)$에서 표본크기 $n = 50$으로 MLE와 적률법을 비교하는 Monte Carlo 모의실험을 수행하라. $\alpha$를 추정할 때 어느 방법의 평균제곱오차가 더 작은가?

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    a_true, b_true, n, n_sim = 2, 5, 50, 10_000
    mle_a, mom_a = [], []

    for _ in range(n_sim):
        data = rng.beta(a_true, b_true, n)
        m1 = data.mean()
        m2 = np.mean(data ** 2)
        v = m2 - m1 ** 2
        common = m1 * (1 - m1) / v - 1
        mom_a.append(m1 * common)

        a_mle, b_mle, _, _ = stats.beta.fit(data, floc=0, fscale=1)
        mle_a.append(a_mle)

    mle_a, mom_a = np.array(mle_a), np.array(mom_a)
    print(f"MLE: MSE = {np.mean((mle_a - a_true)**2):.6f}")
    print(f"MoM: MSE = {np.mean((mom_a - a_true)**2):.6f}")
    ```

    대체로 MLE의 평균제곱오차가 더 작으며, 이는 점근 효율성과 일관된다. $\square$

---

**연습문제 4.** MLE가 재모수화에 불변임을 증명하라. 즉 $\hat{\theta}$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat{\theta})$가 $g(\theta)$의 MLE임을 보여라.

??? success "풀이"
    $g$가 일대일 함수일 때 $\eta = g(\theta)$라 하자(일반적인 경우는 유도가능도로 확장된다). $\eta$의 함수로 본 가능도는:

    $$
    L^*(\eta) = L(g^{-1}(\eta))
    $$

    $L(\theta)$가 $\hat{\theta}$에서 최대이므로 모든 $\theta$에 대해 $L^*(g(\hat{\theta})) = L(\hat{\theta}) \geq L(\theta)$이다. 따라서 $g$의 치역에 속하는 모든 $\eta$에 대해 $L^*(\eta) \leq L^*(g(\hat{\theta}))$이므로 $g(\hat{\theta})$가 $L^*$을 최대화한다.

    일대일이 아닐 수도 있는 일반적인 $g$에 대해서는 $\hat{\eta} = \sup_{\{\theta: g(\theta) = \eta\}} L(\theta)$로 정의하고 그 최대점을 취하면, 구성상 그것이 $g(\hat{\theta})$와 같다. $\square$

---

**연습문제 5.** Bernoulli 모수 $p$의 Fisher 정보량은 $I(p) = 1/[p(1-p)]$이다. $p = 0.3$, $n = 100$일 때 표본비율 $\hat{p} = \bar{X}$의 분산이 Cramér-Rao 한계 $1/[nI(p)]$를 달성함을 수치적으로 확인하라.

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    p, n, n_sim = 0.3, 100, 100_000
    p_hats = np.array([rng.binomial(n, p) / n for _ in range(n_sim)])

    crlb = p * (1 - p) / n
    empirical_var = p_hats.var()
    print(f"CRLB = p(1-p)/n = {crlb:.6f}")
    print(f"Var(p_hat)      = {empirical_var:.6f}")
    print(f"Ratio           = {empirical_var / crlb:.4f}")
    ```

    비가 1.0에 매우 가깝게 나와 $\hat{p}$가 CRLB를 달성하는 효율적 추정량임을 확인해 준다. $\square$
