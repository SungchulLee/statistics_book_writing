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

<div class="codebox" markdown>

### 예제 1. 분산추정량 셋의 MSE 비교 { .eg }

```python
import numpy as np

def compare_variance_estimators(mu=5, sigma2=4, n=10, n_sim=50_000):
    sigma = np.sqrt(sigma2)
    rng = np.random.default_rng(42)

    results = {}
    # 크기 n인 표본을 5만 개 만든다. 행 하나가 표본 하나다.
    samples = rng.normal(mu, sigma, (n_sim, n))

    # 제곱합 SS = sum (x_i - x_bar)^2 을 표본마다 구한다.
    # keepdims=True 로 (n_sim, 1) 모양을 유지해야 브로드캐스팅으로 빼진다.
    # 세 추정량은 이 SS 를 **무엇으로 나누는가**만 다르다.
    ss = np.sum((samples - samples.mean(axis=1, keepdims=True)) ** 2, axis=1)

    estimators = {
        # n으로 나눔: 최대가능도추정량. 아래로 편향된다(과소추정).
        "MLE (n)":      ss / n,
        # n-1로 나눔: 베셀 보정. 편향이 정확히 0이 된다.
        "Bessel (n-1)": ss / (n - 1),
        # n+1로 나눔: 편향은 더 커지지만 분산이 더 줄어 MSE가 최소가 된다.
        "MSE-opt (n+1)": ss / (n + 1),
    }

    # 아래 세 값이 MSE = 편향^2 + 분산 을 이룬다.
    # "불편이 언제나 최선은 아니다"가 이 표의 요점이다.
    for name, vals in estimators.items():
        bias = vals.mean() - sigma2
        var = vals.var()
        mse = np.mean((vals - sigma2) ** 2)
        results[name] = {"bias": bias, "var": var, "mse": mse}
        print(f"{name:18s}  bias={bias:+.4f}  var={var:.4f}  MSE={mse:.4f}")

    return results

compare_variance_estimators()
```

출력:

```
MLE (n)             bias=-0.3949  var=2.9106  MSE=3.0665
Bessel (n-1)        bias=+0.0057  var=3.5933  MSE=3.5934
MSE-opt (n+1)       bias=-0.7226  var=2.4054  MSE=2.9276
```

</div>

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

<div class="codebox" markdown>

### 예제 2. 축소추정량의 MSE { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

def shrinkage_mse(mu_true=3, sigma2=4, n=20):
    """축소추정량 lambda * x_bar 의 MSE를 lambda의 함수로 본다.

    lambda = 1 이면 보통의 표본평균(불편),
    lambda < 1 이면 추정값을 0 쪽으로 끌어당긴다(편향되지만 분산이 준다).
    """
    lambdas = np.linspace(0.01, 1.5, 200)

    # MSE = 편향^2 + 분산.  E[lambda * x_bar] = lambda * mu 이므로
    #   편향 = (lambda - 1) * mu
    #   분산 = lambda^2 * sigma^2/n
    bias_sq = (lambdas - 1) ** 2 * mu_true ** 2
    variance = lambdas ** 2 * sigma2 / n
    mse = bias_sq + variance

    # MSE를 lambda에 대해 미분해 0으로 두면 이 값이 나온다.
    # 분모가 분자보다 크므로 언제나 lambda* < 1 이다.
    # 즉 **불편추정량(lambda=1)은 결코 MSE 최소가 아니다.**
    # 다만 이 최적값은 미지의 mu에 의존하므로 실제로 쓸 수는 없다.
    lambda_opt = mu_true ** 2 / (mu_true ** 2 + sigma2 / n)
    print(f"Optimal lambda = {lambda_opt:.4f}")
    print(f"MSE at lambda=1 (unbiased): {sigma2 / n:.4f}")
    print(f"MSE at lambda*:             {lambda_opt**2 * sigma2/n + (1 - lambda_opt)**2 * mu_true**2:.4f}")

shrinkage_mse()
```

출력:

```
Optimal lambda = 0.9783
MSE at lambda=1 (unbiased): 0.2000
MSE at lambda*:             0.1957
```

</div>

## 정규분포의 MLE

$X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에 대해 MLE는 닫힌 형태의 해를 갖는다:

$$
\hat{\mu}_{\text{MLE}} = \bar{X}, \qquad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

음의 로그가능도를 수치적으로 최적화하여 확인할 수도 있다:

$$
-\ell(\mu, \sigma^2) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

<div class="codebox" markdown>

### 예제 3. 정규분포 모수의 최대가능도추정 { .eg }

```python
import numpy as np
from scipy import optimize

def mle_normal_demo(n=100):
    rng = np.random.default_rng(42)
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)

    # 닫힌 형태의 MLE — 공식으로 바로 구한다.
    mu_hat = data.mean()
    sigma2_hat = np.mean((data - mu_hat) ** 2)

    # 수치 최적화로 구한 MLE. 위 공식과 같은 값이 나와야 한다.
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

출력:

```
True:        mu = 5.0000, sigma^2 = 4.0000
Closed-form: mu = 4.8995, sigma^2 = 2.3888
Numerical:   mu = 4.8995, sigma^2 = 2.3889
```

</div>

## 감마분포에서 MLE와 적률법

$E[X] = \alpha\beta$이고 $\text{Var}(X) = \alpha\beta^2$인 $X \sim \text{Gamma}(\alpha, \beta)$에 대해 적률법 추정량은:

$$
\hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}, \qquad \hat{\beta}_{\text{MoM}} = \frac{S^2}{\bar{X}}
$$

MLE는 닫힌 형태가 없어 수치 최적화가 필요하다.

<div class="codebox" markdown>

### 예제 4. 감마분포에서 MLE와 적률법 비교 { .eg }

```python
import numpy as np
from scipy import stats

def mle_vs_mom_gamma(alpha_true=3, beta_true=2, n=200, n_sim=5000):
    rng = np.random.default_rng(42)
    mle_alpha, mom_alpha = [], []

    for _ in range(n_sim):
        data = rng.gamma(alpha_true, beta_true, n)

        # 적률법: 표본의 1차·2차 적률을 이론값과 맞춘다.
        # E[X] = a*b, Var(X) = a*b^2 이므로 a = E[X]^2 / Var(X) 다.
        # 닫힌 식이라 계산이 즉시 끝나는 것이 장점이다.
        m1 = data.mean()
        v = data.var(ddof=0)
        mom_alpha.append(m1 ** 2 / v)

        # MLE: 감마분포는 닫힌 해가 없어 scipy가 수치적으로 푼다.
        # floc=0 은 위치모수를 0으로 **고정**한다는 뜻이다.
        # 이것을 빼면 scipy가 위치까지 추정해 모수가 셋이 되고,
        # 적률법과 같은 조건이 아니게 되어 비교가 성립하지 않는다.
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

출력:

```
MLE: bias=+0.0399, MSE=0.088900
MoM: bias=+0.0616, MSE=0.129929
```

</div>

!!! success "평균제곱오차에서 MLE의 승리"
    감마분포에서 $\alpha$와 $\beta$ 모두에 대해 MLE의 평균제곱오차가 적률법보다 작다. 점근이론과 일치하는 결과이다. MLE는 (Cramér–Rao 한계를 달성하여) 효율적인 반면 적률법은 일반적으로 그렇지 않다.

## Cramér-Rao 하한 확인

Cramér-Rao 부등식은 임의의 불편추정량 $\hat{\theta}$에 대해 다음을 말한다:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

여기서 $I(\theta)$는 Fisher 정보량이다. $N(\mu, \sigma^2)$의 평균을 추정할 때 CRLB는 $\sigma^2/n$이고 표본평균이 이 한계를 정확히 달성한다.

<div class="codebox" markdown>

### 예제 5. 크라메르-라오 하한 확인 { .eg }

```python
import numpy as np

def cramer_rao_demo(n=50, n_sim=20_000):
    rng = np.random.default_rng(42)
    mu_true, sigma = 5.0, 2.0
    # 정규분포 평균에 대한 크라메르-라오 하한
    crlb = sigma ** 2 / n

    # 같은 모수를 추정하는 두 불편추정량을 비교한다.
    #   표본평균  : 하한을 달성한다 (비율 ≈ 1.00). 효율적이다.
    #   표본중앙값: 하한보다 분산이 크다 (비율 ≈ 1.52). 정보를 버린 셈이다.
    # 정규모집단에서 중앙값의 점근 상대효율은 2/pi ≈ 0.637 이고,
    # 그 역수 pi/2 ≈ 1.571 이 아래 비율과 맞아떨어진다.
    #
    # 그렇다고 중앙값이 나쁜 것은 아니다. 이상치가 섞이면 순위가 뒤바뀐다.
    # 효율성은 **모형이 맞다는 전제 아래에서의** 성적이다.
    means = np.array([rng.normal(mu_true, sigma, n).mean() for _ in range(n_sim)])
    medians = np.array([np.median(rng.normal(mu_true, sigma, n)) for _ in range(n_sim)])

    print(f"CRLB = sigma^2/n = {crlb:.6f}")
    print(f"Var(X_bar)       = {means.var():.6f}  (ratio to CRLB: {means.var()/crlb:.4f})")
    print(f"Var(median)      = {medians.var():.6f}  (ratio to CRLB: {medians.var()/crlb:.4f})")

cramer_rao_demo()
```

출력:

```
CRLB = sigma^2/n = 0.080000
Var(X_bar)       = 0.078777  (ratio to CRLB: 0.9847)
Var(median)      = 0.121813  (ratio to CRLB: 1.5227)
```

</div>

## 해석

모의실험은 여러 이론적 결과를 확인해 준다:

1. **편향–분산 맞바꿈은 실재한다**: 평균제곱오차가 최적인 분산추정량은 편향되어 있음에도 $n-1$이 아니라 $n+1$로 나눈다.
2. **MLE는 점근적으로 효율적이다**: 감마분포에서 이론이 예측한 대로 MLE가 적률법보다 낮은 평균제곱오차를 달성한다.
3. **축소가 도움이 될 수 있다**: 신호 대 잡음비 $\mu/(\sigma/\sqrt{n})$이 중간 정도일 때 표본평균을 0 쪽으로 당기면 평균제곱오차가 줄어든다.
4. **표본평균은 CRLB를 달성한다**: 정규분포 평균에 대해 그 분산이 Cramér-Rao 하한과 정확히 일치한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$에 대해 $\lambda$의 적률법 추정량과 MLE를 유도하라. 둘은 같은가?

</div>

??? success "풀이"
    지수분포는 $E[X] = 1/\lambda$이므로 적률법은 $\bar{X} = 1/\hat{\lambda}$로 두어 $\hat{\lambda}_{\text{MoM}} = 1/\bar{X}$를 준다.

    로그가능도는 $\ell(\lambda) = n\log\lambda - \lambda \sum x_i$이다. $\ell'(\lambda) = n/\lambda - \sum x_i = 0$으로 두면 $\hat{\lambda}_{\text{MLE}} = n/\sum x_i = 1/\bar{X}$이다.

    지수분포에서 두 추정량은 동일하다. 모수가 하나이고 그것이 하나의 적률로 결정되기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span> 정규모집단에서 $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n(X_i - \bar{X})^2$ 계열 중 평균제곱오차가 최적인 추정량이 $c^* = n+1$임을 보여라.

</div>

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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\alpha = 2, \beta = 5$인 베타분포 $\text{Beta}(\alpha, \beta)$에서 표본크기 $n = 50$으로 MLE와 적률법을 비교하는 Monte Carlo 모의실험을 수행하라. $\alpha$를 추정할 때 어느 방법의 평균제곱오차가 더 작은가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    a_true, b_true, n, n_sim = 2, 5, 50, 10_000
    mle_a, mom_a = [], []

    for _ in range(n_sim):
        data = rng.beta(a_true, b_true, n)

        # 적률법: 베타분포의 평균과 분산을 표본값과 맞추어 푼다.
        #   E[X] = a/(a+b),  Var(X) = ab / [(a+b)^2 (a+b+1)] 를 뒤집어 푼다.
        # 이를 a, b에 대해 풀면 공통 인자 [E(1-E)/V - 1] 이 나오고
        #   a = E * common,  b = (1-E) * common 이 그 해다.
        m1 = data.mean()
        m2 = np.mean(data ** 2)
        v = m2 - m1 ** 2
        common = m1 * (1 - m1) / v - 1
        mom_a.append(m1 * common)

        # MLE. floc=0, fscale=1 로 지지구간을 [0,1]에 고정한다.
        # 베타분포의 표준 정의가 [0,1] 위이므로 이것이 맞는 설정이며,
        # 고정하지 않으면 scipy가 구간의 양 끝까지 추정하려 든다.
        a_mle, b_mle, _, _ = stats.beta.fit(data, floc=0, fscale=1)
        mle_a.append(a_mle)

    mle_a, mom_a = np.array(mle_a), np.array(mom_a)
    print(f"MLE: MSE = {np.mean((mle_a - a_true)**2):.6f}")
    print(f"MoM: MSE = {np.mean((mom_a - a_true)**2):.6f}")
    ```

    출력:

    ```
    MLE: MSE = 0.183749
    MoM: MSE = 0.213541
    ```

    대체로 MLE의 평균제곱오차가 더 작으며, 이는 점근 효율성과 일관된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> MLE가 재모수화에 불변임을 증명하라. 즉 $\hat{\theta}$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat{\theta})$가 $g(\theta)$의 MLE임을 보여라.

</div>

??? success "풀이"
    $g$가 일대일 함수일 때 $\eta = g(\theta)$라 하자(일반적인 경우는 유도가능도로 확장된다). $\eta$의 함수로 본 가능도는:

    $$
    L^*(\eta) = L(g^{-1}(\eta))
    $$

    $L(\theta)$가 $\hat{\theta}$에서 최대이므로 모든 $\theta$에 대해 $L^*(g(\hat{\theta})) = L(\hat{\theta}) \geq L(\theta)$이다. 따라서 $g$의 치역에 속하는 모든 $\eta$에 대해 $L^*(\eta) \leq L^*(g(\hat{\theta}))$이므로 $g(\hat{\theta})$가 $L^*$을 최대화한다.

    일대일이 아닐 수도 있는 일반적인 $g$에 대해서는 $\hat{\eta} = \sup_{\{\theta: g(\theta) = \eta\}} L(\theta)$로 정의하고 그 최대점을 취하면, 구성상 그것이 $g(\hat{\theta})$와 같다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 베르누이 모수 $p$의 Fisher 정보량은 $I(p) = 1/[p(1-p)]$이다. $p = 0.3$, $n = 100$일 때 표본비율 $\hat{p} = \bar{X}$의 분산이 Cramér-Rao 한계 $1/[nI(p)]$를 달성함을 수치적으로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    p, n, n_sim = 0.3, 100, 100_000
    # 크기 100짜리 표본의 표본비율을 10만 번 만든다.
    # binomial(n, p)가 성공 횟수를 주므로 n으로 나누면 비율이 된다.
    p_hats = np.array([rng.binomial(n, p) / n for _ in range(n_sim)])

    # 크라메르-라오 하한. 불편추정량이 가질 수 있는 분산의 이론적 최솟값이다.
    # 베르누이의 피셔정보가 I(p) = 1/[p(1-p)] 이므로 1/[n I(p)] = p(1-p)/n.
    # 아래 비율이 1.00 에 가까우면 표본비율이 이 한계를 **달성**한다는 뜻이며,
    # 그런 추정량을 효율적(efficient)이라고 부른다.
    crlb = p * (1 - p) / n
    empirical_var = p_hats.var()
    print(f"CRLB = p(1-p)/n = {crlb:.6f}")
    print(f"Var(p_hat)      = {empirical_var:.6f}")
    print(f"Ratio           = {empirical_var / crlb:.4f}")
    ```

    출력:

    ```
    CRLB = p(1-p)/n = 0.002100
    Var(p_hat)      = 0.002103
    Ratio           = 1.0016
    ```

    비가 1.0에 매우 가깝게 나와 $\hat{p}$가 CRLB를 달성하는 효율적 추정량임을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
적률법, 최대가능도, 최소제곱, 베이즈 네 가지 추정 방법을 **무엇을 최적화하는가**의 관점에서 한 표로 정리하고, 각각이 필요로 하는 가정을 적어라.

</div>

??? success "풀이"

    | 방법 | 최적화 대상 | 필요한 가정 |
    |---|---|---|
    | 적률법 | 표본적률 = 모형적률 (방정식 풀이) | 적률이 존재하고 모수를 결정할 것 |
    | 최대가능도 | $\ell(\theta) = \sum\ln f(x_i;\theta)$ 최대화 | **분포 전체**를 지정 |
    | 최소제곱 | $\sum(y_i-f(x_i;\theta))^2$ 최소화 | 평균 구조만 지정, 등분산이면 효율적 |
    | 베이즈 | 사후위험 최소화(손실에 따라 평균·중앙값·최빈값) | 분포 전체 + **사전분포** |

    **가정의 강도 순서.** 적률법·최소제곱 < 최대가능도 < 베이즈.

    적률법과 최소제곱은 **분포를 지정하지 않는다.** 처음 몇 개의 적률이나 평균 구조만 맞으면 된다. 그래서 모형 오설정에 강건하지만 효율을 잃는다.

    최대가능도는 분포 전체를 쓰므로 **가정이 맞으면 가장 효율적**이고, 틀리면 유사참값으로 수렴한다.

    베이즈는 사전분포까지 요구하지만, 그 대가로 소표본에서의 안정성과 불확실성의 완결된 표현(사후분포)을 얻는다.

    **서로 겹치는 지점.**

    - **정규 오차 아래에서 최소제곱 = 최대가능도**다.
    - **일모수 지수족에서 적률법 = 최대가능도**인 경우가 많다(지수, 포아송, 베르누이).
    - **평평한 사전분포에서 MAP = 최대가능도**다.
    - **$n\to\infty$이면 베이즈 사후평균과 MLE가 같아진다**(베른슈타인-폰 미제스).

    **고르는 기준.** 분포를 믿을 수 있고 표본이 적당하면 최대가능도, 분포가 미덥지 않으면 적률법·최소제곱에 강건 표준오차, 표본이 아주 작거나 사전 정보가 있으면 베이즈가 자연스러운 선택이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
같은 자료에 네 방법을 적용했더니 추정값이 서로 꽤 달랐다. 이것이 문제인지 판단하는 절차를 적어라.

</div>

??? success "풀이"
    **먼저 나눠 볼 것.** 차이가 **표집변동 수준인가, 그 이상인가.**

    각 추정값의 표준오차를 계산해 차이가 표준오차의 몇 배인지 본다. $\hat\theta_1-\hat\theta_2$가 각각의 표준오차보다 작으면 우연으로 설명된다. 두 추정량이 상관되어 있으므로 차의 표준오차를 정확히 구하려면 부트스트랩으로 함께 재표집하는 것이 낫다.

    **차이가 크다면 원인을 좁힌다.**

    1. **모형이 틀렸는가.** 적률법과 최대가능도가 크게 다르면 **분포 가정이 의심스럽다**는 강한 신호다. 모형이 맞으면 둘 다 같은 참값으로 수렴하기 때문이다. 이 아이디어를 형식화한 것이 **하우스만 검정**이다.

       $$
       H = \frac{(\hat\theta_{\text{eff}}-\hat\theta_{\text{rob}})^2}{\operatorname{Var}(\hat\theta_{\text{rob}})-\operatorname{Var}(\hat\theta_{\text{eff}})}\ \sim\ \chi^2
       $$

       효율적이지만 가정에 민감한 추정량과, 덜 효율적이지만 강건한 추정량을 비교한다.

    2. **이상치가 있는가.** 최대가능도(정규 가정)만 크게 움직이면 몇몇 관측값이 끌고 있을 가능성이 높다. 관측값을 하나씩 빼 보는 영향력 진단을 한다.

    3. **사전분포가 지배하는가.** 베이즈 추정값만 다르면 사전분포의 영향이다. 사전분포를 바꿔 가며 사후분포가 얼마나 움직이는지 보는 **민감도 분석**이 필요하다. 자료가 적을수록 이 영향이 크다.

    4. **최적화가 수렴했는가.** 최대가능도만 이상하면 국소해에 빠졌을 수 있다. 여러 초기값에서 다시 돌린다.

    **결론을 내는 법.** 원인을 특정했으면 그에 맞는 방법을 고르고 **왜 골랐는지 밝힌다.** 특정하지 못했다면 여러 추정값을 모두 보고하는 것이 정직하다. "모형 선택에 따라 결과가 $x$에서 $y$까지 달라진다"는 진술이 하나의 값을 자신 있게 보고하는 것보다 나은 경우가 많다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**M-추정량**이 위 네 방법 중 셋을 하나의 틀로 묶는다. $\hat\theta = \arg\min_\theta\sum_i\rho(x_i;\theta)$라는 정의에서 $\rho$를 어떻게 고르면 각각이 나오는지 보이고, 이 틀의 장점을 적어라.

</div>

??? success "풀이"
    **$\rho$의 선택.**

    | $\rho(x;\theta)$ | 얻어지는 추정량 |
    |---|---|
    | $-\ln f(x;\theta)$ | 최대가능도 |
    | $(x-\theta)^2$ | 최소제곱(평균) |
    | $|x-\theta|$ | 중앙값 |
    | $\rho_\tau$ (핀볼 손실) | $\tau$ 분위수 |
    | 후버 $\rho$ | 강건 위치 추정 |
    | $-\ln f - \ln\pi(\theta)/n$ | MAP |

    적률법도 추정방정식 $\sum_i\psi(x_i;\theta)=0$ 꼴로 쓰면 같은 틀(Z-추정량)에 들어온다.

    **틀의 장점.**

    1. **점근이론을 한 번만 세우면 된다.** 정칙 조건 아래에서

       $$
       \sqrt n(\hat\theta-\theta_0) \xrightarrow{d} N\!\left(0,\ A^{-1}BA^{-1}\right)
       $$

       이고 $A = E[\psi'], B = E[\psi^2]$이다($\psi = \partial\rho/\partial\theta$). **샌드위치 형태가 기본**이고, 최대가능도에서만 $A=B$가 되어 $I^{-1}$로 줄어든다.

    2. **강건성을 설계할 수 있다.** $\psi$가 유계이면 이상치의 영향이 제한된다. 영향함수가 $\psi$에 비례하므로, **원하는 강건성을 $\psi$의 모양으로 직접 지정**할 수 있다.

    3. **모형 오설정을 자연스럽게 다룬다.** $\rho$가 로그가능도가 아니어도 이론이 그대로 적용된다. 유사최대가능도, 준가능도, 추정방정식이 모두 특수한 경우다.

    4. **계산이 통일된다.** 대부분 IRLS나 뉴턴류로 풀리고, 표준오차도 같은 공식으로 나온다.

    **대가.** 일반성을 얻는 대신 **효율의 최적성을 잃는다.** 모형이 정확히 맞으면 최대가능도가 최선이고, M-추정량은 그보다 못하다. 그 차이가 강건성의 가격이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
예제 1의 분산추정량 세 가지($n$, $n-1$, $n+1$로 나누는 것)를 **정규가 아닌 모집단**에서 비교하면 순위가 달라질 수 있다. 왜 그런지 설명하고 어떻게 확인할지 적어라.

</div>

??? success "풀이"
    **정규 가정에 기댄 결론.** $c^*=n+1$이 최적이라는 결과는 $Q/\sigma^2\sim\chi^2_{n-1}$, 즉 $\operatorname{Var}(Q)=2(n-1)\sigma^4$을 썼다. 이 관계는 **정규모집단에서만** 성립한다.

    **일반 모집단.** 앞서 본 대로

    $$
    \operatorname{Var}(S^2) \approx \frac{\sigma^4}{n}\left(\gamma_2+2\right)
    $$

    이고 $\gamma_2$는 초과첨도다. 같은 계산을 반복하면 최적 $c$가

    $$
    c^* \approx n + 1 + \frac{n\gamma_2}{2}
    $$

    꼴로 **첨도에 의존**한다.

    - **$\gamma_2 > 0$(두꺼운 꼬리)**: $c^*$가 커진다. 분산 추정값을 더 많이 축소해야 한다. 분산 추정 자체가 불안정하므로 편향을 더 감수하는 것이 이득이기 때문이다.
    - **$\gamma_2 < 0$(균등분포 등)**: $c^*$가 작아진다. $\gamma_2 = -1.2$이고 $n=10$이면 $c^* \approx 11-6 = 5$로, $n-1=9$보다도 작다.

    **순위가 뒤집힐 수 있다.** 두꺼운 꼬리에서는 $n+1$도 부족하고, 가벼운 꼬리에서는 $n-1$이 오히려 나을 수 있다.

    **확인 방법.**

    ```python
    rng = np.random.default_rng(0)
    n, B = 10, 50_000
    samplers = {
        "정규": lambda: rng.normal(0, 1, n),
        "균등": lambda: rng.uniform(-np.sqrt(3), np.sqrt(3), n),
        "t5":   lambda: rng.standard_t(5, n) / np.sqrt(5 / 3),
    }
    for name, draw in samplers.items():
        Q = np.empty(B)
        for b in range(B):
            x = draw()
            Q[b] = ((x - x.mean()) ** 2).sum()
        for c in [n - 1, n, n + 1]:
            print(name, c, np.mean((Q / c - 1) ** 2))   # 참 분산을 1로 맞춰 두었다
    ```

    **핵심은 세 모집단의 분산을 1로 맞춰 두는 것**이다. 그래야 MSE를 직접 비교할 수 있다. $t_5$는 분산이 $5/3$이므로 그 제곱근으로 나눈다.

    **교훈.** 교재에서 "$n+1$이 최적"이라고 배우는 결과는 **정규모집단이라는 조건부 진술**이다. 이 조건을 밝히지 않고 인용하는 것이 흔한 오류이며, 실제 자료에서 꼬리가 두꺼우면 결론이 달라진다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
추정 방법을 모의실험으로 비교할 때 저지르기 쉬운 실수를 세 가지 들고 각각의 대처법을 적어라.

</div>

??? success "풀이"
    **(1) 한 가지 참값에서만 비교한다.** $\theta=1$에서 방법 A가 낫다고 모든 $\theta$에서 그런 것이 아니다. 축소추정량은 참값이 축소 목표 근처면 좋고 멀면 나쁘다.

    → **$\theta$를 격자로 바꿔 가며 MSE 곡선을 그린다.** 곡선이 교차하면 그 사실 자체를 보고한다.

    **(2) 모의실험 오차를 무시한다.** $B=1000$에서 MSE가 $0.021$ 대 $0.019$로 나왔다고 후자가 낫다고 할 수 없다. MSE 추정값의 상대 표준오차가 대략 $\sqrt{2/B} = 4.5\%$이므로 이 차이는 잡음 안에 있다.

    → **$B$를 충분히 키우고**(최소 10,000) MSE 추정값의 표준오차를 함께 보고한다. 더 나은 방법은 **같은 난수 흐름으로 두 방법을 평가**하는 것이다(공통난수). 차이의 분산이 크게 줄어 훨씬 적은 $B$로 구별할 수 있다.

    **(3) 유리한 조건에서만 본다.** 새 방법의 가정이 정확히 성립하는 자료만 생성하면 당연히 이긴다.

    → **가정을 깨뜨려 가며 본다.** 이상치를 섞고, 분포를 바꾸고, 표본크기를 줄이고, 모형을 오설정한다. 실무의 관심사는 "가정이 맞을 때 얼마나 좋은가"보다 **"가정이 틀렸을 때 얼마나 나빠지는가"**인 경우가 많다.

    **그 밖에 흔한 실수.**

    - **평균만 본다.** 꼬리가 두꺼운 추정량은 평균 MSE가 몇 번의 큰 실패에 지배된다. 오차의 중앙값·분위수·상자그림을 함께 본다.
    - **정의되지 않는 경우를 조용히 버린다.** 앞서 본 $t=0$에서의 링컨-피터슨처럼, 실패한 반복을 제외하면 남은 것만으로 계산한 성능이 낙관적으로 나온다. **실패율을 반드시 보고**한다.
    - **난수 씨앗을 고정하지 않는다.** 재현 불가능한 결과가 된다.

---

## 정리하며

두 전략을 **해석적 유도와 몬테카를로 모의실험**으로 나란히 견주었다.

- **편향·분산·MSE 를 모두 재는 것이 요점이다.** 어느 한 지표만 보면 판단이 갈리며, 실무에서 중요한 것은 대개 MSE 다.
- **모형이 맞으면 최대가능도의 MSE 가 작다.** 점근 효율성이 유한표본에서도 대체로 유지된다는 것을 모의실험이 확인해 준다.
- **차이의 크기가 분포에 따라 다르다.** 정규분포처럼 모수가 곧 적률인 경우에는 두 방법이 사실상 같은 답을 주고, 감마·베타처럼 적률과 모수의 관계가 비선형일 때 격차가 벌어진다.
- **계산 비용도 함께 재야 공정하다.** 적률법은 닫힌 형태로 즉시 나오고 최대가능도는 반복이 필요하다. 자료가 아주 크거나 추정을 수없이 반복해야 하는 상황에서는 이 차이가 결정적일 수 있다.
- **모의실험으로 추정량을 평가하는 절차 자체가 이 절의 교훈이다.** 같은 조건에서 여러 번 반복해 추정량의 분포를 만들고 그 중심과 퍼짐을 재는 것이 6.1절에서 세운 기준을 실제로 적용하는 방법이다.

**이것으로 6.3절이 끝난다.** 적률법과 최대가능도는 모두 자료만으로 모수를 정하는 **빈도주의** 방법이다.

다음 절 **사전분포, 가능도, 사후분포**부터는 세 번째 길로 넘어간다. 모수 자체에 확률분포를 부여하는 베이즈 접근이며, 이전 지식을 추정에 들여오는 방법이다.
