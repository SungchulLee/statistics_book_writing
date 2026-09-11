# Fisher 정보량 계산

## 개요

**Fisher 정보량**은 확률표본이 미지 모수에 관해 얼마나 많은 정보를 담고 있는지를 정량화한다. 추정이론에서 중심적인 역할을 한다. 달성 가능한 최선의 정밀도(Cramér-Rao 하한)를 결정하고, MLE의 점근분산을 지배하며, 실험설계를 이끈다. 이 페이지에서는 Fisher 정보량의 해석적 계산과 수치적 계산을 모두 보인다.

## 정의

$X$의 밀도(또는 PMF)가 $f(x; \theta)$라 하자. **점수함수**는:

$$
S(\theta) = \frac{\partial}{\partial\theta}\log f(X; \theta)
$$

관측값 하나에 대한 **Fisher 정보량**은:

$$
I(\theta) = E\left[S(\theta)^2\right] = \text{Var}\left[S(\theta)\right]
$$

정칙 조건 아래에서 이는 2계도함수의 기댓값에 음수를 취한 것과 같다:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X; \theta)\right]
$$

$n$개의 i.i.d. 관측값에서 전체 Fisher 정보량은 $I_n(\theta) = nI(\theta)$이다.

## 주요 이론 결과

!!! info "Cramér-Rao 하한"
    임의의 불편추정량 $\hat{\theta}$에 대해:

    $$
    \text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}
    $$

    이 한계를 달성하는 추정량을 **효율적**이라 한다.

!!! info "MLE의 점근정규성"
    정칙 조건 아래에서:

    $$
    \hat{\theta}_{\text{MLE}} \overset{d}{\to} N\!\left(\theta,\, \frac{1}{nI(\theta)}\right)
    $$

    MLE는 점근적으로 효율적이며 그 분산이 CRLB를 달성한다.

## 해석적 예제

### 정규분포의 평균

$\sigma^2$이 알려진 $X \sim N(\mu, \sigma^2)$에 대해:

$$
\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
$$

$$
S(\mu) = \frac{x - \mu}{\sigma^2}, \qquad -\frac{\partial^2}{\partial\mu^2}\log f = \frac{1}{\sigma^2}
$$

$$
I(\mu) = \frac{1}{\sigma^2}
$$

### Bernoulli

$X \sim \text{Bernoulli}(p)$에 대해:

$$
\log f(x; p) = x\log p + (1-x)\log(1-p)
$$

$$
I(p) = \frac{1}{p(1-p)}
$$

### Poisson

$X \sim \text{Poisson}(\lambda)$에 대해:

$$
\log f(x; \lambda) = x\log\lambda - \lambda - \log(x!)
$$

$$
I(\lambda) = \frac{1}{\lambda}
$$

### Exponential

(비율 모수화를 쓴) $X \sim \text{Exp}(\lambda)$에 대해:

$$
I(\lambda) = \frac{1}{\lambda^2}
$$

## Fisher 정보량의 수치적 계산

Fisher 정보량을 닫힌 형태로 계산할 수 없을 때는 다음 방법으로 수치적으로 추정할 수 있다:

1. **점수 분산법**: $f(x; \theta)$에서 $X_1, \ldots, X_N$을 표본추출하고 각 점에서 점수를 계산한 뒤 $I(\theta) \approx \text{Var}(\{S_i\})$로 추정한다.
2. **유한차분법**: 점수를 $S(\theta) \approx [\log f(X; \theta + \delta) - \log f(X; \theta - \delta)]/(2\delta)$로 근사한다.

```python
import numpy as np
from scipy import stats

def fisher_information_numerical(dist_name="norm", true_params=None,
                                  param_name="loc", n_samples=100_000, delta=1e-5):
    """Compute Fisher information numerically via the score variance."""
    if true_params is None:
        true_params = {"loc": 5, "scale": 2}

    dist = getattr(stats, dist_name)
    rng = np.random.default_rng(42)
    data = dist.rvs(size=n_samples, random_state=rng, **true_params)

    theta = true_params[param_name]

    # 점수함수 = 로그밀도를 모수로 미분한 것.
    # 해석적으로 미분하는 대신 중심차분으로 근사한다.
    # 그래서 이 코드는 분포가 무엇이든 그대로 쓸 수 있다.
    params_plus = {**true_params, param_name: theta + delta}
    params_minus = {**true_params, param_name: theta - delta}

    logf_plus = dist.logpdf(data, **params_plus)
    logf_minus = dist.logpdf(data, **params_minus)
    score = (logf_plus - logf_minus) / (2 * delta)

    # 피셔정보는 점수함수의 **분산**이다.
    # 참 모수에서 점수의 기댓값이 0이므로 분산 = 2차 적률이 되고,
    # 그래서 var를 그대로 쓰면 된다.
    #
    # 직관: 점수가 크게 흔들린다는 것은 모수를 조금만 바꿔도 가능도가
    # 크게 변한다는 뜻이고, 그만큼 자료가 모수를 잘 짚어낸다는 뜻이다.
    I_numerical = np.var(score)
    return I_numerical

# Normal mean: theoretical I(mu) = 1/sigma^2
I_num = fisher_information_numerical("norm", {"loc": 5, "scale": 2}, "loc")
I_theory = 1 / 2**2
print(f"Normal mean Fisher information:")
print(f"  Numerical:   I(mu) = {I_num:.6f}")
print(f"  Theoretical: I(mu) = {I_theory:.6f}")
```

출력:

```
Normal mean Fisher information:
  Numerical:   I(mu) = 0.251855
  Theoretical: I(mu) = 0.250000
```

## Cramér-Rao 한계 확인

정규분포의 평균에 대해 표본평균이 CRLB를 달성하는지 확인할 수 있다.

```python
import numpy as np

def verify_crlb(mu_true=5.0, sigma=2.0, n=50, n_sim=20_000):
    """Verify that the sample mean achieves the CRLB."""
    rng = np.random.default_rng(42)
    crlb = sigma**2 / n

    means = np.array([rng.normal(mu_true, sigma, n).mean() for _ in range(n_sim)])
    empirical_var = means.var()

    print(f"CRLB = sigma^2/n = {crlb:.6f}")
    print(f"Var(X_bar)       = {empirical_var:.6f}")
    print(f"Ratio            = {empirical_var / crlb:.4f}")

    # 비교: 중앙값은 하한을 달성하지 못한다.
    # 정규모집단에서 중앙값의 점근 효율은 2/pi ≈ 0.637 이다.
    # 즉 같은 정밀도를 얻으려면 표본이 약 1.57배 더 필요하다.
    medians = np.array([np.median(rng.normal(mu_true, sigma, n)) for _ in range(n_sim)])
    print(f"\nVar(median) = {medians.var():.6f}")
    print(f"Efficiency of median = {crlb / medians.var():.4f}")

verify_crlb()
```

출력:

```
CRLB = sigma^2/n = 0.080000
Var(X_bar)       = 0.078777
Ratio            = 0.9847

Var(median) = 0.121813
Efficiency of median = 0.6567
```

!!! note "중앙값의 효율"
    정규분포에서 평균 대비 중앙값의 점근 상대효율은 $2/\pi \approx 0.637$이다. 중앙값은 자료가 담은 정보의 약 64%만 사용한다.

## 다모수 Fisher 정보행렬

모수 벡터 $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)$에 대해 Fisher 정보량은 $k \times k$ 행렬이다:

$$
[I(\boldsymbol{\theta})]_{ij} = -E\left[\frac{\partial^2}{\partial\theta_i\,\partial\theta_j}\log f(X; \boldsymbol{\theta})\right]
$$

$\mu$와 $\sigma^2$이 모두 미지인 정규분포에서:

$$
I(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}
$$

비대각 성분이 0이라는 것은 $\mu$와 $\sigma^2$이 정보적으로 직교함을 보여 준다.

## 해석

- Fisher 정보량은 참 모수에서 로그가능도의 **곡률**을 잰다. 곡률이 크면 자료가 정보를 많이 담고 있고 MLE가 정밀하다.
- $I(\theta)$가 클수록 Cramér-Rao 한계가 좁아지므로 추정량이 더 정밀할 수 있다.
- Fisher 정보량은 참 모수값에 의존한다. 예를 들어 Bernoulli의 $I(p) = 1/[p(1-p)]$는 $p$가 0이나 1에 가까울 때 가장 크고(관측값 하나하나가 많은 정보를 준다) $p = 0.5$에서 가장 작다(불확실성이 최대이다).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** Poisson 분포 $P(\lambda)$의 Fisher 정보량을 점수 분산의 정의와 2계도함수 기댓값의 음수, 두 가지 방식으로 유도하라. 둘이 일치함을 확인하라.

</div>

??? success "풀이"
    로그 PMF는 $\log f(x; \lambda) = x\log\lambda - \lambda - \log(x!)$이다.

    **점수:** $S(\lambda) = X/\lambda - 1$.

    **점수의 분산:** $\text{Var}(S) = \text{Var}(X/\lambda) = \text{Var}(X)/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    **2계도함수의 음수:** $-\partial^2\log f/\partial\lambda^2 = X/\lambda^2$이므로 $E[-\partial^2\log f/\partial\lambda^2] = E[X]/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    둘 다 $I(\lambda) = 1/\lambda$를 준다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 비율이 $\lambda$인 Exponential 분포($x > 0$에서 밀도 $f(x; \lambda) = \lambda e^{-\lambda x}$)에 대해 Fisher 정보량과, $n$개의 관측값으로 $\lambda$를 추정할 때의 Cramér-Rao 하한을 계산하라.

</div>

??? success "풀이"
    로그밀도는 $\log f(x; \lambda) = \log\lambda - \lambda x$이다.

    점수: $S(\lambda) = 1/\lambda - X$. 2계도함수: $\partial^2\log f/\partial\lambda^2 = -1/\lambda^2$.

    Fisher 정보량: $I(\lambda) = 1/\lambda^2$.

    $n$개 관측값에 대한 CRLB는:

    $$
    \text{Var}(\hat{\lambda}) \geq \frac{1}{nI(\lambda)} = \frac{\lambda^2}{n}
    $$

    MLE는 $\hat{\lambda} = 1/\bar{X}$이다. 델타 방법에 의해 $n$이 크면 $\text{Var}(\hat{\lambda}) \approx \lambda^2/n$이므로 MLE가 점근적으로 이 한계를 달성한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** Bernoulli 분포에서 Fisher 정보량 $I(p) = 1/[p(1-p)]$이 $p = 1/2$에서 최소가 됨을 보여라. 동전 던지기의 관점에서 이 결과를 해석하라.

</div>

??? success "풀이"
    미분하면 $\frac{d}{dp}I(p) = \frac{d}{dp}[p(1-p)]^{-1} = -\frac{1-2p}{[p(1-p)]^2}$이다.

    0으로 두면 $p = 1/2$이다. $p \to 0$이나 $p \to 1$일 때 $I(p) \to \infty$이고 $I(1/2) = 4$가 유한한 최솟값이므로 $p = 1/2$에서 Fisher 정보량이 최소가 된다.

    **해석:** 공정한 동전($p = 1/2$)이 인접한 값들과 구별하기 가장 어렵다. 결과가 가장 불확실할 때 던지기 한 번이 $p$에 관해 가장 적은 정보를 준다. 반대로 $p$가 0이나 1에 가까우면 결과가 매우 예측 가능하므로 던지기 한 번이 큰 정보를 주며, 그 양상에서 벗어나는 결과는 강한 진단 신호가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** $\text{Gamma}(\alpha, \beta)$ 분포의 $\alpha$에 대한 Fisher 정보량은 trigamma 함수 $\psi_1(\alpha)$를 포함한다: $I_{\alpha\alpha} = \psi_1(\alpha)$. `scipy.special.polygamma(1, alpha)`를 사용하여 $\alpha = 3$에서 $n = 100$개의 관측값으로 $\alpha$를 추정할 때의 CRLB를 수치적으로 계산하라.

</div>

??? success "풀이"
    ```python
    from scipy.special import polygamma

    alpha = 3.0
    n = 100
    I_alpha = polygamma(1, alpha)  # trigamma function
    crlb = 1 / (n * I_alpha)
    print(f"Trigamma(alpha=3) = {I_alpha:.6f}")
    print(f"CRLB for alpha:     {crlb:.6f}")
    ```

    출력:

    ```
    Trigamma(alpha=3) = 0.394934
    CRLB for alpha:     0.025321
    ```

    trigamma 함수 값이 $\psi_1(3) \approx 0.3949$이므로 CRLB는 약 $1/(100 \times 0.3949) \approx 0.0253$이다. 관측값 100개로는 $\alpha$의 어떤 불편추정량도 분산이 약 0.025보다 작을 수 없다는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** Fisher 정보량이 가법성을 만족함을 증명하라. $n$개의 i.i.d. 관측값에서 $I_n(\theta) = nI_1(\theta)$이다. 이것이 직관적으로 타당한 이유는 무엇인가?

</div>

??? success "풀이"
    $X_1, \ldots, X_n$을 밀도가 $f(x; \theta)$인 i.i.d. 확률변수라 하자. 결합 로그가능도는:

    $$
    \ell_n(\theta) = \sum_{i=1}^n \log f(X_i; \theta)
    $$

    $S_i(\theta) = \partial\log f(X_i; \theta)/\partial\theta$일 때 전체 점수는 $S_n(\theta) = \sum_{i=1}^n S_i(\theta)$이다.

    $X_i$들이 독립이므로 $S_i$들도 독립이고 $E[S_i] = 0$, $\text{Var}(S_i) = I_1(\theta)$이다. 따라서:

    $$
    I_n(\theta) = \text{Var}(S_n) = \sum_{i=1}^n \text{Var}(S_i) = nI_1(\theta)
    $$

    **직관:** 독립인 각 관측값이 $\theta$에 관해 같은 양의 정보를 기여한다. 표본크기를 두 배로 하면 전체 정보량이 두 배가 되고, 그러면 CRLB가 절반이 되어 달성 가능한 최소 분산도 절반이 된다. $\square$
