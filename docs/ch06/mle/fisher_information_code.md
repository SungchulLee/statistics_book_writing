# Fisher 정보량 계산

## 개요

**Fisher 정보량**은 확률표본이 미지 모수에 관해 얼마나 많은 정보를 담고 있는지를 정량화한다. 추정이론에서 중심적인 역할을 한다. 달성 가능한 최선의 정밀도(Cramér-Rao 하한)를 결정하고, MLE의 점근분산을 지배하며, 실험설계를 이끈다. 이 페이지에서는 Fisher 정보량의 해석적 계산과 수치적 계산을 모두 보인다.

<div class="defn" markdown>

### 정의 1. 점수함수와 Fisher 정보량 { .dfn }

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

</div>

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

<div class="codebox" markdown>

### 예제 1. 피셔 정보량을 수치로 구하기 { .eg }

```python
import numpy as np
from scipy import stats

def fisher_information_numerical(dist_name="norm", true_params=None,
                                  param_name="loc", n_samples=100_000, delta=1e-5):
    """점수함수의 분산으로 피셔 정보량을 수치적으로 구한다."""
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

# 정규분포 평균의 피셔 정보량은 이론적으로 1/sigma^2 이다.
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

</div>

## Cramér-Rao 한계 확인

정규분포의 평균에 대해 표본평균이 CRLB를 달성하는지 확인할 수 있다.

<div class="codebox" markdown>

### 예제 2. 표본평균이 하한에 도달함을 확인하기 { .eg }

```python
import numpy as np

def verify_crlb(mu_true=5.0, sigma=2.0, n=50, n_sim=20_000):
    """표본평균이 크라메르-라오 하한에 도달함을 확인한다."""
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

</div>

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
- Fisher 정보량은 참 모수값에 의존한다. 예를 들어 베르누이의 $I(p) = 1/[p(1-p)]$는 $p$가 0이나 1에 가까울 때 가장 크고(관측값 하나하나가 많은 정보를 준다) $p = 0.5$에서 가장 작다(불확실성이 최대이다).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 포아송분포 $P(\lambda)$의 Fisher 정보량을 점수 분산의 정의와 2계도함수 기댓값의 음수, 두 가지 방식으로 유도하라. 둘이 일치함을 확인하라.

</div>

??? success "풀이"
    로그 PMF는 $\log f(x; \lambda) = x\log\lambda - \lambda - \log(x!)$이다.

    **점수:** $S(\lambda) = X/\lambda - 1$.

    **점수의 분산:** $\text{Var}(S) = \text{Var}(X/\lambda) = \text{Var}(X)/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    **2계도함수의 음수:** $-\partial^2\log f/\partial\lambda^2 = X/\lambda^2$이므로 $E[-\partial^2\log f/\partial\lambda^2] = E[X]/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    둘 다 $I(\lambda) = 1/\lambda$를 준다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 비율이 $\lambda$인 지수분포($x > 0$에서 밀도 $f(x; \lambda) = \lambda e^{-\lambda x}$)에 대해 Fisher 정보량과, $n$개의 관측값으로 $\lambda$를 추정할 때의 Cramér-Rao 하한을 계산하라.

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

**연습문제 3.** <span class="diff med" title="중간"></span> 베르누이분포에서 Fisher 정보량 $I(p) = 1/[p(1-p)]$이 $p = 1/2$에서 최소가 됨을 보여라. 동전 던지기의 관점에서 이 결과를 해석하라.

</div>

??? success "풀이"
    미분하면 $\frac{d}{dp}I(p) = \frac{d}{dp}[p(1-p)]^{-1} = -\frac{1-2p}{[p(1-p)]^2}$이다.

    0으로 두면 $p = 1/2$이다. $p \to 0$이나 $p \to 1$일 때 $I(p) \to \infty$이고 $I(1/2) = 4$가 유한한 최솟값이므로 $p = 1/2$에서 Fisher 정보량이 최소가 된다.

    **해석:** 공정한 동전($p = 1/2$)이 인접한 값들과 구별하기 가장 어렵다. 결과가 가장 불확실할 때 던지기 한 번이 $p$에 관해 가장 적은 정보를 준다. 반대로 $p$가 0이나 1에 가까우면 결과가 매우 예측 가능하므로 던지기 한 번이 큰 정보를 주며, 그 양상에서 벗어나는 결과는 강한 진단 신호가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> $\text{Gamma}(\alpha, \beta)$ 분포의 $\alpha$에 대한 Fisher 정보량은 trigamma 함수 $\psi_1(\alpha)$를 포함한다: $I_{\alpha\alpha} = \psi_1(\alpha)$. `scipy.special.polygamma(1, alpha)`를 사용하여 $\alpha = 3$에서 $n = 100$개의 관측값으로 $\alpha$를 추정할 때의 CRLB를 수치적으로 계산하라.

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

**연습문제 5.** <span class="diff med" title="중간"></span> Fisher 정보량이 가법성을 만족함을 증명하라. $n$개의 i.i.d. 관측값에서 $I_n(\theta) = nI_1(\theta)$이다. 이것이 직관적으로 타당한 이유는 무엇인가?

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

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
Fisher 정보량을 **수치 미분**으로 계산할 때 생기는 문제를 설명하고, 더 나은 방법을 두 가지 적어라.

</div>

??? success "풀이"
    **수치 미분의 문제.** 중심차분

    $$
    \ell''(\theta) \approx \frac{\ell(\theta+h)-2\ell(\theta)+\ell(\theta-h)}{h^2}
    $$

    은 두 종류의 오차가 **반대 방향으로** 작용한다.

    - **절단오차** $O(h^2)$: $h$가 크면 커진다.
    - **반올림오차** $O(\epsilon/h^2)$: $h$가 작으면 커진다. 분자에서 거의 같은 세 수를 빼기 때문에 상쇄가 일어나고, 그것을 아주 작은 $h^2$으로 나누면 오차가 증폭된다.

    최적 $h$는 대략 $\epsilon^{1/4} \approx 10^{-4}$인데, 그때도 유효숫자를 절반 이상 잃는다. 1계 도함수의 최적 $h \approx \epsilon^{1/3}$보다 나쁘다.

    **더 나은 방법 1 — 해석적 미분.** 손으로 2계 도함수를 구해 코드에 넣는다. 가장 정확하고 빠르지만, 모형이 복잡하면 실수하기 쉽고 모형을 바꿀 때마다 다시 해야 한다.

    **더 나은 방법 2 — 자동미분.** JAX나 PyTorch의 `hessian`을 쓰면 **기계 정밀도까지 정확한** 도함수를 얻는다. 연쇄법칙을 코드 수준에서 적용하는 것이라 수치 근사가 전혀 들어가지 않는다.

    ```python
    import jax, jax.numpy as jnp
    H = jax.hessian(lambda th: -loglik(th, data))(theta_hat)
    ```

    **방법 3 — 복소 계단 미분.** 1계 도함수에는

    $$
    f'(x) \approx \frac{\operatorname{Im}\{f(x+ih)\}}{h}
    $$

    가 뺄셈 없이 계산되어 $h$를 $10^{-20}$까지 줄여도 안전하다. 2계에는 바로 쓸 수 없지만, 점수함수가 해석적으로 있으면 그것에 적용해 헤시안을 얻을 수 있다.

    **실용적 절충.** 많은 최적화 라이브러리가 준뉴턴법(BFGS)으로 헤시안의 근사를 이미 만들어 두므로 그것을 재활용할 수 있다. 다만 BFGS 근사는 최적화를 위한 것이라 표준오차 계산에 쓰기에는 정확도가 부족한 경우가 있어, 최종 표준오차는 위 방법으로 다시 계산하는 편이 안전하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
Fisher 정보량이 **가법적**이라는 성질을 이용해, 서로 다른 실험 두 개에서 같은 모수를 추정할 때 결과를 어떻게 합치는지 설명하라.

</div>

??? success "풀이"
    **가법성.** 두 실험이 독립이면 로그가능도가 더해지고, 2계 도함수도 더해지므로

    $$
    I_{\text{합}}(\theta) = I_1(\theta) + I_2(\theta)
    $$

    이다.

    **합치는 방법.** 각 실험에서 $\hat\theta_j$와 정보량 $I_j$를 얻었다면, 두 로그가능도를 더해 최대화하는 것이 통합 MLE다. 이차 근사 아래에서는

    $$
    \hat\theta_{\text{합}} = \frac{I_1\hat\theta_1 + I_2\hat\theta_2}{I_1+I_2}, \qquad \operatorname{Var}(\hat\theta_{\text{합}}) = \frac{1}{I_1+I_2}
    $$

    로 **정보량 가중평균**이 된다. 앞서 본 역분산 가중과 같은 식이다($I_j = 1/\operatorname{SE}_j^2$).

    **응용.**

    - **메타분석.** 여러 연구의 추정값과 표준오차만 있으면 원자료 없이도 통합할 수 있다. 고정효과 메타분석이 정확히 이 공식이다.
    - **순차 실험.** 1차 실험 결과를 2차 설계에 반영하고, 마지막에 정보량을 더해 합친다.
    - **베이즈 갱신.** 사전분포의 정밀도와 자료의 정보량이 더해져 사후 정밀도가 된다. 정규-정규 켤레의 갱신식이 위 공식과 동일하다.

    **주의.** 가법성은 **독립**을 전제한다. 같은 자료를 두 번 쓰거나 두 연구가 겹치는 표본을 쓰면 정보량을 과대평가해 신뢰구간이 부당하게 좁아진다. 메타분석에서 같은 코호트를 여러 논문이 보고한 경우가 흔한 함정이다.

    또 **연구 간 이질성**이 있으면 $\theta$가 실험마다 다를 수 있고, 그때는 고정효과가 아니라 변량효과 모형이 필요하다. 정보량 가중은 "같은 $\theta$를 추정하고 있다"는 가정 위에서만 옳다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
관측 Fisher 정보량 $\hat I = -\ell''(\hat\theta)$과 $n$개 관측값의 점수 제곱합 $\sum_i s(x_i;\hat\theta)^2$은 둘 다 $I_n$을 추정한다. 둘의 차이와 각각의 쓰임을 밝혀라.

</div>

??? success "풀이"
    **두 추정값.**

    $$
    \hat A = -\sum_i \frac{\partial^2\ln f(x_i;\hat\theta)}{\partial\theta^2}, \qquad \hat B = \sum_i \left(\frac{\partial\ln f(x_i;\hat\theta)}{\partial\theta}\right)^2
    $$

    $\hat B$를 **외적 기울기(OPG)** 추정량 또는 버른트-홀-홀-하우스만 추정량이라 한다.

    **모형이 맞으면 둘이 같은 것을 추정한다.** 정보량 등식 $E[-\ell''] = E[(\ell')^2]$ 때문이다. 다만 유한표본에서 값은 다르고, 일반적으로 $\hat A$가 더 안정적이다. $\hat B$는 점수를 제곱하므로 이상치에 민감하다.

    **모형이 틀리면 갈라진다.** 이 차이 자체가 정보를 담는다.

    - **샌드위치 분산** $\hat A^{-1}\hat B\hat A^{-1}$을 쓰면 모형 오설정에 강건한 표준오차가 나온다.
    - $\hat A$와 $\hat B$가 크게 다르다는 것 자체가 **모형 오설정의 신호**다. 이를 형식적으로 검정하는 것이 화이트의 정보행렬 검정이다.

    **각각의 쓰임.**

    | | $\hat A$(헤시안) | $\hat B$(OPG) |
    |---|---|---|
    | 계산 | 2계 도함수 필요 | 1계 도함수만 필요 |
    | 안정성 | 좋음 | 이상치에 민감 |
    | 양정부호 보장 | 없음(봉우리 밖) | **있음** |
    | 주 용도 | 표준오차 기본값 | 2계 도함수가 어려울 때, 샌드위치의 가운데 |

    OPG가 널리 쓰이는 이유는 **점수만으로 계산되고 언제나 양반정부호**여서다. 복잡한 모형에서 헤시안을 구하기 어려울 때 실용적인 대안이 되며, 계량경제학에서 특히 흔하다. 다만 소표본에서 정보량을 과대평가하는 경향이 있어 표준오차가 너무 작게 나올 수 있다는 점은 알려진 단점이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$\text{Gamma}(\alpha,\beta)$의 Fisher 정보행렬을 구하고, $\alpha$와 $\beta$가 **직교하지 않음**을 확인하라. 이것이 $\alpha$ 추정에 어떤 대가를 치르게 하는가?

</div>

??? success "풀이"
    밀도가 $f(x) = \dfrac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$(비율 모수화)이므로

    $$
    \ln f = \alpha\ln\beta - \ln\Gamma(\alpha) + (\alpha-1)\ln x - \beta x
    $$

    이다. 2계 편도함수를 구하면

    $$
    \frac{\partial^2\ln f}{\partial\alpha^2} = -\psi_1(\alpha), \qquad
    \frac{\partial^2\ln f}{\partial\alpha\partial\beta} = \frac1\beta, \qquad
    \frac{\partial^2\ln f}{\partial\beta^2} = -\frac{\alpha}{\beta^2}
    $$

    이고($\psi_1$은 트라이감마 함수), 자료에 의존하지 않으므로 기대값이 그대로다.

    $$
    I_1 = \begin{pmatrix}\psi_1(\alpha) & -1/\beta \\ -1/\beta & \alpha/\beta^2\end{pmatrix}
    $$

    **비대각이 0이 아니다.** 따라서 두 모수가 직교하지 않는다.

    **$\alpha$ 추정의 대가.** 앞 절의 결과에 따라

    $$
    \operatorname{Var}(\hat\alpha) \ge \frac{(I_1^{-1})_{11}}{n} = \frac{1}{n}\cdot\frac{\alpha/\beta^2}{\psi_1(\alpha)\alpha/\beta^2 - 1/\beta^2} = \frac{1}{n}\cdot\frac{\alpha}{\alpha\psi_1(\alpha)-1}
    $$

    이다. $\beta$를 안다면 하한이 $1/\{n\psi_1(\alpha)\}$이므로, 모르는 대가가

    $$
    \frac{\alpha\psi_1(\alpha)}{\alpha\psi_1(\alpha)-1} > 1
    $$

    배다. $\alpha=1$이면 $\psi_1(1)=\pi^2/6=1.645$이므로 배율이 $1.645/0.645 = 2.55$다. **$\beta$를 모른다는 이유만으로 $\alpha$의 분산이 2.5배가 된다.**

    $\alpha$가 커지면 $\alpha\psi_1(\alpha)\to1+1/(2\alpha)$이 아니라 $\to 1 + \tfrac{1}{2\alpha}$ 꼴로 1에 가까워져 배율이 더 커진다. 형상모수가 큰 감마분포에서 $\alpha$를 정밀하게 추정하기가 특히 어려운 이유다.

    **직교 모수화.** $(\alpha, \mu)$($\mu = \alpha/\beta$가 평균)로 바꾸면 정보행렬이 대각에 가까워진다. 실제로 이 모수화에서 비대각이 0이 되어 두 모수가 직교하며, 일반화선형모형이 감마분포를 평균 모수화로 다루는 이유 중 하나다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
Fisher 정보량은 **표본크기**뿐 아니라 **관측의 질**에도 의존한다. 측정오차가 있는 경우($X$ 대신 $X+\varepsilon$을 관측) 정보량이 어떻게 줄어드는지 정규 모형에서 확인하라.

</div>

??? success "풀이"
    참값이 $X \sim N(\mu, \sigma^2)$인데 측정오차 $\varepsilon \sim N(0,\tau^2)$이 더해진 $Y = X+\varepsilon$을 관측한다고 하자($\varepsilon$은 $X$와 독립).

    **관측되는 분포.** 정규의 합이므로

    $$
    Y \sim N(\mu,\ \sigma^2+\tau^2)
    $$

    이다.

    **정보량.** 정규분포에서 평균에 대한 관측값 하나당 정보량이 분산의 역수이므로

    $$
    I_1^{(Y)}(\mu) = \frac{1}{\sigma^2+\tau^2} < \frac{1}{\sigma^2} = I_1^{(X)}(\mu)
    $$

    이다. **측정오차가 정보를 줄인다.**

    **얼마나 줄어드는가.** 감쇠 계수

    $$
    \frac{I^{(Y)}}{I^{(X)}} = \frac{\sigma^2}{\sigma^2+\tau^2} =: \kappa
    $$

    를 **신뢰도**라 한다. 이 값이 정확히 "관측값 하나가 참값 하나 대비 몇 개분의 정보를 갖는가"이다.

    $\tau = \sigma$이면 $\kappa = 0.5$로, **관측값 두 개가 오차 없는 관측값 하나만큼의 정보**밖에 없다. 유효 표본크기가 절반이 된 셈이다.

    **회귀에서는 더 나쁘다.** 설명변수에 측정오차가 있으면 정보량이 줄 뿐 아니라 **계수 추정이 편향된다.**

    $$
    \hat\beta \xrightarrow{p} \kappa\beta
    $$

    로 0 쪽으로 감쇠한다(회귀 희석). 표본을 늘려도 사라지지 않는 편향이며, 이것이 측정오차의 가장 심각한 결과다.

    **함의.** **표본크기를 늘리는 것과 측정을 정밀하게 하는 것은 서로 대체 가능하다.** 정보량이 $n\kappa/\sigma^2$이므로, 측정 정밀도를 두 배 좋게 하는 것($\tau^2$을 절반으로)과 표본을 일정 비율 늘리는 것이 같은 효과를 낸다. 어느 쪽이 싼지가 설계의 문제다.

    반복 측정도 방법이다. 같은 대상을 $k$번 재어 평균하면 오차분산이 $\tau^2/k$로 줄어 신뢰도가 올라간다. 심리측정에서 문항 수를 늘려 신뢰도를 높이는 것(스피어만-브라운 공식)이 같은 원리다.

---

## 정리하며

이 절은 피셔 정보량을 **해석적으로 구한 값과 수치적으로 구한 값**을 나란히 놓았다.

- **해석적 계산**은 $-\mathbb{E}[\partial^2_\theta\log f]$ 를 손으로 푸는 것이다. 지수족에서는 대개 깔끔한 닫힌 형태가 나온다.
- **수치적 계산**은 로그가능도를 두 번 수치미분하거나, 점수함수를 표본에서 계산해 제곱평균을 취한다. 손으로 풀 수 없는 모형에서 유일한 길이다.
- **둘이 맞는지 확인하는 것이 이 절의 목적이다.** 복잡한 모형에서는 해석적 유도의 검산으로 수치적 값을 쓰고, 반대로 수치 코드의 검증으로 해석적 값을 쓴다.
- **기대정보와 관측정보를 구별하라.** $I(\theta)$ 는 기댓값을 취한 것이고, $-\ell''(\hat\theta)$ 은 이번 자료에서 평가한 것이다. 실무의 표준오차는 대개 후자를 쓰며, 소표본에서 둘은 눈에 띄게 다를 수 있다.
- **수치미분의 간격 선택에 주의한다.** 너무 크면 절단오차가, 너무 작으면 반올림오차가 커진다.

다음 절 **포획–재포획 최대가능도**로 넘어간다. 앞서 비례식으로 얻은 추정량을 이번에는 초기하 가능도에서 제대로 유도한다.
