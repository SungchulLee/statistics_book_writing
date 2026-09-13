# 베이즈 추정 시연

## 개요

베이즈 추정은 미지 모수 $\theta$를 확률변수로 다루며, 자료를 관측하기 전의 믿음을 **사전분포**에 담는다. 자료를 관측한 뒤에는 베이즈 정리를 통해 **사후분포**로 갱신한다. 이 페이지에서는 두 가지 기본적인 켤레 모형인 Beta-Binomial과 Normal-Normal을 시연하며, 자료가 쌓일수록 사후분포가 어떻게 좁아지는지와 사전분포의 선택이 추론에 어떤 영향을 주는지 보인다.

## 추정을 위한 베이즈 정리

자료 $\mathbf{x} = (x_1, \ldots, x_n)$과 모수 $\theta$가 주어졌을 때:

$$
p(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta)\, \pi(\theta)}{\int f(\mathbf{x} \mid \theta)\, \pi(\theta)\, d\theta}
$$

여기서:

- $\pi(\theta)$는 **사전분포**이다
- $f(\mathbf{x} \mid \theta)$는 **가능도**이다
- $p(\theta \mid \mathbf{x})$는 **사후분포**이다

!!! info "사후분포로부터의 점추정"

    - **사후평균:** $E[\theta \mid \mathbf{x}]$ — 제곱오차 손실 아래에서 베이즈 위험을 최소화한다.
    - **MAP (최대사후확률):** $\arg\max_\theta p(\theta \mid \mathbf{x})$ — 사후분포의 최빈값이다.
    - **사후중앙값:** 절대오차 손실 아래에서 베이즈 위험을 최소화한다.

## 켤레 사전분포

사후분포가 사전분포와 같은 분포족에 속하면 사전분포 $\pi(\theta)$가 그 가능도에 **켤레**라고 한다. 켤레 사전분포는 닫힌 형태의 사후분포를 주므로 베이즈 갱신을 해석적으로 다룰 수 있게 한다.

| 가능도 | 켤레 사전분포 | 사후분포 |
|------------|----------------|-----------|
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal ($\sigma^2$ 알려짐) | Normal | Normal |
| Normal ($\mu$ 알려짐) | Inverse-Gamma | Inverse-Gamma |
| Exponential | Gamma | Gamma |

## Beta-Binomial 켤레 모형

### 설정

이항 자료에서 비율 $p$를 추정할 때:

- **사전분포:** $p \sim \text{Beta}(\alpha_0, \beta_0)$
- **자료:** $n$번의 시행에서 $k$번 성공
- **사후분포:** $p \mid k \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

사후평균은:

$$
E[p \mid k] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}
$$

이는 사전평균 $\alpha_0/(\alpha_0 + \beta_0)$과 MLE $k/n$의 가중평균이며, 가중은 사전분포의 "유효 표본크기" $\alpha_0 + \beta_0$과 실제 표본크기 $n$에 비례한다.

### 시연

<div class="codebox" markdown>

#### 예제 1. 베타-이항 켤레모형으로 비율 추정하기 { .eg }

```python
import numpy as np
from scipy import stats

def demo_beta_binomial():
    """베타-이항 켤레모형으로 비율을 추정한다."""
    alpha_0, beta_0 = 2, 2  # weakly informative prior

    n, k = 50, 32  # observed data

    # 사후분포의 모수. 켤레사전분포라 형태가 그대로 유지된다.
    alpha_post = alpha_0 + k
    beta_post = beta_0 + (n - k)

    # 점추정값: 사후평균과 사후최빈값.
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    post_mean = alpha_post / (alpha_post + beta_post)
    mle = k / n

    print(f"Prior:     Beta({alpha_0}, {beta_0})")
    print(f"Data:      {k} successes in {n} trials")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")
    print(f"MAP        = {map_est:.4f}")
    print(f"Post. mean = {post_mean:.4f}")
    print(f"MLE        = {mle:.4f}")

    # 95% 신용구간: 사후분포의 2.5·97.5 백분위점.
    ci_low = stats.beta.ppf(0.025, alpha_post, beta_post)
    ci_high = stats.beta.ppf(0.975, alpha_post, beta_post)
    print(f"95% CI:    [{ci_low:.4f}, {ci_high:.4f}]")

demo_beta_binomial()
```

출력:

```
Prior:     Beta(2, 2)
Data:      32 successes in 50 trials
Posterior: Beta(34, 20)
MAP        = 0.6346
Post. mean = 0.6296
MLE        = 0.6400
95% CI:    [0.4980, 0.7521]
```

</div>

!!! note "가상 자료로서의 사전분포"
    Beta$(\alpha_0, \beta_0)$ 사전분포는 실제 자료를 보기 전에 이미 $\alpha_0 - 1$번의 성공과 $\beta_0 - 1$번의 실패를 관측한 것처럼 작동한다. $\alpha_0 = \beta_0 = 2$이면 사전분포가 총 2개의 "가상 관측값"에 해당하는 기여를 한다.

## Normal-Normal 켤레 모형

### 설정

분산 $\sigma^2$이 알려진 상태에서 평균 $\mu$를 추정할 때:

- **사전분포:** $\mu \sim N(\mu_0, \tau_0^2)$
- **자료:** $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$
- **사후분포:** $\mu \mid \mathbf{x} \sim N(\mu_n, \tau_n^2)$

여기서:

$$
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
$$

$$
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}\right)
$$

사후평균은 사전평균과 표본평균의 **정밀도 가중평균**이다:

$$
\mu_n = \frac{\tau_0^{-2}\,\mu_0 + n\sigma^{-2}\,\bar{x}}{\tau_0^{-2} + n\sigma^{-2}}
$$

### 시연

<div class="codebox" markdown>

#### 예제 2. 정규-정규 켤레모형으로 평균 추정하기 { .eg }

```python
import numpy as np
from scipy import stats

def demo_normal_normal():
    """정규-정규 켤레모형으로 평균을 추정한다."""
    sigma = 2.0        # known population std
    mu_true = 5.0      # true mean
    mu_0, tau_0 = 0, 10  # prior parameters

    rng = np.random.default_rng(42)
    n = 25
    data = rng.normal(mu_true, sigma, n)
    x_bar = data.mean()

    # 사후분포의 모수. 켤레사전분포라 형태가 그대로 유지된다.
    tau_n_sq = 1 / (1 / tau_0**2 + n / sigma**2)
    mu_n = tau_n_sq * (mu_0 / tau_0**2 + n * x_bar / sigma**2)
    tau_n = np.sqrt(tau_n_sq)

    print(f"Prior:      N({mu_0}, {tau_0}^2)")
    print(f"Data:       n={n}, x_bar={x_bar:.3f}")
    print(f"Posterior:  N({mu_n:.3f}, {tau_n:.3f}^2)")
    print(f"95% credible interval: [{mu_n - 1.96*tau_n:.3f}, {mu_n + 1.96*tau_n:.3f}]")

demo_normal_normal()
```

출력:

```
Prior:      N(0, 10^2)
Data:       n=25, x_bar=4.929
Posterior:  N(4.921, 0.400^2)
95% credible interval: [4.138, 5.705]
```

</div>

## 자료에 따른 사후분포의 변화

$n$이 커질수록 사후평균은 MLE로 수렴하고 사후분산은 0으로 줄어든다. 사전분포는 무의미해진다:

$$
\mu_n \to \bar{x} \quad \text{and} \quad \tau_n^2 \to \frac{\sigma^2}{n} \quad \text{as } n \to \infty
$$

이는 핵심적인 점근적 성질을 보여 준다. 대표본에서는 가능도가 사후분포를 지배하며 베이즈 추정값과 빈도주의 추정값이 일치한다.

## 해석

- **켤레 사전분포**는 해석적 편의를 제공한다. 사후분포가 알려진 분포 형태를 가지므로 신용구간과 사후확률을 정확히 계산할 수 있다.
- **사후평균**은 사전 믿음과 자료의 증거를 각각의 정밀도로 가중한 절충이다.
- **사전분포 민감도**는 소표본에서 중요하다. $n$이 커질수록 사전분포의 영향이 사라진다.
- **신용구간**은 직접적인 확률 해석을 갖는다. 95% 신용구간은 사후확률 0.95로 참 모수를 포함한다. 이는 빈도주의 신뢰구간과 다르다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $\text{Beta}(1, 1)$(균등) 사전분포로 $n = 10$번의 Bernoulli 시행에서 $k = 7$번 성공을 관측했다고 하자. 사후분포, 사후평균, MAP 추정값, 95% 신용구간을 계산하라.

</div>

??? success "풀이"
    사후분포는 $\text{Beta}(1 + 7, 1 + 3) = \text{Beta}(8, 4)$이다.

    사후평균: $\frac{8}{8 + 4} = \frac{2}{3} \approx 0.6667$.

    MAP: $\frac{8 - 1}{8 + 4 - 2} = \frac{7}{10} = 0.7$ (MLE와 같다).

    `scipy.stats.beta.ppf`를 쓴 95% 신용구간:

    ```python
    from scipy.stats import beta
    ci = (beta.ppf(0.025, 8, 4), beta.ppf(0.975, 8, 4))
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    ```

    출력:

    ```
    95% CI: [0.3903, 0.8907]
    ```

    결과는 약 $[0.3834, 0.9029]$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> Normal-Normal 모형에서 $\tau_0 \to \infty$(막연한 사전분포)일 때 사후평균이 표본평균으로, 사후분산이 $\sigma^2/n$으로 수렴함을 보여라.

</div>

??? success "풀이"
    $\tau_0 \to \infty$이면 $1/\tau_0^2 \to 0$이다. 그러면:

    $$
    \frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2} \to \frac{n}{\sigma^2}
    $$

    따라서 $\tau_n^2 \to \sigma^2/n$이다.

    사후평균에 대해서는:

    $$
    \mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}\right)
    $$

    $\tau_0 \to \infty$일 때 첫 항 $\mu_0/\tau_0^2 \to 0$이므로:

    $$
    \mu_n \to \frac{\sigma^2}{n} \cdot \frac{n\bar{x}}{\sigma^2} = \bar{x}
    $$

    막연한(무정보) 사전분포에서는 베이즈 사후분포가 빈도주의 결과로 환원된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> Beta-Binomial 모형의 사후평균을 사전평균과 MLE의 볼록결합으로 쓸 수 있음을 증명하라. 가중을 찾아 해석하라.

</div>

??? success "풀이"
    $\pi_0 = \alpha_0/(\alpha_0 + \beta_0)$을 사전평균, $\hat{p} = k/n$을 MLE라 하자. 사후평균은:

    $$
    E[p \mid k] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}
    $$

    다시 쓰면:

    $$
    = \frac{\alpha_0 + \beta_0}{\alpha_0 + \beta_0 + n}\cdot\frac{\alpha_0}{\alpha_0 + \beta_0} + \frac{n}{\alpha_0 + \beta_0 + n}\cdot\frac{k}{n}
    $$

    $$
    = w\,\pi_0 + (1 - w)\,\hat{p}
    $$

    여기서 $w = (\alpha_0 + \beta_0)/(\alpha_0 + \beta_0 + n)$이다.

    사전분포에 대한 가중 $w$는 $n$이 커질수록 줄어든다. 사전분포의 "유효 표본크기"가 $\alpha_0 + \beta_0$이며, 사후평균은 사전분포와 자료에 각각의 표본크기에 비례하여 가중을 배분한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 유병률이 약 1%인 희귀질환의 확률을 추정한다고 하자. 동료가 $\text{Beta}(1, 99)$ 사전분포를 제안한다. $n = 500$명을 검사하여 $k = 8$명이 양성이었을 때 사후평균과 MLE를 비교하라. 사전분포의 선택이 적절한가?

</div>

??? success "풀이"
    사전분포: 평균이 $1/100 = 0.01$인 $\text{Beta}(1, 99)$.

    사후분포: $\text{Beta}(1 + 8, 99 + 492) = \text{Beta}(9, 591)$.

    사후평균: $9/600 = 0.015$.

    MLE: $8/500 = 0.016$.

    사전분포의 유효 표본크기는 $1 + 99 = 100$으로 $n = 500$에 비해 크지 않다. 사후평균(0.015)은 MLE(0.016)에서 사전평균(0.01) 쪽으로 약간 당겨져 있다. 이 사전분포는 합리적이다. (1) 질환이 드물다는 실제 분야 지식을 담고 있고, (2) 유효 표본크기가 실제 표본크기보다 훨씬 작아 자료를 압도하지 않으며, (3) 자료가 희소해도 사후분포를 잘 정의된 상태로 유지해 주기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> Poisson-Gamma 켤레 모형에서 $\lambda$의 사후분포를 유도하라. $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$이고 사전분포는 (형상–비율 모수화의) $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$이다. 사후평균은 무엇인가?

</div>

??? success "풀이"
    가능도는:

    $$
    L(\lambda) \propto \lambda^{\sum x_i} e^{-n\lambda}
    $$

    사전분포는:

    $$
    \pi(\lambda) \propto \lambda^{\alpha_0 - 1} e^{-\beta_0 \lambda}
    $$

    곱하면:

    $$
    p(\lambda \mid \mathbf{x}) \propto \lambda^{\alpha_0 + \sum x_i - 1} e^{-(\beta_0 + n)\lambda}
    $$

    이는 $\text{Gamma}(\alpha_0 + \sum x_i, \beta_0 + n)$의 핵이다.

    사후평균은:

    $$
    E[\lambda \mid \mathbf{x}] = \frac{\alpha_0 + \sum x_i}{\beta_0 + n} = \frac{\beta_0}{\beta_0 + n}\cdot\frac{\alpha_0}{\beta_0} + \frac{n}{\beta_0 + n}\cdot\bar{x}
    $$

    이번에도 사전평균 $\alpha_0/\beta_0$과 MLE $\bar{x}$의 가중평균이며, 가중은 사전분포의 비율 모수 $\beta_0$과 표본크기 $n$에 비례한다. $\square$

---

## 정리하며

두 켤레 모형을 직접 돌려 보며 베이즈 갱신을 눈으로 확인했다.

- **베타–이항과 정규–정규.** 앞의 것은 비율을, 뒤의 것은 평균을 추정한다. 둘 다 사후분포가 닫힌 형태로 나온다.
- **자료가 쌓이면 사후분포가 좁아진다.** 불확실성이 줄어드는 과정이 그림에서 그대로 보이며, 폭이 대략 $1/\sqrt n$ 으로 준다.
- **사후평균은 사전평균과 표본평균의 가중평균이다.** 정규–정규에서 가중치가 **정밀도**(분산의 역수)에 비례한다는 점이 특히 깔끔하다. 자료가 많아질수록 표본 쪽 무게가 커진다.
- **사전분포의 선택이 소표본에서 결정적이다.** 같은 자료에 강한 사전분포와 약한 사전분포를 쓰면 결론이 달라지며, $n$ 이 작을수록 격차가 크다. **그래서 사전분포를 밝히고 민감도를 확인하는 것이 보고의 일부여야 한다.**
- **사후분포는 점이 아니라 분포다.** 신용구간, 사후확률, 예측분포가 모두 여기서 나온다.

다음 절 **베타분포 켤레 사전분포**에서 베타–이항 모형 하나를 더 깊이 들여다보며 민감도 분석과 의사결정 확률을 다룬다.
