# Gaussian 최대가능도

## 개요

정규분포의 최대가능도추정(MLE)은 닫힌 형태의 추정량을 준다: 평균은 $\hat{\mu} = \bar{X}$, 분산은 $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$이다. 이 페이지에서는 해석적 MLE를 수치최적화와 대조해 확인하고, 로그가능도 곡면을 시각화하며, 유한표본 편향을 정량화하고, Cramer-Rao 하한을 유도하며, 신뢰구간의 포함확률을 검증하고, Gaussian MLE를 VaR 추정에 적용한다.

## 해석적 MLE

i.i.d. 관측값 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$에 대해 로그가능도는:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

편도함수를 0으로 놓으면 MLE를 얻는다:

$$\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$

유의: 분산의 MLE는 $n-1$이 아니라 $n$으로 나눈다.

```python
import numpy as np
from scipy import optimize

def mle_analytical_vs_numerical(seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 2.0
    n = 50
    data = rng.normal(mu_true, sigma_true, n)

    # Analytical MLE
    mu_mle = data.mean()
    sigma2_mle = np.mean((data - mu_mle)**2)

    # Numerical MLE (parameterize log(sigma^2) for unconstrained optimization)
    def neg_ll(params):
        mu, ls2 = params
        s2 = np.exp(ls2)
        return n/2*np.log(2*np.pi*s2) + np.sum((data-mu)**2)/(2*s2)

    res = optimize.minimize(neg_ll, [0, 0], method='Nelder-Mead')
    mu_num, s2_num = res.x[0], np.exp(res.x[1])

    print(f"Analytical: mu={mu_mle:.6f}, sigma²={sigma2_mle:.6f}")
    print(f"Numerical:  mu={mu_num:.6f}, sigma²={s2_num:.6f}")
```

!!! tip "일치"
    해석적 해와 수치해가 소수점 아래 여러 자리까지 일치하여 닫힌 형태 유도가 확인된다.

## 로그가능도 곡면

로그가능도는 $(\hat{\mu}, \hat{\sigma}^2)$에서 유일한 최댓값을 갖는 매끄러운 오목 곡면을 이룬다. 프로파일 가능도를 쓰면 각 모수를 따로 시각화할 수 있다.

```python
import matplotlib.pyplot as plt
from scipy import stats

def loglikelihood_surface(seed=42):
    rng = np.random.default_rng(seed)
    n = 30
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)

    mu_mle = data.mean()
    s2_mle = np.mean((data - mu_mle)**2)

    mu_r = np.linspace(mu_mle - 2, mu_mle + 2, 200)
    s2_r = np.linspace(s2_mle * 0.3, s2_mle * 3, 200)
    MU, S2 = np.meshgrid(mu_r, s2_r)

    LL = np.zeros_like(MU)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            LL[i, j] = (-n/2*np.log(2*np.pi*S2[i,j])
                        - np.sum((data-MU[i,j])**2)/(2*S2[i,j]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Contour plot
    axes[0].contour(MU, S2, LL, levels=30, cmap='viridis')
    axes[0].plot(mu_mle, s2_mle, 'r*', ms=15, label='MLE')
    axes[0].set_xlabel('mu'); axes[0].set_ylabel('sigma²')
    axes[0].set_title('Log-Likelihood Contours')
    axes[0].legend()

    # Profile for mu
    prof_mu = [-np.sum((data-m)**2)/(2*s2_mle) for m in mu_r]
    prof_mu = np.array(prof_mu) - max(prof_mu)
    axes[1].plot(mu_r, prof_mu, 'b-', lw=2)
    axes[1].axvline(mu_mle, color='red', ls='--')
    axes[1].set_xlabel('mu'); axes[1].set_title('Profile for mu')

    # Profile for sigma²
    prof_s = [-n/2*np.log(s)-np.sum((data-mu_mle)**2)/(2*s) for s in s2_r]
    prof_s = np.array(prof_s) - max(prof_s)
    axes[2].plot(s2_r, prof_s, 'b-', lw=2)
    axes[2].axvline(s2_mle, color='red', ls='--')
    axes[2].set_xlabel('sigma²'); axes[2].set_title('Profile for sigma²')

    plt.tight_layout()
    plt.show()
```

## 유한표본 편향

평균의 MLE $\hat{\mu}$은 불편이지만 분산의 MLE $\hat{\sigma}^2_{\text{MLE}}$은 아래로 편향되어 있다:

$$E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$$

편향은 $-\sigma^2/n$으로 $n \to \infty$일 때 사라진다(따라서 MLE는 점근적으로 불편이다).

```python
def finite_sample_bias(n_sim=200_000, seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 3.0
    sigma2 = sigma_true**2
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]

    for n in sample_sizes:
        samp = rng.normal(mu_true, sigma_true, (n_sim, n))
        s2_mle = np.var(samp, axis=1, ddof=0)
        s2_ub  = np.var(samp, axis=1, ddof=1)
        print(f"n={n:>4}  E[sigma²_MLE]={s2_mle.mean():.4f}  "
              f"E[S²]={s2_ub.mean():.4f}  Bias(MLE)={s2_mle.mean()-sigma2:.4f}")
```

## Fisher 정보량과 Cramer-Rao 하한

$N(\mu, \sigma^2)$의 **Fisher 정보행렬**은:

$$I_n(\mu, \sigma^2) = \begin{pmatrix} n/\sigma^2 & 0 \\ 0 & n/(2\sigma^4) \end{pmatrix}$$

**Cramer-Rao 하한(CRLB)**은 임의의 불편추정량이 가질 수 있는 최소 분산을 준다:

$$\text{Var}(\hat{\mu}) \geq \frac{\sigma^2}{n}, \qquad \text{Var}(\hat{\sigma}^2) \geq \frac{2\sigma^4}{n}$$

평균의 MLE는 CRLB를 정확히 달성한다. 분산의 MLE는 점근적으로 도달한다.

```python
def fisher_information_crlb(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [10, 25, 50, 100, 500]

    print("For mu: CRLB = sigma²/n")
    for n in sample_sizes:
        mu_h = np.array([rng.normal(5, sigma, n).mean() for _ in range(n_sim)])
        print(f"  n={n:>4}  Var(mu_hat)={mu_h.var():.6f}  "
              f"CRLB={sigma**2/n:.6f}  Ratio={mu_h.var()/(sigma**2/n):.4f}")

    print("\nFor sigma²: CRLB = 2*sigma⁴/n")
    for n in sample_sizes:
        s2_h = np.array([np.var(rng.normal(5, sigma, n)) for _ in range(n_sim)])
        print(f"  n={n:>4}  Var(sigma²_hat)={s2_h.var():.6f}  "
              f"CRLB={2*sigma**4/n:.6f}  Ratio={s2_h.var()/(2*sigma**4/n):.4f}")
```

!!! info "효율성"
    비 $\text{Var}/\text{CRLB}$는 $\hat{\mu}$에서 정확히 1이고(모든 표본크기에서 효율적이다), $\hat{\sigma}^2$에서는 $n \to \infty$일 때 1로 수렴한다(점근적으로 효율적이다).

## 신뢰구간의 포함확률

Gaussian 모형에서는 세 종류의 신뢰구간이 나온다:

| 모수 | 알려진 것 | 구간의 종류 | 추축량 |
|-----------|-------|---------------|-----------------|
| $\mu$ | $\sigma$를 앎 | $z$-구간 | $\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \sim N(0,1)$ |
| $\mu$ | $\sigma$를 모름 | $t$-구간 | $\frac{\bar{X}-\mu}{S/\sqrt{n}} \sim t_{n-1}$ |
| $\sigma^2$ | $\mu$를 모름 | $\chi^2$-구간 | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ |

```python
def confidence_interval_coverage(seed=42):
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 10.0, 3.0
    n, alpha, n_sim = 25, 0.05, 50_000

    z_ok = t_ok = chi_ok = 0
    for _ in range(n_sim):
        d = rng.normal(mu_true, sigma_true, n)
        xb, s, s2 = d.mean(), d.std(ddof=1), d.var(ddof=1)

        # z-interval (sigma known)
        z_c = stats.norm.ppf(1 - alpha/2)
        if xb - z_c*sigma_true/np.sqrt(n) <= mu_true <= xb + z_c*sigma_true/np.sqrt(n):
            z_ok += 1

        # t-interval (sigma unknown)
        t_c = stats.t.ppf(1 - alpha/2, n-1)
        if xb - t_c*s/np.sqrt(n) <= mu_true <= xb + t_c*s/np.sqrt(n):
            t_ok += 1

        # chi-squared interval for sigma²
        lo = (n-1)*s2 / stats.chi2.ppf(1-alpha/2, n-1)
        hi = (n-1)*s2 / stats.chi2.ppf(alpha/2, n-1)
        if lo <= sigma_true**2 <= hi:
            chi_ok += 1

    print(f"z-interval (mu, sigma known):  {z_ok/n_sim:.1%} (target: {1-alpha:.1%})")
    print(f"t-interval (mu, sigma unknown): {t_ok/n_sim:.1%} (target: {1-alpha:.1%})")
    print(f"chi²-interval (sigma²):        {chi_ok/n_sim:.1%} (target: {1-alpha:.1%})")
```

!!! success "포함확률이 맞는다"
    세 구간 모두 명목 95% 포함확률을 달성하여 이론적 유도가 확인된다.

## 금융 응용: Value at Risk

수준 $\alpha$에서의 **VaR(Value at Risk)**는 확률 $\alpha$로 초과되는 손실이다. 일별 수익률에 대한 Gaussian 모형 $R \sim N(\hat{\mu}, \hat{\sigma}^2)$ 아래에서:

$$\text{VaR}_\alpha = -(\hat{\mu} + z_\alpha \hat{\sigma})$$

여기서 $z_\alpha = \mathcal{N}^{-1}(\alpha)$는 정규분위수이다.

```python
def var_estimation_finance(seed=42):
    rng = np.random.default_rng(seed)
    mu_d = 0.08/252
    sig_d = 0.20/np.sqrt(252)
    n = 504
    df = 5

    # Simulate t-distributed returns (heavier tails than normal)
    returns = mu_d + sig_d * rng.standard_t(df, n) / np.sqrt(df/(df-2))
    mu_hat = returns.mean()
    sig_hat = np.sqrt(np.mean((returns - mu_hat)**2))

    for alpha in [0.01, 0.025, 0.05, 0.10]:
        v_p = -(mu_hat + stats.norm.ppf(alpha) * sig_hat)
        v_h = -np.percentile(returns, alpha * 100)
        print(f"alpha={alpha:.3f}  Parametric VaR={v_p*100:.3f}%  "
              f"Historical VaR={v_h*100:.3f}%  Ratio={v_h/v_p:.3f}")
```

!!! warning "모형 위험"
    참 수익률 분포의 꼬리가 (금융에서 흔하듯) 정규보다 두꺼우면 Gaussian VaR는 꼬리 위험을 **과소평가**한다. 1% 수준의 역사적 VaR가 대개 모수적 VaR보다 크며, 이는 참 분포의 두꺼운 꼬리를 반영한다.

## 해석

- Gaussian MLE는 우아한 **닫힌 형태의 해**를 가지며 평균추정량은 (CRLB를 달성하여) 전역적으로 효율적이다.
- **분산의 MLE는 편향**되어 있어 $(n-1)/n$배가 되지만, 이 편향은 점근적으로 사라지고 Bessel 인자로 보정할 수 있다.
- **로그가능도 곡면**은 오목이며 최댓값이 유일하여 최적화가 쉽다.
- Fisher 정보량은 **Cramer-Rao 하한**을 통해 추정 정밀도의 근본적인 한계를 준다.
- 정규성 아래에서 세 가지 표준 신뢰구간($z$, $t$, $\chi^2$) 모두 명목 포함확률을 달성한다.
- 금융에서 Gaussian 가정은 간단한 VaR 공식을 주지만 **꼬리 위험을 체계적으로 과소평가**한다.

## 연습문제

**연습문제 1.**
로그가능도를 미분하고 1계 조건을 풀어 $\mu$와 $\sigma^2$의 MLE를 유도하라.

??? success "연습문제 1 풀이"
    i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$의 로그가능도는:

    $$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(X_i - \mu)^2$$

    **$\mu$에 대해:** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(X_i - \mu) = 0$에서 $\sum X_i = n\mu$이므로 $\hat{\mu} = \bar{X}$이다.

    **$\sigma^2$에 대해:** $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n(X_i - \mu)^2 = 0$.

    풀면 $n\sigma^2 = \sum(X_i - \mu)^2$이고, $\hat{\mu} = \bar{X}$을 대입하면:

    $$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$$

    2계 조건이 이것이 최댓값임을 확인해 준다(MLE에서 Hessian이 음정부호이다). $\square$

---

**연습문제 2.**
$N(\mu, \sigma^2)$ 모형에서 $\mu$에 대한 관측값당 Fisher 정보량이 $I(\mu) = 1/\sigma^2$이고, 관측값 $n$개로 $\mu$를 추정할 때의 CRLB가 $\sigma^2/n$임을 보여라.

??? success "연습문제 2 풀이"
    관측값 하나의 로그가능도는:

    $$\ell(\mu; x) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

    점수함수는:

    $$\frac{\partial \ell}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$

    관측값당 Fisher 정보량은:

    $$I_1(\mu) = E\left[\left(\frac{\partial \ell}{\partial \mu}\right)^2\right] = E\left[\frac{(X-\mu)^2}{\sigma^4}\right] = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}$$

    i.i.d. 관측값 $n$개에 대해 $I_n(\mu) = nI_1(\mu) = n/\sigma^2$이다. CRLB는:

    $$\text{Var}(\hat{\mu}) \geq \frac{1}{I_n(\mu)} = \frac{\sigma^2}{n}$$

    $\text{Var}(\bar{X}) = \sigma^2/n$이므로 표본평균은 CRLB를 정확히 달성하며 따라서 **효율적인** 추정량이다. $\square$

---

**연습문제 3.**
$n = 25$, $\bar{x} = 12.4$, $s = 3.1$일 때 정규모집단 평균의 95% 신뢰구간을 구성하라. ($\sigma$를 아는 척한) $z$-구간과 올바른 $t$-구간을 비교하라.

??? success "연습문제 3 풀이"
    **$z$-구간** ($s$를 $\sigma$로 취급): $z_{0.025} = 1.960$.

    $$\bar{x} \pm z_{0.025}\frac{s}{\sqrt{n}} = 12.4 \pm 1.960 \times \frac{3.1}{\sqrt{25}} = 12.4 \pm 1.216$$

    $$\text{CI}_z = [11.184, 13.616]$$

    **$t$-구간** (올바른 방법): $t_{24, 0.025} = 2.064$.

    $$\bar{x} \pm t_{24, 0.025}\frac{s}{\sqrt{n}} = 12.4 \pm 2.064 \times \frac{3.1}{\sqrt{25}} = 12.4 \pm 1.280$$

    $$\text{CI}_t = [11.120, 13.680]$$

    $t$-구간이 (약 5%) 더 넓은데, $\sigma$를 추정하는 데서 오는 추가 불확실성을 반영하기 때문이다. $n = 25$에서는 차이가 크지 않지만 $n$이 작으면 훨씬 커진다. $\square$

---

**연습문제 4.**
어떤 포트폴리오의 일별 수익률 504개에서 표본평균 $\hat{\mu} = 0.035\%$, 표본표준편차 $\hat{\sigma} = 1.30\%$를 얻었다. 1%와 5% 모수적(Gaussian) VaR를 계산하라. 참 수익률이 자유도 5인 $t$-분포를 따른다면 Gaussian VaR가 참 VaR를 과대평가할 것 같은가, 과소평가할 것 같은가?

??? success "연습문제 4 풀이"
    **Gaussian VaR:**

    $$\text{VaR}_{1\%} = -(\hat{\mu} + z_{0.01}\hat{\sigma}) = -(0.035\% + (-2.326)(1.30\%)) = -(0.035\% - 3.024\%) = 2.989\%$$

    $$\text{VaR}_{5\%} = -(\hat{\mu} + z_{0.05}\hat{\sigma}) = -(0.035\% + (-1.645)(1.30\%)) = -(0.035\% - 2.139\%) = 2.103\%$$

    **두꺼운 꼬리의 효과:** 자유도 5인 $t$-분포는 정규보다 꼬리가 두껍다. 그 1번째 백분위수는 $t_{5, 0.01} = -3.365$로 ($z_{0.01} = -2.326$과 비교된다). Gaussian VaR는 정규 모형이 꼬리의 초과 확률을 담지 못하므로 참 꼬리 위험을 **과소평가**한다.

    이는 체계적인 문제이다: Gaussian VaR는 꼬리가 두꺼운 분포에서 보수적이지 않은데, 금융 수익률이 정확히 그런 상황이다. $\square$

---

**연습문제 5.**
$\sigma^2$의 MLE가 점근적으로 효율적임을, 즉 $n \to \infty$일 때 $n \cdot \text{Var}(\hat{\sigma}^2_{\text{MLE}}) \to 2\sigma^4$임을 증명하라.

??? success "연습문제 5 풀이"
    $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2 = \frac{n-1}{n}S^2$이다.

    정규 자료에서 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$이므로 $\text{Var}(S^2) = 2\sigma^4/(n-1)$이다.

    $$\text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \left(\frac{n-1}{n}\right)^2 \text{Var}(S^2) = \left(\frac{n-1}{n}\right)^2 \cdot \frac{2\sigma^4}{n-1} = \frac{2(n-1)\sigma^4}{n^2}$$

    따라서:

    $$n \cdot \text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \frac{2(n-1)\sigma^4}{n} \to 2\sigma^4 \quad (n \to \infty)$$

    $\sigma^2$에 대한 CRLB는 $1/I_n(\sigma^2) = 2\sigma^4/n$이므로 $n \cdot \text{CRLB} = 2\sigma^4$이다.

    점근분산이 CRLB와 같으므로 $\hat{\sigma}^2_{\text{MLE}}$은 점근적으로 효율적이다. $\square$
