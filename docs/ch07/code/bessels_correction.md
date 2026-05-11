# Bessel's Correction

## Overview

Bessel's correction replaces the divisor $n$ with $n - 1$ in the sample variance formula, yielding an unbiased estimator of $\sigma^2$. This page verifies unbiasedness across distributions, confirms the chi-squared distributional result for normal data, demonstrates the independence of $\bar{X}$ and $S^2$ (a property unique to the normal distribution), explores the standard deviation bias from Jensen's inequality, highlights software default pitfalls, and applies these ideas to financial tracking error estimation.

## Unbiasedness Across Distributions

The Bessel-corrected sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2$ satisfies:

$$E[S^2] = \sigma^2$$

for **any** distribution with finite variance, not just the normal.

```python
import numpy as np

def unbiasedness_across_distributions(n_sim=200_000, seed=42):
    rng = np.random.default_rng(seed)
    sigma = 4.0
    sigma2 = sigma**2
    n = 20

    distributions = {
        f'Normal(0, {sigma2})':    (lambda: rng.normal(0, sigma, n), sigma2),
        f'Exp(scale={sigma})':     (lambda: rng.exponential(sigma, n), sigma2),
        f'Uniform':                (lambda: rng.uniform(0, 2*sigma*np.sqrt(3), n), sigma2),
        f'Chi²(df={int(sigma2)})': (lambda: rng.chisquare(int(sigma2), n), 2*sigma2),
    }

    for name, (sampler, true_var) in distributions.items():
        s2_vals = np.array([np.var(sampler(), ddof=1) for _ in range(n_sim)])
        print(f"{name:<25} True σ²={true_var:.2f}  "
              f"E[S²]={s2_vals.mean():.4f}  Bias={s2_vals.mean()-true_var:.4f}")
```

!!! tip "Distribution-free result"
    The proof of $E[S^2] = \sigma^2$ uses only the identity $\sum(X_i - \bar{X})^2 = \sum(X_i - \mu)^2 - n(\bar{X} - \mu)^2$ and linearity of expectation. No distributional assumption is needed beyond finite variance.

## Chi-Squared Distribution

For normal data $X_i \sim N(\mu, \sigma^2)$, the scaled sample variance follows a chi-squared distribution:

$$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$

This exact distributional result is the foundation for chi-squared tests and confidence intervals for $\sigma^2$.

```python
import matplotlib.pyplot as plt
from scipy import stats

def chi_squared_verification(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 25, 50]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, n in zip(axes.flat, sample_sizes):
        samples = rng.normal(0, sigma, (n_sim, n))
        s2 = np.var(samples, axis=1, ddof=1)
        chi2_vals = (n - 1) * s2 / sigma**2

        ax.hist(chi2_vals, bins=80, density=True, alpha=0.6, color='steelblue')
        x = np.linspace(0, stats.chi2.ppf(0.999, n-1), 200)
        ax.plot(x, stats.chi2.pdf(x, n-1), 'r-', linewidth=2, label=f'chi²(df={n-1})')
        ax.set_title(f'n = {n}')
        ax.legend()
    plt.suptitle('(n-1)S²/σ² ~ chi²(n-1) for Normal Data')
    plt.tight_layout()
    plt.show()
```

From the chi-squared distribution, we can immediately derive:

$$E[S^2] = \sigma^2, \qquad \text{Var}(S^2) = \frac{2\sigma^4}{n-1}$$

## Independence of X-bar and S-squared

**Cochran's theorem** states that for normal data, $\bar{X}$ and $S^2$ are independent. This is a remarkable property that does **not** hold for non-normal distributions.

```python
def independence_xbar_s2(sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    n = 20

    # Normal
    samp_n = rng.normal(5, sigma, (n_sim, n))
    xbar_n = samp_n.mean(axis=1)
    s2_n   = np.var(samp_n, axis=1, ddof=1)
    corr_n = np.corrcoef(xbar_n, s2_n)[0, 1]

    # Exponential
    samp_e = rng.exponential(sigma, (n_sim, n))
    xbar_e = samp_e.mean(axis=1)
    s2_e   = np.var(samp_e, axis=1, ddof=1)
    corr_e = np.corrcoef(xbar_e, s2_e)[0, 1]

    print(f"Normal:      Corr(X̄, S²) = {corr_n:.6f}  (≈ 0)")
    print(f"Exponential: Corr(X̄, S²) = {corr_e:.6f}  (≠ 0)")
```

!!! note "Why this matters"
    The independence of $\bar{X}$ and $S^2$ is what makes the $t$-distribution derivation work. The $t$-statistic $T = \frac{\bar{X} - \mu}{S/\sqrt{n}}$ involves the ratio of $\bar{X} - \mu$ (related to a normal) and $S$ (related to a chi-squared). Independence ensures this ratio has the $t$-distribution.

## Standard Deviation Bias

Although $S^2$ is unbiased for $\sigma^2$, its square root $S$ is **biased** for $\sigma$. By Jensen's inequality (since $\sqrt{\cdot}$ is concave):

$$E[S] = E[\sqrt{S^2}] < \sqrt{E[S^2]} = \sigma$$

The correction factor $c_4$ depends on $n$:

$$c_4(n) = \sqrt{\frac{2}{n-1}} \cdot \frac{\Gamma(n/2)}{\Gamma((n-1)/2)}$$

and an unbiased estimator of $\sigma$ is $S/c_4$.

```python
from scipy.special import gamma as gamma_func

def std_deviation_bias(sigma=3.0, n_sim=200_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]

    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s = np.std(samples, axis=1, ddof=1)
        c4 = np.sqrt(2 / (n - 1)) * gamma_func(n / 2) / gamma_func((n - 1) / 2)
        print(f"n={n:>4}  E[S]={s.mean():.4f}  σ={sigma:.4f}  "
              f"Bias={s.mean()-sigma:.4f}  c₄={c4:.4f}  E[S/c₄]={(s/c4).mean():.4f}")
```

!!! warning "Bias is largest for small samples"
    For $n = 3$, $c_4 \approx 0.886$, so $E[S] \approx 0.886\sigma$ — the standard deviation is underestimated by about 11%. By $n = 50$, the bias is less than 0.5%.

## Software Defaults Pitfall

Different software packages use different defaults for the variance divisor:

```python
import numpy as np

data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
n = len(data)

print(f"np.var(data)          = {np.var(data):.4f}  <- divides by n={n}  (BIASED)")
print(f"np.var(data, ddof=1)  = {np.var(data, ddof=1):.4f}  <- divides by n-1={n-1}  (UNBIASED)")
```

!!! danger "Always check your divisor"
    NumPy defaults to `ddof=0` (biased), while R and pandas default to `ddof=1` (unbiased). Always explicitly specify `ddof=1` in NumPy when computing sample variance.

## Financial Application: Tracking Error

**Tracking error** measures how closely a portfolio follows its benchmark, defined as the standard deviation of excess returns (portfolio return minus benchmark return). Bessel's correction matters when estimating tracking error from short histories.

```python
def tracking_error_estimation(seed=42):
    rng = np.random.default_rng(seed)
    n_months = 36
    te_true_monthly = 0.01
    te_true_annual = te_true_monthly * np.sqrt(12)
    n_sim = 50_000

    te_n, te_n1 = [], []
    for _ in range(n_sim):
        excess = rng.normal(0.002, te_true_monthly, n_months)
        te_n.append(np.std(excess, ddof=0) * np.sqrt(12))
        te_n1.append(np.std(excess, ddof=1) * np.sqrt(12))

    te_n, te_n1 = np.array(te_n), np.array(te_n1)
    for name, est in [('ddof=0', te_n), ('ddof=1', te_n1)]:
        print(f"{name:<10} Mean={est.mean()*100:.3f}%  "
              f"Bias={(est.mean()-te_true_annual)*100:.3f}%  "
              f"RMSE={np.sqrt(np.mean((est-te_true_annual)**2))*100:.3f}%")
    print(f"True TE: {te_true_annual*100:.3f}%")
```

## Interpretation

- Bessel's correction produces an **unbiased** estimator of $\sigma^2$ for any distribution, but the $\chi^2$ distributional result requires normality.
- The **independence** of $\bar{X}$ and $S^2$ is specific to normal populations and is the key ingredient for Student's $t$-test.
- Unbiasedness of $S^2$ does **not** imply unbiasedness of $S$. Jensen's inequality causes $S$ to underestimate $\sigma$, especially for small $n$.
- Always be explicit about the `ddof` parameter in NumPy to avoid silent errors.
- In finance, tracking error estimation from short windows benefits meaningfully from Bessel's correction.

## Exercises

**Exercise 1.**
Show that $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ when $X_i \sim N(\mu, \sigma^2)$, by writing the sum of squares in terms of independent standard normals.

??? success "Solution to Exercise 1"
    Let $Z_i = (X_i - \mu)/\sigma \sim N(0,1)$ iid. Then:

    $$\frac{1}{\sigma^2}\sum_{i=1}^n(X_i - \bar{X})^2 = \sum_{i=1}^n Z_i^2 - n\bar{Z}^2$$

    where $\bar{Z} = \frac{1}{n}\sum Z_i$. Now $\sum Z_i^2 \sim \chi^2_n$ and $n\bar{Z}^2 = \left(\sqrt{n}\bar{Z}\right)^2 \sim \chi^2_1$ since $\sqrt{n}\bar{Z} \sim N(0,1)$.

    By Cochran's theorem, since the quadratic forms are based on an orthogonal decomposition of $\mathbb{R}^n$ into complementary subspaces of dimensions $n-1$ and $1$:

    $$\sum Z_i^2 - n\bar{Z}^2 \sim \chi^2_{n-1}$$

    and the two components are independent. Therefore $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$. $\square$

---

**Exercise 2.**
Prove that for exponential data $X_i \sim \text{Exp}(\lambda)$, the sample mean $\bar{X}$ and sample variance $S^2$ are **not** independent. (Hint: compute $\text{Cov}(\bar{X}, S^2)$ using the third central moment.)

??? success "Solution to Exercise 2"
    For the exponential distribution with rate $\lambda$: $\mu = 1/\lambda$, $\sigma^2 = 1/\lambda^2$, and the third central moment $\mu_3 = E[(X - \mu)^3] = 2/\lambda^3$.

    We can show that:

    $$\text{Cov}(\bar{X}, S^2) = \frac{\mu_3}{n}$$

    This is a general result. The proof uses:

    $$\text{Cov}(\bar{X}, S^2) = E[\bar{X} \cdot S^2] - E[\bar{X}]\cdot E[S^2]$$

    Expanding $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ and using $\bar{X} = \frac{1}{n}\sum X_i$, after algebraic manipulation:

    $$\text{Cov}(\bar{X}, S^2) = \frac{1}{n}E[(X_1 - \mu)^3] = \frac{\mu_3}{n} = \frac{2}{n\lambda^3}$$

    Since $\mu_3 \neq 0$ for the exponential (it is positively skewed), we have $\text{Cov}(\bar{X}, S^2) \neq 0$, so they are not independent.

    For the normal distribution, $\mu_3 = 0$ (symmetric), so this covariance is zero. Zero covariance combined with joint normality of the underlying quadratic forms gives full independence. $\square$

---

**Exercise 3.**
Using Jensen's inequality, explain why $E[\sqrt{S^2}] < \sigma$. For $n = 5$ and normal data, compute the exact value of $c_4$ and the percentage bias in $S$ as an estimator of $\sigma$.

??? success "Solution to Exercise 3"
    Jensen's inequality states that for a concave function $g$ (such as $g(x) = \sqrt{x}$):

    $$E[g(X)] \leq g(E[X])$$

    with strict inequality when $X$ is non-degenerate. Applying this to $S^2$:

    $$E[S] = E[\sqrt{S^2}] < \sqrt{E[S^2]} = \sqrt{\sigma^2} = \sigma$$

    For $n = 5$:

    $$c_4 = \sqrt{\frac{2}{4}} \cdot \frac{\Gamma(5/2)}{\Gamma(2)} = \sqrt{\frac{1}{2}} \cdot \frac{\frac{3}{4}\sqrt{\pi}}{1} = \frac{1}{\sqrt{2}} \cdot \frac{3\sqrt{\pi}}{4}$$

    Computing: $\Gamma(5/2) = \frac{3}{2}\cdot\frac{1}{2}\cdot\sqrt{\pi} = \frac{3\sqrt{\pi}}{4}$ and $\Gamma(2) = 1! = 1$.

    $$c_4 = \frac{1}{\sqrt{2}} \cdot \frac{3\sqrt{\pi}}{4} = \frac{3\sqrt{\pi}}{4\sqrt{2}} \approx \frac{3 \times 1.7725}{5.6569} \approx 0.9400$$

    The percentage bias is $(c_4 - 1) \times 100\% \approx -6.0\%$. So $S$ underestimates $\sigma$ by about 6% on average when $n = 5$. $\square$

---

**Exercise 4.**
A portfolio tracker has 36 months of excess returns. The estimated annualized tracking error using `ddof=1` is 3.8%. Construct a 95% confidence interval for the true annualized tracking error, assuming normality.

??? success "Solution to Exercise 4"
    Monthly tracking error estimate: $\hat{\sigma}_m = 3.8\%/\sqrt{12} \approx 1.097\%$. The sample variance is $\hat{\sigma}_m^2$.

    With $n = 36$ months and $\nu = n - 1 = 35$ degrees of freedom:

    $$\frac{(n-1)\hat{\sigma}_m^2}{\sigma_m^2} \sim \chi^2_{35}$$

    The 95% CI for $\sigma_m^2$ is:

    $$\left[\frac{35 \hat{\sigma}_m^2}{\chi^2_{35, 0.975}}, \frac{35 \hat{\sigma}_m^2}{\chi^2_{35, 0.025}}\right]$$

    Using $\chi^2_{35, 0.975} = 53.20$ and $\chi^2_{35, 0.025} = 20.57$:

    $$\sigma_m^2 \in \left[\frac{35 \times 1.097^2}{53.20}, \frac{35 \times 1.097^2}{20.57}\right] = [0.7916, 2.0477]$$

    Taking square roots and annualizing (multiply by $\sqrt{12}$):

    $$\sigma_{\text{annual}} \in [\sqrt{0.7916} \times \sqrt{12}, \sqrt{2.0477} \times \sqrt{12}] = [3.08\%, 4.96\%]$$

    This is a wide interval, reflecting the imprecision of volatility estimates from only 3 years of monthly data. $\square$

---

**Exercise 5.**
Explain the "general principle" of degrees of freedom: when estimating variance after fitting a model with $k$ parameters, we divide by $n - k$. Give three examples.

??? success "Solution to Exercise 5"
    **General principle:** Fitting a model with $k$ estimated parameters imposes $k$ constraints on the residuals (analogous to $\sum(X_i - \bar{X}) = 0$ for $k=1$). The residuals have only $n - k$ degrees of freedom, so dividing the residual sum of squares by $n - k$ gives an unbiased variance estimate.

    **Example 1: One-sample variance.** With $k = 1$ (estimating $\mu$ by $\bar{X}$), the constraint is $\sum(X_i - \bar{X}) = 0$, and we divide by $n - 1$.

    **Example 2: Simple linear regression.** With $Y_i = \beta_0 + \beta_1 x_i + \epsilon_i$, we estimate $k = 2$ parameters. The residual variance is:

    $$\hat{\sigma}^2 = \frac{\sum(Y_i - \hat{Y}_i)^2}{n - 2}$$

    **Example 3: Multiple regression with $p$ predictors.** With $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$ and $k = p$ coefficients (including intercept):

    $$\hat{\sigma}^2 = \frac{\|\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2}{n - p}$$

    In each case, the denominator equals the dimension of the residual space (the orthogonal complement of the column space of the design matrix), ensuring unbiasedness. $\square$
