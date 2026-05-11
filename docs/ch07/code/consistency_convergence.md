# Consistency and Convergence

## Overview

Consistency means that an estimator converges to the true parameter value as the sample size grows. For the sample mean, this is guaranteed by the Law of Large Numbers when the population has a finite mean. The Central Limit Theorem further describes the rate and shape of this convergence. This page demonstrates these convergence properties, explores when they fail (Cauchy distribution), and examines practical complications such as autocorrelation and financial estimation horizons.

## Strong Law of Large Numbers

The **Strong Law of Large Numbers (SLLN)** states that for iid observations with $E[|X|] < \infty$:

$$\bar{X}_n \xrightarrow{\text{a.s.}} \mu \quad \text{as } n \to \infty$$

This means that with probability 1, the running average converges to $\mu$. The following simulation plots running averages for 20 independent sequences across four distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def consistency_visualization(seed=42):
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 20

    distributions = {
        'Normal(5, 9)':    (lambda: rng.normal(5, 3, N), 5.0),
        'Exp(λ=0.5)':      (lambda: rng.exponential(2, N), 2.0),
        'Uniform(0, 10)':  (lambda: rng.uniform(0, 10, N), 5.0),
        'Chi²(df=5)':      (lambda: rng.chisquare(5, N), 5.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, (sampler, true_mu)) in zip(axes.flat, distributions.items()):
        for _ in range(n_runs):
            data = sampler()
            running_mean = np.cumsum(data) / np.arange(1, N + 1)
            ax.plot(running_mean, alpha=0.2, linewidth=0.5)
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'μ = {true_mu}')
        ax.set_xscale('log')
        ax.set_xlabel('n')
        ax.set_ylabel('X̄ₙ')
        ax.set_title(f'{name}: SLLN')
        ax.legend()
    plt.tight_layout()
    plt.show()
```

!!! tip "Visual pattern"
    All 20 sample paths converge to the red dashed line ($\mu$) as $n$ grows, regardless of the population distribution. This is the SLLN in action.

## Central Limit Theorem

The **Central Limit Theorem (CLT)** describes the distribution of $\bar{X}_n$ for large $n$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

or equivalently $\bar{X}_n \approx N(\mu, \sigma^2/n)$ for large $n$. This holds for **any** population distribution with finite variance.

```python
from scipy import stats

def clt_demonstration(seed=42):
    rng = np.random.default_rng(seed)
    n_sim = 20_000

    populations = {
        'Normal(5, 4)':     (lambda n: rng.normal(5, 2, n), 5.0, 4.0),
        'Exp(λ=0.5)':       (lambda n: rng.exponential(2, n), 2.0, 4.0),
        'Uniform(0, 10)':   (lambda n: rng.uniform(0, 10, n), 5.0, 100/12),
        'Bernoulli(0.3)':   (lambda n: rng.binomial(1, 0.3, n), 0.3, 0.21),
    }

    sample_sizes = [2, 5, 30]
    fig, axes = plt.subplots(len(populations), len(sample_sizes), figsize=(15, 12))

    for i, (pop_name, (sampler, mu, sigma2)) in enumerate(populations.items()):
        for j, n in enumerate(sample_sizes):
            x_bars = np.array([sampler(n).mean() for _ in range(n_sim)])
            ax = axes[i, j]
            ax.hist(x_bars, bins=60, density=True, alpha=0.6, color='steelblue')
            x = np.linspace(x_bars.min(), x_bars.max(), 200)
            se = np.sqrt(sigma2 / n)
            ax.plot(x, stats.norm.pdf(x, mu, se), 'r-', linewidth=2)
            if i == 0:
                ax.set_title(f'n = {n}')
            if j == 0:
                ax.set_ylabel(pop_name)
    plt.suptitle('Central Limit Theorem')
    plt.tight_layout()
    plt.show()
```

!!! note "Rate of convergence to normality"
    Symmetric distributions (Normal, Uniform) reach normality quickly. Skewed distributions (Exponential, Bernoulli with $p$ far from 0.5) require larger $n$. By $n = 30$, the normal approximation is adequate for most distributions.

## Cauchy Distribution: Convergence Failure

The **Cauchy distribution** has density $f(x) = \frac{1}{\pi(1 + x^2)}$ and no finite mean ($E[|X|] = \infty$). As a result, the sample mean does **not** converge:

$$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{for all } n$$

Averaging more Cauchy observations provides no improvement whatsoever.

```python
def cauchy_failure(seed=42):
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normal — converges
    ax = axes[0]
    for _ in range(n_runs):
        data = rng.standard_normal(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylim(-2, 2)
    ax.set_title('Normal: Converges')

    # Cauchy — does NOT converge
    ax = axes[1]
    for _ in range(n_runs):
        data = rng.standard_cauchy(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Cauchy: Does NOT Converge')

    plt.suptitle('Consistency Failure: Cauchy (E[|X|] = ∞)')
    plt.tight_layout()
    plt.show()
```

!!! warning "The LLN requires finite mean"
    The Cauchy sample mean wanders erratically no matter how large $n$ is. The sample **median**, however, is consistent for the Cauchy location parameter since it does not require a finite mean.

## Autocorrelation Effects

When observations are dependent (e.g., from an AR(1) process $X_t = \rho X_{t-1} + \epsilon_t$), the variance of the sample mean is no longer $\sigma^2/n$. The correction factor is approximately:

$$\text{Var}(\bar{X}) \approx \frac{\sigma^2}{n} \cdot \frac{1 + \rho}{1 - \rho}$$

Positive autocorrelation **inflates** the variance; negative autocorrelation **deflates** it.

```python
def autocorrelation_effect(n=100, n_sim=30_000, seed=42):
    rng = np.random.default_rng(seed)
    sigma = 1.0
    rho_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95]

    for rho in rho_values:
        x_bars = []
        innov_sig = sigma * np.sqrt(max(1 - rho**2, 0.01))
        for _ in range(n_sim):
            x = np.zeros(n)
            x[0] = rng.normal(0, sigma)
            for t in range(1, n):
                x[t] = rho * x[t - 1] + rng.normal(0, innov_sig)
            x_bars.append(x.mean())

        var_emp = np.var(x_bars)
        var_iid = sigma**2 / n
        ratio = var_emp / var_iid
        print(f"ρ={rho:>5.2f}  Var(X̄)={var_emp:.6f}  σ²/n={var_iid:.6f}  Ratio={ratio:.2f}")
```

!!! danger "Financial time series"
    Financial returns often exhibit positive autocorrelation in volatility (and sometimes weak autocorrelation in returns). Ignoring this dependence leads to understating the uncertainty of the sample mean, which makes confidence intervals too narrow and hypothesis tests too liberal.

## Estimation Horizon for Detecting Positive Returns

A fundamental challenge in finance: how many years of data are needed to detect that the expected excess return is positive? With a 3% equity premium and 20% annual volatility, the probability of correctly concluding $\mu > r_f$ after $T$ years is:

$$P(\bar{X}_T > r_f) = \mathcal{N}\left(\frac{\mu - r_f}{\sigma / \sqrt{T}}\right)$$

```python
def estimation_horizon_analysis(seed=42):
    mu_annual = 0.06    # 6% expected return
    sigma_annual = 0.20  # 20% volatility
    rf = 0.03            # risk-free rate

    years = np.arange(1, 101)
    prob_detect = [stats.norm.cdf((mu_annual - rf) / (sigma_annual / np.sqrt(T)))
                   for T in years]

    for target in [0.80, 0.90, 0.95]:
        idx = np.argmax(np.array(prob_detect) >= target)
        print(f"  {target*100:.0f}% power: ~{years[idx]} years of data needed")
```

## Interpretation

- **Consistency** guarantees that the sample mean eventually gets close to $\mu$, but the convergence rate $1/\sqrt{n}$ is slow.
- The **CLT** provides the shape of the sampling distribution (approximately normal) and makes inference possible even when the population is non-normal.
- The Cauchy example is a stark reminder that the **LLN and CLT require finite moments**. Without a finite mean, the sample mean is useless.
- **Autocorrelation** is common in real data and can dramatically inflate or deflate the effective sample size.
- In finance, the **signal-to-noise ratio** for expected returns is so poor that decades of data are needed to distinguish skill from luck.

## Exercises

**Exercise 1.**
State the Weak Law of Large Numbers (WLLN) and explain how it differs from the Strong Law (SLLN). What minimal moment condition does each require?

??? success "Solution to Exercise 1"
    **WLLN:** For iid $X_1, X_2, \ldots$ with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$:

    $$\bar{X}_n \xrightarrow{P} \mu \quad \text{(convergence in probability)}$$

    This means: for every $\epsilon > 0$, $P(|\bar{X}_n - \mu| > \epsilon) \to 0$ as $n \to \infty$.

    **SLLN:** Under the weaker condition $E[|X_i|] < \infty$ (finite first moment, no variance needed):

    $$\bar{X}_n \xrightarrow{\text{a.s.}} \mu \quad \text{(almost sure convergence)}$$

    Almost sure convergence means $P(\lim_{n\to\infty} \bar{X}_n = \mu) = 1$.

    The SLLN is stronger: a.s. convergence implies convergence in probability, but not vice versa. The SLLN requires only $E[|X|] < \infty$, while the simple proof of the WLLN via Chebyshev's inequality requires finite variance. $\square$

---

**Exercise 2.**
Show that $\bar{X}_n$ of iid Cauchy random variables has the same distribution as a single Cauchy observation. (Hint: use characteristic functions.)

??? success "Solution to Exercise 2"
    The characteristic function of a standard Cauchy random variable is $\varphi_X(t) = e^{-|t|}$.

    For iid Cauchy $X_1, \ldots, X_n$, the characteristic function of $S_n = \sum X_i$ is:

    $$\varphi_{S_n}(t) = \left(e^{-|t|}\right)^n = e^{-n|t|}$$

    The characteristic function of $\bar{X}_n = S_n/n$ is:

    $$\varphi_{\bar{X}_n}(t) = \varphi_{S_n}(t/n) = e^{-n|t/n|} = e^{-|t|}$$

    This is exactly the characteristic function of a standard Cauchy. Since the characteristic function uniquely determines the distribution, $\bar{X}_n \sim \text{Cauchy}(0,1)$ for every $n$.

    This means the sample mean is no more concentrated than a single observation — averaging is futile. $\square$

---

**Exercise 3.**
For an AR(1) process $X_t = \rho X_{t-1} + \epsilon_t$ with $|\rho| < 1$ and $\epsilon_t \sim N(0, \sigma_\epsilon^2)$, derive the variance of $\bar{X}_n$ and show it is approximately $\frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$ for large $n$, where $\sigma^2 = \sigma_\epsilon^2/(1-\rho^2)$.

??? success "Solution to Exercise 3"
    The stationary variance is $\gamma_0 = \text{Var}(X_t) = \sigma_\epsilon^2/(1-\rho^2)$. The autocovariance at lag $h$ is $\gamma_h = \gamma_0 \rho^{|h|}$.

    $$\text{Var}(\bar{X}_n) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \text{Cov}(X_i, X_j) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \gamma_0 \rho^{|i-j|}$$

    $$= \frac{\gamma_0}{n^2}\left(n + 2\sum_{h=1}^{n-1}(n-h)\rho^h\right)$$

    For large $n$, the sum $\sum_{h=1}^{n-1}(n-h)\rho^h \approx n\sum_{h=1}^{\infty}\rho^h = \frac{n\rho}{1-\rho}$. Therefore:

    $$\text{Var}(\bar{X}_n) \approx \frac{\gamma_0}{n}\left(1 + \frac{2\rho}{1-\rho}\right) = \frac{\gamma_0}{n}\cdot\frac{1+\rho}{1-\rho}$$

    Since $\gamma_0 = \sigma^2$, we get $\text{Var}(\bar{X}_n) \approx \frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$.

    For $\rho = 0.8$, the inflation factor is $1.8/0.2 = 9$, meaning the effective sample size is only $n/9$. $\square$

---

**Exercise 4.**
A fund has a true annual expected excess return of 3% with annual volatility 20%. How many years of data are needed so that the probability of the sample mean excess return being positive exceeds 90%?

??? success "Solution to Exercise 4"
    We need $P(\bar{X}_T > 0) \geq 0.90$, where $\bar{X}_T \sim N(\mu, \sigma^2/T)$ with $\mu = 0.03$ and $\sigma = 0.20$.

    $$P(\bar{X}_T > 0) = P\left(Z > \frac{-\mu}{\sigma/\sqrt{T}}\right) = \mathcal{N}\left(\frac{\mu\sqrt{T}}{\sigma}\right) \geq 0.90$$

    Since $\mathcal{N}^{-1}(0.90) = 1.282$:

    $$\frac{0.03\sqrt{T}}{0.20} \geq 1.282 \implies \sqrt{T} \geq \frac{1.282 \times 0.20}{0.03} = 8.547 \implies T \geq 73.1$$

    So approximately **74 years** of data are needed. This starkly illustrates why distinguishing managerial skill from luck is so difficult in finance. $\square$

---

**Exercise 5.**
Explain why the CLT does not apply to the Cauchy distribution. Does this mean no limit theorem applies to Cauchy sample averages?

??? success "Solution to Exercise 5"
    The CLT requires $\text{Var}(X) < \infty$. The Cauchy distribution has no finite variance (indeed, no finite mean), so the CLT does not apply.

    However, a different limit theorem does apply. By the **Generalized Central Limit Theorem** (for stable distributions), normalized partial sums of iid Cauchy variables converge to a Cauchy distribution. In fact, the result is exact: $\bar{X}_n$ has the same Cauchy distribution as a single observation, for every finite $n$.

    More broadly, the Cauchy distribution belongs to the family of **stable distributions** with stability index $\alpha = 1$. For stable distributions with index $\alpha < 2$, the CLT fails but the Generalized CLT gives convergence to a non-Gaussian stable law. The Gaussian is the special case $\alpha = 2$. $\square$
