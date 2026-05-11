# Return Estimation

## Overview

Estimating expected returns and volatility from financial data is one of the most important — and most challenging — applications of estimation theory. Expected returns are notoriously imprecise to estimate (the signal-to-noise ratio is very low), Sharpe ratios inherit this imprecision, and volatility estimates depend heavily on the choice of estimation window. This page explores these challenges through simulation, covering estimation precision, Sharpe ratio uncertainty, realized volatility windows, and annualization conventions.

## Expected Return Precision

The standard error of the estimated annual return from $T$ years of data is:

$$\text{SE}(\hat{\mu}) = \frac{\sigma}{\sqrt{T}}$$

With typical equity parameters ($\mu = 8\%$, $\sigma = 20\%$), the standard error is large relative to the quantity being estimated.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def expected_return_precision(seed=42):
    rng = np.random.default_rng(seed)
    mu_annual = 0.08
    sigma_annual = 0.20

    years = [5, 10, 20, 30, 50, 100]
    for T in years:
        se = sigma_annual / np.sqrt(T)
        lo = mu_annual - 1.96 * se
        hi = mu_annual + 1.96 * se
        print(f"T={T:>4} years  SE={se*100:.2f}%  "
              f"95% CI=[{lo*100:.2f}%, {hi*100:.2f}%]  Width={2*1.96*se*100:.2f}%")
```

!!! danger "The fundamental problem"
    With 10 years of data, the 95% confidence interval for the mean return is roughly $[-4.4\%, 20.4\%]$ — so wide that it includes zero. Even 50 years of data gives a standard error of 2.8%, barely enough to distinguish the expected return from zero.

The following simulation shows the distribution of estimated annual returns from 10 and 50 years of monthly data:

```python
def return_precision_simulation(seed=42):
    rng = np.random.default_rng(seed)
    mu_annual, sigma_annual = 0.08, 0.20
    n_sim = 20_000

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, T, title in [(axes[0], 10, '10 Years'), (axes[1], 50, '50 Years')]:
        n_m = T * 12
        mu_m = mu_annual / 12
        sig_m = sigma_annual / np.sqrt(12)
        ests = np.array([rng.normal(mu_m, sig_m, n_m).mean() * 12
                         for _ in range(n_sim)])
        ax.hist(ests, bins=60, density=True, alpha=0.6, color='steelblue')
        ax.axvline(mu_annual, color='red', ls='--', lw=2, label=f'True mu = {mu_annual*100:.0f}%')
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Estimated Annual Return')
        ax.set_title(f'{title} of Monthly Data')
        ax.legend()
    plt.suptitle('Distribution of Expected Return Estimates')
    plt.tight_layout()
    plt.show()
```

## Sharpe Ratio Uncertainty

The **Sharpe ratio** $\text{SR} = \mu/\sigma$ (or excess return divided by volatility) is the standard measure of risk-adjusted performance. Its estimation uncertainty is approximately:

$$\text{SE}(\widehat{\text{SR}}) \approx \frac{1}{\sqrt{T}} \sqrt{1 + \frac{\text{SR}^2}{2}}$$

where $T$ is the number of periods. For annual Sharpe ratios near 0.5, you need roughly 16 years for a $t$-statistic of 2.

```python
def sharpe_ratio_uncertainty(seed=42):
    rng = np.random.default_rng(seed)
    true_sr = 0.5
    mu_annual = 0.08
    sigma_annual = mu_annual / true_sr
    n_sim = 30_000

    horizons = [3, 5, 10, 20, 50]
    for T in horizons:
        n_m = T * 12
        mu_m, sig_m = mu_annual / 12, sigma_annual / np.sqrt(12)
        sr_ests = []
        for _ in range(n_sim):
            r = rng.normal(mu_m, sig_m, n_m)
            sr_ests.append(r.mean() / r.std() * np.sqrt(12))
        sr_ests = np.array(sr_ests)
        print(f"T={T:>3} years  E[SR]={sr_ests.mean():.3f}  "
              f"SD(SR)={sr_ests.std():.3f}  P(SR<0)={( sr_ests < 0).mean():.1%}")
```

!!! note "Implications for fund evaluation"
    With only 3 years of data, a fund with a true Sharpe ratio of 0.5 has a roughly 20% probability of showing a *negative* estimated Sharpe ratio. Even 10 years gives a standard deviation of about 0.3 around the true value. Reliably distinguishing skilled from unskilled managers requires decades of data.

## Realized Volatility Windows

Volatility is time-varying in practice (GARCH effects). The choice of estimation window involves a **bias-variance tradeoff**:

- **Short windows** (5-21 days): responsive to recent changes but noisy
- **Long windows** (126-252 days): smooth but lagging

```python
def realized_volatility_windows(seed=42):
    rng = np.random.default_rng(seed)

    # Simulate GARCH(1,1) returns
    T = 756  # 3 years daily
    omega, alpha, beta = 0.00001, 0.08, 0.90
    sigma2 = np.zeros(T)
    returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))

    windows = [5, 10, 21, 63, 126, 252]

    fig, ax = plt.subplots(figsize=(12, 5))
    true_vol = np.sqrt(sigma2 * 252) * 100
    ax.plot(true_vol, 'k-', alpha=0.3, lw=0.8, label='True vol (GARCH)')

    for w, color in zip([21, 63, 252], ['blue', 'red', 'green']):
        rv = np.array([np.std(returns[max(0,t-w):t], ddof=1) * np.sqrt(252) * 100
                       for t in range(w, T)])
        ax.plot(range(w, T), rv, color=color, alpha=0.7, lw=0.8, label=f'{w}d window')

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Annualized Vol (%)')
    ax.set_title('Realized Volatility: Window Size Comparison')
    ax.legend()
    plt.tight_layout()
    plt.show()
```

!!! info "Practical guidance"
    There is no single "correct" window. Practitioners often use 21-day (monthly) windows for short-term risk management and 252-day (annual) windows for strategic allocation. More sophisticated approaches (exponentially weighted, GARCH models) address the tradeoff more explicitly.

## Annualization Conventions

Financial data is collected at different frequencies. Standard annualization assumes iid returns:

| Frequency | Mean | Volatility | Sharpe Ratio |
|-----------|------|------------|--------------|
| Daily to Annual | $\mu_a = \mu_d \times 252$ | $\sigma_a = \sigma_d \times \sqrt{252}$ | $\text{SR}_a = \text{SR}_d \times \sqrt{252}$ |
| Monthly to Annual | $\mu_a = \mu_m \times 12$ | $\sigma_a = \sigma_m \times \sqrt{12}$ | $\text{SR}_a = \text{SR}_m \times \sqrt{12}$ |

```python
def annualization_conventions():
    mu_d = 0.0003     # Daily mean return
    sigma_d = 0.012   # Daily volatility

    print(f"Daily: mu = {mu_d*100:.4f}%, sigma = {sigma_d*100:.4f}%")
    print(f"Annualized (252 trading days):")
    print(f"  mu_annual   = {mu_d*252*100:.2f}%")
    print(f"  sigma_annual = {sigma_d*np.sqrt(252)*100:.2f}%")
    print(f"  SR_annual   = {mu_d/sigma_d*np.sqrt(252):.3f}")
```

!!! warning "iid assumption"
    Annualization formulas assume returns are iid. Autocorrelation in returns (momentum or mean-reversion) and volatility clustering (GARCH effects) invalidate the simple $\sqrt{T}$ scaling rules. In practice, these are useful approximations but should be used with awareness of their limitations.

## Interpretation

- **Expected return estimation** is fundamentally imprecise in finance. The signal-to-noise ratio $\mu/\sigma$ is typically small (around 0.03 daily), requiring decades of data for meaningful precision.
- **Sharpe ratios** are noisy estimates. A 3-year track record is insufficient to reliably judge whether a manager has skill.
- **Volatility estimation** is much more precise than return estimation (volatility is observable from high-frequency data, but expected return is not). This is why risk models are more reliable than return forecasts.
- The **bias-variance tradeoff** in window selection is a practical manifestation of the classical tradeoff: short windows have low bias but high variance, and vice versa.
- **Annualization** is straightforward under iid assumptions but should be interpreted cautiously when returns exhibit serial dependence.

## Exercises

**Exercise 1.**
A fund has a true annual expected return of 10% and annual volatility of 15%. Compute the standard error of the estimated annual return from 5 years of monthly data. What is the probability that the estimated return is negative?

??? success "Solution to Exercise 1"
    Monthly parameters: $\mu_m = 10\%/12 \approx 0.833\%$ and $\sigma_m = 15\%/\sqrt{12} \approx 4.330\%$.

    With $n = 60$ months, $\text{SE}(\hat{\mu}_m) = \sigma_m/\sqrt{60}$. Annualized:

    $$\text{SE}(\hat{\mu}_a) = \sigma_a / \sqrt{T} = 15\% / \sqrt{5} = 6.71\%$$

    The probability of a negative estimated return:

    $$P(\hat{\mu}_a < 0) = P\left(Z < \frac{0 - 10\%}{6.71\%}\right) = P(Z < -1.49) = \mathcal{N}(-1.49) \approx 6.8\%$$

    Even with a true 10% expected return, there is about a 7% chance of estimating a negative value from 5 years of data. $\square$

---

**Exercise 2.**
Derive the approximate standard error of the estimated Sharpe ratio, $\text{SE}(\widehat{\text{SR}}) \approx \sqrt{(1 + \text{SR}^2/2)/T}$, using the delta method.

??? success "Solution to Exercise 2"
    The Sharpe ratio is $\text{SR} = \mu/\sigma = g(\mu, \sigma^2)$ where $g(a, b) = a/\sqrt{b}$.

    By the delta method, for $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = S^2$:

    $$\text{Var}(\widehat{\text{SR}}) \approx \nabla g^T \Sigma \nabla g$$

    where $\Sigma = \text{Cov}(\hat{\mu}, \hat{\sigma}^2)$. For normal data, $\hat{\mu}$ and $\hat{\sigma}^2$ are independent, so $\Sigma$ is diagonal:

    $$\Sigma = \begin{pmatrix} \sigma^2/n & 0 \\ 0 & 2\sigma^4/(n-1) \end{pmatrix}$$

    The gradient of $g(\mu, \sigma^2) = \mu(\sigma^2)^{-1/2}$:

    $$\frac{\partial g}{\partial \mu} = \frac{1}{\sigma}, \qquad \frac{\partial g}{\partial \sigma^2} = -\frac{\mu}{2\sigma^3}$$

    Therefore:

    $$\text{Var}(\widehat{\text{SR}}) \approx \frac{1}{\sigma^2}\cdot\frac{\sigma^2}{n} + \frac{\mu^2}{4\sigma^6}\cdot\frac{2\sigma^4}{n} = \frac{1}{n}\left(1 + \frac{\mu^2}{2\sigma^2}\right) = \frac{1}{n}\left(1 + \frac{\text{SR}^2}{2}\right)$$

    Taking the square root: $\text{SE}(\widehat{\text{SR}}) \approx \sqrt{(1 + \text{SR}^2/2)/n}$.

    For annual data, $n = T$ (years), giving the stated formula. $\square$

---

**Exercise 3.**
Explain why increasing the sampling frequency (e.g., from monthly to daily) does not improve the precision of expected return estimation but does improve volatility estimation.

??? success "Solution to Exercise 3"
    **Expected return:** Under iid assumptions, the annual standard error is $\text{SE} = \sigma_a/\sqrt{T}$, where $T$ is the number of *years* of data. If we sample daily instead of monthly within the same $T$ years, we get more observations but each has proportionally smaller mean and variance. The net effect cancels:

    - Daily: $n = 252T$ observations, $\mu_d = \mu_a/252$, $\sigma_d = \sigma_a/\sqrt{252}$. SE of annualized mean: $\sigma_d\sqrt{252}/\sqrt{n} = \sigma_a/\sqrt{T}$.

    The precision depends only on the time span $T$, not the sampling frequency. This is because the mean return accumulates linearly with time.

    **Volatility:** Volatility estimation precision improves with more observations. The standard error of $\hat{\sigma}^2$ is proportional to $1/\sqrt{n}$, where $n$ is the number of observations. Daily data gives 252 observations per year instead of 12, improving volatility estimation by a factor of $\sqrt{252/12} \approx 4.6$.

    Intuitively, each daily return reveals information about the current variance (through its squared magnitude), so more frequent sampling genuinely helps. In contrast, each daily return carries only a tiny signal about the drift, and that signal does not compound faster with more frequent observation. $\square$

---

**Exercise 4.**
A GARCH(1,1) model has parameters $\omega = 0.00001$, $\alpha = 0.08$, $\beta = 0.90$. Compute the unconditional (long-run) annualized volatility. Is the process stationary?

??? success "Solution to Exercise 4"
    The GARCH(1,1) model is $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$.

    Stationarity requires $\alpha + \beta < 1$. Here $\alpha + \beta = 0.08 + 0.90 = 0.98 < 1$, so the process is stationary.

    The unconditional variance is:

    $$\sigma^2 = \frac{\omega}{1 - \alpha - \beta} = \frac{0.00001}{1 - 0.98} = \frac{0.00001}{0.02} = 0.0005$$

    The unconditional daily volatility is $\sigma = \sqrt{0.0005} \approx 0.02236$ or $2.236\%$.

    Annualized: $\sigma_a = 0.02236 \times \sqrt{252} \approx 35.5\%$.

    Note that $\alpha + \beta = 0.98$ is close to 1, indicating high volatility persistence — shocks to volatility decay slowly. $\square$

---

**Exercise 5.**
A practitioner claims that because daily returns are roughly iid, the monthly Sharpe ratio multiplied by $\sqrt{12}$ gives the annual Sharpe ratio. Under what conditions is this correct, and when might it fail?

??? success "Solution to Exercise 5"
    The claim relies on the iid assumption for daily returns. Under iid:

    $$\text{SR}_{\text{annual}} = \text{SR}_{\text{monthly}} \times \sqrt{12} = \text{SR}_{\text{daily}} \times \sqrt{252}$$

    This is correct because under iid: $\mu_a = 12\mu_m$ and $\sigma_a = \sqrt{12}\sigma_m$, so $\text{SR}_a = 12\mu_m/(\sqrt{12}\sigma_m) = \sqrt{12}\cdot\text{SR}_m$.

    **Conditions for failure:**

    1. **Serial correlation in returns:** If returns are positively autocorrelated (momentum), the true annual volatility is higher than $\sqrt{12}\sigma_m$, so the $\sqrt{12}$ scaling **overstates** the annual Sharpe ratio. Negative autocorrelation (mean reversion) leads to understating it.

    2. **Volatility clustering (GARCH):** Time-varying volatility means that the compounding of daily returns does not follow the simple $\sqrt{T}$ rule. The annual distribution has fatter tails than the scaled daily distribution.

    3. **Non-zero serial correlation in squared returns:** Even if returns are uncorrelated, autocorrelation in $r_t^2$ (which is present in GARCH models) affects the annualized volatility.

    In practice, the $\sqrt{T}$ rule is a useful approximation but should be validated against direct computation from the relevant frequency. $\square$
