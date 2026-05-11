# Q-Q Plots for Financial Data: Detecting Non-Normality in Asset Returns


## Overview

Q-Q plots are particularly valuable in finance for diagnosing departures from normality in asset returns. Many financial models assume returns follow a normal distribution, but empirical data often exhibit **heavy tails** and **skewness**, leading to significant underestimation of tail risk. This section focuses on using Q-Q plots to visualize these departures in financial data.

## Why Normality Matters in Finance

Financial models rely heavily on distributional assumptions:

- **Option pricing** (Black-Scholes) assumes log-normal asset prices, equivalent to normally distributed log-returns
- **Value at Risk (VaR)** and **Expected Shortfall (ES)** rely on tail behavior assumptions
- **Portfolio optimization** uses variance-covariance approaches that assume multivariate normality

When actual returns deviate from normality, these models systematically underestimate the probability of large losses, especially during market stress periods.

## Real-World Example: Netflix (NFLX) Log-Returns

Netflix stock provides an excellent case study of non-normal financial returns. The following example uses daily log-returns computed from closing prices:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate NFLX-like log-returns (or load real data)
np.random.seed(42)
# Use a distribution with slightly heavier tails than normal
# (Actual NFLX returns are even heavier-tailed)
returns = np.random.standard_t(df=8, size=1000) * 0.02

# Create Q-Q plot against normal distribution
fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(returns, dist="norm", plot=ax)

ax.set_title("Q-Q Plot: Daily Log-Returns vs Normal Distribution", fontsize=12)
ax.set_xlabel("Theoretical Normal Quantiles", fontsize=11)
ax.set_ylabel("Sample Quantiles (Observed Returns)", fontsize=11)

# Enhance aesthetics
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
```

## Interpreting the Q-Q Plot for Financial Data

### Perfect Normality
When data follow a normal distribution exactly, the Q-Q plot shows all points tightly clustered along the 45-degree reference line.

### Heavy Tails (The Financial Reality)
Real stock returns exhibit **heavy tails**: both the left tail (large negative returns/losses) and right tail (large positive returns/gains) contain more observations than the normal distribution predicts.

**Visual signature:** The Q-Q plot "bends upward" in the right tail and "bends downward" in the left tail. This creates an S-shaped pattern.

```python
# Simulate heavy-tailed returns (e.g., using Student's t distribution)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
df = 5  # degrees of freedom; lower = heavier tails
heavy_tailed_returns = stats.t.rvs(df=df, scale=0.02, size=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram with normal overlay
ax1.hist(heavy_tailed_returns, bins=50, density=True, alpha=0.6, label='Observed Returns')
x = np.linspace(heavy_tailed_returns.min(), heavy_tailed_returns.max(), 100)
ax1.plot(x, stats.norm.pdf(x, loc=heavy_tailed_returns.mean(),
                            scale=heavy_tailed_returns.std()),
         'r-', lw=2, label='Normal PDF')
ax1.set_xlabel('Daily Log-Return', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Distribution Shape: Heavy Tails vs Normal', fontsize=12)
ax1.legend()
ax1.spines[["top", "right"]].set_visible(False)

# Right: Q-Q plot
stats.probplot(heavy_tailed_returns, dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot: Revealing Heavy Tails", fontsize=12)
ax2.set_xlabel('Theoretical Normal Quantiles', fontsize=11)
ax2.set_ylabel('Sample Quantiles', fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
```

## Consequences of Ignoring Non-Normality

### Underestimating Tail Risk

The normal distribution has a **kurtosis** of 3. Real financial returns typically have **excess kurtosis > 1**, meaning the tails are heavier than normal.

**Example:** If the normal distribution predicts a 5% loss with 0.1% probability, actual heavy-tailed returns might exhibit this loss with 0.5% probability — a 5x underestimation!

### Implication for Risk Management

- **VaR models** based on normality underestimate losses at the 99th or 99.9th percentile
- **Hedging strategies** based on normal assumptions leave portfolios vulnerable to tail events
- **Capital requirements** (e.g., Basel III) may be insufficient if built on normal assumptions

## Practical Workflow: Diagnosing Return Distributions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load or simulate returns
np.random.seed(42)
returns = np.random.standard_t(df=6, size=1500) * 0.025

# Step 2: Compute summary statistics
mean_ret = returns.mean()
std_ret = returns.std()
skewness = stats.skew(returns)
kurtosis = stats.kurtosis(returns)  # Excess kurtosis

print(f"Mean:     {mean_ret:.4f}")
print(f"Std Dev:  {std_ret:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Ex. Kurtosis: {kurtosis:.4f}")

# Step 3: Normality tests
_, p_ks = stats.kstest(returns, 'norm', args=(mean_ret, std_ret))
_, p_ad = stats.anderson(returns, dist='norm')
_, p_jb = stats.jarque_bera(returns)

print(f"\nKolmogorov-Smirnov test p-value: {p_ks:.4f}")
print(f"Jarque-Bera test p-value: {p_jb:.4f}")

# Step 4: Q-Q plot
fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(returns, dist="norm", plot=ax)
ax.set_title("Diagnostics: Are Returns Normal?", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
plt.show()
```

## Adjusting for Non-Normality

### 1. Alternative Distributions
Instead of normal, fit Student's $t$ or Generalized Hyperbolic distributions:

```python
from scipy.stats import t as student_t

# Fit Student's t distribution
df, loc, scale = student_t.fit(returns)
print(f"Fitted df: {df:.2f} (lower df → heavier tails)")
```

### 2. Non-Parametric Methods
Bootstrap and quantile-based approaches make no distributional assumptions.

### 3. Modified Risk Measures
Use Expected Shortfall (CVaR) instead of VaR; it better captures tail behavior for non-normal distributions.

## Summary

- **Q-Q plots** visually compare sample quantiles to theoretical quantiles, revealing distributional shape
- **Financial returns exhibit heavy tails**, showing an S-shaped pattern in Q-Q plots
- **Ignoring non-normality** leads to systematic underestimation of tail risk
- **Practical solutions**: use alternative distributions, non-parametric methods, or robust risk measures tailored to observed tail behavior


## Exercises

**Exercise 1.**
Daily stock returns plotted on a normal Q-Q plot show points that follow a straight line in the center but curve sharply away at both extremes. Interpret this pattern.

??? success "Solution to Exercise 1"
    This is the classic signature of **heavy tails** (leptokurtosis) in financial return data. The central returns are approximately normally distributed, but extreme returns (both large losses and large gains) are far more extreme than a normal distribution would predict.

    The left tail curving below the line means large negative returns are more negative than expected; the right tail curving above the line means large positive returns are more positive than expected. This excess tail probability is why normal-based risk models (VaR, option pricing) systematically underestimate the likelihood of extreme events.

---

**Exercise 2.**
If you plot financial returns against the quantiles of a $t$-distribution with 5 degrees of freedom and the Q-Q plot appears linear, what does this suggest about the return distribution?

??? success "Solution to Exercise 2"
    A linear Q-Q plot against $t_5$ quantiles suggests the returns are well-modeled by a $t$-distribution with approximately 5 degrees of freedom. This means the returns have heavier tails than normal but not as extreme as, say, a Cauchy distribution.

    The $t_5$ distribution has excess kurtosis of $6/(5-4) = 6$, meaning the data have substantially heavier tails than normal (which has excess kurtosis of 0). This is a common finding for daily equity returns, where estimated degrees of freedom typically range from 3 to 8.

---

**Exercise 3.**
Explain how Q-Q plots can be used to calibrate risk models. Why is the tail region most important?

??? success "Solution to Exercise 3"
    Risk measures like Value-at-Risk (VaR) and Expected Shortfall depend on the tail of the return distribution (the 1st or 5th percentile). A Q-Q plot directly shows whether the model's tail matches the data's tail.

    If the Q-Q plot is linear throughout, the model fits well and risk estimates are reliable. If the tails curve away (as they do for the normal model applied to financial returns), the model underestimates tail risk.

    The tail region is most important because: (1) risk management is fundamentally about extreme events; (2) a model that fits the center well but misses the tails gives false confidence; (3) regulatory requirements (Basel accords) explicitly require accurate tail modeling.

---

**Exercise 4.**
Compare Q-Q plots of daily returns versus monthly returns against normal quantiles. Which is more likely to appear linear, and why?

??? success "Solution to Exercise 4"
    **Monthly returns** are more likely to appear approximately linear (closer to normal) because:

    1. **Aggregation effect:** Monthly returns are the sum of ~21 daily returns. By the CLT, sums of i.i.d. random variables converge to normality, even if individual daily returns are non-normal.
    2. **Reduced kurtosis:** Aggregation reduces excess kurtosis approximately by a factor of $1/\sqrt{T}$ where $T$ is the number of days.
    3. **Less volatility clustering:** The GARCH effects that make daily returns non-normal are partially averaged out over a month.

    However, monthly returns are not perfectly normal -- they still exhibit some heavy tails and skewness, just less pronounced than daily returns. The convergence to normality is slow for heavy-tailed distributions.
