# Heavy-Tailed Distributions

## Overview

When the underlying distribution has heavy tails, the sample mean can have poor performance due to extreme observations. This is particularly relevant in finance, where asset returns exhibit substantially heavier tails than the normal distribution.

## Characteristics

Heavy-tailed distributions have tails that decay slower than exponential. Examples include:

- **Student's $t$ distribution** (with low degrees of freedom)
- **Cauchy distribution** (undefined mean and variance)
- **Pareto distribution** (power-law tails)
- **Log-normal distribution** (right-skewed)
- **Financial returns** (empirically observed in stock, currency, and commodity markets)

## Kurtosis and Tail Weight

Tail weight is often quantified by **kurtosis**. The normal distribution has kurtosis of 3 (excess kurtosis of 0). Distributions with excess kurtosis > 1 are considered heavy-tailed.

$$\text{Excess Kurtosis} = E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

**Examples**:

- Normal distribution: excess kurtosis = 0
- Student's $t$ with df=5: excess kurtosis ≈ 6 (much heavier tails)
- Real stock returns: excess kurtosis typically 3-10 (depend on frequency and asset)

## Impact on the Sample Mean

When data come from heavy-tailed distributions:

1. **Higher variance of $\bar{X}$**: The standard error is larger than predicted by the normal approximation
2. **Slower convergence to normality**: The Central Limit Theorem still applies, but convergence is slower; $n = 30$ may not be sufficient
3. **Outliers have outsized influence**: A single extreme observation can substantially shift the sample mean
4. **Confidence intervals underestimate uncertainty**: Intervals based on normal theory are too narrow, leading to undercoverage

## Financial Context: Asset Returns

Empirical evidence consistently shows that financial returns exhibit heavy tails:

- **Daily stock returns**: Excess kurtosis 3-6 (typically)
- **Intraday returns**: Even heavier tails
- **Commodity prices**: Very heavy tails during supply shocks
- **Foreign exchange**: Moderate excess kurtosis

### Why Do Financial Returns Have Heavy Tails?

1. **Rare events cluster**: Market crashes and rallies happen in waves, not uniformly
2. **Volatility clustering**: High volatility periods attract more large moves
3. **Information asymmetries**: Sudden news creates discontinuous jumps
4. **Leverage and margin calls**: Can amplify downward moves

### Consequences for Risk Management

If you assume normality when returns are heavy-tailed:

- **Value at Risk (VaR)** at the 99th percentile is severely underestimated
- **Expected Shortfall (CVaR)** is underestimated
- **Hedging ratios** are too small, leaving positions under-protected
- **Capital requirements** are inadequate during stress

**Example**: A normal distribution predicts a 5% loss occurs with probability 0.01%. Heavy-tailed financial returns might exhibit this loss with probability 0.1% — a 10x underestimation!

## Visual Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate samples
normal_returns = np.random.normal(loc=0, scale=0.02, size=5000)
heavy_tailed_returns = stats.t.rvs(df=6, scale=0.02, size=5000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Row 1: Histograms
ax = axes[0, 0]
ax.hist(normal_returns, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
x = np.linspace(-0.08, 0.08, 200)
ax.plot(x, stats.norm.pdf(x, 0, 0.02), 'b-', linewidth=2, label='Normal PDF')
ax.set_title('Normal Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

ax = axes[0, 1]
ax.hist(heavy_tailed_returns, bins=50, alpha=0.6, label='Heavy-tailed', color='red', density=True)
x = np.linspace(-0.08, 0.08, 200)
ax.plot(x, stats.t.pdf(x, df=6, loc=0, scale=0.02), 'r-', linewidth=2, label="Student's t PDF")
ax.set_title("Heavy-Tailed (Student's t) Distribution", fontsize=12, fontweight='bold')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

# Row 2: Q-Q plots
ax = axes[1, 0]
stats.probplot(normal_returns, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normal Data', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1, 1]
stats.probplot(heavy_tailed_returns, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Heavy-Tailed Data', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

# Print statistics
print("Normal Distribution:")
print(f"  Excess Kurtosis: {stats.kurtosis(normal_returns):.2f}")
print()
print("Heavy-Tailed (t) Distribution:")
print(f"  Excess Kurtosis: {stats.kurtosis(heavy_tailed_returns):.2f}")
```

## Robust Alternatives to the Mean

For heavy-tailed data, consider these alternatives:

### 1. Median
- **Robustness**: Unaffected by extreme values
- **Efficiency**: Lower efficiency than mean for normal data, comparable for heavy-tailed
- **Inference**: Use bootstrap for standard errors and confidence intervals

```python
import numpy as np
from sklearn.utils import resample

data = stats.t.rvs(df=5, size=100)
original_median = np.median(data)

# Bootstrap standard error
bootstrap_medians = [np.median(resample(data)) for _ in range(1000)]
se_median = np.std(bootstrap_medians)
print(f"Median: {original_median:.4f} ± {se_median:.4f}")
```

### 2. Trimmed Mean
Remove a fixed percentage from each tail before computing mean:

$$\bar{X}_{\text{trim}, \alpha} = \frac{1}{n(1-2\alpha)} \sum_{i=\lceil n\alpha \rceil}^{\lfloor n(1-\alpha) \rfloor} X_{(i)}$$

where $X_{(i)}$ are order statistics and $\alpha$ is the trim fraction (e.g., 0.1 for 10%).

```python
from scipy.stats import trim_mean

data = stats.t.rvs(df=5, size=100)
mean_trim10 = trim_mean(data, 0.1)  # 10% trimmed mean
```

### 3. Winsorized Mean
Replace extreme values with the $\alpha$-quantiles rather than discarding them:

```python
def winsorize_mean(data, alpha=0.1):
    """Compute mean after Winsorizing tails."""
    lower = np.quantile(data, alpha)
    upper = np.quantile(data, 1 - alpha)
    winsorized = np.clip(data, lower, upper)
    return np.mean(winsorized)

data = stats.t.rvs(df=5, size=100)
mean_wins = winsorize_mean(data, alpha=0.1)
```

### 4. M-Estimators (Huber's Estimator)
Downweight extreme values smoothly using a loss function that transitions from quadratic (for small errors) to absolute (for large errors):

```python
from scipy.stats import huber

data = stats.t.rvs(df=5, size=100)
result = huber(0.1, data)  # Tuning parameter 0.1
print(f"Huber M-estimate: {result.estimate:.4f}")
```

## Comparison of Estimators

```python
import numpy as np
from scipy import stats
from scipy.stats import trim_mean, huber

np.random.seed(42)
data = stats.t.rvs(df=5, loc=0, scale=1, size=500)

print("Estimator Comparison (Population mean = 0):")
print(f"  Sample mean:         {np.mean(data):7.4f}")
print(f"  Median:              {np.median(data):7.4f}")
print(f"  10% Trimmed mean:    {trim_mean(data, 0.1):7.4f}")
print(f"  Winsorized mean:     {winsorize_mean(data, 0.1):7.4f}")
print(f"  Huber's estimator:   {huber(0.1, data).estimate:7.4f}")
```

## Summary

Heavy-tailed distributions present significant challenges for statistical inference:

- The sample mean is inefficient and unstable
- Normal-theory confidence intervals are too narrow
- Risk measures based on normality are dangerously optimistic

For financial data especially, robust alternatives like the median, trimmed means, or M-estimators provide more reliable inference. Bootstrap methods (which require no distributional assumptions) are ideal for constructing confidence intervals around any of these estimators.

## Exercises

**Exercise 1.**
The Cauchy distribution has PDF $f(x) = \frac{1}{\pi(1+x^2)}$. Show that its mean does not exist by demonstrating that $\int_{-\infty}^{\infty} |x| f(x)\,dx$ diverges.

??? success "Solution to Exercise 1"
    $$
    \int_{-\infty}^{\infty} |x| f(x)\,dx = \frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx
    $$

    Using the substitution $u = 1 + x^2$, $du = 2x\,dx$:

    $$
    = \frac{2}{\pi} \cdot \frac{1}{2}\int_1^{\infty} \frac{du}{u} = \frac{1}{\pi}\left[\log u\right]_1^{\infty} = \frac{1}{\pi}(\infty - 0) = \infty
    $$

    Since $E[|X|] = \infty$, the mean $E[X]$ does not exist. The integral diverges logarithmically, which means even averaging over very large samples does not stabilize.

---

**Exercise 2.**
A Student-$t$ distribution with $\nu$ degrees of freedom has finite variance only when $\nu > 2$. For $\nu = 3$, the variance is $\sigma^2 = \nu/(\nu-2) = 3$. Compare the sample mean's behavior for $n = 100$ observations from $t_3$ versus $N(0,3)$ in terms of the standard error of $\bar{X}$.

??? success "Solution to Exercise 2"
    For $N(0,3)$: the standard error is $\text{SE} = \sqrt{3/100} = \sqrt{0.03} \approx 0.173$. The CLT applies perfectly since the population is normal.

    For $t_3$: the population variance is also 3, so the theoretical standard error is the same: $\text{SE} = \sqrt{3/100} \approx 0.173$. However, the $t_3$ distribution has excess kurtosis $\kappa = 6(\nu-2)^{-1} = 6$ (when $\nu > 4$ kurtosis is $6/(\nu-4)$; for $\nu = 3$, kurtosis is technically infinite since the fourth moment does not exist).

    In practice, the sample mean from $t_3$ will show much more variability than predicted by the standard error formula because occasional extreme observations inflate $\bar{X}$ dramatically. The CLT convergence is very slow for $t_3$, and the normal approximation for $\bar{X}$ at $n = 100$ will be poor — confidence intervals based on normality will have coverage well below the nominal level.

---

**Exercise 3.**
Explain why the median is a better estimator of the center of a Cauchy distribution than the sample mean, despite the sample mean being the standard choice for normal data.

??? success "Solution to Exercise 3"
    The **sample mean** of Cauchy data does not converge: by a remarkable property, the sample mean of $n$ i.i.d. Cauchy observations has the same Cauchy distribution regardless of $n$. Averaging does not reduce variability at all because the LLN does not apply (the mean does not exist).

    The **sample median**, in contrast, is consistent for the location parameter of the Cauchy distribution and has asymptotic variance $\pi^2/(4n)$, which decreases at the standard $1/n$ rate. The median is unaffected by extreme observations in the tails, making it robust to the heavy tails that render the mean useless. For the Cauchy distribution specifically, the median is the MLE of the location parameter.

---

**Exercise 4.**
A risk manager estimates the 99th percentile of daily portfolio losses using a normal model and obtains \$2.33 million. If the true distribution of losses follows a $t_5$ distribution with the same scale, how much larger is the true 99th percentile?

??? success "Solution to Exercise 4"
    For a standard normal, the 99th percentile is $z_{0.99} = 2.326$. For a $t_5$ distribution, the 99th percentile is $t_{5, 0.99} \approx 3.365$.

    The ratio is:

    $$
    \frac{t_{5,0.99}}{z_{0.99}} = \frac{3.365}{2.326} \approx 1.447
    $$

    The true 99th percentile under the $t_5$ model is approximately 44.7% larger: $\$2.33\text{M} \times 1.447 \approx \$3.37\text{M}$. This underestimation of tail risk by the normal model is a well-known danger in financial risk management and was a contributing factor in the 2008 financial crisis.
