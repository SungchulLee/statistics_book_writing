# Normality in Financial Data

## Motivation

The normal distribution plays a central role in finance. The foundational models of portfolio theory, option pricing, and risk management all assume that asset returns are normally distributed. However, decades of empirical evidence show that real financial returns deviate from normality in systematic and consequential ways. Understanding these deviations is essential for anyone applying statistical methods to financial data.

## The Normal Assumption in Finance

If $R_t$ denotes the log-return of an asset at time $t$, the normal model assumes

$$
R_t \overset{\text{iid}}{\sim} N(\mu, \sigma^2)
$$

where $\mu$ is the expected return and $\sigma$ is the volatility. Under this assumption, portfolio returns are also normal (as linear combinations of normals), and risk measures such as Value at Risk (VaR) can be computed in closed form. For example, the $\alpha$-level VaR under normality is

$$
\text{VaR}_\alpha = -\left(\mu + z_\alpha \, \sigma\right)
$$

where $z_\alpha$ is the $\alpha$-quantile of the standard normal distribution.

## Stylized Facts of Financial Returns

Empirical studies of financial return data consistently document several departures from normality, often called **stylized facts**:

### Fat Tails (Excess Kurtosis)

Financial returns exhibit heavier tails than the normal distribution. The **kurtosis** of a random variable $X$ with mean $\mu$ and standard deviation $\sigma$ is

$$
\kappa = \frac{E\left[(X - \mu)^4\right]}{\sigma^4}
$$

For the normal distribution, $\kappa = 3$. The **excess kurtosis** is $\kappa - 3$, so a normal distribution has excess kurtosis of zero. Financial returns typically have positive excess kurtosis, meaning extreme events occur more frequently than the normal model predicts. Daily stock returns often exhibit excess kurtosis values between 3 and 50, depending on the asset and time period.

??? warning "Underestimating Tail Risk"
    A normal model with the same mean and variance as the actual return distribution will systematically underestimate the probability of large losses. For example, a 4-standard-deviation move occurs once every 63 years under the normal distribution but may occur several times per decade in actual financial markets.

### Skewness

The **skewness** of a distribution is

$$
\gamma = \frac{E\left[(X - \mu)^3\right]}{\sigma^3}
$$

For the normal distribution, $\gamma = 0$. Equity returns often exhibit negative skewness, meaning large losses are more frequent than large gains of the same magnitude. This asymmetry is particularly pronounced during market stress.

### Volatility Clustering

Financial returns display **volatility clustering**: periods of high volatility tend to be followed by high volatility, and periods of low volatility by low volatility. Formally, while the returns $R_t$ may be approximately uncorrelated, the squared returns $R_t^2$ exhibit significant autocorrelation. This violates the independence assumption in the iid normal model, because the variance is not constant over time.

### Aggregational Gaussianity

As the return horizon increases from daily to weekly to monthly, the distribution of returns becomes closer to normal. This is consistent with the CLT: if daily returns are weakly dependent but have finite variance, the sum over longer horizons converges toward normality. However, the convergence is slow, and even monthly returns may still show excess kurtosis.

## Consequences for Statistical Inference

The non-normality of financial data has direct consequences for statistical procedures:

**Confidence intervals for the mean return.** If returns have heavier tails than the normal distribution, $t$-based confidence intervals may understate the true uncertainty. The standard error $S / \sqrt{n}$ may be an unreliable estimate of the variability of $\bar{R}$ when extreme observations inflate $S$.

**Hypothesis tests.** Tests of market efficiency, CAPM beta, or portfolio performance that rely on the normality of returns may produce distorted $p$-values. The actual size of a $t$-test can exceed the nominal $\alpha$ when the return distribution has heavy tails and $n$ is moderate.

**Risk measures.** The normal VaR formula underestimates tail risk. If the true return distribution has excess kurtosis $\kappa - 3 > 0$, the Cornish-Fisher expansion provides a correction:

$$
\text{VaR}_\alpha^{\text{CF}} \approx -\left(\mu + \tilde{z}_\alpha \, \sigma\right)
$$

where

$$
\tilde{z}_\alpha = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)\gamma + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)(\kappa - 3) - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)\gamma^2
$$

and $\gamma$ is the skewness and $\kappa$ is the kurtosis of the return distribution.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Compare normal and actual tail behavior of financial returns
# ===================================================================

np.random.seed(42)

# Simulate returns from a t-distribution (fat tails) with df = 5
n = 2000
df = 5
returns = stats.t.rvs(df=df, size=n) * 0.01  # scale to ~1% daily vol

# Compute descriptive statistics
mean_r = np.mean(returns)
std_r = np.std(returns, ddof=1)
skew_r = stats.skew(returns)
kurt_r = stats.kurtosis(returns)  # excess kurtosis

# Compare tail probabilities
threshold = 3 * std_r
empirical_tail = np.mean(np.abs(returns - mean_r) > threshold)
normal_tail = 2 * (1 - stats.norm.cdf(3))

if __name__ == "__main__":
    print(f"Sample mean:           {mean_r:.6f}")
    print(f"Sample std dev:        {std_r:.6f}")
    print(f"Sample skewness:       {skew_r:.4f}")
    print(f"Sample excess kurtosis: {kurt_r:.4f}")
    print(f"\nP(|R - mean| > 3 std):")
    print(f"  Empirical:  {empirical_tail:.4f}")
    print(f"  Normal:     {normal_tail:.4f}")
```

The output shows that the empirical tail probability exceeds the normal prediction, consistent with the fat-tailed nature of financial returns.

## Summary

Financial returns violate the normality assumption through excess kurtosis, negative skewness, and volatility clustering. These departures are not minor statistical curiosities; they have direct consequences for risk measurement, hypothesis testing, and confidence interval construction. When working with financial data, practitioners should test for normality, consider robust alternatives, and use risk models that account for heavy tails.
