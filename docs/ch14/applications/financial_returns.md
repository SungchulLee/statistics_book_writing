# Normality of Financial Returns

## Motivation

Financial returns are among the most heavily studied datasets in applied statistics, and the question of whether they are normally distributed has profound implications. Portfolio optimization, option pricing, and risk management all depend on assumptions about the return distribution. If returns are normal, closed-form solutions exist for many problems. If they are not, these solutions may underestimate risk and lead to poor decisions. This section examines the empirical evidence on return distributions and its consequences.

## Log-Returns and the Normal Model

The **log-return** (or continuously compounded return) of an asset over one period is

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

where $P_t$ is the asset price at time $t$. The normal model assumes

$$
r_t \overset{\text{iid}}{\sim} N(\mu, \sigma^2)
$$

This model implies that multi-period returns are also normal (since sums of normals are normal), and that the price process follows a geometric Brownian motion:

$$
\ln P_T - \ln P_0 = \sum_{t=1}^{T} r_t \sim N(T\mu, T\sigma^2)
$$

The normal model is mathematically convenient, but empirical data consistently violate its predictions.

## Empirical Evidence Against Normality

### Fat Tails

The most robust empirical finding is that financial return distributions have **fatter tails** than the normal distribution. If returns were truly normal with mean $\mu$ and standard deviation $\sigma$, then the probability of a return more than 4 standard deviations from the mean would be

$$
P(|r_t - \mu| > 4\sigma) \approx 6.3 \times 10^{-5}
$$

or about once every 63 years for daily data. In practice, such events occur far more frequently. The stock market crash of October 19, 1987, saw a daily return of approximately $-22\%$, which corresponds to roughly 20 standard deviations under a normal model -- an event with essentially zero probability.

The **excess kurtosis** of daily equity returns typically ranges from 3 to 50, far above the normal value of zero. This means the distribution has more mass in the tails and at the center, with less in the intermediate regions, compared to the normal.

### Negative Skewness

Equity returns tend to exhibit **negative skewness**, meaning large negative returns are more common than large positive returns of equal magnitude. This asymmetry violates the symmetry of the normal distribution. Typical values of skewness for daily stock returns are between $-0.5$ and $-1.0$.

### Volatility Clustering

Returns exhibit **volatility clustering**: large returns (of either sign) tend to be followed by large returns, and small returns by small returns. This means that $r_t$ and $r_{t+1}$ are not independent, even though they may be approximately uncorrelated. Formally, the autocorrelation of $|r_t|$ or $r_t^2$ is significantly positive at many lags:

$$
\text{Corr}(r_t^2, r_{t+h}^2) > 0 \quad \text{for } h = 1, 2, \ldots
$$

This pattern violates the iid assumption in the normal model and motivates models such as GARCH that allow the variance to change over time.

### Aggregational Gaussianity

As the return horizon increases (daily to weekly to monthly to annual), the distribution of returns becomes closer to normal. This is consistent with a version of the CLT: if daily returns are weakly dependent with finite variance, sums over longer horizons converge toward normality. However, convergence is slow, and even monthly returns often exhibit detectable excess kurtosis.

## Implications for Value at Risk

**Value at Risk** (VaR) at confidence level $1 - \alpha$ is the loss threshold that is exceeded with probability $\alpha$. Under the normal model:

$$
\text{VaR}_\alpha = -(\mu + z_\alpha \sigma)
$$

where $z_\alpha$ is the $\alpha$-quantile of the standard normal. Because the normal distribution underestimates tail probabilities, this formula systematically underestimates VaR.

For example, at the 1% level, the normal model gives $z_{0.01} = -2.326$, so

$$
\text{VaR}_{0.01} = -\mu + 2.326\,\sigma
$$

If returns actually follow a $t$-distribution with $\nu = 5$ degrees of freedom (a common empirical finding), the corresponding quantile is approximately $-3.365$ (in standardized units), which is 45% larger. The normal model would underestimate the true 1% VaR by a substantial margin.

??? warning "Normal VaR Underestimates Tail Risk"
    Using the normal distribution for VaR calculations systematically underestimates the frequency and magnitude of extreme losses. Regulatory frameworks such as Basel III require banks to account for fat tails in their risk models.

## Testing Normality of Returns

To test whether a return series is normally distributed, apply the standard normality tests:

1. **Jarque-Bera test**: tests whether the skewness and kurtosis match normal values. This is the most commonly used test in finance because it directly targets the two most prominent departures.
2. **Shapiro-Wilk test**: a general-purpose normality test with high power.
3. **Anderson-Darling test**: particularly sensitive to tail departures, making it well-suited for financial data.
4. **Q-Q plot**: visually reveals fat tails as upward curvature in the right tail and downward curvature in the left tail.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Test normality of simulated financial returns
# ===================================================================

np.random.seed(42)

# Simulate daily returns from a t-distribution (realistic fat tails)
n_days = 1000
df = 5
daily_returns = stats.t.rvs(df=df, size=n_days) * 0.01

# Descriptive statistics
skew = stats.skew(daily_returns)
kurt = stats.kurtosis(daily_returns)  # excess kurtosis

# Normality tests
sw_stat, sw_p = stats.shapiro(daily_returns)
jb_stat, jb_p = stats.jarque_bera(daily_returns)
ad_result = stats.anderson(daily_returns, dist="norm")

# VaR comparison: normal vs empirical
alpha = 0.01
var_normal = -(np.mean(daily_returns)
               + stats.norm.ppf(alpha) * np.std(daily_returns, ddof=1))
var_empirical = -np.quantile(daily_returns, alpha)

if __name__ == "__main__":
    print(f"Simulated daily returns (t-distribution, df={df})")
    print(f"  Skewness:        {skew:.4f}")
    print(f"  Excess kurtosis: {kurt:.4f}")
    print(f"\nNormality tests:")
    print(f"  Shapiro-Wilk:  W = {sw_stat:.4f}, p = {sw_p:.4f}")
    print(f"  Jarque-Bera:   JB = {jb_stat:.4f}, p = {jb_p:.4f}")
    print(f"  Anderson-Darling: A2 = {ad_result.statistic:.4f}")
    print(f"\n1% VaR comparison:")
    print(f"  Normal VaR:    {var_normal:.6f}")
    print(f"  Empirical VaR: {var_empirical:.6f}")
    print(f"  Ratio:         {var_empirical / var_normal:.2f}")
```

The output demonstrates that normality tests strongly reject for the simulated return data, and the empirical VaR exceeds the normal VaR, illustrating the practical consequence of assuming normality when tails are fat.

## Summary

Empirical financial returns violate normality through fat tails, negative skewness, and volatility clustering. These departures are not marginal: they have direct consequences for risk measurement (VaR underestimation), option pricing (mispriced tail risk), and hypothesis testing (distorted $p$-values). The Jarque-Bera and Anderson-Darling tests are particularly well-suited for detecting these departures. Practitioners working with financial data should routinely test for normality and use models that accommodate heavy tails and time-varying volatility.

## Exercises

**Exercise 1.**
Daily returns for a stock have sample skewness $-0.3$ and sample excess kurtosis $4.2$. Based on these descriptive statistics, would you expect a normal Q-Q plot to be linear? Explain.

??? success "Solution to Exercise 1"
    No. Normal data have skewness $= 0$ and excess kurtosis $= 0$. Excess kurtosis of 4.2 indicates much heavier tails than normal (leptokurtic), meaning extreme returns occur more frequently than a normal model predicts. The negative skewness of $-0.3$ indicates a slight left tail asymmetry (large negative returns are more extreme than large positive ones).

    On a Q-Q plot, the heavy tails would appear as points curving away from the reference line at both ends (below the line on the left, above on the right for the positive kurtosis), and the negative skewness would make the left-tail departure more pronounced. The central portion might look approximately linear.

---

**Exercise 2.**
Explain why the normal distribution is a poor model for daily stock returns. What stylized facts of financial returns violate normality?

??? success "Solution to Exercise 2"
    Key stylized facts that violate normality:

    1. **Heavy tails (excess kurtosis):** Extreme returns (crashes, rallies) occur far more frequently than a normal distribution predicts. Daily returns typically have excess kurtosis of 3-10+.
    2. **Negative skewness:** Large negative returns (crashes) tend to be more extreme than large positive returns.
    3. **Volatility clustering:** Periods of high volatility tend to cluster together (GARCH effects), violating the i.i.d. assumption underlying the normal model.
    4. **Time-varying parameters:** The mean and variance of returns change over time.

    These features mean that risk measures based on normality (e.g., VaR computed from normal quantiles) systematically underestimate tail risk.

---

**Exercise 3.**
The $t$-distribution with $\nu$ degrees of freedom is sometimes used as an alternative to the normal for modeling returns. Why does it better capture heavy tails?

??? success "Solution to Exercise 3"
    The $t$-distribution has heavier tails than the normal, with the heaviness controlled by the degrees of freedom $\nu$. For small $\nu$ (e.g., 3-5), the tails are much heavier; as $\nu \to \infty$, the $t$-distribution converges to the normal.

    The tail probability $P(|X| > x)$ decays polynomially ($\sim x^{-\nu}$) for the $t$-distribution versus exponentially ($\sim e^{-x^2/2}$) for the normal. This means the $t$-distribution assigns much higher probability to extreme events, better matching the observed frequency of large stock moves.

    Fitting a $t$-distribution to daily returns typically yields $\hat{\nu} \approx 3\text{-}8$, producing more realistic VaR and Expected Shortfall estimates than the normal.

---

**Exercise 4.**
A risk manager uses the Jarque-Bera test on 252 daily returns and obtains $p < 0.001$. What should they conclude and what action should they take?

??? success "Solution to Exercise 4"
    The Jarque-Bera test strongly rejects normality, confirming what is nearly universally true for daily financial returns. The conclusion is that normal-based risk models will underestimate tail risk.

    Actions:

    1. **Use heavy-tailed distributions** (Student's $t$, generalized hyperbolic) for VaR and Expected Shortfall calculations.
    2. **Apply historical simulation** or **filtered historical simulation** instead of parametric normal methods.
    3. **Consider GARCH models** to account for volatility clustering (conditional normality with time-varying variance).
    4. **Stress testing:** Supplement statistical models with scenario-based stress tests for extreme events.
