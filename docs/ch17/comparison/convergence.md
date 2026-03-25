# Number of Resamples and Convergence

## Motivation

Every bootstrap or permutation procedure involves a choice: how many resamples $B$ should we use? Too few resamples introduce excessive **Monte Carlo error** — the variability due to the random resampling process itself, distinct from the statistical variability we are trying to estimate. Too many resamples waste computation without meaningful improvement. This section quantifies the Monte Carlo error, provides guidelines for choosing $B$, and describes methods for monitoring convergence.

## Monte Carlo Error

The bootstrap estimate of any quantity (standard error, confidence interval endpoint, $p$-value) is a random variable that depends on which bootstrap samples happen to be drawn. Running the same bootstrap procedure twice with different random seeds produces slightly different results. This variability is the **Monte Carlo error**.

Monte Carlo error decreases at rate $O(1/\sqrt{B})$: doubling $B$ reduces the Monte Carlo standard deviation by a factor of $\sqrt{2} \approx 1.41$.

## Monte Carlo Error for the Standard Error

The bootstrap standard error is:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat{\theta}^{*(b)} - \bar{\hat{\theta}}^*\right)^2}
$$

The Monte Carlo standard error of this estimate (i.e., the standard error of the standard error) is approximately:

$$
\text{SE}_{\text{MC}}(\widehat{\text{SE}}_{\text{boot}}) \approx \frac{\widehat{\text{SE}}_{\text{boot}}}{\sqrt{2B}}
$$

For example, with $B = 1{,}000$ and $\widehat{\text{SE}}_{\text{boot}} = 2.5$:

$$
\text{SE}_{\text{MC}} \approx \frac{2.5}{\sqrt{2000}} \approx 0.056
$$

The bootstrap standard error is determined to within about $\pm 0.11$ (two Monte Carlo standard errors). This level of precision is adequate for most purposes.

## Monte Carlo Error for Quantiles

Confidence interval endpoints depend on the quantiles of the bootstrap distribution. The Monte Carlo error for the $q$-th quantile $\hat{\theta}^*_{(q)}$ is approximately:

$$
\text{SE}_{\text{MC}}(\hat{\theta}^*_{(q)}) \approx \sqrt{\frac{q(1-q)}{B}} \cdot \frac{1}{f(\hat{\theta}^*_{(q)})}
$$

where $f$ is the density of the bootstrap distribution evaluated at the quantile. Extreme quantiles (near 0 or 1) require more resamples because the numerator $q(1-q)$ is small but the density $f$ at the tails is also small, so the ratio can be large.

!!! note "Quantiles Need More Resamples Than Means"
    Estimating the 2.5th and 97.5th percentiles (for a 95% CI) requires substantially more bootstrap replicates than estimating the standard error. With $B = 1{,}000$, the 2.5th percentile is determined by only 25 bootstrap values, making it noisy. With $B = 10{,}000$, it is determined by 250 values, which is far more stable.

## Monte Carlo Error for p-Values

The bootstrap $p$-value $\hat{p} = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(|t^{*(b)}| \ge |t_{\text{obs}}|)$ is a sample proportion with Monte Carlo standard error:

$$
\text{SE}_{\text{MC}}(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{B}}
$$

| True $p$ | $B = 1{,}000$ | $B = 5{,}000$ | $B = 10{,}000$ |
|---|---|---|---|
| 0.05 | 0.0069 | 0.0031 | 0.0022 |
| 0.01 | 0.0031 | 0.0014 | 0.0010 |
| 0.001 | 0.0010 | 0.0004 | 0.0003 |

!!! warning "p-Values Near the Significance Level"
    When the true $p$-value is close to the significance level $\alpha$, the Monte Carlo error can cause the bootstrap to make the wrong decision. If $p \approx 0.05$ and $B = 1{,}000$, the estimated $p$-value fluctuates in the range roughly $[0.036, 0.064]$. Use $B \ge 10{,}000$ when precise $p$-values are needed.

## Guidelines for Choosing B

The appropriate number of resamples depends on the inferential goal:

| Purpose | Recommended $B$ | Rationale |
|---|---|---|
| Standard error estimation | $1{,}000$ | $\text{SE}_{\text{MC}} \approx \widehat{\text{SE}}_{\text{boot}} / 45$ |
| Percentile CI (95%) | $5{,}000$ to $10{,}000$ | Stable tail quantile estimates |
| BCa CI | $5{,}000$ to $10{,}000$ | Same as percentile (plus jackknife) |
| $p$-value at $\alpha = 0.05$ | $10{,}000$ | $\text{SE}_{\text{MC}}(\hat{p}) \approx 0.002$ |
| $p$-value at $\alpha = 0.01$ | $50{,}000$ to $100{,}000$ | Precise estimation of small $p$ |
| Publication-quality results | $10{,}000$ or more | Ensures reproducibility |

These are minimum recommendations. Using more resamples than needed has no statistical cost beyond computation time.

## Monitoring Convergence

Rather than choosing $B$ in advance, a practical approach is to **monitor convergence** by tracking the stability of the bootstrap estimate as $B$ increases.

**Running estimate plot.** Compute the bootstrap standard error (or confidence interval endpoint) cumulatively as $b$ increases from $1$ to $B$. Plot the running estimate against $b$. When the curve stabilizes (fluctuations become negligible), $B$ is large enough.

**Repeat-and-compare.** Run the bootstrap procedure twice with different random seeds. If the two estimates of the standard error or CI agree to the desired precision, $B$ is adequate.

**Coefficient of variation.** Compute the ratio $\text{SE}_{\text{MC}} / \widehat{\text{SE}}_{\text{boot}}$. This gives the relative Monte Carlo error. A target of 1-2% (i.e., $B \ge 2{,}500$) is reasonable for standard errors; tighter targets require proportionally more resamples.

!!! tip "Practical Convergence Check"
    A simple rule of thumb: run the bootstrap with $B = 1{,}000$, then again with $B = 2{,}000$. If the two standard error estimates agree to within 5%, $B = 1{,}000$ is sufficient. If they disagree substantially, increase $B$ and repeat.

## Diminishing Returns

The Monte Carlo standard error decreases as $1/\sqrt{B}$. This means:

- Going from $B = 100$ to $B = 1{,}000$: reduces Monte Carlo error by a factor of $\sqrt{10} \approx 3.2$
- Going from $B = 1{,}000$ to $B = 10{,}000$: reduces by another factor of $\sqrt{10} \approx 3.2$
- Going from $B = 10{,}000$ to $B = 100{,}000$: reduces by another factor of $\sqrt{10} \approx 3.2$

Beyond $B = 10{,}000$, the improvement is rarely noticeable for standard errors and confidence intervals. For $p$-values near common thresholds, larger $B$ can still be worthwhile.

## Summary

The number of bootstrap resamples $B$ controls the Monte Carlo error in the bootstrap approximation. Standard error estimates converge quickly ($B = 1{,}000$ is usually sufficient), while confidence interval endpoints and $p$-values require more resamples ($B = 5{,}000$ to $10{,}000$ or more). Monte Carlo error decreases at rate $1/\sqrt{B}$, so there are diminishing returns to increasing $B$ indefinitely. Monitoring convergence through running estimate plots or repeat-and-compare checks is a practical alternative to choosing $B$ by rule of thumb.
