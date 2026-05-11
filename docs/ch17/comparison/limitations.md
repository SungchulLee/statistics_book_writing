# When Resampling Fails

## Motivation

The bootstrap and permutation methods are remarkably versatile, but they are not universally valid. Under certain conditions, resampling produces misleading results — confidence intervals with incorrect coverage, biased standard errors, or invalid $p$-values. Recognizing these failure modes is essential for applying resampling methods responsibly.

This section catalogs the main situations where resampling breaks down, explains why each failure occurs, and suggests alternatives.

## Dependent Data

The standard (iid) bootstrap assumes that the observations $x_1, \ldots, x_n$ are independent and identically distributed. When the data have temporal, spatial, or clustered dependence, resampling individual observations destroys the dependence structure.

**Why it fails.** If $x_1, \ldots, x_n$ is a time series with positive autocorrelation, the true standard error of $\bar{x}$ exceeds $\sigma/\sqrt{n}$ because consecutive observations carry less independent information. The iid bootstrap underestimates this standard error because it treats every observation as independent.

**Remedies:**

- **Block bootstrap** (Kunsch, 1989): resample contiguous blocks of observations to preserve local dependence. Block length $\ell \approx n^{1/3}$ is a common choice.
- **Moving block bootstrap**: uses all $n - \ell + 1$ overlapping blocks.
- **Circular block bootstrap**: wraps the series into a circle so all observations appear in the same number of blocks.
- **Stationary bootstrap** (Politis and Romano, 1994): uses random block lengths drawn from a geometric distribution to produce stationary bootstrap samples.

!!! warning "Ignoring Dependence"
    Applying the standard bootstrap to dependent data typically produces confidence intervals that are too narrow and $p$-values that are too small. The results look more precise than they actually are, leading to overconfident conclusions.

## Extreme Order Statistics

The bootstrap fails for statistics that depend on the extreme values of the sample, such as the sample maximum $X_{(n)}$ or the sample minimum $X_{(1)}$.

**Classical example.** Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$ and estimate $\theta$ by $\hat{\theta} = X_{(n)}$. The true distribution of $n(\theta - X_{(n)})$ is $\text{Exp}(1)$, but the bootstrap distribution of $n(\hat{\theta} - X^*_{(n)})$ does not converge to $\text{Exp}(1)$.

**Why it fails.** The bootstrap sample maximum $X^*_{(n)}$ can never exceed $X_{(n)}$ (since bootstrap values are drawn from the original data). The true $X_{(n)}$ can approach $\theta$ from below at rate $1/n$, but the bootstrap maximum is "stuck" at the observed maximum with positive probability.

**Remedies:**

- **Parametric bootstrap**: fit the parametric model and simulate from it
- **Subsampling**: resample without replacement at a smaller size $m < n$ (Politis, Romano, and Wolf, 1999)
- **$m$-out-of-$n$ bootstrap**: resample $m < n$ observations with replacement, where $m/n \to 0$

## Heavy-Tailed Distributions

When the population has infinite variance (e.g., stable distributions with index $\alpha < 2$ or Pareto distributions with shape parameter $\le 2$), the sample mean does not satisfy the CLT in its usual form.

**Why it fails.** The bootstrap approximation to the distribution of $\sqrt{n}(\bar{X} - \mu)$ is based on the empirical distribution, which inherits the heavy tails. However, the bootstrap variance $\text{Var}^*(\bar{X}^*) = s^2/n$ converges to infinity in probability, while the true limiting distribution of $\bar{X}$ is a stable law, not a normal distribution. The bootstrap standard error can fluctuate wildly across runs.

**Remedies:**

- **Subsampling**: valid under weaker moment conditions
- **Robust statistics**: use the median or trimmed mean instead of the mean
- **Tail-specific methods**: estimate the tail index and use extreme value theory

!!! note "Finite vs Infinite Variance"
    If the population has finite variance but very heavy tails (e.g., $t_3$ distribution), the bootstrap is still valid but converges slowly. Larger $B$ and larger $n$ are needed. The bootstrap is inconsistent only when the variance is truly infinite.

## Small Sample Sizes

With very small $n$ (say $n < 10$), the empirical distribution $\hat{F}_n$ is a coarse approximation to $F$. The bootstrap distribution inherits this coarseness, and the resulting standard errors and confidence intervals can be unreliable.

**Specific problems with small $n$:**

- The bootstrap distribution is discrete with at most $\binom{2n-1}{n}$ distinct resamples
- Confidence interval coverage can be far from nominal
- The BCa acceleration estimate $\hat{a}$ based on $n$ jackknife values is noisy
- The bootstrap-$t$ can have extreme outliers when $\hat{\text{se}}^*$ is near zero

**Remedies:**

- **Parametric bootstrap**: if a parametric model is justifiable, it provides a smoother approximation to $F$
- **Exact methods**: for simple statistics (mean, proportion), use exact distributional results when available
- **Bayesian approaches**: incorporate prior information to regularize inference

## Non-Smooth Statistics

The bootstrap requires the statistic $\hat{\theta} = T(\hat{F}_n)$ to be a smooth functional of the empirical distribution. When $T$ is non-smooth (discontinuous or non-differentiable), the bootstrap can be inconsistent or have slow convergence.

**Examples of non-smooth statistics:**

- **Sample median** with tied or discrete data
- **Mode** of a distribution
- **Quantiles** at points where the density is zero
- **Indicator functions** such as $\mathbf{1}(\hat{\theta} > c)$

For the sample median with continuous data, the bootstrap is consistent but converges slowly. For the mode, the bootstrap can be inconsistent.

**Remedies:**

- **Smoothed bootstrap**: add a small amount of noise to each resampled observation to smooth the empirical distribution
- **Subsampling**: provides valid inference for a broader class of statistics
- **$m$-out-of-$n$ bootstrap**: valid for some non-regular statistics with appropriate $m$

## Estimating Extreme Quantiles

Bootstrapping extreme quantiles (e.g., the 99th or 99.9th percentile) is unreliable because the bootstrap has limited information about the tails of the distribution. With $n$ observations, the empirical distribution has no data beyond the sample maximum and minimum.

**Why it fails.** The bootstrap can only generate values that appear in the original sample. Extreme quantiles depend on the tail behavior of $F$, which is poorly captured by $\hat{F}_n$ when $n$ is moderate.

**Remedies:**

- **Extreme value theory**: fit a generalized Pareto distribution to the tail and extrapolate
- **Parametric bootstrap**: use a parametric model that captures the tail behavior
- **Peaks-over-threshold** methods

!!! tip "Rule of Thumb for Quantile Estimation"
    The bootstrap is reliable for estimating quantiles between the $1/\sqrt{n}$ and $1 - 1/\sqrt{n}$ levels. For $n = 100$, this means quantiles between 10% and 90% are reasonably estimated; for more extreme quantiles, use tail-specific methods.

## Infinite-Dimensional Parameters

The bootstrap assumes that the parameter of interest can be expressed as a smooth functional of the data distribution. For infinite-dimensional objects (e.g., the entire CDF, a nonparametric density estimate, or a functional data curve), the bootstrap requires additional regularity conditions that may not hold.

## Summary Table

| Failure Mode | Symptom | Alternative |
|---|---|---|
| Dependent data | CI too narrow, $p$-values too small | Block bootstrap, stationary bootstrap |
| Extreme order statistics | Bootstrap stuck at observed extremes | Parametric bootstrap, subsampling |
| Infinite variance | Wildly fluctuating SE estimates | Subsampling, robust statistics |
| Small $n$ ($< 10$) | Poor CI coverage, noisy estimates | Parametric bootstrap, exact methods |
| Non-smooth statistics | Slow convergence, inconsistency | Smoothed bootstrap, subsampling |
| Extreme quantiles | No data in tails | Extreme value theory |

## Summary

Resampling methods fail when the standard assumptions (independence, finite variance, smoothness, adequate sample size) are violated. The most common failure modes are dependent data (where the iid bootstrap underestimates uncertainty), extreme order statistics (where the bootstrap distribution cannot reach beyond the observed data range), heavy tails (where variance estimates are unstable), and small samples (where the empirical distribution is too coarse). Recognizing these limitations and knowing the appropriate remedies — block bootstrap, subsampling, parametric bootstrap, or specialized tail methods — is essential for responsible use of resampling techniques.

## Exercises

**Exercise 1.**
Simulate an AR(1) process $X_t = 0.7 X_{t-1} + \varepsilon_t$ with $\varepsilon_t \sim N(0, 1)$ and $n = 200$.

(a) Apply the standard (iid) bootstrap to estimate the 95% CI for $E[X]$. What coverage do you observe in simulation?

(b) Apply the **moving block bootstrap** with block lengths $\ell = 5, 10, 20$.

(c) Compare the widths and coverage of both approaches. Explain why the iid bootstrap fails for dependent data.
