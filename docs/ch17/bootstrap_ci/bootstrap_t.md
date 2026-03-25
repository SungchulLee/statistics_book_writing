# Bootstrap-t Method

## Motivation

The classical $t$-interval $\hat{\theta} \pm t_{\alpha/2} \cdot \hat{\text{se}}$ assumes the pivot $(\hat{\theta} - \theta)/\hat{\text{se}}$ follows a known distribution (Student's $t$ or standard normal). The **bootstrap-$t$ method** (also called the **studentized bootstrap**) avoids this assumption by using the bootstrap to estimate the distribution of the pivot itself. This produces a confidence interval with **second-order accuracy**, matching the BCa method in coverage precision while providing an interval that is not transformation invariant.

The key idea is to bootstrap not just the statistic $\hat{\theta}$ but the entire $t$-statistic, including its denominator.

## The Studentized Bootstrap Statistic

For each bootstrap sample, compute both the estimate and its estimated standard error:

$$
t^{*(b)} = \frac{\hat{\theta}^{*(b)} - \hat{\theta}}{\hat{\text{se}}^{*(b)}}
$$

where $\hat{\theta}^{*(b)}$ is the statistic computed from the $b$-th bootstrap sample, and $\hat{\text{se}}^{*(b)}$ is the estimated standard error of $\hat{\theta}$ computed from the same bootstrap sample.

The quantity $t^{*(b)}$ is a bootstrap version of the pivot $(\hat{\theta} - \theta)/\hat{\text{se}}$. Its distribution across bootstrap samples approximates the true distribution of the pivot.

## Algorithm

1. Compute the observed statistic $\hat{\theta}$ and its estimated standard error $\hat{\text{se}}$ from the original sample
2. **For** $b = 1, \ldots, B$:
    - Draw a bootstrap sample of size $n$ with replacement
    - Compute $\hat{\theta}^{*(b)}$ from the bootstrap sample
    - Compute $\hat{\text{se}}^{*(b)}$ from the bootstrap sample (see note below)
    - Compute $t^{*(b)} = (\hat{\theta}^{*(b)} - \hat{\theta}) / \hat{\text{se}}^{*(b)}$
3. Let $t^*_{(q)}$ denote the $q$-th quantile of $\{t^{*(1)}, \ldots, t^{*(B)}\}$
4. The $100(1-\alpha)\%$ bootstrap-$t$ confidence interval is:

$$
\left[\hat{\theta} - t^*_{(1-\alpha/2)} \cdot \hat{\text{se}}, \quad \hat{\theta} - t^*_{(\alpha/2)} \cdot \hat{\text{se}}\right]
$$

Note the reversal of quantiles: the upper quantile of $t^*$ is used for the lower bound of the interval and vice versa. This follows from inverting the pivot inequality $t^*_{(\alpha/2)} \le (\hat{\theta} - \theta)/\hat{\text{se}} \le t^*_{(1-\alpha/2)}$.

## Estimating the Inner Standard Error

The most critical and computationally demanding aspect of the bootstrap-$t$ is computing $\hat{\text{se}}^{*(b)}$ for each bootstrap sample. Three common approaches exist:

**Formula-based.** When a closed-form formula for $\hat{\text{se}}$ is available (e.g., $s/\sqrt{n}$ for the mean), apply the same formula to each bootstrap sample. This is fast and numerically stable.

**Jackknife within bootstrap.** For each bootstrap sample, compute the jackknife standard error by removing one observation at a time from that bootstrap sample. This requires $n$ additional computations per bootstrap replicate.

**Nested bootstrap (double bootstrap).** For each bootstrap sample, draw $B_2$ second-level bootstrap resamples to estimate $\hat{\text{se}}^{*(b)}$. This produces $B \times B_2$ total computations and is usually prohibitively expensive.

!!! tip "Practical Recommendation for the Inner Standard Error"
    Use a formula-based estimate when one exists. Otherwise, use the jackknife within each bootstrap sample. The nested bootstrap is rarely needed in practice and its computational cost ($B \times B_2$ evaluations) is seldom justified.

## Why It Achieves Second-Order Accuracy

The bootstrap-$t$ interval has coverage error of $O(n^{-1})$ compared to $O(n^{-1/2})$ for the percentile interval. The reason is that the studentized statistic $t^* = (\hat{\theta}^* - \hat{\theta})/\hat{\text{se}}^*$ approximates the distribution of $(\hat{\theta} - \theta)/\hat{\text{se}}$ to a higher order than the unstudentized $\hat{\theta}^* - \hat{\theta}$ approximates $\hat{\theta} - \theta$.

Studentization absorbs the leading-order effect of non-constant variance. If $\text{Var}(\hat{\theta})$ depends on $\theta$, the unstudentized bootstrap distribution has the wrong spread; the studentized version self-corrects by dividing by $\hat{\text{se}}^*$.

!!! note "Comparison with BCa"
    Both BCa and bootstrap-$t$ achieve second-order accuracy. The BCa interval is transformation invariant but requires jackknife calculations for the acceleration. The bootstrap-$t$ is not transformation invariant but directly uses the pivot, which can be more natural in certain contexts. In simulation studies, both methods typically give similar coverage for smooth statistics.

## Example: Bootstrap-t Interval for the Mean

Consider a sample of $n = 20$ observations with $\bar{x} = 7.3$ and $s = 2.1$, giving $\hat{\text{se}} = s/\sqrt{n} = 0.470$.

**Bootstrap-$t$ procedure:**

1. For $b = 1, \ldots, 10{,}000$: draw 20 observations with replacement, compute $\bar{x}^{*(b)}$ and $s^{*(b)}/\sqrt{20}$
2. Compute $t^{*(b)} = (\bar{x}^{*(b)} - \bar{x}) / (s^{*(b)}/\sqrt{20})$
3. Suppose the 2.5th and 97.5th percentiles of $\{t^{*(b)}\}$ are $t^*_{(0.025)} = -2.18$ and $t^*_{(0.975)} = 2.31$
4. The 95% bootstrap-$t$ interval is:

$$
[7.3 - 2.31 \times 0.470, \quad 7.3 - (-2.18) \times 0.470] = [6.21, \; 8.32]
$$

Compare this with the classical $t$-interval: $7.3 \pm 2.093 \times 0.470 = [6.32, 8.28]$. The bootstrap-$t$ interval is asymmetric, reflecting the slight skewness of the sampling distribution.

## Advantages

- **Second-order accuracy**: coverage error of $O(n^{-1})$
- **Natural for pivot-based inference**: directly extends the classical $t$-statistic approach
- **No jackknife needed** (unlike BCa) when a formula for $\hat{\text{se}}$ is available

## Limitations

**Computational cost.** Computing $\hat{\text{se}}^{*(b)}$ inside each bootstrap loop can be expensive when no closed-form formula exists.

**Not transformation invariant.** The bootstrap-$t$ interval for $\phi = m(\theta)$ is generally not $[m(L), m(U)]$. If transformation invariance is important, BCa is preferred.

**Unstable tails.** When $\hat{\text{se}}^{*(b)}$ is occasionally very small (e.g., in small samples), $t^{*(b)}$ can be extremely large, producing erratic quantile estimates. Winsorizing or trimming extreme $t^*$ values can help.

!!! warning "Small Standard Errors in Bootstrap Samples"
    If a bootstrap sample happens to have very low variability (e.g., many repeated values), $\hat{\text{se}}^{*(b)}$ can be near zero, producing extreme $t^{*(b)}$ values. These outliers inflate the interval width. Monitoring for and addressing extreme $t^*$ values is important in practice.

## Summary

The bootstrap-$t$ method estimates the distribution of the studentized pivot $(\hat{\theta} - \theta)/\hat{\text{se}}$ via bootstrap resampling. By incorporating the standard error estimate into each bootstrap replicate, it achieves second-order accurate coverage. The main challenge is computing the inner standard error $\hat{\text{se}}^{*(b)}$ efficiently. When a formula for the standard error exists, the bootstrap-$t$ is straightforward and highly effective; otherwise, the BCa method may be more practical.
