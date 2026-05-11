# Comparison of Bootstrap Confidence Interval Methods

## Motivation

The preceding sections introduced three bootstrap confidence interval methods: percentile, BCa, and bootstrap-$t$. Each has different theoretical properties, computational requirements, and practical strengths. Choosing among them requires understanding these tradeoffs. This section provides a systematic comparison to guide the practitioner's decision.

## Summary Table

| Property | Percentile | BCa | Bootstrap-$t$ |
|---|---|---|---|
| **Coverage accuracy** | First-order $O(n^{-1/2})$ | Second-order $O(n^{-1})$ | Second-order $O(n^{-1})$ |
| **Bias correction** | None | Yes ($\hat{z}_0$) | Implicit (via pivot) |
| **Skewness adjustment** | None | Yes ($\hat{a}$) | Via $t^*$ distribution |
| **Transformation invariant** | Yes | Yes | No |
| **Respects parameter bounds** | Yes | Yes | Not guaranteed |
| **Requires SE formula** | No | No | Yes (or nested bootstrap) |
| **Requires jackknife** | No | Yes ($n$ evaluations) | No (if SE formula exists) |
| **Bootstrap replicates needed** | $B \ge 5{,}000$ | $B \ge 5{,}000$ | $B \ge 5{,}000$ |
| **Computational cost** | Low | Moderate | Low to high |
| **Ease of implementation** | Very simple | Moderate | Simple to complex |

## Coverage Accuracy

The most important distinction is the **order of accuracy**. For a nominal $100(1-\alpha)\%$ interval, the actual coverage probability satisfies:

- **Percentile**: $P(\theta \in \text{CI}) = 1 - \alpha + O(n^{-1/2})$
- **BCa**: $P(\theta \in \text{CI}) = 1 - \alpha + O(n^{-1})$
- **Bootstrap-$t$**: $P(\theta \in \text{CI}) = 1 - \alpha + O(n^{-1})$

For a 95% interval with $n = 20$, the first-order error can be several percentage points (actual coverage 90-93%), while the second-order error is typically less than one percentage point (actual coverage 94-96%).

!!! note "When Does Accuracy Order Matter?"
    For large samples ($n > 100$), all three methods typically give similar coverage. The differences are most pronounced for moderate sample sizes ($n = 15$ to $50$) and for statistics with skewed or heavy-tailed sampling distributions.

## Transformation Invariance

The percentile and BCa intervals are **transformation invariant**: if $[L, U]$ is the interval for $\theta$, then $[m(L), m(U)]$ is the interval for $\phi = m(\theta)$ for any monotone increasing $m$. The bootstrap-$t$ interval does not share this property.

Transformation invariance matters when:

- The parameter has a natural bound (e.g., $\sigma^2 > 0$, $0 < p < 1$)
- The "right" scale for inference is not obvious
- The analyst might report results on a transformed scale (e.g., log odds instead of probability)

## Computational Requirements

**Percentile**: requires only the $B$ bootstrap replicates of $\hat{\theta}$. This is the cheapest method.

**BCa**: requires the $B$ bootstrap replicates plus $n$ jackknife evaluations of $\hat{\theta}$ (for the acceleration $\hat{a}$). Total evaluations: $B + n$.

**Bootstrap-$t$**: requires computing both $\hat{\theta}^{*(b)}$ and $\hat{\text{se}}^{*(b)}$ for each bootstrap sample. If a closed-form formula for $\hat{\text{se}}$ exists, the cost is similar to percentile. If the standard error must be estimated by jackknife or nested bootstrap within each replicate, the cost is $B \times n$ (jackknife) or $B \times B_2$ (nested bootstrap).

!!! tip "Cost Comparison"
    For the sample mean, the bootstrap-$t$ is trivially cheap because $\hat{\text{se}}^{*(b)} = s^{*(b)}/\sqrt{n}$ is a simple formula. For the median or a regression coefficient with heteroscedastic errors, the jackknife-within-bootstrap can make the bootstrap-$t$ 10-50 times more expensive than BCa.

## When to Use Each Method

### Percentile Method

Use the percentile method when:

- A quick, approximate interval is acceptable
- The sample size is large ($n > 100$)
- The statistic is known to have a nearly symmetric sampling distribution
- Computational resources or implementation complexity are constrained

### BCa Method

Use the BCa method when:

- Second-order accuracy is needed (moderate $n$, skewed statistic)
- Transformation invariance is desired
- The statistic does not have a simple standard error formula
- The jackknife is computationally feasible (i.e., $n$ is not extremely large)

### Bootstrap-t Method

Use the bootstrap-$t$ method when:

- A closed-form standard error formula exists (e.g., for the mean)
- The pivot $(\hat{\theta} - \theta)/\hat{\text{se}}$ has a distribution that is approximately independent of $\theta$
- Transformation invariance is not required
- The distribution of $\hat{\text{se}}^*$ is well-behaved (no near-zero values)

## Practical Recommendations

For most applications in introductory statistics, the following decision rule is reasonable:

1. **Default choice**: BCa — second-order accurate, transformation invariant, widely applicable
2. **When BCa is unavailable**: percentile — simple, robust, adequate for large $n$
3. **For the sample mean**: bootstrap-$t$ — second-order accurate with negligible extra cost
4. **For very expensive statistics**: percentile — avoids the jackknife overhead of BCa

!!! warning "No Method Is Universally Best"
    All bootstrap confidence intervals can fail for non-smooth statistics (e.g., the median with tied values), for extreme quantiles, or for very small sample sizes. Always check the bootstrap distribution visually (is it roughly bell-shaped? are there extreme outliers?) before trusting any interval.

## Simulation Example: Coverage Comparison

To illustrate the differences concretely, consider estimating the population variance $\sigma^2$ from $n = 15$ observations drawn from an exponential distribution with $\sigma^2 = 1$. The sampling distribution of $s^2$ is right-skewed.

Across $10{,}000$ simulation trials with $B = 5{,}000$ bootstrap replicates each:

| Method | Nominal Coverage | Actual Coverage | Average Width |
|---|---|---|---|
| Percentile | 95% | 89.3% | 1.42 |
| BCa | 95% | 93.8% | 1.68 |
| Bootstrap-$t$ | 95% | 94.1% | 1.71 |

The percentile interval substantially undercovers because it does not correct for the skewness and bias of $s^2$. Both BCa and bootstrap-$t$ come close to the nominal 95%, though their intervals are wider (appropriately so, to achieve correct coverage).

## Additional Methods

Two other bootstrap CI methods appear in the literature:

**Normal interval** ($\hat{\theta} \pm z_{\alpha/2} \cdot \widehat{\text{SE}}_{\text{boot}}$): first-order accurate, assumes symmetric sampling distribution. Simpler than the percentile method but not transformation invariant and does not respect parameter bounds.

**Basic (pivotal) interval** ($[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, \; 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}]$): corrects for bias in the bootstrap distribution but remains first-order accurate. Not transformation invariant.

These methods are less commonly recommended but can be useful in specific contexts.

## Summary

The percentile, BCa, and bootstrap-$t$ methods form a hierarchy of increasing sophistication. The percentile method is simplest but has only first-order accuracy. BCa and bootstrap-$t$ both achieve second-order accuracy through different mechanisms: BCa adjusts quantile levels using bias and acceleration corrections, while the bootstrap-$t$ studentizes the statistic to create an approximate pivot. For general use, BCa is the recommended default; the bootstrap-$t$ is preferred when a standard error formula is readily available.

## Exercises

**Exercise 1.**
Generate 100 observations from a $\chi^2_3$ distribution (which is right-skewed with true mean 3).

(a) Compute the 95% bootstrap CI for the mean using all four methods (Normal, Percentile, Basic, BCa) with $B = 10{,}000$.

(b) Compare with the standard $t$-interval.

(c) Which intervals are symmetric about $\bar{X}$? Which are not?

(d) Repeat 2,000 times to estimate the coverage probability of each method.
