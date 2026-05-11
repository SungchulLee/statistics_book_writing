# Percentile Method

## Motivation

The simplest bootstrap confidence interval uses the quantiles of the bootstrap distribution directly. Instead of assuming that $\hat{\theta}$ has a normal sampling distribution, the **percentile method** reads the interval endpoints straight from the bootstrap replicates. This makes the interval inherently adaptive: it respects skewness, respects natural parameter bounds, and requires no formula for the standard error.

Despite its simplicity, the percentile method has well-understood theoretical limitations. This section presents the method, explains when it works well, and identifies the situations where more sophisticated methods are needed.

## Definition

Let $\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}$ be the bootstrap replicates obtained by resampling with replacement from the observed data and computing the statistic $\hat{\theta}$ on each resample. The **percentile bootstrap confidence interval** at level $1 - \alpha$ is:

$$
\left[\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}\right]
$$

where $\hat{\theta}^*_{(q)}$ denotes the $q$-th quantile of the bootstrap distribution. For a 95% confidence interval ($\alpha = 0.05$), this is the interval from the 2.5th percentile to the 97.5th percentile of the bootstrap replicates.

In practice, with $B$ replicates sorted in increasing order $\hat{\theta}^*_{[1]} \le \hat{\theta}^*_{[2]} \le \cdots \le \hat{\theta}^*_{[B]}$, the lower bound is $\hat{\theta}^*_{[\lfloor B \cdot \alpha/2 \rfloor]}$ and the upper bound is $\hat{\theta}^*_{[\lceil B \cdot (1-\alpha/2) \rceil]}$.

## Algorithm

1. Compute the observed statistic $\hat{\theta} = g(x_1, \ldots, x_n)$
2. Generate $B$ bootstrap replicates $\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}$
3. Sort the replicates in increasing order
4. The $100(1-\alpha)\%$ confidence interval is:

$$
\left[\hat{\theta}^*_{[\lfloor B \cdot \alpha/2 \rfloor]}, \quad \hat{\theta}^*_{[\lceil B \cdot (1 - \alpha/2) \rceil]}\right]
$$

!!! tip "Choosing B for Percentile Intervals"
    For the percentile method, $B$ should be large enough that the extreme quantiles are stable. With $B = 1{,}000$, the 2.5th percentile is the 25th smallest value — reasonable but noisy. With $B = 10{,}000$, it is the 250th smallest value, giving a much more stable estimate. A minimum of $B = 5{,}000$ is recommended for confidence intervals.

## Why the Percentile Method Works

The percentile method has an elegant justification when there exists a monotone transformation $\phi = m(\theta)$ such that $\hat{\phi} = m(\hat{\theta})$ is normally distributed with constant variance. In that case, $\hat{\phi}$ is a **pivotal quantity** (up to a location shift), and the percentile interval for $\phi$ has correct coverage. Because $m$ is monotone, transforming back gives:

$$
[m^{-1}(\hat{\phi}^*_{(\alpha/2)}), \quad m^{-1}(\hat{\phi}^*_{(1-\alpha/2)})] = [\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}]
$$

The percentile interval on the $\theta$ scale is identical to the one on the $\phi$ scale transformed back, so it automatically respects the normalizing transformation without the user needing to know $m$.

!!! note "Transformation Invariance"
    The percentile method is **transformation invariant**: if $\phi = m(\theta)$ for any monotone increasing function $m$, then the percentile interval for $\phi$ is exactly $[m(\hat{\theta}^*_{(\alpha/2)}), m(\hat{\theta}^*_{(1-\alpha/2)})]$. This is a property that the normal-approximation interval $\hat{\theta} \pm z_{\alpha/2} \cdot \widehat{\text{SE}}$ does not share.

## Advantages

1. **Simplicity**: no formula for the standard error is needed
2. **Transformation invariance**: the interval automatically adapts to any monotone reparameterization
3. **Respects parameter bounds**: if $\theta \ge 0$ and all bootstrap replicates are non-negative, the interval stays non-negative
4. **Captures shape**: an asymmetric bootstrap distribution produces an asymmetric confidence interval

## Limitations

The percentile method has **first-order accuracy**, meaning its coverage error is $O(n^{-1/2})$. This can result in noticeable undercoverage or overcoverage for moderate sample sizes. The main sources of error are:

**Bias.** If the bootstrap distribution of $\hat{\theta}^*$ is centered away from $\hat{\theta}$ (i.e., the bootstrap is biased), the percentile interval shifts in the wrong direction. For example, if $\hat{\theta}$ systematically overestimates $\theta$, the bootstrap replicates will be centered above $\theta$, and the percentile interval will be too high.

**Skewness without a normalizing transformation.** When no monotone transformation makes the sampling distribution of $\hat{\theta}$ approximately normal, the percentile interval can have substantially incorrect coverage on one side.

!!! warning "Coverage Can Be Poor"
    Simulation studies show that the percentile method can have actual coverage of 85-90% when the nominal level is 95%, particularly for skewed statistics like the sample variance, odds ratios, or correlation coefficients with small $n$. The BCa and bootstrap-$t$ methods address these deficiencies.

## Example: Confidence Interval for the Median

Consider a sample of $n = 25$ observations from a right-skewed distribution. The sample median is $\hat{\theta} = 14.3$.

**Percentile bootstrap procedure:**

1. Generate $B = 10{,}000$ bootstrap replicates of the median
2. Sort the replicates
3. The 95% confidence interval is $[\hat{\theta}^*_{[250]}, \hat{\theta}^*_{[9750]}]$

Suppose the sorted bootstrap medians give $\hat{\theta}^*_{[250]} = 11.8$ and $\hat{\theta}^*_{[9750]} = 16.5$. The 95% percentile interval is $[11.8, 16.5]$.

Note that this interval is asymmetric around the observed median $14.3$: the distance to the lower bound ($2.5$) differs from the distance to the upper bound ($2.2$), reflecting the skewness of the sampling distribution.

## Comparison with the Normal Interval

The **normal bootstrap interval** uses the bootstrap standard error:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \widehat{\text{SE}}_{\text{boot}}
$$

This interval is always symmetric around $\hat{\theta}$ and can extend beyond natural parameter bounds. The percentile interval avoids both issues but can suffer from bias in the bootstrap distribution.

The **basic (pivotal) bootstrap interval** attempts to correct for bias:

$$
\left[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, \quad 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}\right]
$$

Note the reversed quantiles. The basic interval corrects the first-order bias of the percentile interval but sacrifices transformation invariance.

## Summary

The percentile method constructs confidence intervals by reading quantiles directly from the bootstrap distribution. Its simplicity, transformation invariance, and ability to respect parameter bounds make it a natural starting point. However, its first-order accuracy means that coverage can be poor when the bootstrap distribution is biased or when no normalizing transformation exists. For improved coverage, the BCa and bootstrap-$t$ methods (covered in the following sections) provide second-order corrections.


## Exercises

**Exercise 1.**
Describe the main concept of Percentile Method and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Percentile Method is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
