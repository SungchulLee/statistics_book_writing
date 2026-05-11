# Bootstrap vs Permutation Tests

## Motivation

Both bootstrap and permutation methods use resampling to perform inference without relying on parametric distributional assumptions. However, they answer fundamentally different questions and operate under different mechanisms. Understanding these differences is essential for choosing the right tool for a given problem.

This section clarifies what each method estimates, when each is appropriate, and how they relate to each other.

## The Fundamental Distinction

The bootstrap and the permutation test approximate different distributions:

- **Bootstrap**: approximates the **sampling distribution** of a statistic under the true (unknown) data-generating process
- **Permutation test**: approximates the **null distribution** of a test statistic under a specific null hypothesis (typically exchangeability or independence)

This distinction has far-reaching consequences for how each method is used and interpreted.

## Side-by-Side Comparison

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **Goal** | Estimate sampling distribution | Estimate null distribution |
| **Resampling** | With replacement | Without replacement (shuffle labels) |
| **Null hypothesis** | Not required | Required and enforced |
| **Produces** | SE, CI, bias, distribution shape | $p$-value |
| **Confidence intervals** | Yes (percentile, BCa, bootstrap-$t$) | Only by inversion |
| **Exactness** | Approximate (Monte Carlo) | Exact (conditional on data) |
| **Sample values** | Can repeat observations | Each observation appears exactly once |
| **Validity condition** | iid (or appropriate dependence structure) | Exchangeability under $H_0$ |

## What the Bootstrap Approximates

The bootstrap draws samples with replacement from the observed data (or from a fitted model). Each bootstrap sample has the same size $n$ as the original data, but some observations are repeated and others are omitted. The bootstrap distribution of $\hat{\theta}^*$ approximates the sampling distribution of $\hat{\theta}$ — that is, the variability of $\hat{\theta}$ across hypothetical repeated samples from the population.

Because the bootstrap does not enforce any null hypothesis, it is naturally suited for:

- Estimating standard errors
- Constructing confidence intervals
- Estimating bias
- Assessing the shape of the sampling distribution

For hypothesis testing, the bootstrap must be modified (e.g., by centering or pooling) to generate samples under $H_0$.

## What the Permutation Test Approximates

The permutation test shuffles group labels (or breaks pairings) to generate the distribution of the test statistic under $H_0$. Each permuted dataset uses the exact same observations, just with reassigned labels. No observations are duplicated or omitted.

The permutation distribution is the **exact conditional distribution** of the test statistic given the observed data, under the null hypothesis. This means the permutation $p$-value is exact (not approximate) for testing $H_0$, conditional on the sufficient statistic of the combined data.

The permutation test is naturally suited for:

- Testing hypotheses about group differences (two-sample, multi-sample)
- Testing independence or association
- Situations where the null hypothesis implies exchangeability

!!! note "Exactness of Permutation Tests"
    The permutation test provides an exact $p$-value in the sense that it controls the Type I error rate at exactly level $\alpha$ (conditional on the data). The only approximation arises when we use a random subset of all possible permutations rather than enumerating all $\binom{m+n}{m}$ arrangements. With $B = 10{,}000$ or more random permutations, this Monte Carlo approximation is negligible.

## When to Use the Bootstrap

Choose the bootstrap when:

- **Confidence intervals** are the primary goal (not just a $p$-value)
- **Standard error estimation** is needed for a complex statistic
- **The null hypothesis does not imply exchangeability** (e.g., testing $H_0: \rho = 0.5$ rather than $\rho = 0$)
- **The parameter of interest is not a simple group comparison** (e.g., regression coefficients, variance ratios)
- **Bias estimation** or distributional shape assessment is needed

## When to Use the Permutation Test

Choose the permutation test when:

- **Testing a sharp null hypothesis** that implies exchangeability (e.g., $H_0: F_X = F_Y$)
- **Exact Type I error control** is important (e.g., in regulatory or confirmatory settings)
- **The sample size is small** and the bootstrap distribution may be unreliable
- **The test statistic is complex** but the null hypothesis is simple (exchangeability)
- **No parametric model is assumed** and you want a distribution-free test

## Bootstrap for Hypothesis Testing

When the bootstrap is used for hypothesis testing, it must generate samples that respect $H_0$. The two main approaches are:

**Centering.** Shift the data so that $H_0$ is satisfied, then resample with replacement from the shifted data. Example: to test $H_0: \mu = \mu_0$, shift $\tilde{x}_i = x_i - \bar{x} + \mu_0$ and resample from $\{\tilde{x}_1, \ldots, \tilde{x}_n\}$.

**Pooling.** Combine the groups under $H_0$ and resample from the pooled sample. Example: to test $H_0: \mu_X = \mu_Y$, pool all observations and draw two bootstrap samples of sizes $m$ and $n$.

Both approaches introduce approximation beyond the exact conditional inference of the permutation test. The bootstrap hypothesis test is consistent but not exact.

!!! tip "When Both Are Valid, Which Is Better?"
    For testing $H_0: F_X = F_Y$ with two independent samples, both the permutation test and the pooled bootstrap are valid. The permutation test has exact Type I error control and is generally preferred. The bootstrap is preferred when you also want confidence intervals or when the null hypothesis is more nuanced than simple exchangeability.

## Hybrid Approaches

In practice, both methods are often used together:

1. Use the **permutation test** to obtain an exact $p$-value for $H_0$
2. Use the **bootstrap** to construct a confidence interval for the effect size
3. Report both: the permutation $p$-value quantifies evidence against $H_0$, and the bootstrap CI quantifies the magnitude and uncertainty of the effect

## Common Misconceptions

**"The bootstrap tests a null hypothesis."** Without modification (centering or pooling), the standard bootstrap does not generate a null distribution. It approximates the sampling distribution under the true parameter value.

**"Permutation tests give confidence intervals."** While permutation $p$-values can be inverted to form confidence intervals, this is computationally expensive and rarely done in practice. The bootstrap is far more natural for interval estimation.

**"Both methods give the same answer."** For testing $H_0: \mu_X = \mu_Y$, the pooled bootstrap and the permutation test often give similar $p$-values, but they can disagree when the groups have different variances or when the sample sizes are small.

## Summary

The bootstrap approximates the sampling distribution and is best for confidence intervals, standard errors, and bias estimation. The permutation test approximates the null distribution under exchangeability and is best for exact hypothesis testing. For two-sample comparisons under a strong null ($F_X = F_Y$), both methods are valid and the permutation test offers exact Type I error control. For more general inference — confidence intervals, testing non-exchangeable nulls, or estimating distributional properties — the bootstrap is the appropriate tool.

## Exercises

**Exercise 1.**
Download daily returns for two stocks (e.g., AAPL and MSFT) over the past year.

(a) Use a permutation test to test whether the mean daily returns differ.

(b) Use the bootstrap to compute 95% CIs for the Sharpe ratio of each stock.

(c) Test whether the Sharpe ratios differ using a permutation test with $\text{SR}_1 - \text{SR}_2$ as the test statistic.

(d) Why is the bootstrap especially useful for the Sharpe ratio? (Hint: consider the sampling distribution of a ratio.)
