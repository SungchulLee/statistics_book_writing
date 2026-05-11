# Bootstrap Test for Correlation

## Motivation

The classical test for the Pearson correlation coefficient $r$ assumes bivariate normality. Under this assumption and $H_0: \rho = 0$, the statistic $t = r\sqrt{(n-2)/(1-r^2)}$ follows a $t_{n-2}$ distribution. When the data are non-normal — heavy-tailed, skewed, or contain outliers — this distributional result may not hold, and the classical $p$-value can be unreliable.

The bootstrap provides two complementary approaches to testing correlation: (1) a **bootstrap confidence interval** approach that inverts the interval to test any hypothesized value $\rho_0$, and (2) a **permutation-style bootstrap** that directly estimates the null distribution under $H_0: \rho = 0$.

## Testing Whether the Correlation Is Zero

The most common hypothesis is:

$$
H_0: \rho = 0 \quad \text{vs} \quad H_1: \rho \neq 0
$$

Under $H_0$, $X$ and $Y$ are uncorrelated (and, if we assume nothing else, exchangeable in their pairing). This motivates a **permutation approach**: break the association between $X$ and $Y$ by shuffling one variable while keeping the other fixed.

## Algorithm: Permutation Approach

Given paired observations $(x_1, y_1), \ldots, (x_n, y_n)$:

1. Compute the observed correlation $r_{\text{obs}}$ from the original data
2. **For** $b = 1, \ldots, B$:
    - Randomly permute the $y$-values: $(x_1, y_{\pi(1)}), \ldots, (x_n, y_{\pi(n)})$
    - Compute $r^{*(b)}$ from the permuted data
3. The two-sided $p$-value is:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\left(|r^{*(b)}| \ge |r_{\text{obs}}|\right)
$$

This is technically a permutation test rather than a bootstrap test (it samples without replacement from the permutation distribution). It provides an exact test of the null hypothesis that $X$ and $Y$ are independent, which is stronger than $\rho = 0$ alone.

!!! note "Permutation vs Bootstrap for Testing Correlation"
    For testing $H_0: \rho = 0$, the permutation approach is often preferred because it directly enforces independence under the null. The bootstrap approach (resampling pairs) is better suited for constructing confidence intervals for $\rho$ or testing $H_0: \rho = \rho_0$ for nonzero $\rho_0$.

## Algorithm: Bootstrap Confidence Interval Approach

To test $H_0: \rho = \rho_0$ for any value $\rho_0$ (not just zero), construct a bootstrap confidence interval for $\rho$ and reject if $\rho_0$ falls outside the interval.

1. Compute $r_{\text{obs}}$ from the original paired data
2. **For** $b = 1, \ldots, B$:
    - Resample $n$ pairs $(x_i, y_i)$ **with replacement** (keeping pairs intact)
    - Compute $r^{*(b)}$ from the bootstrap sample
3. Construct a $100(1-\alpha)\%$ bootstrap confidence interval (percentile, BCa, or bootstrap-$t$)
4. Reject $H_0: \rho = \rho_0$ at level $\alpha$ if $\rho_0$ is not in the interval

!!! warning "Keep Pairs Together"
    When resampling for correlation, always resample entire pairs $(x_i, y_i)$. Resampling $x$-values and $y$-values separately would destroy the dependence structure and produce meaningless bootstrap correlations.

## Bootstrap Standard Error for the Correlation

The bootstrap distribution of $r^*$ also provides a standard error estimate:

$$
\widehat{\text{SE}}_{\text{boot}}(r) = \text{sd}(r^{*(1)}, \ldots, r^{*(B)})
$$

Under bivariate normality, the asymptotic standard error of $r$ is approximately $(1-\rho^2)/\sqrt{n}$, which can be estimated as $(1-r^2)/\sqrt{n}$. The bootstrap standard error does not rely on normality and is valid more broadly.

## Fisher's z-Transform Bootstrap

For correlation coefficients, Fisher's z-transform often improves the bootstrap approximation:

$$
z = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right) = \text{arctanh}(r)
$$

Under bivariate normality, $z$ is approximately $N(\text{arctanh}(\rho), 1/(n-3))$. The transformed statistic has a sampling distribution closer to normal, which improves the coverage of percentile and BCa intervals.

**Procedure:**

1. For each bootstrap replicate, compute $z^{*(b)} = \text{arctanh}(r^{*(b)})$
2. Construct the bootstrap confidence interval on the $z$-scale
3. Transform back to the $r$-scale using $r = \tanh(z)$

Because the percentile and BCa methods are transformation invariant, this transformation does not change their intervals. However, it can improve the performance of the bootstrap-$t$ and normal intervals.

## Example

A dataset of $n = 25$ students shows the correlation between study hours and exam scores as $r_{\text{obs}} = 0.47$.

**Test $H_0: \rho = 0$ using the permutation approach:**

1. Permute the exam scores $B = 10{,}000$ times, computing $r^{*(b)}$ each time
2. Count how many times $|r^{*(b)}| \ge 0.47$
3. Suppose 178 out of $10{,}000$ permutations satisfy this condition
4. The $p$-value is $178/10{,}000 = 0.018$

**Construct a 95% bootstrap CI for $\rho$:**

1. Resample 25 pairs with replacement $B = 10{,}000$ times
2. Compute $r^{*(b)}$ for each resample
3. The BCa 95% interval is $[0.11, 0.72]$
4. Since $0 \notin [0.11, 0.72]$, this also rejects $H_0: \rho = 0$ at the 5% level

!!! example "Interpreting the Bootstrap Distribution of r"
    The bootstrap distribution of $r^*$ from the 25-pair resample is likely left-skewed (since $r_{\text{obs}} = 0.47$ is moderately positive, the distribution is bounded above by 1). This skewness is why the BCa interval is preferred over the percentile interval for correlation coefficients.

## Comparison of Approaches

| Approach | Tests | Assumptions | Preserves |
|---|---|---|---|
| Permutation | $H_0: \rho = 0$ only | Exchangeability under $H_0$ | Original data values |
| Bootstrap CI | Any $H_0: \rho = \rho_0$ | iid pairs | Pair structure |
| Classical $t$-test | $H_0: \rho = 0$ | Bivariate normality | Parametric efficiency |

The permutation approach is the most powerful for testing zero correlation because it uses the exact null distribution. The bootstrap CI approach is more flexible, allowing tests of any hypothesized value and providing interval estimates simultaneously.

## Summary

Bootstrap methods for testing correlation take two forms. For testing $H_0: \rho = 0$, the permutation approach (shuffling one variable) is preferred because it directly enforces independence under the null. For testing $H_0: \rho = \rho_0$ with nonzero $\rho_0$ or for constructing confidence intervals, the bootstrap resamples pairs with replacement and applies percentile, BCa, or bootstrap-$t$ intervals. Both approaches avoid the bivariate normality assumption required by the classical test.

## Exercises

**Exercise 1.**
Use the bootstrap to estimate the standard error and 95% CI for the **correlation coefficient** between two variables. Generate $n = 50$ observations from a bivariate normal with $\rho = 0.6$.

(a) Compute the bootstrap SE with $B = 5{,}000$.

(b) Compare the percentile CI with Fisher's $z$-transformation CI.

(c) Repeat with $\rho = 0.95$. Does the percentile interval capture the skewness near the boundary?
