# Permutation Test for Correlation

## Motivation

The classical test for the Pearson correlation uses the statistic $t = r\sqrt{(n-2)/(1-r^2)}$, which follows a $t_{n-2}$ distribution under $H_0: \rho = 0$ when the data are bivariate normal. When the normality assumption fails, this $p$-value can be inaccurate.

The permutation test for correlation provides an exact, distribution-free test of independence between two variables. By randomly shuffling one variable while keeping the other fixed, it destroys any association and generates the null distribution of the test statistic directly from the data.

## The Hypothesis

We test:

$$
H_0: X \text{ and } Y \text{ are independent} \quad \text{vs} \quad H_1: X \text{ and } Y \text{ are associated}
$$

Under $H_0$, any pairing of the $x$-values with the $y$-values is equally likely. The test statistic is the Pearson correlation coefficient (though Spearman's $\rho$ or Kendall's $\tau$ can be used in the same framework):

$$
r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}
$$

!!! note "Independence vs Zero Correlation"
    The permutation test actually tests **independence**, which is a stronger condition than $\rho = 0$. Two variables can have $\rho = 0$ while still being dependent (e.g., $Y = X^2$ with $X$ symmetric around zero). However, for most practical purposes and for Pearson's $r$, the permutation test is used and interpreted as a test of $H_0: \rho = 0$.

## Algorithm

Given paired observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$:

1. Compute the observed correlation $r_{\text{obs}}$ from the original paired data
2. **For** $b = 1, 2, \ldots, B$:
    - Generate a random permutation $\pi$ of $\{1, 2, \ldots, n\}$
    - Form the permuted pairs $(x_1, y_{\pi(1)}), (x_2, y_{\pi(2)}), \ldots, (x_n, y_{\pi(n)})$
    - Compute $r^{*(b)}$ from the permuted pairs
3. The two-sided $p$-value is:

$$
p = \frac{\#\{b : |r^{*(b)}| \ge |r_{\text{obs}}|\} + 1}{B + 1}
$$

The "$+1$" in both numerator and denominator includes the observed data as one of the permutations, ensuring the $p$-value is never exactly zero and making the test exact.

For one-sided alternatives:

- $H_1: \rho > 0$: $\displaystyle p = \frac{\#\{b : r^{*(b)} \ge r_{\text{obs}}\} + 1}{B + 1}$
- $H_1: \rho < 0$: $\displaystyle p = \frac{\#\{b : r^{*(b)} \le r_{\text{obs}}\} + 1}{B + 1}$

## The Exact Permutation Distribution

There are $n!$ possible permutations of $n$ observations. For small $n$, all permutations can be enumerated to compute the exact $p$-value:

| $n$ | Number of permutations ($n!$) | Feasibility |
|---|---|---|
| 5 | 120 | Exact enumeration trivial |
| 8 | 40,320 | Exact enumeration easy |
| 10 | 3,628,800 | Exact enumeration feasible |
| 12 | 479,001,600 | Borderline |
| 15 | $> 10^{12}$ | Random sampling required |

For $n \ge 12$, random sampling of $B$ permutations is standard. With $B = 10{,}000$, the Monte Carlo error in the $p$-value is approximately $\sqrt{p(1-p)/B}$.

## Properties

**Exactness.** Under $H_0$, the permutation test has exact Type I error rate $\alpha$ (conditional on the observed data values). This holds regardless of the marginal distributions of $X$ and $Y$.

**Distribution-free.** No assumptions about the shape of the joint or marginal distributions are needed. The test is valid for continuous, discrete, or mixed data.

**Consistency.** The permutation test is consistent: as $n \to \infty$, the power against any fixed alternative with $\rho \neq 0$ approaches 1.

**Finite-sample validity.** Unlike asymptotic tests, the permutation test controls Type I error exactly for any sample size $n$.

!!! tip "Choice of Test Statistic"
    The permutation framework is agnostic to the choice of test statistic. Using Pearson's $r$ tests for linear association; using Spearman's $\rho$ or Kendall's $\tau$ tests for monotone association; using distance correlation tests for any type of dependence. The permutation mechanism (shuffling $y$-values) is the same regardless.

## Example

A dataset of $n = 20$ cities records average temperature ($x$) and ice cream sales ($y$). The observed Pearson correlation is $r_{\text{obs}} = 0.71$.

**Permutation test procedure:**

1. Randomly shuffle the sales values $B = 10{,}000$ times
2. For each permutation, compute $r^{*(b)}$
3. Count how many times $|r^{*(b)}| \ge 0.71$
4. Suppose 3 out of $10{,}000$ permutations satisfy this condition

The $p$-value is $(3 + 1)/(10{,}000 + 1) = 0.0004$.

The classical $t$-test gives $t = 0.71\sqrt{18/(1-0.71^2)} = 4.28$ with $p < 0.001$ from the $t_{18}$ distribution. The two approaches agree closely, but the permutation test makes no normality assumption.

## Visualization: The Null Distribution

The histogram of $\{r^{*(1)}, \ldots, r^{*(B)}\}$ shows the distribution of correlation values expected under independence. This distribution is symmetric around zero (because random permutations are equally likely to produce positive and negative correlations) and approximately normal for moderate $n$ (by the central limit theorem applied to the permutation distribution).

The observed $r_{\text{obs}}$ is marked on the histogram. If it falls far into the tails, the evidence against $H_0$ is strong.

## Comparison with the Bootstrap Test

| Aspect | Permutation Test | Bootstrap Test |
|---|---|---|
| Resampling | Without replacement (shuffle) | With replacement (resample pairs) |
| Tests | Independence ($H_0: X \perp Y$) | Any $H_0: \rho = \rho_0$ (via CI) |
| Exactness | Exact conditional $p$-value | Approximate |
| Confidence interval | Only by inversion | Directly available |
| Best for | Testing zero correlation | CI for $\rho$, testing nonzero $\rho_0$ |

For the specific hypothesis $H_0: \rho = 0$, the permutation test is preferred because of its exact Type I error control. For confidence intervals or testing $H_0: \rho = \rho_0$ with $\rho_0 \neq 0$, the bootstrap is more natural.

## Summary

The permutation test for correlation tests $H_0: X$ and $Y$ are independent by randomly shuffling one variable and recomputing the correlation on each permuted dataset. The proportion of permuted correlations as extreme as the observed value gives the $p$-value. This test is exact, distribution-free, and valid for any sample size. It works with Pearson's $r$, Spearman's $\rho$, Kendall's $\tau$, or any other measure of association. For small $n$, all permutations can be enumerated; for larger $n$, random sampling provides a close approximation.


## Exercises

**Exercise 1.**
Describe the main concept of Permutation Test for Correlation and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Permutation Test for Correlation is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
