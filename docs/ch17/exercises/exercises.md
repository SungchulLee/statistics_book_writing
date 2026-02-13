# Exercises: Resampling Methods

## Conceptual

**Exercise 1.** Explain why bootstrap samples are drawn **with replacement** rather than without replacement. What would happen if we drew samples of size $n$ without replacement from the original data?

**Exercise 2.** A bootstrap sample of size $n$ drawn from a dataset of size $n$ contains, on average, only about 63.2% of the unique original observations.

(a) Derive this result. (Hint: The probability that observation $i$ is *not* selected in any of the $n$ draws is $(1 - 1/n)^n$.)

(b) What happens to this percentage as $n \to \infty$?

(c) The "out-of-bag" observations (those not in the bootstrap sample) are used in random forests for validation. Why does this work?

**Exercise 3.** Consider a permutation test for $H_0: \mu_X = \mu_Y$ when the two populations have the same mean but different variances ($\sigma_X^2 \neq \sigma_Y^2$).

(a) Is the exchangeability assumption satisfied under $H_0$?

(b) Can this lead to inflated Type I error? Explain.

(c) How would you modify the test to handle this case?

**Exercise 4.** Explain the difference between the **percentile** and **BCa** bootstrap confidence intervals. When does the BCa interval substantially differ from the percentile interval?

## Computation

**Exercise 5.** Generate 100 observations from a $\chi^2_3$ distribution (which is right-skewed with true mean 3).

(a) Compute the 95% bootstrap CI for the mean using all four methods (Normal, Percentile, Basic, BCa) with $B = 10{,}000$.

(b) Compare with the standard $t$-interval.

(c) Which intervals are symmetric about $\bar{X}$? Which are not?

(d) Repeat 2,000 times to estimate the coverage probability of each method.

**Exercise 6.** Given two samples:

- Group A: 14.2, 16.8, 13.5, 15.9, 17.3, 12.8, 16.1, 14.7
- Group B: 11.3, 13.6, 10.9, 12.4, 14.1, 11.8, 13.2, 12.7

(a) Perform a two-sample permutation test for the difference in means.

(b) Perform a permutation test using the **median** difference as the test statistic.

(c) Compare both p-values with the Welch $t$-test.

**Exercise 7.** Implement the **paired sign-flip permutation test** for the following before/after data (blood pressure readings):

| Patient | Before | After |
|---|---|---|
| 1 | 148 | 140 |
| 2 | 142 | 138 |
| 3 | 136 | 132 |
| 4 | 155 | 147 |
| 5 | 129 | 131 |
| 6 | 161 | 152 |
| 7 | 138 | 135 |
| 8 | 144 | 139 |

(a) Compute the exact p-value by enumerating all $2^8 = 256$ sign-flip permutations.

(b) Compare with the paired $t$-test and the Wilcoxon signed-rank test.

**Exercise 8.** Use the bootstrap to estimate the standard error and 95% CI for the **correlation coefficient** between two variables. Generate $n = 50$ observations from a bivariate normal with $\rho = 0.6$.

(a) Compute the bootstrap SE with $B = 5{,}000$.

(b) Compare the percentile CI with Fisher's $z$-transformation CI.

(c) Repeat with $\rho = 0.95$. Does the percentile interval capture the skewness near the boundary?

## Applied

**Exercise 9 (Finance).** Download daily returns for two stocks (e.g., AAPL and MSFT) over the past year.

(a) Use a permutation test to test whether the mean daily returns differ.

(b) Use the bootstrap to compute 95% CIs for the Sharpe ratio of each stock.

(c) Test whether the Sharpe ratios differ using a permutation test with $\text{SR}_1 - \text{SR}_2$ as the test statistic.

(d) Why is the bootstrap especially useful for the Sharpe ratio? (Hint: consider the sampling distribution of a ratio.)

**Exercise 10 (Block Bootstrap).** Simulate an AR(1) process $X_t = 0.7 X_{t-1} + \varepsilon_t$ with $\varepsilon_t \sim N(0, 1)$ and $n = 200$.

(a) Apply the standard (iid) bootstrap to estimate the 95% CI for $E[X]$. What coverage do you observe in simulation?

(b) Apply the **moving block bootstrap** with block lengths $\ell = 5, 10, 20$.

(c) Compare the widths and coverage of both approaches. Explain why the iid bootstrap fails for dependent data.
