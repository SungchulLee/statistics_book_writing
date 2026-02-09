# Chi-Square Distribution

## Overview

The **chi-square distribution** arises naturally when summing squares of independent standard normal random variables. If $Z_1, Z_2, \dots, Z_k$ are independent standard normal variables, then

$$
Q = \sum_{i=1}^{k} Z_i^2 \sim \chi^2_k
$$

follows a chi-square distribution with $k$ degrees of freedom.

This distribution is fundamental to several hypothesis tests for categorical data, including the **goodness-of-fit test**, the **test of independence**, and the **test of homogeneity**.

## Connection to Categorical Data

We have a categorical variable with $k$ possible outcomes.

- Observed counts: $O_1, O_2, \dots, O_k$
- Expected counts (under $H_0$): $E_1, E_2, \dots, E_k$

Under the null hypothesis $H_0$, the probabilities are fixed at $p_1, p_2, \dots, p_k$.
We want to see whether the deviations $O_i - E_i$ are small enough to be attributed to chance.

## Normal Approximation to Counts

If the total number of observations is $n$, then the vector of counts

$$
(O_1, O_2, \dots, O_k)
$$

follows a **multinomial distribution** with parameters $n$ and $(p_1, p_2, \dots, p_k)$.

For large $n$, by the **multivariate Central Limit Theorem**, this multinomial distribution can be approximated by a **multivariate normal distribution**:

$$
O_i \approx N(E_i, \operatorname{Var}(O_i))
$$

with:

$$
E[O_i] = n p_i, \quad \operatorname{Var}(O_i) = n p_i (1 - p_i), \quad \operatorname{Cov}(O_i, O_j) = -n p_i p_j
$$

That negative covariance reflects the fact that the counts must sum to $n$ — if one category has more counts, others must have fewer.

## Standardizing the Deviations

Define standardized residuals:

$$
Z_i = \frac{O_i - E_i}{\sqrt{E_i}}
$$

If the categories were *independent*, each $Z_i$ would be approximately standard normal ($N(0,1)$).
Then the sum of their squares

$$
\sum Z_i^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

would approximately follow a $\chi^2$ distribution with $k$ degrees of freedom.

## The Constraint

Here is the key insight: because the total count $n$ is fixed,

$$
\sum_{i=1}^{k} (O_i - E_i) = 0.
$$

That is a **constraint**, meaning only $k - 1$ of the $O_i - E_i$ values are free to vary. One of them is always determined by the others.

## The Chi-Square Test Statistic

Therefore, the test statistic

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

is approximately distributed as a **chi-square with $k - 1$ degrees of freedom** because:

- Each standardized term behaves approximately like a squared standard normal variable.
- The sum involves $k$ such terms.
- One degree of freedom is lost due to the constraint that total counts must add up to $n$.

Hence:

$$
\boxed{\chi^2 \sim \chi^2_{(k-1)} \text{ approximately under } H_0.}
$$

## About the Denominator

### The True Standardization

For each category $i$:

$$
Z_i = \frac{O_i - E_i}{\sqrt{\operatorname{Var}(O_i)}}
$$

and under the multinomial model:

$$
\operatorname{Var}(O_i) = n p_i (1 - p_i)
$$

### The Practical Approximation

In practice, when $n$ is large and each $p_i$ is small or moderate,
$1 - p_i \approx 1$.

That leads to the approximation:

$$
\sqrt{\operatorname{Var}(O_i)} = \sqrt{n p_i (1 - p_i)} \approx \sqrt{n p_i} = \sqrt{E_i}
$$

So the **true denominator** should be $\sqrt{n p_i (1 - p_i)}$, but because $(1 - p_i)$ is close to 1 and we want a test that sums over all categories, we simplify to $\sqrt{E_i}$.

This simplification is part of what makes the test statistic

$$
\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}
$$

work so cleanly in practice.

## Historical Note

In 1900, Pearson published a paper claiming that as $n \rightarrow \infty$

$$
\sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i} \Rightarrow \chi^2_{k-1}
$$

However, there was some controversy in practical applications, and it was not settled for 20 years until Fisher's 1922 and 1924 papers.
