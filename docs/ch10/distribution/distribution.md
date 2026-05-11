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
\sum_{i=1}^{k} (O_i - E_i) = 0
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

## Exercises

**Exercise 1.**
A fair die is rolled 120 times, producing the following counts: 1 (18), 2 (22), 3 (17), 4 (25), 5 (19), 6 (19). Compute the chi-square test statistic. With 5 degrees of freedom, the critical value at $\alpha = 0.05$ is 11.07. Is there evidence the die is unfair?

??? success "Solution to Exercise 1"
    Under $H_0$ (fair die), each expected count is $E_i = 120/6 = 20$.

    $$
    \chi^2 = \frac{(18-20)^2}{20} + \frac{(22-20)^2}{20} + \frac{(17-20)^2}{20} + \frac{(25-20)^2}{20} + \frac{(19-20)^2}{20} + \frac{(19-20)^2}{20}
    $$

    $$
    = \frac{4+4+9+25+1+1}{20} = \frac{44}{20} = 2.2
    $$

    Since $\chi^2 = 2.2 < 11.07$, we **fail to reject** $H_0$. There is no evidence that the die is unfair.

---

**Exercise 2.**
Explain why the chi-square statistic uses $E_i$ in the denominator rather than $O_i$.

??? success "Solution to Exercise 2"
    The denominator $E_i$ serves as a standardization factor. Each term $(O_i - E_i)^2 / E_i$ is approximately a squared standard normal variable because:

    $$
    \text{Var}(O_i) \approx E_i \quad \text{(when } np_i(1-p_i) \approx np_i = E_i\text{)}
    $$

    Using $E_i$ ensures that categories with larger expected counts contribute to the test statistic in proportion to the surprise of the deviation. A deviation of 5 from an expected count of 100 is much less noteworthy than a deviation of 5 from an expected count of 10, and the $E_i$ denominator appropriately downweights the former.

---

**Exercise 3.**
The chi-square test statistic is derived as an approximation. What assumption makes the approximation $\sqrt{np_i(1-p_i)} \approx \sqrt{np_i} = \sqrt{E_i}$ valid?

??? success "Solution to Exercise 3"
    The approximation requires that each $p_i$ is moderate to small, so that $1 - p_i \approx 1$. This holds when the number of categories $k$ is reasonably large and probabilities are spread out (no single category dominates). For example, if $k = 6$ and $p_i = 1/6$, then $1 - p_i = 5/6 \approx 0.83$, which is close enough to 1 for practical purposes.

    The approximation breaks down when a category has very high probability (e.g., $p_i = 0.9$, making $1 - p_i = 0.1$ far from 1). In such cases, the actual variance $np_i(1-p_i)$ is much smaller than $np_i$, and the simple chi-square test statistic may not follow the $\chi^2$ distribution as closely.

---

**Exercise 4.**
Pearson's chi-square test was published in 1900, but the correct degrees of freedom were not established until Fisher's work in 1922-1924. Explain intuitively why the degrees of freedom for a goodness-of-fit test with $k$ categories is $k - 1$, not $k$.

??? success "Solution to Exercise 4"
    The degrees of freedom are $k - 1$ because the $k$ observed counts are subject to one constraint: they must sum to $n$ (the total sample size). That is, $\sum_{i=1}^k O_i = n$. This means only $k-1$ of the observed counts are free to vary independently; the last one is determined by $n - \sum_{i=1}^{k-1} O_i$.

    Similarly, the deviations $O_i - E_i$ satisfy $\sum_{i=1}^k (O_i - E_i) = 0$, so the $k$ standardized deviations are not independent. This constraint reduces the dimension of the variation from $k$ to $k-1$, which is why the limiting distribution is $\chi^2_{k-1}$ rather than $\chi^2_k$.
