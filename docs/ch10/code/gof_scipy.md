# Chi-Square Goodness-of-Fit (scipy)

## Overview

This page shows how to perform a chi-square goodness-of-fit test using the convenience function `scipy.stats.chisquare`. Rather than computing the statistic and p-value by hand, `chisquare` accepts observed and expected frequency arrays and returns both values directly. This is the recommended approach for production code once you understand the underlying mathematics.

## Hypotheses

- **Null Hypothesis** ($H_0$): The observed frequencies follow the specified expected distribution.
- **Alternative Hypothesis** ($H_A$): The observed frequencies do not follow the specified expected distribution.

## Test Statistic

The function internally computes

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

with $\text{df} = k - 1$ degrees of freedom, where $k$ is the number of categories.

## Code

```python
from scipy import stats

# Observed frequencies for each outcome: Win, Loss, Tie
observed_frequencies = [4, 13, 7]

# Expected frequencies assuming an even distribution
total_games = sum(observed_frequencies)
expected_frequencies = [total_games / 3] * 3

# Perform the chi-square goodness-of-fit test
chi_square_statistic, p_value = stats.chisquare(
    f_obs=observed_frequencies, f_exp=expected_frequencies
)

# Output results
print(f"{chi_square_statistic = }")
print(f"{p_value = }")
```

**Key parameters of `stats.chisquare`:**

| Parameter | Description |
|-----------|-------------|
| `f_obs` | Array of observed frequencies |
| `f_exp` | Array of expected frequencies (must sum to same total as `f_obs`). If omitted, a uniform distribution is assumed. |
| `ddof` | Adjustment to degrees of freedom. Default is 0, giving $\text{df} = k - 1$. If parameters were estimated from data, set `ddof` accordingly. |

**Output for this example:**

- `chi_square_statistic = 5.25`
- `p_value = 0.07249...`

## When to Omit f_exp

If the null hypothesis specifies a uniform distribution, you can omit `f_exp` entirely:

```python
statistic, p = stats.chisquare(f_obs=[4, 13, 7])
```

SciPy will automatically set each expected frequency to $n / k$ where $n$ is the total count and $k$ is the length of `f_obs`.

## Non-Uniform Expected Proportions

When the null hypothesis specifies unequal proportions $p_1, p_2, \ldots, p_k$, compute expected counts as $E_i = n \cdot p_i$ and pass them explicitly:

```python
n = 300
proportions = [0.4, 0.3, 0.3]
expected = [n * p for p in proportions]
stat, pval = stats.chisquare(f_obs=[130, 85, 85], f_exp=expected)
```

## Interpretation

For the rock-paper-scissors example, the function returns $\chi^2 = 5.25$ with $p \approx 0.0725$. At significance level $\alpha = 0.05$, we fail to reject $H_0$. There is insufficient evidence that the game outcomes deviate from an equal distribution.

The `scipy.stats.chisquare` function is a thin wrapper around the manual calculation. Its main advantages are brevity and fewer opportunities for arithmetic mistakes.

## Exercises

**1.** Run `stats.chisquare` with only `f_obs=[10, 20, 30]` (no `f_exp`). What expected frequencies does SciPy assume, and what are the resulting statistic and p-value?

??? success "Solution to Exercise 1"

    SciPy assumes a uniform distribution, so $E_i = 60/3 = 20$ for each category. The statistic is

    $$
    \chi^2 = \frac{(10-20)^2}{20} + \frac{(20-20)^2}{20} + \frac{(30-20)^2}{20} = 5 + 0 + 5 = 10.0
    $$

    With $\text{df} = 2$, the p-value is $P(\chi^2_2 \ge 10) \approx 0.0067$. We would reject $H_0$ at $\alpha = 0.05$. $\square$

---

**2.** A researcher fits a Poisson model to data and estimates the parameter $\lambda$ from the sample. The data has 5 categories. What value should be passed to the `ddof` parameter and why?

??? success "Solution to Exercise 2"

    One parameter ($\lambda$) was estimated from the data, so we lose one additional degree of freedom. Pass `ddof=1`, which makes $\text{df} = k - 1 - \text{ddof} = 5 - 1 - 1 = 3$. The `ddof` parameter accounts for parameters estimated from the data that reduce the degrees of freedom beyond the standard $k-1$ baseline. $\square$

---

**3.** What happens if the sum of `f_exp` does not equal the sum of `f_obs`? Test this with `stats.chisquare(f_obs=[10, 20], f_exp=[5, 5])` and explain the result.

??? success "Solution to Exercise 3"

    SciPy will raise an error or produce a misleading result because the total expected count (10) does not match the total observed count (30). Specifically, `stats.chisquare` does **not** automatically rescale `f_exp`. Before the test is meaningful, the expected counts must sum to the same total as the observed counts. The correct call would rescale: `f_exp=[15, 15]` or use proportions multiplied by the observed total. $\square$

---

**4.** Show algebraically that the chi-square statistic can be rewritten as

$$
\chi^2 = \sum_{i=1}^{k} \frac{O_i^2}{E_i} - n
$$

where $n = \sum_{i=1}^{k} O_i = \sum_{i=1}^{k} E_i$.

??? success "Solution to Exercise 4"

    Expand the standard formula:

    $$
    \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} = \sum_{i=1}^{k} \frac{O_i^2 - 2O_iE_i + E_i^2}{E_i}
    $$

    $$
    = \sum_{i=1}^{k} \frac{O_i^2}{E_i} - 2\sum_{i=1}^{k} O_i + \sum_{i=1}^{k} E_i
    $$

    Since $\sum O_i = \sum E_i = n$, this simplifies to

    $$
    \chi^2 = \sum_{i=1}^{k} \frac{O_i^2}{E_i} - 2n + n = \sum_{i=1}^{k} \frac{O_i^2}{E_i} - n
    $$

    $\square$

---

**5.** A bag is claimed to contain 50% red, 30% blue, and 20% green marbles. You draw 200 marbles (with replacement) and observe $[90, 70, 40]$. Use `stats.chisquare` to test the claim at $\alpha = 0.01$. State your conclusion.

??? success "Solution to Exercise 5"

    Expected counts: $E = [200 \times 0.5,\; 200 \times 0.3,\; 200 \times 0.2] = [100, 60, 40]$.

    ```python
    from scipy import stats
    stat, p = stats.chisquare(f_obs=[90, 70, 40], f_exp=[100, 60, 40])
    ```

    $$
    \chi^2 = \frac{(90-100)^2}{100} + \frac{(70-60)^2}{60} + \frac{(40-40)^2}{40} = 1.0 + 1.667 + 0 = 2.667
    $$

    With $\text{df} = 2$, the p-value is approximately $0.2636$. Since $p = 0.2636 > 0.01 = \alpha$, we **fail to reject** $H_0$. The data is consistent with the claimed proportions at the 1% significance level. $\square$
