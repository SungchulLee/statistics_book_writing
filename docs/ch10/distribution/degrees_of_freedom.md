# Degrees of Freedom and Asymptotic Theory

## Degrees of Freedom in Chi-Square Tests

The degrees of freedom determine the shape of the chi-square distribution used as the reference distribution under $H_0$. The calculation depends on which chi-square test is being performed.

### Goodness-of-Fit Test

For the goodness-of-fit test with $k$ categories:

$$
\text{df} = k - 1
$$

The single constraint arises because the observed counts must sum to the total sample size $n$:

$$
\sum_{i=1}^{k} O_i = n = \sum_{i=1}^{k} E_i
$$

This means only $k - 1$ of the deviations $O_i - E_i$ are free to vary; the last is determined by the others.

### Test of Independence and Homogeneity

For an $r \times c$ contingency table:

$$
\text{df} = (r - 1) \times (c - 1)
$$

where $r$ is the number of rows and $c$ is the number of columns.

The constraints are:

- Row totals must match: $r - 1$ independent constraints from rows.
- Column totals must match: $c - 1$ independent constraints from columns.
- One constraint is redundant (the grand total).

So the total number of free cells in the table is:

$$
rc - 1 - (r - 1) - (c - 1) = (r - 1)(c - 1)
$$

## Asymptotic Theory

### The Core Result

Under $H_0$, for large sample sizes, the chi-square test statistic

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

converges in distribution to a chi-square random variable with the appropriate degrees of freedom. This is an **asymptotic** result — it holds approximately for finite samples, and the approximation improves as the sample size grows.

### Derivation Sketch for Goodness-of-Fit

$$
\sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}
= \sum_{i=1}^k \frac{\left(\left(\sum_{j=1}^{n} X_j\right) - np_i\right)^2}{np_i}
= \sum_{i=1}^k \left(\frac{\left(\sum_{j=1}^{n} X_j\right) - np_i}{\sqrt{np_i}}\right)^2
$$

Approximating the denominator:

$$
\approx \sum_{i=1}^k \left(\frac{\left(\sum_{j=1}^{n} X_j\right) - np_i}{\sqrt{np_i(1-p_i)}}\right)^2
\approx \sum_{i=1}^k Z_i^2
= \chi^2_{k-1}
$$

The final step uses the fact that the $Z_i$ are not fully independent (they satisfy a linear constraint), reducing the effective degrees of freedom from $k$ to $k-1$.

### Rate of Convergence

The chi-square approximation improves with:

- Larger total sample size $n$.
- More uniform expected cell counts.
- Fewer categories with very small expected frequencies.

As a practical guideline, the approximation is generally reliable when all expected frequencies are at least 5.

## Exercises

**Exercise 1.**
A contingency table has 4 rows and 3 columns. How many degrees of freedom does the chi-square test of independence have? Explain the formula.

??? success "Solution to Exercise 1"
    The degrees of freedom are:

    $$
    df = (r-1)(c-1) = (4-1)(3-1) = 3 \times 2 = 6
    $$

    The formula arises because the expected frequencies are computed from the marginal totals. With $r$ rows and $c$ columns, there are $r + c$ marginal totals, but they are subject to the constraint that row and column marginals both sum to $n$, giving $r + c - 1$ constraints. The number of free parameters is $(r \times c) - 1 - (r + c - 1) = rc - r - c + 1 = (r-1)(c-1)$.

---

**Exercise 2.**
A goodness-of-fit test for a normal distribution uses 8 bins. Two parameters ($\mu$ and $\sigma$) are estimated from the data. What are the degrees of freedom?

??? success "Solution to Exercise 2"
    For a goodness-of-fit test with $k$ bins and $p$ estimated parameters:

    $$
    df = k - 1 - p = 8 - 1 - 2 = 5
    $$

    Each estimated parameter reduces the degrees of freedom by 1 because it introduces an additional constraint: the expected frequencies are computed using the estimated (not hypothesized) parameter values, which reduces the discrepancy between observed and expected.

---

**Exercise 3.**
A chi-square test has $n = 50$ observations in 10 categories with equal expected frequencies. Check whether the rule-of-thumb condition (all expected frequencies at least 5) is satisfied.

??? success "Solution to Exercise 3"
    With $n = 50$ and $k = 10$ equal categories, each expected frequency is:

    $$
    E_i = \frac{n}{k} = \frac{50}{10} = 5
    $$

    The condition $E_i \geq 5$ is satisfied (exactly at the boundary). The chi-square approximation should be adequate, though some statisticians recommend $E_i \geq 5$ as a minimum and prefer $E_i \geq 10$ for greater reliability.

---

**Exercise 4.**
If one of the 10 categories in Exercise 3 had only 2 expected observations, what remedial action could be taken?

??? success "Solution to Exercise 4"
    When expected frequencies are too small, common remedies include:

    1. **Combine adjacent categories**: Merge the low-frequency category with a neighboring category to create a combined category with a larger expected count. This reduces $k$ and increases the expected frequency in the combined cell.

    2. **Use an exact test**: Fisher's exact test or a permutation test does not rely on the chi-square approximation and can handle small expected frequencies.

    3. **Collect more data**: If possible, increasing $n$ proportionally increases all expected frequencies.

    The first option (combining categories) is most common in practice. The combined categories should be scientifically meaningful — adjacent or similar categories should be merged rather than arbitrary ones.
