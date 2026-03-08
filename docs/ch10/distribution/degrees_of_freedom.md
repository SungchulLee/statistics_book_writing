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
