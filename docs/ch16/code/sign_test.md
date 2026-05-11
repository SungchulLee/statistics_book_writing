# Sign Test

## Overview

The **sign test** is one of the simplest nonparametric tests for paired data. It assesses
whether the median difference between two related measurements is zero by examining only the
*signs* of the pairwise differences. Because it discards magnitude information entirely, it
requires very few assumptions---just that the differences are independent and come from a
continuous distribution---making it highly robust though less powerful than alternatives such
as the Wilcoxon signed-rank test.

## Hypotheses and Test Statistic

Given $n$ paired observations $(X_i, Y_i)$, define the differences
$D_i = X_i - Y_i$. Let $n_+$ be the number of positive differences and $n_-$ the number
of negative differences. Ties ($D_i = 0$) are excluded, leaving an effective sample size
$n = n_+ + n_-$.

Under $H_0{:}\; \text{median}(D) = 0$, each nonzero difference is equally likely to be
positive or negative, so $n_+$ follows a $\text{Binomial}(n, 1/2)$ distribution. The
sample proportion of positive signs is

$$
\hat{p} = \frac{n_+}{n}.
$$

For the normal approximation the standardized statistic is

$$
Z = \frac{\hat{p} - 0.5}{\sqrt{0.5 \cdot 0.5 \,/\, n}}
  = \frac{\hat{p} - 0.5}{\,1 / (2\sqrt{n}\,)}.
$$

The $p$-value depends on the alternative hypothesis:

| Alternative | $p$-value |
|---|---|
| $H_1{:}\; \text{median}(D) \neq 0$ | $2\,\mathcal{N}(-\lvert Z \rvert)$ |
| $H_1{:}\; \text{median}(D) > 0$ | $1 - \mathcal{N}(Z)$ |
| $H_1{:}\; \text{median}(D) < 0$ | $\mathcal{N}(Z)$ |

## Implementation

```python
import numpy as np
import scipy.stats as stats


def sign_test(paired_data, test_type="two-sided"):
    """
    Sign test for paired observations.

    Parameters
    ----------
    paired_data : ndarray of shape (n, 2)
        Column 0 is post-treatment, column 1 is pre-treatment.
    test_type : str
        One of "less", "two-sided", "greater".

    Returns
    -------
    z : float
        The Z test statistic.
    p_value : float
    """
    p_0, q_0 = 0.5, 0.5

    n_plus = np.sum(paired_data[:, 0] > paired_data[:, 1])
    n_minus = np.sum(paired_data[:, 0] < paired_data[:, 1])
    n = n_plus + n_minus  # ties excluded
    p_hat = n_plus / n

    z = (p_hat - p_0) / np.sqrt(p_0 * q_0 / n)

    if test_type == "less":
        p_value = stats.norm.cdf(z)
    elif test_type == "two-sided":
        p_value = 2 * stats.norm.cdf(-abs(z))
    elif test_type == "greater":
        p_value = stats.norm.sf(z)

    return z, p_value
```

### Example: Student Pre/Post Scores

Fifteen students are measured before and after a treatment program.

```python
paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

z, p_value = sign_test(paired_data, test_type="two-sided")
print(f"Z = {z:.4f}, p = {p_value:.4f}")
```

Among the 15 pairs, two are tied ($D_i = 0$) and are excluded, leaving $n = 13$. Of
those, $n_+ = 10$ and $n_- = 3$, so
$\hat{p} = 10/13 \approx 0.769$, yielding $Z \approx 1.94$ and a two-sided
$p$-value of approximately $0.052$.

## Interpretation

- The sign test converts a paired comparison into a sequence of Bernoulli trials. It is
  essentially asking: "Is the coin fair?" where heads means "post > pre."
- Because it uses only sign information, it is **distribution-free**: no normality or
  symmetry assumption is needed for the differences.
- The trade-off is lower **power**---the test ignores how large the differences are. When
  differences are roughly symmetric, the Wilcoxon signed-rank test is more powerful.
- For very small samples ($n < 20$), the exact binomial distribution should be used instead
  of the normal approximation.

## Exercises

**Exercise 1.** Eight subjects are measured before and after a diet. The weight changes
(after $-$ before) in kg are: $-3, +1, -2, 0, -4, -1, +2, -5$. Carry out the sign test at
$\alpha = 0.05$ using the exact binomial distribution.

??? success "Solution to Exercise 1"

    Drop the tie ($D = 0$), leaving $n = 7$. Among the nonzero differences we have
    $n_+ = 2$ and $n_- = 5$.

    Under $H_0$, $n_+ \sim \text{Binomial}(7, 0.5)$. The two-sided $p$-value is

    $$
    p = 2 \cdot P(n_+ \leq 2) = 2 \sum_{k=0}^{2} \binom{7}{k} (0.5)^7
      = 2 \cdot \frac{1 + 7 + 21}{128} = \frac{58}{128} \approx 0.453.
    $$

    Since $p = 0.453 > 0.05$, we fail to reject $H_0$. There is no significant evidence
    that the median weight change differs from zero. $\square$

---

**Exercise 2.** In the student data example, recompute the sign test as a one-sided test
with $H_1{:}\; \text{median}(D) > 0$ (treatment improves scores). State your conclusion at
$\alpha = 0.05$.

??? success "Solution to Exercise 2"

    From the example, $n_+ = 10$, $n = 13$, $\hat{p} = 10/13 \approx 0.769$, and
    $Z \approx 1.94$.

    For the one-sided alternative (greater), the $p$-value is

    $$
    p = 1 - \mathcal{N}(Z) = 1 - \mathcal{N}(1.94) \approx 1 - 0.9738 = 0.026.
    $$

    Since $0.026 < 0.05$, we reject $H_0$ and conclude the treatment significantly
    improves scores in the direction of positive differences. $\square$

---

**Exercise 3.** Show that the sign test statistic $Z$ can be written as

$$
Z = \frac{2\,n_+ - n}{\sqrt{n}}.
$$

??? success "Solution to Exercise 3"

    Starting from the definition:

    $$
    Z = \frac{\hat{p} - 0.5}{\sqrt{0.25/n}}
      = \frac{n_+/n - 0.5}{\,1/(2\sqrt{n}\,)}
      = \frac{(n_+ - n/2)/n}{\,1/(2\sqrt{n}\,)}
      = \frac{2\sqrt{n}\,(n_+ - n/2)}{n}
      = \frac{2\,n_+ - n}{\sqrt{n}}.
    $$

    $\square$

---

**Exercise 4.** Explain why the sign test is valid even when the distribution of the
differences is heavily skewed, and give one scenario where the Wilcoxon signed-rank test
would *not* be appropriate but the sign test would.

??? success "Solution to Exercise 4"

    The sign test depends only on $P(D_i > 0) = P(D_i < 0) = 0.5$ under $H_0$, which
    holds whenever the distribution of $D_i$ is continuous with median zero. No symmetry
    or finite-moment assumption is required.

    The Wilcoxon signed-rank test additionally requires the distribution of $|D_i|$ to be
    symmetric about the median. If the differences come from, say, an exponential-like
    distribution shifted to have median zero (highly right-skewed magnitudes), the
    symmetry assumption of the signed-rank test is violated and its null distribution
    is no longer correct. The sign test remains valid in this case. $\square$

---

**Exercise 5.** Write a Python function that performs the sign test using the exact
binomial distribution (not the normal approximation) and returns the two-sided $p$-value.
Test it on the student data.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy.stats import binom

    def sign_test_exact(paired_data):
        """Exact two-sided sign test for paired observations."""
        diffs = paired_data[:, 0] - paired_data[:, 1]
        nonzero = diffs[diffs != 0]
        n = len(nonzero)
        n_plus = (nonzero > 0).sum()

        # Two-sided p-value: 2 * min(P(X <= n_+), P(X >= n_+))
        p_left = binom.cdf(n_plus, n, 0.5)
        p_right = binom.sf(n_plus - 1, n, 0.5)  # P(X >= n_+)
        p_value = 2 * min(p_left, p_right)
        p_value = min(p_value, 1.0)  # cap at 1

        return n_plus, n, p_value

    paired_data = np.array([
        [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
        [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
        [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
    ])

    n_plus, n, p = sign_test_exact(paired_data)
    print(f"n_+ = {n_plus}, n = {n}, exact p = {p:.4f}")
    # n_+ = 10, n = 13, exact p ≈ 0.0923
    ```

    The exact two-sided $p$-value is approximately $0.092$, which is slightly larger than
    the normal-approximation result because the discrete binomial distribution is better
    captured by the exact calculation for this small sample size. $\square$
