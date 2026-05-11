# McNemar's Test (Paired Binary)

## Overview

**McNemar's test** is used to analyze paired binary data, typically arising from before-and-after studies where the same subjects are measured under two conditions. The test focuses on the **discordant pairs** -- subjects who changed outcome between conditions -- and determines whether the change is symmetric or whether one direction of change is significantly more common than the other.

## Study Design

McNemar's test applies when:

- Each subject is observed under **two conditions** (e.g., before/after treatment, two diagnostic tests).
- The outcome at each condition is **binary** (e.g., positive/negative, success/failure).
- The observations are **paired** (the same subject provides both measurements).

The data is organized in a $2 \times 2$ table of paired counts:

$$
\begin{array}{c|cc}
 & \text{After +} & \text{After -} \\
\hline
\text{Before +} & a & b \\
\text{Before -} & c & d
\end{array}
$$

- $a$: positive at both times (concordant)
- $d$: negative at both times (concordant)
- $b$: positive before, negative after (discordant)
- $c$: negative before, positive after (discordant)

## Hypotheses

- **Null Hypothesis** ($H_0$): The probability of changing in one direction equals the probability of changing in the other direction, i.e., $P(b) = P(c)$.
- **Alternative Hypothesis** ($H_A$): The probabilities of change differ, i.e., $P(b) \ne P(c)$.

## Test Statistic

The McNemar statistic with **continuity correction** is

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$

The version **without continuity correction** is

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

Under $H_0$, this statistic follows a $\chi^2(1)$ distribution approximately, provided $b + c$ is sufficiently large (typically $b + c \ge 25$).

## Code

### Custom Implementation

```python
import numpy as np
from scipy import stats

def mcnemar_test(table):
    """
    Perform McNemar's test on a 2x2 table of paired counts.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Contingency table where off-diagonal cells (b, c)
        represent discordant pairs:
            [[a, b],
             [c, d]]

    Returns
    -------
    statistic : float   McNemar chi-square statistic (continuity-corrected)
    p_value   : float   Two-sided p-value from chi-square(1)
    """
    table = np.asarray(table)
    b = table[0, 1]
    c = table[1, 0]
    # Continuity-corrected McNemar statistic
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value
```

### Running the Test

```python
# Disease status before/after treatment
#                  After+   After-
# Before+           101      121
# Before-            59       33
table = np.array([[101, 121],
                  [ 59,  33]])

chi2, p = mcnemar_test(table)

print(f"McNemar chi2 = {chi2:.4f}")
print(f"p-value      = {p:.4e}")

if p < 0.05:
    print("Reject H0: significant change after treatment (alpha = 0.05).")
else:
    print("Fail to reject H0: no significant change (alpha = 0.05).")
```

**Key values for this example:**

- Discordant pair counts: $b = 121$, $c = 59$.
- The statistic tests whether the 121 subjects who went from positive to negative differ significantly from the 59 who went from negative to positive.

## Interpretation

The discordant pairs are the only cells that matter for McNemar's test. Concordant pairs ($a$ and $d$) provide no information about differential change.

For the disease example:

- $b = 121$ patients improved (positive to negative).
- $c = 59$ patients worsened (negative to positive).
- The asymmetry is substantial: more patients improved than worsened.
- If the p-value is less than $0.05$, we conclude that the treatment produced a statistically significant change in disease status.

## Exercises

**1.** A diagnostic study compares two tests on 200 patients. The paired results are:

$$
\begin{pmatrix} 80 & 15 \\ 25 & 80 \end{pmatrix}
$$

Compute the McNemar statistic (with continuity correction) and determine whether the two tests differ significantly at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    The discordant counts are $b = 15$ and $c = 25$.

    $$
    \chi^2 = \frac{(|15 - 25| - 1)^2}{15 + 25} = \frac{(10 - 1)^2}{40} = \frac{81}{40} = 2.025
    $$

    With $\text{df} = 1$, the critical value at $\alpha = 0.05$ is $3.841$. Since $2.025 < 3.841$, we **fail to reject** $H_0$. There is no significant difference between the two diagnostic tests. $\square$

---

**2.** Explain why the concordant pairs ($a$ and $d$) do not contribute to the McNemar test statistic.

??? success "Solution to Exercise 2"

    Concordant pairs are subjects whose outcome did not change between conditions (positive-positive or negative-negative). These subjects provide no evidence about whether the conditions differ because their outcome is the same regardless. The question of interest is whether the *changes* are symmetric: among subjects who *did* change, did they change equally in both directions? Only the discordant pairs ($b$ and $c$) carry information about this asymmetry. Including concordant pairs would dilute the test with irrelevant information and reduce power. $\square$

---

**3.** Compute the McNemar statistic both with and without continuity correction for $b = 30$, $c = 10$. How much do the two versions differ?

??? success "Solution to Exercise 3"

    Without continuity correction:

    $$
    \chi^2 = \frac{(30 - 10)^2}{30 + 10} = \frac{400}{40} = 10.0
    $$

    With continuity correction:

    $$
    \chi^2 = \frac{(|30 - 10| - 1)^2}{30 + 10} = \frac{(19)^2}{40} = \frac{361}{40} = 9.025
    $$

    The corrected version is smaller by $0.975$. Both yield highly significant p-values (well below $0.001$ for $\chi^2(1)$), so in this case the correction does not change the conclusion. The correction matters more when $b + c$ is small and the statistic is near the critical value. $\square$

---

**4.** When $b + c$ is small (say, less than 25), the chi-square approximation for McNemar's test may be poor. Describe an exact alternative based on the binomial distribution.

??? success "Solution to Exercise 4"

    Under $H_0$, each discordant pair is equally likely to be of type $b$ or type $c$, so $b \sim \text{Binomial}(b + c, 0.5)$. The exact p-value for a two-sided test is

    $$
    p = 2 \cdot \min\bigl[P(X \le b),\; P(X \ge b)\bigr]
    $$

    where $X \sim \text{Binomial}(b + c, 0.5)$. In Python:

    ```python
    from scipy import stats
    n_discordant = b + c
    p_exact = stats.binom_test(b, n_discordant, 0.5)
    # or equivalently:
    p_exact = stats.binomtest(b, n_discordant, 0.5).pvalue
    ```

    This exact binomial test makes no distributional approximation and is valid for any sample size. It is the recommended approach when the number of discordant pairs is small. $\square$

---

**5.** Prove that under $H_0$ ($P(b) = P(c) = 0.5$ among discordant pairs), the uncorrected McNemar statistic converges in distribution to $\chi^2(1)$ as $b + c \to \infty$.

??? success "Solution to Exercise 5"

    Let $n = b + c$ be the total number of discordant pairs. Under $H_0$, $b \sim \text{Binomial}(n, 0.5)$, so $E[b] = n/2$ and $\text{Var}(b) = n/4$.

    The uncorrected McNemar statistic can be written as

    $$
    \chi^2 = \frac{(b - c)^2}{b + c} = \frac{(b - (n - b))^2}{n} = \frac{(2b - n)^2}{n}
    $$

    Now define $Z = (b - n/2) / \sqrt{n/4} = (2b - n) / \sqrt{n}$. By the Central Limit Theorem, $Z \xrightarrow{d} N(0, 1)$ as $n \to \infty$.

    The statistic is

    $$
    \chi^2 = \frac{(2b - n)^2}{n} = \left(\frac{2b - n}{\sqrt{n}}\right)^2 = Z^2
    $$

    Since $Z^2 \xrightarrow{d} \chi^2(1)$ when $Z \xrightarrow{d} N(0,1)$, the McNemar statistic converges in distribution to $\chi^2(1)$. $\square$
