# F Distribution

## Overview

The **F distribution** arises as the ratio of two independent chi-square random variables, each divided by their degrees of freedom. It is fundamental for comparing variances between two populations and for Analysis of Variance (ANOVA).

---

## Definition

Let $X_1^2 \sim \chi^2_{d_1}$ and $X_2^2 \sim \chi^2_{d_2}$ be independent. Then:

$$
F = \frac{X_1^2 / d_1}{X_2^2 / d_2} \sim F_{d_1, d_2}
$$

where $d_1$ is the numerator degrees of freedom and $d_2$ is the denominator degrees of freedom.

Since each $\chi^2$ is a sum of squared standard normals:

$$
X_1^2 = \sum_{i=1}^{d_1} Z_i^2, \qquad X_2^2 = \sum_{i=1}^{d_2} Z_i'^2
$$

---

## Degrees of Freedom

The F distribution depends on **two** sets of degrees of freedom, which distinguishes it from the chi-square distribution:

### Numerator (d_1) and Denominator (d_2)
- **Low $d_1$ and $d_2$:** Highly right-skewed.
- **Increasing $d_1$ and $d_2$:** Distribution becomes more symmetric.
- **Special case:** $F(1, d_2) = \frac{\chi^2(1)/1}{\chi^2(d_2)/d_2}$ relates directly to a squared $t$ variable.

### Comparison with Chi-Square

The chi-square distribution has a single degrees-of-freedom parameter controlling its shape. The F distribution's dual dependency creates a richer family of shapes due to the **ratio** of two independent variance-like quantities.

---

## Properties

- **Non-negativity:** $F \geq 0$ (ratio of non-negative quantities).
- **Asymmetry:** Positively skewed, especially for small degrees of freedom.
- **Mean:** $\frac{d_2}{d_2 - 2}$ for $d_2 > 2$.
- **Mode:** For $d_1 > 2$:

$$
\text{Mode} = \frac{d_2(d_1 - 2)}{d_1(d_2 + 2)}
$$

---

## Random Samples

### Direct Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
d1, d2 = 5, 10
data = stats.f(d1, d2).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.linspace(0, 5, 100)
ax.hist(data, bins=bins, density=True, alpha=0.7, label='F Samples')
ax.plot(bins, stats.f(d1, d2).pdf(bins), '--r', lw=3, label=f'F({d1},{d2}) PDF')
ax.legend()
plt.show()
```

### Sampling from Definition (Ratio of Chi-Squares)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
d1, d2, n = 5, 10, 10_000

f_data = (stats.chi2(d1).rvs(n) / d1) / (stats.chi2(d2).rvs(n) / d2)

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.linspace(0, 5, 100)
ax.hist(f_data, bins=bins, density=True, alpha=0.7, label='χ²-Ratio Samples')
ax.plot(bins, stats.f(d1, d2).pdf(bins), '--r', lw=3, label=f'F({d1},{d2}) PDF')
ax.legend()
plt.show()
```

---

## Why F?

The F distribution arises naturally when comparing the variances of two independent normal populations.

### Step 1: Distribution of Scaled Variances

For samples from normal populations:

$$
\frac{(n_1-1)S_1^2}{\sigma_1^2} \sim \chi^2_{n_1-1}, \qquad \frac{(n_2-1)S_2^2}{\sigma_2^2} \sim \chi^2_{n_2-1}
$$

These are independent because the two samples are independent.

### Step 2: Ratio of Chi-Squares

Dividing each by its degrees of freedom and taking the ratio:

$$
\frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n_1-1, \, n_2-1}
$$

### Step 3: Under the Null Hypothesis

If we test $H_0: \sigma_1^2 = \sigma_2^2$, the population variances cancel:

$$
\frac{S_1^2}{S_2^2} \sim F_{n_1-1, \, n_2-1}
$$

This is the **F-test for equality of two variances**.

### Why Not Something Else?

The F distribution is the unique distribution that arises from the ratio of independent chi-square variables divided by their degrees of freedom. This same logic extends to ANOVA, where F ratios measure whether between-group variability is significantly larger than within-group variability.

---

## Limitations

The F distribution result is **exact only under normality**:

- **Normality required:** The chi-square results for each $S^2$ depend on the population being normal. Without normality, the distribution of $S_1^2/S_2^2$ deviates from $F$.
- **Small samples, non-normal:** The F-test is unreliable. For skewed or heavy-tailed populations, the true Type I error rate can be much higher than the nominal level.
- **Robust alternatives:** Levene's test, Brown–Forsythe test, and Fligner–Killeen test maintain validity under broader distributional conditions and are widely preferred in practice.

| Scenario | F-Test Validity |
|:---|:---|
| Normal populations | Exact |
| Large samples, mild non-normality | Approximately valid |
| Small samples, skewed/heavy-tailed | Unreliable; use robust alternatives |

---

## PPF Example

```python
from scipy import stats

# 95th percentile of F(5, 20)
f_95 = stats.f(5, 20).ppf(0.95)
print(f"F_0.95(5, 20) = {f_95:.4f}")
```

---

## Key Takeaways

- The F distribution is the ratio of two independent chi-square variables, each divided by their degrees of freedom.
- It governs comparisons of variances and is the foundation of ANOVA.
- Both numerator and denominator degrees of freedom affect the shape of the distribution.
- The exactness of the F-test depends critically on normality; robust alternatives are often preferred in practice.

## Exercises

**Exercise 1.**
Two independent normal samples with common $\sigma^2$, $n_1 = 15$, $n_2 = 10$. Compute $P(S_1^2/S_2^2 > 1.5)$.

??? success "Solution to Exercise 1"
    Under equal variances, $F = S_1^2/S_2^2 \sim F_{14, 9}$ (numerator df = $n_1 - 1 = 14$, denominator df = $n_2 - 1 = 9$).

    Derivation: $S_i^2 \sim \sigma^2 \chi^2_{n_i - 1}/(n_i - 1)$. Ratio of independent scaled chi-squares each divided by df is exactly $F$.

    Numerical: $P(F_{14, 9} > 1.5) \approx 0.274$ — about 27%.

    The $F$ distribution is highly skewed for small df; non-trivial probability in the tails even for modest values of the ratio.

---

**Exercise 2.**
**Construction.** Derive $F = (U/d_1)/(V/d_2)$ where $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ independent. State the mean and what it tells you.

??? success "Solution to Exercise 2"
    Definition: if $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ are independent, then $F = (U/d_1)/(V/d_2) \sim F_{d_1, d_2}$.

    Mean: $\mathbb{E}[F] = \mathbb{E}[U/d_1] / \mathbb{E}[V/d_2]$ — wait, this is wrong; for ratios we cannot split the expectation. Actually $\mathbb{E}[U/d_1] = 1$, and $\mathbb{E}[1/(V/d_2)] = d_2 \cdot \mathbb{E}[1/V] = d_2 / (d_2 - 2)$ for $d_2 > 2$ (using moment of inverse chi-square).

    So $\mathbb{E}[F_{d_1, d_2}] = d_2/(d_2 - 2)$ for $d_2 > 2$, slightly greater than 1.

    **Asymmetry:** $F > 1$ on average, even under the null of equal variances. This is because $1/V$ has positive bias relative to $1/\mathbb{E}[V]$ — Jensen's inequality applied to the convex function $1/v$.

    For inference, one-tailed tests at the $\alpha$ level use $F$ critical values from tables; two-tailed tests are uncommon because of $F$'s asymmetry.

---

**Exercise 3.**
**ANOVA $F$-test.** State the F-statistic for testing equality of $k$ group means under one-way ANOVA. What are the degrees of freedom?

??? success "Solution to Exercise 3"
    One-way ANOVA: groups $i = 1, \ldots, k$ with $n_i$ observations each, total $n = \sum n_i$. Total/between/within sum of squares:

    - $\mathrm{SST} = \sum_{ij}(Y_{ij} - \bar Y_{..})^2$
    - $\mathrm{SSB} = \sum_i n_i (\bar Y_{i\cdot} - \bar Y_{..})^2$
    - $\mathrm{SSW} = \sum_{ij}(Y_{ij} - \bar Y_{i\cdot})^2$

    F-statistic:

    $$
    F = \frac{\mathrm{SSB}/(k-1)}{\mathrm{SSW}/(n-k)} \sim F_{k-1, n-k}
    $$

    Under $H_0$: equal means, both numerator and denominator estimate $\sigma^2$; under $H_1$: numerator is inflated by between-group variance, denominator unchanged. Large $F$ → reject.

    Degrees of freedom: $k - 1$ for between (one less than the number of group means), $n - k$ for within (each group contributes $n_i - 1$, summing to $n - k$).

---

**Exercise 4.**
**$F$ vs. $t$.** For $k = 2$ groups, show that $F_{1, n-2}$ has the same distribution as the square of $t_{n-2}$.

??? success "Solution to Exercise 4"
    From the definitions: $t = Z/\sqrt{V/(n-2)}$ where $Z \sim N(0,1)$ and $V \sim \chi^2_{n-2}$ independent.

    So $t^2 = Z^2/(V/(n-2))$. Now $Z^2 \sim \chi^2_1$ (square of standard normal).

    $$
    t^2 = \frac{Z^2/1}{V/(n-2)} \sim F_{1, n-2}
    $$

    by the definition of $F$. $\square$

    **Practical implication:** for $k = 2$ groups, the ANOVA $F$-test and the two-sample $t$-test are equivalent: $F_{\text{obs}} = t_{\text{obs}}^2$. Reject $H_0$ at level $\alpha$ in ANOVA iff $F > F_{1, n-2, 1-\alpha} = t_{n-2, 1-\alpha/2}^2$. Same decision; the two tests are different formulations of the same procedure.

---

**Exercise 5.**
**Reciprocal relationship.** Show that $F_{d_1, d_2}$ and $1/F_{d_2, d_1}$ have the same distribution.

??? success "Solution to Exercise 5"
    Let $F = (U/d_1)/(V/d_2)$ with $U \sim \chi^2_{d_1}$, $V \sim \chi^2_{d_2}$ independent. Then

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1} \sim F_{d_2, d_1}
    $$

    by definition (with degrees of freedom swapped).

    **Practical:** to compute $P(F_{d_1, d_2} < c)$, use $P(1/F_{d_1, d_2} > 1/c) = P(F_{d_2, d_1} > 1/c)$ — convenient when tables only list upper-tail critical values. Lower-tail of $F_{d_1, d_2}$ at $\alpha$ equals reciprocal of upper-tail of $F_{d_2, d_1}$ at $\alpha$.

---

**Exercise 6.**
**Robustness to non-normality.** The $F$-test assumes normality of the underlying populations. Discuss the impact of violations and propose alternatives.

??? success "Solution to Exercise 6"
    **Non-normality effects:**

    - **Light tails or symmetric distributions:** $F$ remains approximately valid for moderate $n$.
    - **Heavy tails:** $F$-test is non-robust — the sample variances $s_i^2$ have inflated variability, distorting the null distribution.
    - **Strong skew:** $F$ distribution under $H_0$ no longer matches the data's behavior.

    **Alternatives:**

    - **Levene's test** for equal variances: uses $|X_{ij} - \tilde X_i|$ (absolute deviations from group median) rather than squared deviations. More robust to non-normality.
    - **Bartlett's test:** more powerful for normal data but sensitive to non-normality.
    - **Brown-Forsythe test:** variant of Levene using median rather than mean — most robust against heavy tails.
    - **Permutation test:** non-parametric alternative; relabel observations and recompute $F$ to build null distribution.

    Practical recommendation: in industrial quality control or biological experiments with moderate $n$ and approximate normality, $F$-test is fine. For heavy-tailed data (financial returns, biological counts), use Levene/Brown-Forsythe.
