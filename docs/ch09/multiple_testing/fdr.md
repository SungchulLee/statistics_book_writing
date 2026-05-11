# False Discovery Rate (Benjamini-Hochberg)

## From FWER to FDR

The [family-wise error rate](fwer.md) controls the probability of making even a single false rejection. While appropriate when each false positive carries serious consequences, FWER control becomes overly conservative when the number of tests is large. In a genome-wide association study with $m = 500{,}000$ tests, the Bonferroni threshold $\alpha / m$ is so stringent that many real effects go undetected. The **false discovery rate (FDR)**, introduced by Benjamini and Hochberg in 1995, offers a less conservative alternative: instead of preventing any false positives at all, it controls the expected **proportion** of false positives among the rejected hypotheses.

## Notation and Setup

Suppose we test $m$ null hypotheses $H_1, \ldots, H_m$ simultaneously. Of these, $m_0$ are truly null and $m - m_0$ are truly alternative. After testing, we can classify the outcomes as follows:

| | Not rejected | Rejected | Total |
|---|---|---|---|
| True null | $U$ | $V$ | $m_0$ |
| False null | $T$ | $S$ | $m - m_0$ |
| Total | $m - R$ | $R$ | $m$ |

Here $R$ is the total number of rejections, $V$ is the number of false discoveries (true nulls that were rejected), and $S$ is the number of true discoveries. The quantities $V$, $S$, $U$, $T$, and $R$ are random variables -- they depend on the data and the decision rule.

## Definition of FDR

The **false discovery proportion (FDP)** is the fraction of rejections that are false:

$$
\text{FDP} = \frac{V}{R}
$$

When $R = 0$ (nothing is rejected), we define $\text{FDP} = 0$ by convention. The **false discovery rate** is the expected value of this proportion:

$$
\text{FDR} = E\!\left[\frac{V}{\max(R, 1)}\right]
$$

The $\max(R, 1)$ in the denominator avoids division by zero. Equivalently, $\text{FDR} = E[V/R \mid R > 0] \cdot P(R > 0)$.

??? note "FDR vs FWER"
    FWER = $P(V \geq 1)$ asks whether **any** false rejection occurs. FDR = $E[V / \max(R, 1)]$ asks what **fraction** of rejections are false, on average. When all null hypotheses are true ($m_0 = m$), controlling FDR at level $\alpha$ also controls FWER at level $\alpha$, because every rejection is a false discovery and $\text{FDR} = P(R > 0) = \text{FWER}$. When some nulls are false, FDR is typically smaller than FWER.

## The Benjamini-Hochberg Procedure

The Benjamini-Hochberg (BH) procedure is a **step-up** method for controlling FDR. Given $m$ p-values $p_1, p_2, \ldots, p_m$:

**Step 1.** Sort the p-values in ascending order:

$$
p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}
$$

Let $H_{(i)}$ denote the null hypothesis corresponding to $p_{(i)}$.

**Step 2.** Find the largest index $k$ such that:

$$
p_{(k)} \leq \frac{k}{m} \alpha
$$

**Step 3.** Reject all hypotheses $H_{(1)}, H_{(2)}, \ldots, H_{(k)}$.

If no such $k$ exists, reject nothing.

The threshold $k\alpha / m$ increases with $k$, forming a line from $(1, \alpha/m)$ to $(m, \alpha)$. The procedure finds the rightmost p-value that falls below this line and rejects everything to its left.

### Why the BH Procedure Works (Intuition)

The key insight is that the step-up thresholds are calibrated so that, on average, the fraction of false discoveries among all rejections does not exceed $\alpha$. When most rejections are true discoveries (small p-values from real effects), the procedure allows a generous threshold for additional discoveries. When few hypotheses are rejected, the threshold automatically becomes more stringent, mimicking Bonferroni-like protection.

## BH Theorem

!!! info "Benjamini-Hochberg Theorem (1995)"
    If the $m$ test statistics are **independent**, the BH procedure controls the FDR at level:

    $$
    \text{FDR} \leq \frac{m_0}{m} \alpha \leq \alpha
    $$

    where $m_0$ is the number of true null hypotheses.

The factor $m_0 / m \leq 1$ means the BH procedure is actually slightly conservative: the true FDR is at most $\alpha$ times the fraction of null hypotheses that are true.

Benjamini and Yekutieli (2001) later showed that the BH procedure also controls FDR under **positive regression dependence on each one from a subset** (PRDS), a condition satisfied by many common multivariate distributions including the multivariate normal with non-negative correlations.

## Example: Drug Screening

A pharmaceutical company screens $m = 10$ drug compounds for activity against a target, performing one hypothesis test per compound at $\alpha = 0.10$. The resulting p-values are:

| Compound | $p$-value |
|---|---|
| A | 0.005 |
| B | 0.011 |
| C | 0.032 |
| D | 0.048 |
| E | 0.065 |
| F | 0.100 |
| G | 0.180 |
| H | 0.350 |
| I | 0.560 |
| J | 0.920 |

The p-values are already sorted. Compute the BH threshold $k\alpha / m$ for each rank:

| Rank $k$ | $p_{(k)}$ | BH threshold $k \times 0.10 / 10$ | $p_{(k)} \leq$ threshold? |
|---|---|---|---|
| 1 | 0.005 | 0.010 | Yes |
| 2 | 0.011 | 0.020 | Yes |
| 3 | 0.032 | 0.030 | No |
| 4 | 0.048 | 0.040 | No |
| 5 | 0.065 | 0.050 | No |
| 6 | 0.100 | 0.060 | No |
| 7 | 0.180 | 0.070 | No |
| 8 | 0.350 | 0.080 | No |
| 9 | 0.560 | 0.090 | No |
| 10 | 0.920 | 0.100 | No |

The largest $k$ with $p_{(k)} \leq k\alpha / m$ is $k = 2$. The BH procedure rejects $H_{(1)}$ and $H_{(2)}$, declaring compounds A and B as active.

For comparison, the [Bonferroni correction](bonferroni_holm.md) at level $\alpha = 0.10$ uses the threshold $0.10 / 10 = 0.01$. Only compound A ($p = 0.005$) passes this threshold. The BH procedure identifies one additional discovery (compound B) while maintaining the expected false discovery proportion at or below 10%.

## Adjusted P-values

Rather than comparing each p-value to its BH threshold, it is often more convenient to compute **BH-adjusted p-values** (also called q-values). The adjusted p-value for the $i$-th ordered hypothesis is:

$$
\tilde{p}_{(i)} = \min_{j \geq i} \left\{\frac{m}{j} \, p_{(j)}\right\}
$$

processed from the largest rank downward, with the constraint that adjusted p-values are non-decreasing. A hypothesis is rejected by the BH procedure at level $\alpha$ if and only if its adjusted p-value $\tilde{p}_{(i)} \leq \alpha$.

## When to Use FDR vs FWER

| Criterion | FWER control | FDR control |
|---|---|---|
| Error guarantee | No false positives (with high probability) | Controlled proportion of false positives |
| Conservatism | High (especially for large $m$) | Moderate |
| Power | Lower | Higher |
| Best for | Confirmatory studies, small $m$ | Exploratory studies, large $m$ |
| Examples | Clinical trial endpoints, regulatory submissions | Genomics, proteomics, neuroimaging |

FDR control is the standard approach in high-dimensional screening problems where the goal is to identify a set of promising candidates for follow-up investigation, and a small proportion of false leads is acceptable.

## Python Implementation

```python
import numpy as np

def benjamini_hochberg(p_values, alpha=0.05):
    """Apply the Benjamini-Hochberg procedure.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from m hypothesis tests.
    alpha : float
        Target FDR level.

    Returns
    -------
    rejected : ndarray of bool
        True for hypotheses that are rejected.
    adjusted : ndarray of float
        BH-adjusted p-values.
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    # BH thresholds
    thresholds = np.arange(1, m + 1) / m * alpha

    # Find largest k where p_(k) <= k/m * alpha
    below = sorted_p <= thresholds
    if not below.any():
        k = 0
    else:
        k = np.max(np.where(below)[0]) + 1

    # Rejection decisions
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k]] = True

    # Adjusted p-values (processed from largest to smallest)
    adjusted_sorted = np.minimum(1, sorted_p * m / np.arange(1, m + 1))
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    return rejected, adjusted
```

## Connection to Other Topics

- For the definition of FWER and why unadjusted testing is problematic, see [Family-Wise Error Rate](fwer.md).
- For FWER-controlling procedures (Bonferroni and Holm), see [Bonferroni and Holm Corrections](bonferroni_holm.md).
- The BH procedure is implemented in `scipy.stats.false_discovery_control` (SciPy 1.11+) and `statsmodels.stats.multitest.multipletests`.

## Exercises

**Exercise 1.**
Apply the Benjamini-Hochberg (BH) procedure at FDR level $q = 0.10$ to the following 6 p-values: 0.001, 0.008, 0.039, 0.041, 0.23, 0.76. Which hypotheses are rejected?

??? success "Solution to Exercise 1"
    Sort the p-values and compute BH thresholds $q \cdot j/m$:

    | Rank $j$ | $p_{(j)}$ | Threshold $q \cdot j/m = 0.10 \cdot j/6$ | $p_{(j)} \leq$ threshold? |
    |---|---|---|---|
    | 1 | 0.001 | 0.0167 | Yes |
    | 2 | 0.008 | 0.0333 | Yes |
    | 3 | 0.039 | 0.0500 | Yes |
    | 4 | 0.041 | 0.0667 | Yes |
    | 5 | 0.230 | 0.0833 | No |
    | 6 | 0.760 | 0.1000 | No |

    The largest $j$ with $p_{(j)} \leq q \cdot j/m$ is $j = 4$. Reject the 4 hypotheses corresponding to $p_{(1)}$ through $p_{(4)}$. Under BH with $q = 0.10$, the expected false discovery proportion among these 4 rejections is at most 10%.

---

**Exercise 2.**
Define the False Discovery Rate (FDR) and explain how it differs from the Family-Wise Error Rate (FWER).

??? success "Solution to Exercise 2"
    The **FDR** is the expected proportion of false discoveries among all rejected hypotheses:

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right]
    $$

    where $V$ is the number of false rejections, $R$ is the total number of rejections, and $R \vee 1 = \max(R, 1)$ avoids division by zero.

    The **FWER** is the probability of making at least one false rejection: $\text{FWER} = P(V \geq 1)$.

    Key differences:

    - FWER is more stringent: it controls the probability of *any* false positive, while FDR allows some false positives as long as they are a small fraction of all discoveries.
    - FDR is more powerful: it rejects more hypotheses because it tolerates a controlled rate of errors among discoveries.
    - When all nulls are true ($m_0 = m$), FDR = FWER (since any rejection is false). When many alternatives are true, FDR $\ll$ FWER.
    - FDR is preferred in large-scale testing (genomics, neuroimaging) where thousands of tests are performed and some false discoveries are acceptable.

---

**Exercise 3.**
Prove that the BH procedure controls FDR at level $q$ when the test statistics are independent.

??? success "Solution to Exercise 3"
    Under independence, the BH procedure controls FDR at exactly $q \cdot m_0/m \leq q$, where $m_0$ is the number of true nulls.

    The key insight: for each true null $H_i$, the p-value $p_i$ is Uniform$(0,1)$ and independent of the others. The BH procedure rejects $H_i$ when $p_i \leq q \cdot R_i/m$ for some data-dependent rank $R_i$.

    The formal proof (Benjamini & Hochberg, 1995) proceeds by showing:

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right] = \sum_{i \in \mathcal{H}_0} E\!\left[\frac{1}{R \vee 1} \cdot \mathbf{1}(H_i \text{ rejected})\right] = \frac{m_0}{m} \cdot q \leq q
    $$

    The final equality relies on the independence of p-values corresponding to true nulls from those corresponding to false nulls. $\square$

---

**Exercise 4.**
A genomics study tests 10,000 genes. Using BH at $q = 0.05$, the procedure rejects 500 hypotheses. How many of these rejections are expected to be false discoveries? How would FWER control (Bonferroni) compare?

??? success "Solution to Exercise 4"
    Under BH at $q = 0.05$: the expected number of false discoveries is at most $q \times R = 0.05 \times 500 = 25$ genes. This means about 475 out of 500 discoveries are expected to be real.

    Under Bonferroni at $\alpha = 0.05$: the per-test threshold is $0.05/10000 = 5 \times 10^{-6}$. Only p-values below this extremely stringent threshold are rejected. In practice, Bonferroni might reject only 20-50 genes (far fewer than 500), missing many true discoveries.

    This illustrates the power advantage of FDR control: BH discovers 500 genes (accepting ~25 false) while Bonferroni discovers far fewer (accepting ~0 false). The choice depends on the cost of false discoveries versus missed discoveries.
