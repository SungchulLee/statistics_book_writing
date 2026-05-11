# False Discovery Rate and Resampling Multiple Testing

## Overview

When testing many hypotheses simultaneously, the probability of at least one false positive grows rapidly. This page demonstrates three strategies for controlling errors across multiple tests: Bonferroni and Holm corrections for family-wise error rate (FWER) control, the Benjamini-Hochberg (BH) procedure for false discovery rate (FDR) control, and a resampling (permutation) approach that estimates FDR without distributional assumptions.

## Family-Wise Error Rate Growth

If we conduct $m$ independent tests, each at level $\alpha$, the probability of making at least one Type I error is

$$
\text{FWER} = 1 - (1 - \alpha)^m.
$$

For $\alpha = 0.05$ and $m = 100$, this exceeds 0.99 -- virtually guaranteeing a false positive.

## Correction Methods

### Bonferroni Correction

Reject the $i$-th hypothesis only if $p_i < \alpha / m$. This controls FWER at level $\alpha$ but is conservative: power drops as $m$ grows.

### Holm Step-Down Procedure

Sort the $p$-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$. Reject $H_{(k)}$ if

$$
p_{(j)} < \frac{\alpha}{m - j + 1} \quad \text{for all } j = 1, \ldots, k.
$$

Holm controls FWER like Bonferroni but is uniformly more powerful.

### Benjamini-Hochberg Procedure

FDR is defined as the expected proportion of false discoveries among all rejections:

$$
\text{FDR} = E\!\left[\frac{V}{R \vee 1}\right],
$$

where $V$ is the number of false positives and $R$ is the total number of rejections. The BH procedure finds the largest $k$ such that

$$
p_{(k)} \leq \frac{k}{m} \cdot \alpha,
$$

then rejects all $H_{(1)}, \ldots, H_{(k)}$. Under independence, this controls FDR at level $\alpha$.

## Code

### FWER Growth Curve

```python
import numpy as np

m_vals = np.arange(1, 501)
alphas = [0.05, 0.01, 0.001]
for a in alphas:
    fwer = 1 - (1 - a) ** m_vals
    print(f"alpha={a}, m=100: FWER={1 - (1-a)**100:.4f}")
```

### Simulated Multiple Tests with Corrections

```python
from scipy import stats
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

n_tests = 2000
n_true_alt = 200
n_obs = 50
effect_size = 0.5

p_values = np.zeros(n_tests)
truth = np.zeros(n_tests, dtype=int)
truth[:n_true_alt] = 1

for i in range(n_tests):
    mu = effect_size if i < n_true_alt else 0.0
    data = np.random.normal(mu, 1.0, n_obs)
    _, p_values[i] = stats.ttest_1samp(data, 0)

# Apply corrections
_, p_bonf, _, _ = multipletests(p_values, method="bonferroni")
_, p_holm, _, _ = multipletests(p_values, method="holm")
_, p_bh, _, _   = multipletests(p_values, method="fdr_bh")

alpha = 0.05
for name, adj_p in [("Bonferroni", p_bonf), ("Holm", p_holm), ("BH", p_bh)]:
    rejected = adj_p < alpha
    tp = np.sum(rejected & (truth == 1))
    fp = np.sum(rejected & (truth == 0))
    fdr = fp / max(np.sum(rejected), 1)
    power = tp / np.sum(truth == 1)
    print(f"{name:12s}: TP={tp}, FP={fp}, FDR={fdr:.3f}, Power={power:.3f}")
```

### Resampling-Based FDR Estimation

```python
def resampling_fdr(X_group1, X_group2, n_permutations=500):
    n1, n2 = X_group1.shape[0], X_group2.shape[0]
    n_features = X_group1.shape[1]
    X_combined = np.vstack([X_group1, X_group2])

    # Observed test statistics
    t_obs = np.array([
        stats.ttest_ind(X_group1[:, j], X_group2[:, j]).statistic
        for j in range(n_features)
    ])

    # Permutation null distribution
    t_perm = np.zeros((n_permutations, n_features))
    for b in range(n_permutations):
        idx = np.random.permutation(n1 + n2)
        for j in range(n_features):
            t_perm[b, j] = stats.ttest_ind(
                X_combined[idx[:n1], j],
                X_combined[idx[n1:], j]
            ).statistic

    # Estimate FDR at each threshold
    Rs, FDRs = [], []
    for thresh in np.sort(np.abs(t_obs)):
        R = np.sum(np.abs(t_obs) >= thresh)
        V = np.sum(np.abs(t_perm) >= thresh) / n_permutations
        Rs.append(R)
        FDRs.append(V / max(R, 1))
    return np.array(Rs), np.array(FDRs)
```

The algorithm counts how many permuted test statistics exceed each threshold and divides by the number of observed rejections, yielding an FDR estimate at every possible cutoff.

## Interpretation

- **Uncorrected testing** at $\alpha = 0.05$ on 2,000 hypotheses produces many false positives because approximately $0.05 \times 1800 = 90$ null hypotheses are expected to be falsely rejected.
- **Bonferroni** eliminates nearly all false positives but sacrifices power -- many true effects go undetected.
- **Holm** matches Bonferroni's FWER control with slightly better power.
- **BH (FDR)** accepts a small, controlled proportion of false discoveries in exchange for substantially higher power, making it the preferred method in high-dimensional settings such as genomics.
- **Resampling FDR** avoids distributional assumptions altogether and produces a smooth curve of estimated FDR versus number of rejections. It is especially useful when the null distribution of the test statistic is unknown or non-standard.

## Exercises

**Exercise 1.** Prove that the family-wise error rate satisfies $\text{FWER} = 1 - (1 - \alpha)^m$ when the $m$ tests are independent. What happens when the tests are positively correlated?

??? success "Solution to Exercise 1"

    Under independence, the probability that all $m$ tests correctly fail to reject is $(1-\alpha)^m$. By the complement rule,

    $$
    P(\text{at least one rejection}) = 1 - (1 - \alpha)^m.
    $$

    When the tests are positively correlated, the joint probability of no rejections is larger than $(1-\alpha)^m$, so the actual FWER is smaller than the independence formula predicts. Independence gives an upper bound on FWER in this case. $\square$

---

**Exercise 2.** In the simulation with 2,000 tests and 200 true alternatives, compute the expected number of false positives under uncorrected testing and under Bonferroni. Verify your answers against the simulation output.

??? success "Solution to Exercise 2"

    Under uncorrected testing at $\alpha = 0.05$, the expected number of false positives from the 1,800 null hypotheses is

    $$
    E[V] = 1800 \times 0.05 = 90.
    $$

    Under Bonferroni, each test is compared against $\alpha/m = 0.05/2000 = 0.000025$. For a null hypothesis (mean 0, $n=50$), the probability of a one-sample $t$-statistic exceeding the Bonferroni threshold is extremely small, so $E[V] \approx 1800 \times 0.000025 = 0.045$, i.e., nearly zero false positives on average. The simulation results should closely match these expectations. $\square$

---

**Exercise 3.** Explain the BH procedure step by step. Why does sorting p-values and comparing $p_{(k)}$ to $k\alpha/m$ control FDR at level $\alpha$?

??? success "Solution to Exercise 3"

    1. Sort all $m$ p-values in ascending order: $p_{(1)} \leq \cdots \leq p_{(m)}$.
    2. Find the largest index $k$ such that $p_{(k)} \leq k\alpha/m$.
    3. Reject all hypotheses $H_{(1)}, \ldots, H_{(k)}$.

    The intuition is that under the null, p-values are uniformly distributed, so the $k$-th smallest p-value has expected value $k/(m+1)$. The threshold $k\alpha/m$ is a fraction $\alpha$ of this expected spacing. Benjamini and Hochberg (1995) proved that under independence of the test statistics, this procedure guarantees

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right] \leq \frac{m_0}{m}\alpha \leq \alpha,
    $$

    where $m_0$ is the number of true null hypotheses. $\square$

---

**Exercise 4.** In the resampling FDR procedure, why do we divide the number of permutation-based rejections by the number of permutations? What would happen if we used too few permutations?

??? success "Solution to Exercise 4"

    Each permutation provides one realization of the test statistics under $H_0$. The average number of permuted statistics exceeding the threshold estimates $E[V]$, the expected number of false discoveries. Dividing by `n_permutations` converts a count into this average:

    $$
    \hat{V}(c) = \frac{1}{B}\sum_{b=1}^{B} \sum_{j=1}^{m} \mathbf{1}(|T_j^{(b)}| \geq c).
    $$

    With too few permutations, $\hat{V}$ is noisy and the FDR estimate becomes unreliable. In particular, for strict thresholds where $V$ is small, a small $B$ can produce $\hat{V} = 0$ even when the true expected false discoveries are positive, leading to an underestimate of FDR. A practical minimum is $B \geq 200$. $\square$

---

**Exercise 5.** Modify the simulation to compare BH at $\alpha = 0.05$ and $\alpha = 0.10$. How do the number of rejections, FDR, and power change? Explain the trade-off.

??? success "Solution to Exercise 5"

    ```python
    for alpha in [0.05, 0.10]:
        _, p_bh, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        rejected = p_bh < alpha
        tp = np.sum(rejected & (truth == 1))
        fp = np.sum(rejected & (truth == 0))
        fdr = fp / max(np.sum(rejected), 1)
        power = tp / np.sum(truth == 1)
        print(f"alpha={alpha}: Rejections={np.sum(rejected)}, "
              f"FDR={fdr:.3f}, Power={power:.3f}")
    ```

    Increasing $\alpha$ from 0.05 to 0.10 raises the BH threshold, so more hypotheses are rejected. Power increases (more true effects detected), but FDR also rises (a larger fraction of discoveries are false). The trade-off is between discovery rate and reliability: a higher $\alpha$ finds more effects but at the cost of more false leads. $\square$
