# Multiple Testing Corrections

## Overview

When many hypotheses are tested simultaneously, the probability of at least one false positive increases rapidly. Multiple testing corrections adjust p-values or significance thresholds to control error rates such as the **family-wise error rate** (FWER) or the **false discovery rate** (FDR). Understanding these corrections is essential in genomics, neuroimaging, and any setting where thousands of tests are performed at once.

## The Multiple Testing Problem

If $m$ independent tests are each conducted at level $\alpha$, the probability of at least one Type I error is

$$
\text{FWER} = 1 - (1 - \alpha)^m.
$$

For $m = 20$ tests at $\alpha = 0.05$:

$$
\text{FWER} = 1 - 0.95^{20} \approx 0.642.
$$

This means there is about a 64% chance of at least one false positive, even when every null hypothesis is true.

## Common Corrections

### Bonferroni Correction

Reject $H_{0,i}$ if $p_i < \alpha/m$. This controls FWER at level $\alpha$ but is conservative.

### Holm--Bonferroni Method

Order the p-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$. Reject $H_{0,(i)}$ if

$$
p_{(i)} < \frac{\alpha}{m - i + 1}
$$

for all $i = 1, \dots, k$, where $k$ is the largest index satisfying the condition. This is uniformly more powerful than Bonferroni while still controlling FWER.

### Benjamini--Hochberg (BH) Procedure

Controls the FDR at level $q$. Order p-values and find the largest $k$ such that

$$
p_{(k)} \leq \frac{k}{m}\,q.
$$

Reject all hypotheses $H_{0,(1)}, \dots, H_{0,(k)}$.

## Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

print(f"Sample size: {n}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std:  {data.std(ddof=1):.4f}")
```

### Applying Corrections with `statsmodels`

```python
from statsmodels.stats.multitest import multipletests

# Suppose we have m p-values from m independent tests
m = 50
p_values = np.random.uniform(0, 1, m)
# Inject some true signals
p_values[:5] = np.random.uniform(0, 0.005, 5)

# Bonferroni
_, p_bonf, _, _ = multipletests(p_values, method="bonferroni")

# Holm
_, p_holm, _, _ = multipletests(p_values, method="holm")

# Benjamini-Hochberg
_, p_bh, _, _ = multipletests(p_values, method="fdr_bh")

alpha = 0.05
print(f"Rejections (uncorrected): {np.sum(p_values < alpha)}")
print(f"Rejections (Bonferroni):  {np.sum(p_bonf < alpha)}")
print(f"Rejections (Holm):        {np.sum(p_holm < alpha)}")
print(f"Rejections (BH):          {np.sum(p_bh < alpha)}")
```

### Interpretation

- **Bonferroni** is the most conservative: it rejects few hypotheses to ensure FWER control.
- **Holm** is a step-down procedure that is uniformly more powerful than Bonferroni.
- **BH** controls the expected proportion of false discoveries among rejections (FDR), and it tends to be the most liberal of the three, offering greater power at the cost of allowing some false discoveries.

## Exercises

**Exercise 1.** A researcher performs $m = 100$ independent tests at $\alpha = 0.05$. What is the probability of at least one false positive if all null hypotheses are true? What Bonferroni-adjusted threshold should be used?

??? success "Solution to Exercise 1"

    The FWER is

    $$
    1 - (1-0.05)^{100} = 1 - 0.95^{100} \approx 1 - 0.00592 = 0.994.
    $$

    There is about a 99.4% chance of at least one false positive. The Bonferroni threshold is $\alpha/m = 0.05/100 = 0.0005$. $\square$

---

**Exercise 2.** Given ordered p-values $p_{(1)} = 0.001$, $p_{(2)} = 0.008$, $p_{(3)} = 0.039$, $p_{(4)} = 0.041$, $p_{(5)} = 0.23$ from $m=5$ tests, apply the BH procedure at FDR level $q = 0.05$. Which hypotheses are rejected?

??? success "Solution to Exercise 2"

    Compute the BH thresholds $k\,q/m$:

    | $k$ | $p_{(k)}$ | $k \cdot 0.05 / 5$ | $p_{(k)} \leq$ threshold? |
    |-----|-----------|---------------------|---------------------------|
    | 1   | 0.001     | 0.01                | Yes                       |
    | 2   | 0.008     | 0.02                | Yes                       |
    | 3   | 0.039     | 0.03                | No                        |
    | 4   | 0.041     | 0.04                | No                        |
    | 5   | 0.23      | 0.05                | No                        |

    The largest $k$ with $p_{(k)} \leq k\,q/m$ is $k=2$. Reject $H_{0,(1)}$ and $H_{0,(2)}$. $\square$

---

**Exercise 3.** Prove that the Bonferroni correction controls FWER at level $\alpha$. That is, show that if each test is conducted at level $\alpha/m$, then $P(\text{at least one false rejection}) \leq \alpha$.

??? success "Solution to Exercise 3"

    Let $V_i$ be the event that the $i$-th true null hypothesis is falsely rejected. By the union bound (Boole's inequality),

    $$
    P\!\left(\bigcup_{i=1}^{m_0} V_i\right) \leq \sum_{i=1}^{m_0} P(V_i) \leq \sum_{i=1}^{m_0} \frac{\alpha}{m} = \frac{m_0\,\alpha}{m} \leq \alpha,
    $$

    where $m_0 \leq m$ is the number of true null hypotheses. The inequality holds regardless of dependence between tests. $\square$

---

**Exercise 4.** Apply the Holm procedure to the p-values $p_1=0.01$, $p_2=0.04$, $p_3=0.03$, $p_4=0.005$ at $\alpha=0.05$. Which hypotheses are rejected?

??? success "Solution to Exercise 4"

    Order the p-values: $p_{(1)}=0.005$, $p_{(2)}=0.01$, $p_{(3)}=0.03$, $p_{(4)}=0.04$.

    Step 1: Compare $p_{(1)} = 0.005$ with $\alpha/(m-1+1) = 0.05/4 = 0.0125$. Since $0.005 < 0.0125$, reject $H_{0,(1)}$.

    Step 2: Compare $p_{(2)} = 0.01$ with $0.05/3 \approx 0.0167$. Since $0.01 < 0.0167$, reject $H_{0,(2)}$.

    Step 3: Compare $p_{(3)} = 0.03$ with $0.05/2 = 0.025$. Since $0.03 > 0.025$, stop.

    Reject $H_{0,(1)}$ and $H_{0,(2)}$ (corresponding to $p_4=0.005$ and $p_1=0.01$). Fail to reject $H_{0,(3)}$ and $H_{0,(4)}$. $\square$

---

**Exercise 5.** Explain intuitively why the BH procedure is more powerful than Bonferroni. Under what conditions does the BH procedure fail to control the FDR at the nominal level?

??? success "Solution to Exercise 5"

    **Why BH is more powerful:** Bonferroni uses a fixed threshold $\alpha/m$ for every test, which becomes extremely small as $m$ grows. BH uses adaptive thresholds $k\alpha/m$ that increase with rank, so p-values in the middle of the ordered list face a less stringent barrier. When many true signals exist (and their p-values cluster near zero), BH's step-up approach captures them while Bonferroni's uniform threshold misses them.

    **When BH can fail:** The BH procedure is proved to control FDR at level $q$ under independence or positive regression dependency on each one from a subset (PRDS). Under strong negative dependencies between test statistics, the actual FDR can exceed the nominal level $q$. In such cases, the Benjamini--Yekutieli (BY) correction, which replaces $q$ with $q / \sum_{i=1}^m 1/i$, should be used instead. $\square$
