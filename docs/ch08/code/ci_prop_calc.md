# One-Sample Proportion Confidence Interval Computation

## Overview

This page presents the practical computation of a confidence interval for a single population proportion $p$. Four methods are covered: the Wald interval, the Wilson score interval, the Agresti--Coull interval, and the Clopper--Pearson exact interval. The Python implementation accepts either success/failure counts or a CSV file of 0/1 values and supports any confidence level.

## Wald Interval

Given $k$ successes in $n$ independent Bernoulli trials, the sample proportion is $\hat{p} = k/n$. The Wald $(1-\alpha)100\%$ confidence interval is

$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

This is the simplest method but can under-cover badly when $n$ is small or $\hat{p}$ is near 0 or 1, because the normal approximation to the binomial breaks down in those regimes.

## Wilson Score Interval

The Wilson interval inverts the score test. Define $z = z_{\alpha/2}$. The interval endpoints are

$$
\frac{\hat{p} + \dfrac{z^2}{2n}}{1 + \dfrac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \dfrac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

The denominator $1 + z^2/n$ shrinks the interval toward $1/2$, improving coverage across the full range of $p$. The Wilson interval is the recommended default for most practical work.

## Agresti--Coull Interval

Add $z^2/2$ pseudo-successes and $z^2/2$ pseudo-failures to form adjusted quantities:

$$
\tilde{n} = n + z^2, \quad \tilde{p} = \frac{k + z^2/2}{\tilde{n}}
$$

Then apply the Wald formula using $\tilde{p}$ and $\tilde{n}$:

$$
\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}}
$$

At the 95% level this amounts to adding roughly 2 successes and 2 failures (the "plus-four" rule). Coverage is very close to Wilson with an even simpler computation.

## Clopper--Pearson (Exact) Interval

The Clopper--Pearson interval uses Beta quantiles to invert two one-sided binomial tests:

$$
\left(\text{Beta}\!\left(\frac{\alpha}{2};\; k,\; n-k+1\right),\;\;
      \text{Beta}\!\left(1-\frac{\alpha}{2};\; k+1,\; n-k\right)\right)
$$

with the convention that the lower bound is 0 when $k = 0$ and the upper bound is 1 when $k = n$. This interval guarantees at least $(1-\alpha)100\%$ coverage for every $p$, but is conservative (wider than necessary).

## Python Code

### Loading Data from a CSV

```python
import csv
import numpy as np

def load_data(csv_path):
    """Read 0/1 values from a CSV file for proportion estimation."""
    arr = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    v = float(item)
                    if v not in (0, 1):
                        raise ValueError("CSV must contain only 0/1 values.")
                    arr.append(v)
    if len(arr) == 0:
        raise ValueError("No values found in CSV.")
    return np.array(arr, dtype=float)
```

### Computing the Confidence Interval

```python
import math
from scipy.stats import norm, beta

def ci_proportion(k, n, method="wilson", cl=0.95):
    """
    Compute a one-sample CI for a population proportion.

    Parameters
    ----------
    k : int        - number of successes
    n : int        - sample size
    method : str   - 'wald', 'wilson', 'ac', or 'cp'
    cl : float     - confidence level (default 0.95)
    """
    alpha = 1 - cl
    z = norm.ppf(1 - alpha / 2)
    phat = k / n

    if method == "wald":
        se = math.sqrt(phat * (1 - phat) / n)
        lo = phat - z * se
        hi = phat + z * se
    elif method == "wilson":
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        lo, hi = center - half, center + half
    elif method == "ac":
        n_tilde = n + z * z
        p_tilde = (k + 0.5 * z * z) / n_tilde
        se_tilde = math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lo = p_tilde - z * se_tilde
        hi = p_tilde + z * se_tilde
    else:  # cp (Clopper-Pearson)
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)

    lo = max(0.0, lo)
    hi = min(1.0, hi)
    return lo, hi

# Example: Wilson interval for 12 successes in 50 trials
lo, hi = ci_proportion(k=12, n=50, method="wilson", cl=0.95)
print(f"95% Wilson CI: ({lo:.4f}, {hi:.4f})")

# Example: Clopper-Pearson interval at 99% confidence
lo, hi = ci_proportion(k=12, n=50, method="cp", cl=0.99)
print(f"99% Clopper-Pearson CI: ({lo:.4f}, {hi:.4f})")
```

### Command-Line Usage

The companion script `ci_prop_calc.py` supports command-line arguments:

```bash
# Wilson interval from counts
python ci_prop_calc.py --k 12 --n 50 --method wilson

# Clopper-Pearson interval from a CSV of 0/1 values
python ci_prop_calc.py --csv bernoulli.csv --method cp

# 99% confidence level
python ci_prop_calc.py --k 12 --n 50 --method wilson --cl 0.99
```

## Interpretation

- The **Wald interval** is easy to compute but can produce nonsensical results (negative endpoints or endpoints above 1) when $\hat{p}$ is near 0 or 1. The endpoints are clipped to $[0, 1]$, but coverage still suffers in these cases.
- The **Wilson score interval** adjusts the center toward $1/2$ and is recommended for general use. It maintains good coverage even for moderate sample sizes and extreme proportions.
- The **Agresti--Coull interval** achieves coverage comparable to Wilson through a simpler device: inflating the sample size by $z^2$ and recentering $\hat{p}$. It is a practical choice when ease of hand calculation matters.
- The **Clopper--Pearson interval** is the only method that guarantees at least $(1-\alpha)100\%$ coverage for all $p$, but it pays for this guarantee with excess width. For large $n$, the conservatism is mild; for small $n$, it can be substantial.

## Exercises

**Exercise 1.** In a quality-control sample, 8 out of 200 items are defective. Compute the 95% Wald and Wilson confidence intervals for the defect rate $p$. Comment on any differences.

??? success "Solution to Exercise 1"

    Here $k = 8$, $n = 200$, $\hat{p} = 0.04$, and $z = 1.96$.

    **Wald:** $\text{SE} = \sqrt{0.04 \times 0.96 / 200} = \sqrt{0.000192} = 0.01386$. CI: $0.04 \pm 1.96 \times 0.01386 = 0.04 \pm 0.02716 = (0.0128, 0.0672)$.

    **Wilson:** Denominator: $1 + 1.96^2/200 = 1 + 0.01921 = 1.01921$. Center: $(0.04 + 3.8416/400)/1.01921 = 0.04960/1.01921 = 0.04868$. Half-width: $1.96 \times \sqrt{0.04 \times 0.96/200 + 3.8416/160000}/1.01921 = 1.96 \times \sqrt{0.000192 + 0.000024}/1.01921 = 1.96 \times 0.01470/1.01921 = 0.02826$. CI: $(0.0204, 0.0770)$.

    The Wilson interval is shifted to the right (center 0.049 vs. 0.040) and slightly wider. With $n\hat{p} = 8 < 10$, the normal approximation behind the Wald interval is borderline, and the Wilson interval provides more reliable coverage. $\square$

---

**Exercise 2.** Show that the Wald interval has zero width when $\hat{p} = 0$ or $\hat{p} = 1$, and explain why this is problematic.

??? success "Solution to Exercise 2"

    When $\hat{p} = 0$ (i.e., $k = 0$), the standard error is

    $$
    \text{SE} = \sqrt{\frac{0 \cdot 1}{n}} = 0
    $$

    so the Wald interval collapses to the single point $[0, 0]$. Similarly, when $\hat{p} = 1$, the interval is $[1, 1]$.

    This is problematic because observing $k = 0$ successes in $n$ trials does not mean $p = 0$ with certainty. For example, if $p = 0.01$ and $n = 50$, the probability of observing $k = 0$ is $(1 - 0.01)^{50} \approx 0.605$, which is far from negligible. A degenerate interval at zero fails to capture any positive value of $p$, so the coverage probability drops far below the nominal level. Both the Wilson and Clopper--Pearson methods avoid this degeneracy by producing intervals of positive width even when $k = 0$ or $k = n$. $\square$

---

**Exercise 3.** A survey finds that 540 out of 1000 respondents support a policy. Compute the 95% confidence intervals using all four methods (Wald, Wilson, Agresti--Coull, Clopper--Pearson) and compare the widths.

??? success "Solution to Exercise 3"

    Here $k = 540$, $n = 1000$, $\hat{p} = 0.54$, and $z = 1.96$.

    **Wald:** $\text{SE} = \sqrt{0.54 \times 0.46/1000} = \sqrt{0.000248} = 0.01576$. CI: $0.54 \pm 0.03089 = (0.5091, 0.5709)$. Width = 0.0618.

    **Wilson:** Denom $= 1 + 3.8416/1000 = 1.003842$. Center $= (0.54 + 0.001921)/1.003842 = 0.5398$. Half $= 1.96 \times \sqrt{0.000248 + 0.00000096}/1.003842 = 1.96 \times 0.01578/1.003842 = 0.03082$. CI: $(0.5090, 0.5706)$. Width = 0.0616.

    **Agresti--Coull:** $\tilde{n} = 1003.84$, $\tilde{p} = (540 + 1.9208)/1003.84 = 0.5398$. $\text{SE}_{\tilde{}} = \sqrt{0.5398 \times 0.4602/1003.84} = 0.01574$. CI: $0.5398 \pm 0.03085 = (0.5090, 0.5707)$. Width = 0.0617.

    **Clopper--Pearson:** $L = \text{Beta}(0.025;\,540,\,461) = 0.5087$. $U = \text{Beta}(0.975;\,541,\,460) = 0.5712$. Width = 0.0625.

    With $n = 1000$ and $\hat{p}$ near $0.5$, all four methods produce nearly identical intervals. The Clopper--Pearson interval is marginally wider (0.0625 vs. approximately 0.0617 for the others). For large $n$ with $p$ away from the boundaries, the choice of method matters very little. $\square$

---

**Exercise 4.** A clinical trial observes 0 serious adverse events in 30 patients. Compute the one-sided 95% upper bound for $p$ using the Clopper--Pearson method, and state the "rule of three" approximation.

??? success "Solution to Exercise 4"

    With $k = 0$ and $n = 30$, the Clopper--Pearson two-sided 95% interval has lower bound 0. For the upper bound:

    $$
    U = 1 - (\alpha/2)^{1/n}
    $$

    but more precisely, we use $U = \text{Beta}(0.975;\, 1,\, 30) = 1 - 0.025^{1/30}$. Computing: $\ln(0.025)/30 = -3.6889/30 = -0.12296$, so $U = 1 - e^{-0.12296} = 1 - 0.8843 = 0.1157$.

    For a one-sided 95% upper bound (setting $\alpha = 0.05$ for the upper tail alone):

    $$
    U = 1 - (0.05)^{1/30} = 1 - e^{\ln(0.05)/30} = 1 - e^{-0.0999} \approx 0.0951
    $$

    The **rule of three** provides a quick approximation: when $k = 0$, the 95% one-sided upper bound is approximately $3/n$. Here $3/30 = 0.10$, which is close to the exact value of 0.0951. The rule of three follows from the approximation $1 - \alpha^{1/n} \approx -\ln(\alpha)/n$ and the fact that $-\ln(0.05) \approx 3$. $\square$

---

**Exercise 5.** Verify numerically that the Wilson and Agresti--Coull intervals are nearly identical for $k = 7$, $n = 25$ at the 95% level, and explain algebraically why they are close but not exactly equal.

??? success "Solution to Exercise 5"

    With $k = 7$, $n = 25$, $\hat{p} = 0.28$, and $z = 1.96$:

    **Wilson:** Denom $= 1 + 3.8416/25 = 1.15366$. Center $= (0.28 + 0.07683)/1.15366 = 0.3094$. Half $= 1.96 \times \sqrt{0.28 \times 0.72/25 + 3.8416/2500}/1.15366 = 1.96 \times \sqrt{0.008064 + 0.001537}/1.15366 = 1.96 \times 0.09798/1.15366 = 0.1664$. CI: $(0.1430, 0.4758)$.

    **Agresti--Coull:** $\tilde{n} = 25 + 3.8416 = 28.8416$. $\tilde{p} = (7 + 1.9208)/28.8416 = 0.3093$. $\text{SE}_{\tilde{}} = \sqrt{0.3093 \times 0.6907/28.8416} = \sqrt{0.007404} = 0.08605$. CI: $0.3093 \pm 1.96 \times 0.08605 = 0.3093 \pm 0.1687 = (0.1407, 0.4780)$.

    The centers are nearly identical (0.3094 vs. 0.3093), and the half-widths differ by only 0.002.

    **Algebraic explanation.** Both methods adjust the center to $\tilde{p} = (k + z^2/2)/(n + z^2)$. The Wilson interval uses the exact standard error $\sqrt{\hat{p}(1-\hat{p})/n + z^2/(4n^2)}$ divided by $1 + z^2/n$, while Agresti--Coull uses $\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$. These differ because the Wilson half-width involves $\hat{p}$ (the unadjusted proportion) under the radical, whereas Agresti--Coull uses $\tilde{p}$. For moderate $n$, $\hat{p}$ and $\tilde{p}$ are close, so the two intervals nearly coincide. They diverge more noticeably when $n$ is very small or $\hat{p}$ is extreme. $\square$
