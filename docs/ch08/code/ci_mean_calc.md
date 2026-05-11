# One-Sample Mean Confidence Interval Computation

## Overview

This page presents the practical computation of a one-sample confidence interval for the population mean $\mu$. Two methods are covered: the $z$-interval when the population standard deviation $\sigma$ is known, and the $t$-interval when $\sigma$ is unknown and estimated by the sample standard deviation $s$. The Python implementation accepts either raw data or summary statistics.

## z-Interval (Known Variance)

When $\sigma$ is known, the $(1-\alpha)100\%$ confidence interval for $\mu$ is

$$
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

The standard error is $\text{SE} = \sigma / \sqrt{n}$, and the margin of error is $\text{MOE} = z_{\alpha/2} \cdot \text{SE}$.

## t-Interval (Unknown Variance)

When $\sigma$ is unknown, replace it with $s$ and use the $t$-distribution with $\text{df} = n - 1$:

$$
\bar{x} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}
$$

The $t$-distribution has heavier tails than the standard normal, so $t_{\alpha/2,\,n-1} > z_{\alpha/2}$ for all finite $n$, producing a wider interval that accounts for the uncertainty in estimating $\sigma$.

## Python Code

### Loading Data from a CSV

```python
import csv
import numpy as np

def load_data(csv_path):
    """Read a single column of numeric values from a CSV file."""
    arr = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    arr.append(float(item))
    if len(arr) == 0:
        raise ValueError("No numeric values found in CSV.")
    return np.array(arr, dtype=float)
```

### Computing the Confidence Interval

```python
import math
from scipy.stats import norm, t

def ci_mean(n, xbar, s=None, known_sigma=None, method="t", cl=0.95):
    """
    Compute a one-sample CI for the population mean.

    Parameters
    ----------
    n : int            - sample size
    xbar : float       - sample mean
    s : float or None  - sample standard deviation (required for t-interval)
    known_sigma : float or None - known population sigma (required for z-interval)
    method : str       - 't' or 'z'
    cl : float         - confidence level (default 0.95)
    """
    alpha = 1 - cl

    if method == "z":
        se = known_sigma / math.sqrt(n)
        z_star = norm.ppf(1 - alpha / 2)
        moe = z_star * se
    else:
        se = s / math.sqrt(n)
        df = n - 1
        t_star = t.ppf(1 - alpha / 2, df=df)
        moe = t_star * se

    return xbar - moe, xbar + moe

# Example: t-interval from summary statistics
lo, hi = ci_mean(n=25, xbar=3.2, s=1.1, method="t", cl=0.95)
print(f"95% t-interval: ({lo:.4f}, {hi:.4f})")

# Example: z-interval with known sigma
lo, hi = ci_mean(n=25, xbar=3.2, known_sigma=1.0, method="z", cl=0.95)
print(f"95% z-interval: ({lo:.4f}, {hi:.4f})")
```

### Command-Line Usage

The companion script `ci_mean_calc.py` supports command-line arguments:

```bash
# t-interval from a CSV file
python ci_mean_calc.py --csv data.csv

# z-interval from summary statistics
python ci_mean_calc.py --n 25 --mean 3.2 --sd 1.1 --known-sigma 1.0 --method z

# 99% confidence level
python ci_mean_calc.py --csv data.csv --cl 0.99
```

## Interpretation

- The **confidence level** $1 - \alpha$ describes the long-run success rate of the procedure: if we repeat the study many times and construct a CI each time, approximately $(1-\alpha)100\%$ of those intervals will contain $\mu$.
- A **wider interval** reflects greater uncertainty. Width increases when $n$ decreases, $s$ (or $\sigma$) increases, or the confidence level rises.
- The $t$-interval is always at least as wide as the corresponding $z$-interval. For $n \ge 30$, the difference is small because $t_{\alpha/2,\,n-1} \approx z_{\alpha/2}$.
- If the data are strongly non-normal and $n$ is small, the $t$-interval may not achieve the nominal coverage. Consider a nonparametric bootstrap CI in such cases.

## Exercises

**Exercise 1.** A sample of $n = 16$ observations from a normal population gives $\bar{x} = 50$ and $s = 8$. Compute the 95 % and 99 % $t$-confidence intervals for $\mu$. How does the width change?

??? success "Solution to Exercise 1"

    With $\text{df} = 15$:

    **95 % CI:** $t_{0.025,15} = 2.131$. $\text{MOE} = 2.131 \times 8/\sqrt{16} = 2.131 \times 2 = 4.262$. CI: $(45.74, 54.26)$. Width = 8.524.

    **99 % CI:** $t_{0.005,15} = 2.947$. $\text{MOE} = 2.947 \times 2 = 5.894$. CI: $(44.11, 55.89)$. Width = 11.788.

    The 99 % interval is $11.788/8.524 = 1.38$ times as wide as the 95 % interval. Higher confidence requires accepting a wider range of plausible values. $\square$

---

**Exercise 2.** Show that the margin of error of the $z$-interval is a decreasing function of $n$, and find the rate at which it decreases.

??? success "Solution to Exercise 2"

    The margin of error is $\text{MOE} = z_{\alpha/2} \cdot \sigma / \sqrt{n}$. Since $z_{\alpha/2}$ and $\sigma$ are constants:

    $$
    \frac{d(\text{MOE})}{dn} = z_{\alpha/2} \cdot \sigma \cdot \left(-\frac{1}{2}\right) n^{-3/2} < 0
    $$

    so MOE is strictly decreasing in $n$. The rate of decrease is $O(n^{-1/2})$: doubling $n$ reduces the MOE by a factor of $1/\sqrt{2} \approx 0.707$, i.e., a roughly 29 % reduction. To halve the margin of error, the sample size must be quadrupled. $\square$

---

**Exercise 3.** A researcher claims that with $n = 100$ and $s = 10$, the 95 % $t$-interval and $z$-interval for $\mu$ are "practically identical." Verify this numerically.

??? success "Solution to Exercise 3"

    For $n = 100$, $\text{df} = 99$:

    - $z_{0.025} = 1.960$
    - $t_{0.025,99} = 1.984$

    The standard error is $s/\sqrt{n} = 10/10 = 1$.

    - $z$-MOE $= 1.960 \times 1 = 1.960$
    - $t$-MOE $= 1.984 \times 1 = 1.984$

    The difference is $1.984 - 1.960 = 0.024$, less than 1.3 % of the $z$-MOE. For a sample mean of, say, 50, the $z$-interval is $(48.040, 51.960)$ and the $t$-interval is $(48.016, 51.984)$. The researcher's claim is correct: the intervals differ by only 0.024 units on each side. $\square$

---

**Exercise 4.** Prove that $t_{\alpha/2,\,\nu} \to z_{\alpha/2}$ as $\nu \to \infty$.

??? success "Solution to Exercise 4"

    If $T \sim t_\nu$, then $T = Z / \sqrt{V/\nu}$ where $Z \sim N(0,1)$ and $V \sim \chi^2_\nu$ are independent. By the law of large numbers, $V/\nu \xrightarrow{P} 1$ as $\nu \to \infty$. Therefore

    $$
    T = \frac{Z}{\sqrt{V/\nu}} \xrightarrow{d} \frac{Z}{1} = Z \sim N(0,1)
    $$

    by Slutsky's theorem. Since the CDF of $t_\nu$ converges pointwise to the standard normal CDF, the quantiles also converge:

    $$
    t_{\alpha/2,\,\nu} \to z_{\alpha/2} \quad \text{as } \nu \to \infty
    $$

    $\square$

---

**Exercise 5.** Given raw data $\{12, 15, 14, 10, 13, 16, 11, 14, 13, 12\}$, compute by hand (then verify with code) the 90 % $t$-interval for $\mu$.

??? success "Solution to Exercise 5"

    **By hand:** $n = 10$, $\bar{x} = (12+15+14+10+13+16+11+14+13+12)/10 = 130/10 = 13.0$.

    $$
    s^2 = \frac{1}{9}\sum(x_i - 13)^2 = \frac{1}{9}(1+4+1+9+0+9+4+1+0+1) = \frac{30}{9} = 3.333
    $$

    $$
    s = \sqrt{3.333} = 1.826
    $$

    With $\text{df} = 9$, $\alpha = 0.10$, $t_{0.05,9} = 1.833$:

    $$
    13.0 \pm 1.833 \times \frac{1.826}{\sqrt{10}} = 13.0 \pm 1.833 \times 0.5774 = 13.0 \pm 1.059
    $$

    The 90 % CI is $(11.94, 14.06)$.

    **Verification:**

    ```python
    import numpy as np
    from scipy.stats import t
    data = np.array([12, 15, 14, 10, 13, 16, 11, 14, 13, 12])
    n = len(data)
    xbar, s = data.mean(), data.std(ddof=1)
    t_crit = t.ppf(0.95, df=n-1)
    moe = t_crit * s / np.sqrt(n)
    print(f"({xbar - moe:.2f}, {xbar + moe:.2f})")
    # Output: (11.94, 14.06)
    ```

    $\square$
