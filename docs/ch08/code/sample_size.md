# Sample Size Calculations

## Overview

Choosing the right sample size before collecting data is one of the most important steps in study design. A sample that is too small yields wide confidence intervals and low power, while an unnecessarily large sample wastes resources. This page presents the key formulas for determining $n$ when estimating a mean or a proportion, and illustrates them with Python code.

## Sample Size for Estimating a Mean

To estimate the population mean $\mu$ within a margin of error $E$ at confidence level $1 - \alpha$, the minimum sample size is

$$
n = \left\lceil \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2 \right\rceil
$$

where $\sigma$ is the population standard deviation (or a planning estimate of it) and $\lceil \cdot \rceil$ denotes the ceiling function.

### Derivation

Starting from the margin-of-error equation for the $z$-interval,

$$
E = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

solving for $n$ gives

$$
\sqrt{n} = \frac{z_{\alpha/2} \cdot \sigma}{E}
\quad\Longrightarrow\quad
n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2
$$

Since $n$ must be an integer, we round up.

## Sample Size for Estimating a Proportion

For estimating a population proportion $p$ within margin $E$, the conservative formula (using $p = 0.5$, which maximizes $p(1-p)$) is

$$
n = \left\lceil \frac{z_{\alpha/2}^2}{4E^2} \right\rceil
$$

If a planning estimate $p_0$ is available, the tighter formula is

$$
n = \left\lceil \frac{z_{\alpha/2}^2 \, p_0(1 - p_0)}{E^2} \right\rceil
$$

## Python Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- Sample size for estimating a mean ---
sigma_est = 15  # planning estimate of sigma
for E in [1, 2, 3, 5]:
    for conf in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z * sigma_est / E) ** 2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")

# --- Sample size for estimating a proportion (conservative p=0.5) ---
print("\nSample size for proportion (conservative p=0.5):")
for E in [0.01, 0.03, 0.05]:
    for conf in [0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z / (2 * E)) ** 2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")
```

## Interpretation

- The required sample size grows with the **square** of both the critical value $z_{\alpha/2}$ and the population spread $\sigma$, and decreases with the square of the desired margin $E$. Halving the margin of error quadruples the required $n$.
- For proportions, the conservative choice $p = 0.5$ guarantees adequate precision regardless of the true $p$, at the cost of a potentially larger sample.
- In practice, $\sigma$ is rarely known exactly. Common strategies include using a pilot study, prior research, or the range rule $\sigma \approx \text{range}/4$.

## Exercises

**Exercise 1.** An engineer needs to estimate the mean tensile strength of a cable within $\pm 5$ MPa at the 95 % confidence level. A pilot study suggests $\sigma \approx 20$ MPa. What sample size is required?

??? success "Solution to Exercise 1"

    The critical value is $z_{0.025} = 1.96$. Applying the formula,

    $$
    n = \left\lceil \left(\frac{1.96 \times 20}{5}\right)^2 \right\rceil
      = \left\lceil (7.84)^2 \right\rceil
      = \left\lceil 61.47 \right\rceil
      = 62
    $$

    At least 62 observations are needed. $\square$

---

**Exercise 2.** A pollster wants to estimate a candidate's support within $\pm 3$ percentage points at the 99 % confidence level. Using the conservative approach, find the required sample size.

??? success "Solution to Exercise 2"

    With $E = 0.03$ and $z_{0.005} = 2.576$,

    $$
    n = \left\lceil \frac{(2.576)^2}{4 \times (0.03)^2} \right\rceil
      = \left\lceil \frac{6.6358}{0.0036} \right\rceil
      = \left\lceil 1843.3 \right\rceil
      = 1844
    $$

    The pollster needs at least 1844 respondents. $\square$

---

**Exercise 3.** Show that $p(1-p) \le 1/4$ for all $p \in [0,1]$, and explain why $p = 0.5$ yields the most conservative (largest) sample size for a proportion.

??? success "Solution to Exercise 3"

    Write $f(p) = p(1-p) = p - p^2$. Taking the derivative, $f'(p) = 1 - 2p$, which equals zero at $p = 1/2$. Since $f''(p) = -2 < 0$, the point $p = 1/2$ is a global maximum on $[0,1]$ with

    $$
    f(1/2) = \frac{1}{2}\left(1 - \frac{1}{2}\right) = \frac{1}{4}
    $$

    The sample-size formula is $n = z_{\alpha/2}^2 \, p(1-p) / E^2$. Since $p(1-p)$ appears in the numerator, the maximum at $p = 1/2$ produces the largest $n$, guaranteeing adequate precision for any true $p$. $\square$

---

**Exercise 4.** If you initially plan for a margin of error of $\pm 4$ but later decide you need $\pm 2$, by what factor does the required sample size change? Prove this relationship in general.

??? success "Solution to Exercise 4"

    The sample size formula is $n = (z_{\alpha/2} \sigma / E)^2$. Let $n_1$ correspond to margin $E_1$ and $n_2$ to margin $E_2$. Then

    $$
    \frac{n_2}{n_1} = \frac{(z_{\alpha/2} \sigma / E_2)^2}{(z_{\alpha/2} \sigma / E_1)^2} = \left(\frac{E_1}{E_2}\right)^2
    $$

    Halving the margin ($E_2 = E_1/2$) gives $n_2 / n_1 = 4$. The required sample size **quadruples**. In the specific case $E_1 = 4, E_2 = 2$:

    $$
    \frac{n_2}{n_1} = \left(\frac{4}{2}\right)^2 = 4
    $$

    $\square$

---

**Exercise 5.** A researcher has a planning estimate of $p_0 = 0.30$ and wants a 95 % CI for $p$ with margin $\pm 0.04$. Compare the sample size from the planning-estimate formula to the conservative formula. How many observations are saved?

??? success "Solution to Exercise 5"

    With $z_{0.025} = 1.96$ and $E = 0.04$:

    **Conservative** ($p = 0.5$):

    $$
    n_{\text{cons}} = \left\lceil \frac{(1.96)^2}{4(0.04)^2} \right\rceil = \left\lceil \frac{3.8416}{0.0064} \right\rceil = \left\lceil 600.25 \right\rceil = 601
    $$

    **Planning estimate** ($p_0 = 0.30$):

    $$
    n_{\text{plan}} = \left\lceil \frac{(1.96)^2 \times 0.30 \times 0.70}{(0.04)^2} \right\rceil = \left\lceil \frac{3.8416 \times 0.21}{0.0016} \right\rceil = \left\lceil \frac{0.8067}{0.0016} \right\rceil = \left\lceil 504.2 \right\rceil = 505
    $$

    Using the planning estimate saves $601 - 505 = 96$ observations. $\square$
