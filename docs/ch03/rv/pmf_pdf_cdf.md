# PMF, PDF, and CDF

## Overview

The three fundamental functions that characterize the distribution of a random variable are the Probability Mass Function (PMF) for discrete variables, the Probability Density Function (PDF) for continuous variables, and the Cumulative Distribution Function (CDF) for both.

---

## PMF and PDF

$$
\begin{aligned}
\textbf{PMF:} \quad & p_{x_i} = \text{The weight of the brick assigned to the discrete value } x_i \\[8pt]
\textbf{PDF:} \quad & f(x)\,dx = \text{The weight of the bricks within the continuous interval } [x, x + dx]
\end{aligned}
$$

---

## Cumulative Distribution Function (CDF)

The CDF $F(x)$ gives the cumulative probability that the random variable $X$ takes a value less than or equal to $x$:

$$
F(x) = \mathbb{P}(X \leq x) =
\begin{cases}
\displaystyle\sum_{x_i \leq x} p_{x_i}, & \text{if } X \text{ is discrete} \\[10pt]
\displaystyle\int_{-\infty}^x f(s)\,ds, & \text{if } X \text{ is continuous}
\end{cases}
$$

In the brick metaphor: $F(x)$ is the **total weight of all bricks stacked from $-\infty$ up to $x$**.

### Properties of the CDF

- $F(x)$ is non-decreasing
- $\lim_{x \to -\infty} F(x) = 0$
- $\lim_{x \to +\infty} F(x) = 1$
- For continuous $X$: $F'(x) = f(x)$ (the PDF is the derivative of the CDF)

---

## Relationship Between PDF and CDF

The PDF and CDF are related by integration and differentiation:

$$
\text{CDF} = \int \text{PDF} \qquad \text{and} \qquad \text{PDF} = \frac{d}{dx} \text{CDF}
$$

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, (ax_pdf, ax_arrow, ax_cdf) = plt.subplots(1, 3, figsize=(12, 3))

x = np.linspace(-3, 3, 100)

# PDF
ax_pdf.set_title("PDF", fontsize=16)
ax_pdf.plot(x, stats.norm().pdf(x))

# Arrows showing relationship
ax_arrow.arrow(0.1, 0.6, 0.8, 0, width=0.05, length_includes_head=True)
ax_arrow.arrow(0.9, 0.4, -0.8, 0, width=0.05, length_includes_head=True)
ax_arrow.annotate("Integrate", (0.38, 0.75), fontsize=14)
ax_arrow.annotate("Differentiate", (0.30, 0.2), fontsize=14)
for spine in ax_arrow.spines.values():
    spine.set_visible(False)
ax_arrow.set_xticks([])
ax_arrow.set_yticks([])

# CDF
ax_cdf.set_title("CDF", fontsize=16)
ax_cdf.plot(x, stats.norm().cdf(x))

for ax in (ax_pdf, ax_cdf):
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

---

## Percent Point Function (PPF)

The **PPF** is the inverse of the CDF. Given a cumulative probability $p$, the PPF returns the value $x$ such that $P(X \leq x) = p$:

$$
\text{PPF}(p) = F^{-1}(p) = \inf\{x : F(x) \geq p\}
$$

### Example: 95th Percentile of Standard Normal

For $Z \sim N(0, 1)$, the value $z$ such that $P(Z \leq z) = 0.95$ is approximately 1.645:

```python
import scipy.stats as stats

z_95 = stats.norm(0, 1).ppf(0.95)
print(f"95th percentile of N(0,1): {z_95:.4f}")
```

### Example: 97.5th Percentile

The value $z$ such that $P(Z \leq z) = 0.975$ is approximately 1.96, widely used in confidence intervals:

```python
z_975 = stats.norm(0, 1).ppf(0.975)
print(f"97.5th percentile of N(0,1): {z_975:.4f}")
```

---

## CDF and PPF Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(-3, 3)
ax.set_ylim(-0.2, 1.1)

# CDF curve
x = np.linspace(-3, 3, 100)
ax.plot(x, stats.norm().cdf(x), label='CDF')

# PPF demonstration at 0.975
u = 0.975
z = stats.norm().ppf(u)

ax.plot(0, u, 'or', markersize=8)
ax.plot(z, 0, 'or', markersize=8)
ax.annotate(f"U = {u}", (-1.2, u + 0.02), fontsize=14)
ax.annotate(f"Z = {z:.3f}", (z - 0.3, -0.12), fontsize=14)
ax.annotate("PPF →", (0.3, u + 0.03), fontsize=14)
ax.annotate("↓ CDF", (z + 0.1, 0.5), fontsize=14)

ax.spines[['right', 'top']].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.legend(fontsize=14)
plt.show()
```

---

## Generating Random Samples via PPF (Inverse Transform Sampling)

A powerful application of the PPF: if $U \sim \text{Uniform}(0,1)$, then $X = F^{-1}(U)$ has the distribution with CDF $F$:

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

u = stats.uniform().rvs(10_000)
z = stats.norm().ppf(u)

plt.figure(figsize=(12, 3))
plt.hist(z, bins=100, density=True, alpha=0.7, label='Inverse Transform Samples')
x = np.linspace(-4, 4, 200)
plt.plot(x, stats.norm().pdf(x), 'r--', lw=2, label='N(0,1) PDF')
plt.legend()
plt.show()
```

---

## Example: Normal CDF Computation

```python
from scipy import stats

mean, std_dev = 50, 10

# P(40 ≤ X ≤ 60) for X ~ N(50, 10²)
prob = stats.norm(mean, std_dev).cdf(60) - stats.norm(mean, std_dev).cdf(40)
print(f"P(40 ≤ X ≤ 60) = {prob * 100:.2f}%")

# P(X ≤ 55)
prob_55 = stats.norm(mean, std_dev).cdf(55)
print(f"P(X ≤ 55) = {prob_55 * 100:.2f}%")
```

---

## Empirical PMF/PDF and CDF

In practice, we estimate the PDF and CDF from data using histograms and empirical CDFs:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(42)
data = stats.norm.rvs(size=200)

fig, ax = plt.subplots(figsize=(12, 4))

# Empirical PDF (histogram)
counts, bin_edges, _ = ax.hist(data, bins=20, density=True, alpha=0.6, label="Empirical PDF")

# Empirical CDF
empirical_cdf = np.cumsum(counts) / np.sum(counts)
ax.step(bin_edges[1:], empirical_cdf, where='mid', label="Empirical CDF", lw=2)

# Theoretical CDF
ax.plot(bin_edges, stats.norm.cdf(bin_edges), 'r', lw=2, label="Theoretical CDF")

ax.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.show()
```

---

## Summary: scipy.stats Methods

| Method | Description |
|:---|:---|
| `rvs` | Generate random samples |
| `pdf` | Compute the PDF |
| `cdf` | Compute the CDF: $P(X \leq x)$ |
| `sf` | Survival function: $1 - \text{cdf}(x) = P(X > x)$ |
| `ppf` | Percent point function (inverse CDF) |

---

## Key Takeaways

- The **PMF** gives point probabilities for discrete variables; the **PDF** gives probability density for continuous variables.
- The **CDF** accumulates probability from $-\infty$ to $x$ and applies to both types.
- The **PPF** inverts the CDF: given a probability, it returns the corresponding quantile.
- Integration connects PDF → CDF; differentiation connects CDF → PDF.
