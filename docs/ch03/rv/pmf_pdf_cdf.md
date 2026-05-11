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

## Exercises

**Exercise 1.**
$X$ = number of heads in 3 fair coin flips. (a) Write PMF; (b) write CDF; (c) compute $P(1 \le X \le 2)$ two ways.

??? success "Solution to Exercise 1"
    (a) $X \sim \mathrm{Binomial}(3, 1/2)$:

    | $x$ | $p(x)$ |
    |:---:|:---:|
    | 0 | $1/8$ |
    | 1 | $3/8$ |
    | 2 | $3/8$ |
    | 3 | $1/8$ |

    (b) $F(x) = 0, 1/8, 4/8, 7/8, 1$ on intervals $(-\infty, 0), [0, 1), [1, 2), [2, 3), [3, \infty)$.

    (c) PMF: $p(1) + p(2) = 3/8 + 3/8 = 3/4$. CDF: $F(2) - F(0^-) = 7/8 - 0 = 7/8$. Wait — to use CDF for $P(1 \le X \le 2)$ with integer-valued $X$: $P(1 \le X \le 2) = F(2) - F(1^-) = F(2) - F(0) = 7/8 - 1/8 = 6/8 = 3/4$. Both methods agree.

---

**Exercise 2.**
For a continuous RV $X$ with PDF $f(x) = c \cdot x^2$ on $[0, 1]$ and 0 elsewhere: (a) find $c$; (b) compute $F(x)$; (c) find $P(0.3 < X < 0.7)$.

??? success "Solution to Exercise 2"
    (a) $\int_0^1 c x^2 \, dx = c/3 = 1$, so $c = 3$.

    (b) $F(x) = \int_0^x 3 t^2 \, dt = x^3$ for $x \in [0, 1]$; $F(x) = 0$ for $x < 0$; $F(x) = 1$ for $x > 1$.

    (c) $P(0.3 < X < 0.7) = F(0.7) - F(0.3) = 0.343 - 0.027 = 0.316$.

---

**Exercise 3.**
**Differentiate CDF to PDF.** For a continuous $X$ with $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$, compute the PDF $f(x)$. What distribution is this?

??? success "Solution to Exercise 3"
    $f(x) = F'(x) = \lambda e^{-\lambda x}$ for $x \ge 0$.

    This is the **Exponential distribution** with rate $\lambda$. Properties:
    - Mean $1/\lambda$, variance $1/\lambda^2$.
    - Memoryless: $P(X > s + t \mid X > s) = P(X > t)$.
    - Waiting time between events in a Poisson process with rate $\lambda$.

---

**Exercise 4.**
**Inverse transform sampling.** Show that if $U \sim \mathrm{Uniform}(0, 1)$ and $F$ is a continuous strictly-increasing CDF, then $X = F^{-1}(U)$ has CDF $F$.

??? success "Solution to Exercise 4"
    Compute $P(X \le x) = P(F^{-1}(U) \le x)$. Apply $F$ to both sides (which is monotone increasing, preserving inequalities):

    $$
    P(F^{-1}(U) \le x) = P(F(F^{-1}(U)) \le F(x)) = P(U \le F(x)) = F(x)
    $$

    using $F \circ F^{-1} = \mathrm{id}$ for invertible $F$ and the uniform's CDF being $P(U \le u) = u$ for $u \in [0, 1]$.

    So $X$ has CDF $F$. $\square$

    **Use:** to generate samples from any distribution with known $F^{-1}$, draw $U$ uniformly and apply $F^{-1}$. This is how many random-number-generation routines work internally for non-trivial distributions.

---

**Exercise 5.**
**Quantile vs. percentile vs. PPF.** Clarify these three terms with examples. State the relationship between the PPF and the survival function.

??? success "Solution to Exercise 5"
    **Quantile:** for $p \in [0, 1]$, the $p$-th quantile $q_p = F^{-1}(p)$ is the value such that $P(X \le q_p) = p$. Quantile = PPF evaluation.

    **Percentile:** the $p$-percentile for $p \in [0, 100]$ is $q_{p/100}$. Just a unit convention — "95th percentile" means $q_{0.95}$.

    **PPF (Percent Point Function):** the inverse CDF function itself, $\mathrm{PPF}(p) = F^{-1}(p)$.

    **Survival function:** $S(x) = 1 - F(x) = P(X > x)$. The inverse survival function (ISF) gives "the value above which a given probability mass lies": $\mathrm{ISF}(p) = S^{-1}(p) = F^{-1}(1 - p) = \mathrm{PPF}(1 - p)$.

    In scipy.stats: `dist.ppf(0.95)` gives the 95th percentile; `dist.isf(0.05)` gives the value with 5% upper tail probability, which equals the 95th percentile. The ISF avoids numerical loss-of-precision when computing tail quantiles ($1 - F$ near 0 has poor precision; using $S$ directly is better).

---

**Exercise 6.**
**Improper integrals.** A proposed PDF is $f(x) = 1/(x \ln^2 x)$ for $x \ge 2$. Does this define a valid distribution? Compute $\mathbb{E}[X]$.

??? success "Solution to Exercise 6"
    **Normalization:** $\int_2^\infty \frac{1}{x \ln^2 x} dx$. Let $u = \ln x$, $du = dx/x$:

    $$
    \int_{\ln 2}^\infty \frac{1}{u^2} du = \left[-\frac{1}{u}\right]_{\ln 2}^\infty = \frac{1}{\ln 2} \approx 1.443
    $$

    So $f(x)$ is **not** normalized. Define $c = \ln 2$, redefine $f(x) = c/(x \ln^2 x)$, then $\int f = 1$ and we have a valid PDF.

    **Expectation:** $\mathbb{E}[X] = \int_2^\infty x \cdot \frac{c}{x \ln^2 x} dx = c \int_2^\infty \frac{1}{\ln^2 x} dx$.

    The integrand decays like $1/\ln^2 x$, which is *not* integrable at infinity (the integral diverges). So $\mathbb{E}[X] = \infty$ — the distribution has a finite normalization but infinite mean.

    **Lesson:** "valid distribution" (CDF properties hold) is a weaker requirement than "finite expectation". Heavy-tailed distributions like this one need quantile-based summaries instead of mean-based ones. This was the topic of Exercise 5 in the LLN page: distributions with infinite mean break the LLN.
