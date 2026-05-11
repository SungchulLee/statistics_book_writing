# Uniform Distribution

## Overview

The **uniform distribution** assigns equal probability to all values in an interval $[a, b]$. It is the simplest continuous distribution and serves as the foundation for random number generation, simulation, and probability integral transforms.

---

## Definition

A random variable $X$ follows a continuous uniform distribution on $[a, b]$:

$$
X \sim \text{Uniform}(a, b), \qquad f(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}
$$

The PDF is constant over the interval, reflecting equal likelihood for all values.

### CDF

$$
F(x) = \begin{cases} 0 & x < a \\ \frac{x - a}{b - a} & a \leq x \leq b \\ 1 & x > b \end{cases}
$$

---

## Properties

$$
\begin{aligned}
E[X] &= \frac{a + b}{2} \\[4pt]
\text{Var}(X) &= \frac{(b - a)^2}{12} \\[4pt]
\text{SD}(X) &= \frac{b - a}{2\sqrt{3}}
\end{aligned}
$$

### Derivation of Mean

$$
E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\bigg|_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}
$$

### Derivation of Variance

$$
E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^3}{3}\bigg|_a^b = \frac{a^2 + ab + b^2}{3}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2 + ab + b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}
$$

---

## Standard Uniform Distribution

The special case $U \sim \text{Uniform}(0, 1)$ is the **standard uniform distribution**. Any uniform variable can be related to it:

$$
X = a + (b - a)U \sim \text{Uniform}(a, b) \quad \text{where } U \sim \text{Uniform}(0, 1)
$$

Conversely:

$$
U = \frac{X - a}{b - a} \sim \text{Uniform}(0, 1) \quad \text{where } X \sim \text{Uniform}(a, b)
$$

---

## Probability Integral Transform

The uniform distribution plays a central role in simulation through the **probability integral transform**:

**Theorem:** If $X$ is a continuous random variable with CDF $F$, then $F(X) \sim \text{Uniform}(0, 1)$.

**Converse (Inverse Transform Sampling):** If $U \sim \text{Uniform}(0, 1)$, then $X = F^{-1}(U)$ has CDF $F$.

### Proof

$$
P(F(X) \leq u) = P(X \leq F^{-1}(u)) = F(F^{-1}(u)) = u
$$

which is the CDF of $\text{Uniform}(0, 1)$.

This theorem is the basis for generating random samples from any distribution using only a uniform random number generator.

---

## Discrete Uniform Distribution

The discrete counterpart assigns equal probability to a finite set of values $\{a, a+1, \ldots, b\}$:

$$
P(X = k) = \frac{1}{b - a + 1}, \quad k = a, a+1, \ldots, b
$$

$$
E[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{(b - a + 1)^2 - 1}{12}
$$

---

## Worked Example

**Problem:** Daily returns of a certain asset are modeled as uniformly distributed between $-2\%$ and $+3\%$. What is the probability the return exceeds $1\%$? What is the expected return?

**Solution:**

$$
P(X > 1) = \frac{3 - 1}{3 - (-2)} = \frac{2}{5} = 0.40
$$

$$
E[X] = \frac{-2 + 3}{2} = 0.5\%
$$

---

## Python: PDF, CDF, and Sampling

### PDF and CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

a, b = 2, 8
x = np.linspace(a - 1, b + 1, 300)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), label='PDF', lw=2)
ax.plot(x, stats.uniform(loc=a, scale=b-a).cdf(x), label='CDF', lw=2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Sampling with Histogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
a, b = 2, 8
samples = stats.uniform(loc=a, scale=b-a).rvs(50_000)

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(samples, bins=60, density=True, alpha=0.7, label='Samples')
x = np.linspace(a - 1, b + 1, 300)
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), 'r-', lw=2, label='PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### Inverse Transform Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate exponential samples via inverse transform
u = np.random.uniform(0, 1, 50_000)
lam = 2.0
x_exp = -np.log(1 - u) / lam  # F_inv(u) for Exponential(lambda)

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(x_exp, bins=100, density=True, alpha=0.7, label='Inverse transform samples')
t = np.linspace(0, 4, 200)
ax.plot(t, stats.expon(scale=1/lam).pdf(t), 'r-', lw=2, label='Exponential PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

---

## Key Takeaways

- The uniform distribution assigns equal probability to all values in an interval, making it the "maximally uninformative" distribution over a bounded range.
- The standard uniform $U(0,1)$ is the building block for random number generation via the inverse transform method.
- The probability integral transform establishes that applying the CDF to any continuous random variable yields a uniform result.
- Despite its simplicity, the uniform distribution is foundational to Monte Carlo simulation and computational statistics.

## Exercises

**Exercise 1.**
$U \sim \mathrm{Uniform}(0, 1)$, $X = -(1/\lambda)\ln(1 - U)$. (a) Find CDF of $X$. (b) Identify the distribution. (c) Explain inverse transform method. (d) Show $1 - U \sim \mathrm{Uniform}(0, 1)$.

??? success "Solution to Exercise 1"
    (a) $P(X \le x) = P(-(1/\lambda)\ln(1 - U) \le x) = P(U \le 1 - e^{-\lambda x}) = 1 - e^{-\lambda x}$.

    (b) This is the CDF of $\mathrm{Exp}(\lambda)$. So $X \sim \mathrm{Exp}(\lambda)$.

    (c) **Inverse transform method:** for any distribution with invertible CDF $F$, set $X = F^{-1}(U)$ where $U \sim \mathrm{Uniform}(0, 1)$. Result: $X$ has CDF $F$. Universal recipe for sampling from distributions with closed-form quantile functions.

    (d) $P(1 - U \le t) = P(U \ge 1 - t) = 1 - (1 - t) = t$. So $1 - U \sim \mathrm{Uniform}(0, 1)$. The simpler formula $X = -(1/\lambda) \ln U$ is therefore equivalent.

---

**Exercise 2.**
**Mean, variance of Uniform$(a, b)$.** Derive both from the PDF.

??? success "Solution to Exercise 2"
    PDF: $f(x) = 1/(b - a)$ on $[a, b]$.

    $\mathbb{E}[X] = \int_a^b x/(b - a) dx = (b^2 - a^2)/(2(b - a)) = (a + b)/2$.

    $\mathbb{E}[X^2] = \int_a^b x^2/(b - a) dx = (b^3 - a^3)/(3(b - a)) = (a^2 + ab + b^2)/3$.

    $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = (a^2 + ab + b^2)/3 - (a + b)^2/4 = (b - a)^2/12$.

    **Standard cases:** Uniform(0, 1) has mean 1/2 and variance 1/12. Uniform(-1, 1) has mean 0 and variance 1/3.

---

**Exercise 3.**
**Sum of two uniforms.** Show that if $U_1, U_2 \sim \mathrm{Uniform}(0, 1)$ are independent, $U_1 + U_2$ has a **triangular distribution** on $[0, 2]$.

??? success "Solution to Exercise 3"
    Convolve the PDFs:

    $$
    f_{U_1 + U_2}(s) = \int_{-\infty}^\infty f_{U_1}(s - u) f_{U_2}(u) du
    $$

    For $u \in [0, 1]$ and $s - u \in [0, 1]$, the integrand is 1; otherwise 0. The region of integration:

    - For $s \in [0, 1]$: $u \in [0, s]$, integral = $s$.
    - For $s \in [1, 2]$: $u \in [s - 1, 1]$, integral = $2 - s$.

    Result: $f_{U_1 + U_2}(s) = \min(s, 2 - s)$ for $s \in [0, 2]$, a triangle peaking at $s = 1$ with height 1.

    The CLT for sums of uniforms predicts approximate normality for sums of 6+ uniforms — fast convergence. This is the basis of the Marsaglia-Bray algorithm for normal random number generation.

---

**Exercise 4.**
**Order statistics of uniforms.** For $U_1, \ldots, U_n$ i.i.d. $\mathrm{Uniform}(0, 1)$, the $k$-th order statistic $U_{(k)}$ has distribution $\mathrm{Beta}(k, n - k + 1)$. Derive the CDF.

??? success "Solution to Exercise 4"
    $U_{(k)} \le u$ iff at least $k$ of the $U_i$'s are $\le u$. The number of $U_i$'s $\le u$ is $\mathrm{Binomial}(n, u)$ (each independently with probability $u$).

    $$
    P(U_{(k)} \le u) = P(\mathrm{Binomial}(n, u) \ge k) = \sum_{j=k}^n \binom{n}{j} u^j (1 - u)^{n - j}
    $$

    By the incomplete-beta function identity, this equals $I_u(k, n - k + 1)$ — the regularized incomplete beta function. So $U_{(k)} \sim \mathrm{Beta}(k, n - k + 1)$.

    $\mathbb{E}[U_{(k)}] = k/(n + 1)$ (well-known beta mean). The expected $k$-th order statistic divides $[0, 1]$ into $n + 1$ equal pieces, providing the **plotting positions** used in Q-Q plots.

---

**Exercise 5.**
**Maximum-entropy property.** Among all distributions on a bounded interval $[a, b]$, the uniform distribution has the maximum **differential entropy**. State the differential entropy formula and verify.

??? success "Solution to Exercise 5"
    Differential entropy: $h(X) = -\int f(x) \ln f(x) dx$.

    For $X \sim \mathrm{Uniform}(a, b)$: $h(X) = -\int_a^b (1/(b-a)) \ln(1/(b-a)) dx = \ln(b - a)$.

    **Claim:** any other distribution $g$ on $[a, b]$ has $h(g) \le \ln(b - a)$. Proof via the non-negativity of KL divergence:

    $0 \le D_{KL}(g \| f) = \int g \ln(g/f) dx = -h(g) - \int g \ln f \, dx = -h(g) + \ln(b - a)$,

    so $h(g) \le \ln(b - a) = h(f)$, with equality iff $g = f$ a.e.

    **Interpretation:** in the absence of information beyond the support, the uniform distribution is the "least informative" — assigning equal mass everywhere. This makes it the natural prior in non-informative Bayesian analysis on bounded parameters.

---

**Exercise 6.**
**Probability integral transform.** Prove that if $X$ has continuous CDF $F$, then $U = F(X) \sim \mathrm{Uniform}(0, 1)$.

??? success "Solution to Exercise 6"
    $P(U \le u) = P(F(X) \le u) = P(X \le F^{-1}(u)) = F(F^{-1}(u)) = u$, using continuity of $F$ (which makes $F^{-1}$ well-defined and $F \circ F^{-1} = \mathrm{id}$).

    So $U$ has CDF $u$ on $[0, 1]$, i.e., $U \sim \mathrm{Uniform}(0, 1)$.

    **Use:** the probability integral transform underlies many statistical tests:

    - **Kolmogorov-Smirnov test:** transform data using the hypothesized $F$; under the null, the transformed data is uniform.
    - **Copula models:** decompose joint distributions into marginals (via PIT to uniform) and a copula linking the uniforms.
    - **Validation of distributional forecasts:** transformed observations should be uniform if the forecast distribution is correctly specified.

    This is the dual of inverse transform sampling: one transforms uniforms to other distributions, the other transforms other distributions to uniforms. Both rely on the same CDF/inverse CDF machinery.
