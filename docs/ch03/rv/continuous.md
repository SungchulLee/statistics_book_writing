# Continuous Random Variables

## Overview

A **continuous random variable** can take on any value within a continuous range (an interval or union of intervals on the real line). Unlike discrete random variables, the probability of any single specific value is zero—instead, probabilities are defined over intervals.

---

## Definition

A **continuous random variable** $X$ can take on infinitely many possible values within a given range. Examples include heights, weights, temperatures, and waiting times.

For continuous random variables, the weight (probability) is **spread continuously** along the real line rather than concentrated at specific points.

---

## Probability Density Function (PDF)

For a continuous random variable $X$, the **PDF** $f(x)$ describes the density of probability at each point:

$$
f(x)\,dx = \text{Weight of the bricks within the interval } [x, x + dx]
$$

Key properties of the PDF:

1. $f(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f(x)\,dx = 1$
3. $P(a \leq X \leq b) = \int_a^b f(x)\,dx$

Note that $f(x)$ itself is **not** a probability—it can exceed 1. Only the **area** under the curve gives probabilities.

---

## Key Difference from Discrete Variables

For a **discrete** random variable, we can ask $P(X = a)$ and get a positive answer. For a **continuous** random variable:

$$
P(X = a) = 0 \quad \text{for any specific value } a
$$

This is because there are infinitely many possible values, and the "weight" at any single point is zero. We can only meaningfully ask about the probability that $X$ falls within a range.

---

## Example: Normal PDF

The most important continuous distribution is the **normal distribution** with mean $\mu$ and variance $\sigma^2$. Its PDF is:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

The area under this curve between any two values gives the probability that $X$ falls within that range.

---

## Example: Amelia's Maximum Average Wait Time

The distribution of average wait times at drive-through restaurants is approximately normal with mean $\mu = 185$ seconds and standard deviation $\sigma = 11$ seconds. Amelia only uses restaurants in the bottom 10% of wait times. What is her maximum acceptable wait time?

**Solution:**

We need the 10th percentile (PPF at 0.1):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mu = 185
sigma = 11

max_wait = stats.norm(loc=mu, scale=sigma).ppf(0.1)
print(f"Maximum average wait time: {max_wait:.2f} seconds")

# Visualization
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
pdf = stats.norm(loc=mu, scale=sigma).pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf)
x_fill = np.linspace(mu - 3*sigma, max_wait, 100)
ax.fill_between(x_fill, stats.norm(loc=mu, scale=sigma).pdf(x_fill),
                alpha=0.3, color='r', label=f'Bottom 10% (≤ {max_wait:.1f}s)')
ax.spines[['right', 'top']].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.legend()
plt.show()
```

---

## Comparing Discrete and Continuous Distributions

| Feature | Discrete | Continuous |
|:---|:---|:---|
| **Values** | Countable set | Uncountable (interval) |
| **Probability at a point** | $P(X = a) > 0$ possible | $P(X = a) = 0$ always |
| **Probability function** | PMF: $p_{x_i}$ | PDF: $f(x)$ |
| **Probability of a range** | $\sum_{x_i \in [a,b]} p_{x_i}$ | $\int_a^b f(x)\,dx$ |
| **Total probability** | $\sum_i p_{x_i} = 1$ | $\int_{-\infty}^{\infty} f(x)\,dx = 1$ |

---

## Key Takeaways

- Continuous random variables take values in an interval; the probability of any single point is zero.
- The PDF describes the "density" of probability—areas under the PDF curve give probabilities.
- The PDF can exceed 1 at specific points, but the total area under the curve is always 1.
- The normal distribution is the most widely used continuous distribution.

## Exercises

**Exercise 1.**
$f(x) = c x^2$ for $0 \le x \le 2$, 0 else. (a) Find $c$; (b) compute $F(x)$; (c) $P(1 \le X \le 2)$; (d) the median.

??? success "Solution to Exercise 1"
    (a) $\int_0^2 c x^2 dx = 8c/3 = 1 \Rightarrow c = 3/8$.

    (b) $F(x) = \int_0^x (3/8) t^2 dt = x^3/8$ for $x \in [0, 2]$; 0 below, 1 above.

    (c) $P(1 \le X \le 2) = F(2) - F(1) = 1 - 1/8 = 7/8$.

    (d) $F(m) = 1/2 \Rightarrow m^3/8 = 1/2 \Rightarrow m = \sqrt[3]{4} \approx 1.587$.

---

**Exercise 2.**
**Why $P(X = a) = 0$ for continuous $X$.** Rigorously argue this from the CDF.

??? success "Solution to Exercise 2"
    For continuous $X$ with continuous CDF $F$:

    $$
    P(X = a) = \lim_{\varepsilon \to 0^+} P(a - \varepsilon < X \le a) = \lim_{\varepsilon \to 0^+} [F(a) - F(a - \varepsilon)] = F(a) - F(a^-) = 0
    $$

    using continuity of $F$ at $a$.

    **Note:** $P(X = a) = 0$ does *not* mean $X = a$ is impossible. It means the event has probability zero in the Lebesgue sense. Any specific outcome of a continuous random variable is "infinitely unlikely" but the sample space is still uncountable and produces a definite outcome with each draw.

    This is the source of the "$P(0)$ doesn't mean impossible" confusion in elementary probability. The fix is to think in terms of densities, not point probabilities — a continuous random variable concentrates probability in intervals, not at points.

---

**Exercise 3.**
**Transformation rule for continuous PDFs.** If $X$ has density $f_X$ and $Y = g(X)$ with $g$ strictly increasing and differentiable, derive the density of $Y$.

??? success "Solution to Exercise 3"
    Start from the CDF: $F_Y(y) = P(Y \le y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))$.

    Differentiate with chain rule:

    $$
    f_Y(y) = F_Y'(y) = f_X(g^{-1}(y)) \cdot \frac{d}{dy} g^{-1}(y) = \frac{f_X(g^{-1}(y))}{g'(g^{-1}(y))}
    $$

    Or more compactly: with $x = g^{-1}(y)$,

    $$
    f_Y(y) = \frac{f_X(x)}{|g'(x)|}
    $$

    The absolute value handles decreasing $g$ as well.

    **Example:** $X \sim N(0, 1)$, $Y = e^X$. Then $g(x) = e^x$, $g'(x) = e^x = y$, $x = \ln y$. So

    $$
    f_Y(y) = \frac{1}{\sqrt{2\pi}} e^{-(\ln y)^2/2} \cdot \frac{1}{y}
    $$

    which is the **lognormal distribution** density.

---

**Exercise 4.**
**Expected value for continuous RV.** For $X$ with PDF $f(x) = 2x$ on $[0, 1]$: compute $\mathbb{E}[X]$, $\mathbb{E}[X^2]$, and $\mathrm{Var}(X)$.

??? success "Solution to Exercise 4"
    $\mathbb{E}[X] = \int_0^1 x \cdot 2x \, dx = \int_0^1 2x^2 \, dx = 2/3$.

    $\mathbb{E}[X^2] = \int_0^1 x^2 \cdot 2x \, dx = \int_0^1 2x^3 \, dx = 1/2$.

    $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = 1/2 - 4/9 = 9/18 - 8/18 = 1/18 \approx 0.0556$.

    SD $\approx 0.236$.

---

**Exercise 5.**
**Memoryless property of the exponential.** If $X \sim \mathrm{Exp}(\lambda)$, prove $P(X > s + t \mid X > s) = P(X > t)$ for all $s, t \ge 0$. What is its interpretation?

??? success "Solution to Exercise 5"
    Survival function: $P(X > x) = e^{-\lambda x}$. Apply the definition of conditional probability:

    $$
    P(X > s + t \mid X > s) = \frac{P(X > s + t \cap X > s)}{P(X > s)} = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
    $$

    $\square$

    **Interpretation:** if you've already waited $s$ time units for an event, the remaining wait time has the same distribution as if you'd just started waiting. The process "forgets" how long it has been waiting.

    **Real-world implications:**

    - Radioactive decay: the half-life is well-defined because decay is memoryless.
    - Phone-call durations: empirically not exponential because of memory effects (long calls usually continue).
    - Customer arrivals to a counter: often well-modeled as Poisson with exponential inter-arrival times.

    The exponential is the **unique** continuous distribution with the memoryless property — a striking characterization.

---

**Exercise 6.**
**Mixed distributions.** Give an example of a random variable $X$ that is neither purely discrete nor purely continuous. Show how its CDF has both jumps and continuous segments.

??? success "Solution to Exercise 6"
    Example: a measurement that has a chance of being "zero" (e.g., due to a sensor not triggering) plus a continuous component when it does trigger.

    Let $Y \sim \mathrm{Exp}(1)$ and let $A$ be Bernoulli(0.3) independent of $Y$. Define $X = A \cdot Y$:

    - With probability 0.7, $A = 0$, so $X = 0$.
    - With probability 0.3, $A = 1$, so $X = Y \sim \mathrm{Exp}(1)$.

    CDF:

    $$
    F(x) = \begin{cases} 0 & x < 0 \\ 0.7 + 0.3(1 - e^{-x}) & x \ge 0 \end{cases}
    $$

    At $x = 0$: $F(0) = 0.7$ — a **jump** of 0.7 corresponding to the discrete mass at 0.
    For $x > 0$: continuous increase from 0.7 to 1.

    Such mixed distributions appear frequently in practice (zero-inflated count models, insurance claim sizes, censored measurements). The framework of the Lebesgue–Stieltjes integral handles them rigorously; the practical handling decomposes them into discrete and continuous parts and integrates each appropriately.
