# Normal Percent Point Function (Quantile Function)

## Overview

The **percent point function** (PPF), also called the **quantile function** or **inverse CDF**, answers the question: given a cumulative probability $q$, what value $x$ satisfies $P(X \le x) = q$?

$$
\text{ppf}(q) = F^{-1}(q) = \inf\{x : F(x) \ge q\}
$$

For the standard normal, the most important quantile is $\mathcal{N}^{-1}(0.975) \approx 1.96$, the critical value for two-sided 95% confidence intervals.

---

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 0, 1
prob = 0.975

dist = stats.norm(loc=mu, scale=sigma)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
pdf = dist.pdf(x)
z = dist.ppf(prob)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf, color='b', lw=2, label='PDF')
ax.plot([z, z], [0, dist.pdf(z)], color='k', lw=3)
ax.fill_between(x[x <= z], pdf[x <= z], 0,
                interpolate=True, color='r', alpha=0.25,
                label=f"P(X ≤ {z:.2f}) = {prob}")
ax.text(z + 0.05, dist.pdf(z) / 2,
        f"ppf({prob}) = {z:.4f}", fontsize=11, va='center')
ax.set_title(f"Normal({mu}, {sigma}) — PPF (Quantile Function)")
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
```

---

## Common Quantiles of the Standard Normal

| $q$ | $\mathcal{N}^{-1}(q)$ | Usage |
|---|---|---|
| 0.500 | 0 | Median |
| 0.900 | 1.282 | 90% CI one-sided |
| 0.950 | 1.645 | 95% CI one-sided |
| 0.975 | 1.960 | 95% CI two-sided |
| 0.995 | 2.576 | 99% CI two-sided |

---

## Exercises

**Exercise 1.**
Using the PPF, find the value $z$ such that $P(-z \le Z \le z) = 0.99$ for $Z \sim N(0,1)$.

??? success "Solution to Exercise 1"
    We need $P(Z \le z) = 0.995$ (leaving 0.5% in each tail):

    $$
    z = \mathcal{N}^{-1}(0.995) \approx 2.576
    $$

    So 99% of the standard normal distribution lies between $\pm 2.576$.

---

**Exercise 2.**
If $X \sim N(100, 225)$, find the 90th percentile of $X$.

??? success "Solution to Exercise 2"
    Here $\mu = 100$, $\sigma = 15$. The 90th percentile is:

    $$
    x_{0.90} = \mu + \sigma \cdot \mathcal{N}^{-1}(0.90) = 100 + 15 \times 1.282 \approx 119.2
    $$

---

**Exercise 3.**
Prove that for a continuous distribution, $F(F^{-1}(q)) = q$ for all $q \in (0, 1)$.

??? success "Solution to Exercise 3"
    Let $x_q = F^{-1}(q) = \inf\{x : F(x) \ge q\}$. Since $F$ is continuous and nondecreasing with range $(0,1)$, the set $\{x : F(x) \ge q\}$ is a closed half-line $[x_q, \infty)$. Continuity of $F$ ensures $F(x_q) = q$ (the infimum is achieved). Therefore $F(F^{-1}(q)) = F(x_q) = q$. $\square$

---

**Exercise 4.**
Explain the relationship between the PPF and the survival function. How would you compute the value $z$ such that $P(X > z) = 0.05$?

??? success "Solution to Exercise 4"
    The survival function is $S(x) = 1 - F(x)$. If $P(X > z) = 0.05$, then $P(X \le z) = 0.95$, so $z = F^{-1}(0.95)$.

    In SciPy: `z = stats.norm.ppf(0.95)` or equivalently `z = stats.norm.isf(0.05)`, where `isf` is the inverse survival function.
