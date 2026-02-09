# Independence vs Zero Correlation

## Overview

A common misconception is that uncorrelated random variables are independent. While **independence implies zero correlation**, the converse is **false** in general. This section clarifies the distinction with proofs, counterexamples, and the special case where the two notions coincide.

---

## Definitions

### Independence

$X$ and $Y$ are **independent** ($X \perp Y$) if:

$$
P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B) \quad \text{for all sets } A, B
$$

Equivalently, the joint density/PMF factors: $f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)$.

### Zero Correlation (Uncorrelatedness)

$X$ and $Y$ are **uncorrelated** if:

$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 0
$$

Equivalently, $\rho(X,Y) = 0$ or $E[XY] = E[X]E[Y]$.

---

## Independence Implies Zero Correlation

**Theorem:** If $X \perp Y$, then $\text{Cov}(X,Y) = 0$.

**Proof:**

If $X$ and $Y$ are independent, then $E[XY] = E[X] \cdot E[Y]$ (the expectation of a product equals the product of expectations). Therefore:

$$
\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = E[X]E[Y] - E[X]E[Y] = 0
$$

More generally, independence implies $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ for **all** measurable functions $g, h$. Zero correlation only requires this for $g(x) = x$ and $h(y) = y$.

---

## Zero Correlation Does NOT Imply Independence

### Counterexample 1: Symmetric Nonlinear Dependence

Let $X \sim N(0, 1)$ and $Y = X^2$. Then $Y$ is completely determined by $X$ (maximal dependence), yet:

$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = E[X^3] - 0 \cdot E[X^2] = 0
$$

since $E[X^3] = 0$ by the symmetry of the standard normal distribution.

**Why this works:** Correlation measures only **linear** dependence. The relationship $Y = X^2$ is perfectly nonlinear and symmetric, so positive and negative deviations cancel in the covariance calculation.

### Counterexample 2: Discrete Example

Let $X \sim \text{Uniform}\{-1, 0, 1\}$ and $Y = |X|$:

| $X$ | $Y = \|X\|$ | $P$ |
|:---:|:---:|:---:|
| $-1$ | $1$ | $1/3$ |
| $0$ | $0$ | $1/3$ |
| $1$ | $1$ | $1/3$ |

$$
E[X] = 0, \quad E[Y] = \tfrac{2}{3}, \quad E[XY] = (-1)(1)\tfrac{1}{3} + 0 + (1)(1)\tfrac{1}{3} = 0
$$

$$
\text{Cov}(X,Y) = 0 - 0 \cdot \tfrac{2}{3} = 0
$$

But $X$ and $Y$ are **not independent**: $P(Y = 0 \mid X = 0) = 1 \neq P(Y = 0) = 1/3$.

### Counterexample 3: Unit Circle

Let $(X, Y)$ be uniformly distributed on the unit circle. Then $\text{Cov}(X,Y) = 0$ by symmetry, but $X^2 + Y^2 = 1$ makes them perfectly dependent.

---

## When Are They Equivalent?

### Jointly Normal Variables

**Theorem:** If $(X, Y)$ follow a **bivariate normal distribution**, then:

$$
\text{Cov}(X, Y) = 0 \iff X \perp Y
$$

This is a special and very important property of the multivariate normal distribution. The bivariate normal PDF is:

$$
f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)
$$

When $\rho = 0$, the cross term vanishes and the joint PDF factors into the product of two marginal normal PDFs.

### Binary Random Variables

For random variables taking only two values each, zero covariance also implies independence.

---

## Summary Diagram

$$
\boxed{
\text{Independence} \implies \text{Zero Correlation} \implies E[XY] = E[X]E[Y]
}
$$

$$
\text{Zero Correlation} \;\not\!\!\!\implies \text{Independence} \quad \text{(in general)}
$$

$$
\text{Zero Correlation} \iff \text{Independence} \quad \text{(for jointly normal variables)}
$$

---

## Hierarchy of Dependence Concepts

From strongest to weakest:

$$
\begin{aligned}
&\textbf{Functional dependence:} \quad Y = g(X) \\[4pt]
&\textbf{Statistical dependence:} \quad f_{X,Y} \neq f_X \cdot f_Y \\[4pt]
&\textbf{Correlation:} \quad \rho \neq 0 \\[4pt]
&\textbf{Uncorrelated:} \quad \rho = 0 \\[4pt]
&\textbf{Independence:} \quad f_{X,Y} = f_X \cdot f_Y
\end{aligned}
$$

Correlation detects linear patterns. Dependence can take any form. A variable can be functionally dependent on another yet uncorrelated (as shown in the counterexamples).

---

## Python: Demonstrating the Distinction

### Uncorrelated but Dependent: $Y = X^2$

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100_000
X = np.random.normal(0, 1, n)
Y = X**2

corr = np.corrcoef(X, Y)[0, 1]
print(f"Correlation(X, X²) = {corr:.6f}")   # ≈ 0 (uncorrelated)
print(f"But Y is completely determined by X!")  # dependent

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X[:2000], Y[:2000], s=2, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y = X²')
ax.set_title(f'Uncorrelated (ρ={corr:.4f}) but Dependent')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

### Independence Test: Comparing Joint vs Product of Marginals

```python
import numpy as np

np.random.seed(42)
n = 100_000
X = np.random.normal(0, 1, n)
Y = X**2

# If independent: P(X>0, Y>1) = P(X>0) * P(Y>1)
p_joint = np.mean((X > 0) & (Y > 1))
p_x = np.mean(X > 0)
p_y = np.mean(Y > 1)

print(f"P(X>0, Y>1) = {p_joint:.4f}")
print(f"P(X>0) × P(Y>1) = {p_x * p_y:.4f}")
print(f"Equal? {np.isclose(p_joint, p_x * p_y, atol=0.01)}")
print("→ Joint ≠ product of marginals → NOT independent")
```

### Jointly Normal: Zero Correlation ↔ Independence

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100_000

# Correlated normals
rho = 0.8
cov = [[1, rho], [rho, 1]]
corr_data = np.random.multivariate_normal([0, 0], cov, n)

# Uncorrelated normals (independent)
indep_data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in zip(axes,
    [corr_data, indep_data],
    [f'ρ={rho} (correlated, dependent)', 'ρ=0 (uncorrelated, independent)']):
    ax.scatter(data[:2000, 0], data[:2000, 1], s=2, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

# Verify independence for uncorrelated normals
p_joint = np.mean((indep_data[:, 0] > 1) & (indep_data[:, 1] > 1))
p_prod = np.mean(indep_data[:, 0] > 1) * np.mean(indep_data[:, 1] > 1)
print(f"Jointly normal, ρ=0:")
print(f"  P(X>1,Y>1) = {p_joint:.4f}, P(X>1)P(Y>1) = {p_prod:.4f}")
print(f"  Independent? {np.isclose(p_joint, p_prod, atol=0.005)}")
```

---

## Key Takeaways

- Independence is a stronger condition than zero correlation: independence implies zero correlation, but not vice versa.
- Correlation captures only **linear** relationships; variables with nonlinear dependence (e.g., $Y = X^2$) can have zero correlation.
- For jointly normal random variables, zero correlation **does** imply independence — a unique and powerful property.
- Always consider whether the assumption of joint normality holds before equating uncorrelated with independent.
- In practice, checking independence requires examining the full joint distribution, not just the correlation coefficient.
