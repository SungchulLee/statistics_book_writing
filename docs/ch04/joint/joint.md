# Joint Distributions

## Overview

A **joint distribution** describes the probabilistic behavior of two or more random variables simultaneously. While individual (marginal) distributions tell us about each variable in isolation, joint distributions capture how variables relate to and depend on each other.

---

## Joint PMF (Discrete Case)

For discrete random variables $X$ and $Y$, the **joint probability mass function** is:

$$
p_{X,Y}(x, y) = P(X = x, Y = y)
$$

### Requirements

$$
\begin{aligned}
(1) &\quad p_{X,Y}(x, y) \geq 0 \quad \text{for all } (x, y) \\
(2) &\quad \sum_x \sum_y p_{X,Y}(x, y) = 1
\end{aligned}
$$

### Computing Probabilities

For any region $A \subseteq \mathbb{R}^2$:

$$
P((X, Y) \in A) = \sum_{(x,y) \in A} p_{X,Y}(x, y)
$$

---

## Joint PDF (Continuous Case)

For continuous random variables $X$ and $Y$, the **joint probability density function** $f_{X,Y}(x, y)$ satisfies:

$$
P((X, Y) \in A) = \iint_A f_{X,Y}(x, y)\,dx\,dy
$$

### Requirements

$$
\begin{aligned}
(1) &\quad f_{X,Y}(x, y) \geq 0 \quad \text{for all } (x, y) \\
(2) &\quad \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dx\,dy = 1
\end{aligned}
$$

### Joint CDF

$$
F_{X,Y}(x, y) = P(X \leq x, Y \leq y) = \int_{-\infty}^x \int_{-\infty}^y f_{X,Y}(s, t)\,dt\,ds
$$

The PDF is recovered by differentiation:

$$
f_{X,Y}(x, y) = \frac{\partial^2}{\partial x \, \partial y} F_{X,Y}(x, y)
$$

---

## Independence

Two random variables $X$ and $Y$ are **independent** if and only if the joint distribution factors:

$$
\text{Discrete:} \quad p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y
$$

$$
\text{Continuous:} \quad f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y) \quad \text{for all } x, y
$$

Equivalently, $F_{X,Y}(x,y) = F_X(x) \cdot F_Y(y)$ for all $x, y$.

**Key implication:** Under independence, $E[g(X)h(Y)] = E[g(X)] \cdot E[h(Y)]$ for any functions $g, h$.

---

## Expectations from Joint Distributions

For a function $g(X, Y)$:

$$
\text{Discrete:} \quad E[g(X,Y)] = \sum_x \sum_y g(x,y) \cdot p_{X,Y}(x,y)
$$

$$
\text{Continuous:} \quad E[g(X,Y)] = \int\!\!\int g(x,y) \cdot f_{X,Y}(x,y)\,dx\,dy
$$

### Linearity (always holds)

$$
E[aX + bY + c] = aE[X] + bE[Y] + c
$$

### Variance of a Sum

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)
$$

If $X \perp Y$: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

---

## Worked Example: Discrete Joint Distribution

**Problem:** Two assets $X$ and $Y$ have the following joint PMF:

| | $Y=0$ | $Y=1$ | $Y=2$ |
|:---|:---:|:---:|:---:|
| $X=0$ | 0.10 | 0.15 | 0.05 |
| $X=1$ | 0.10 | 0.25 | 0.10 |
| $X=2$ | 0.05 | 0.10 | 0.10 |

Compute $P(X + Y \leq 2)$ and $E[XY]$.

**Solution:**

$$
P(X+Y \leq 2) = p(0,0) + p(0,1) + p(0,2) + p(1,0) + p(1,1) + p(2,0) = 0.10 + 0.15 + 0.05 + 0.10 + 0.25 + 0.05 = 0.70
$$

$$
E[XY] = \sum_x \sum_y xy \cdot p(x,y) = 0 + 0 + 0 + 0 + 1(1)(0.25) + 1(2)(0.10) + 0 + 2(1)(0.10) + 2(2)(0.10) = 0.85
$$

---

## Worked Example: Continuous Joint Distribution

**Problem:** Let $f_{X,Y}(x,y) = 6(1-y)$ for $0 \leq x \leq y \leq 1$. Verify this is a valid PDF and find $P(X < 1/2, Y < 1/2)$.

**Solution:** Verification:

$$
\int_0^1 \int_0^y 6(1-y)\,dx\,dy = \int_0^1 6y(1-y)\,dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6\left(\frac{1}{2} - \frac{1}{3}\right) = 1 \checkmark
$$

$$
P\left(X < \tfrac{1}{2}, Y < \tfrac{1}{2}\right) = \int_0^{1/2}\!\int_0^y 6(1-y)\,dx\,dy = \int_0^{1/2} 6y(1-y)\,dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^{1/2} = \frac{5}{8} \cdot \frac{6}{8} = \frac{5}{16}
$$

Wait — let us recompute carefully:

$$
\int_0^{1/2} 6y(1-y)\,dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^{1/2} = 6\left(\frac{1}{8} - \frac{1}{24}\right) = 6 \cdot \frac{2}{24} = \frac{1}{2}
$$

So $P(X < 1/2, Y < 1/2) = 1/2$.

---

## Python: Joint Distributions

### Discrete Joint PMF Table

```python
import numpy as np
import pandas as pd

# Joint PMF as a 2D array
pmf = np.array([
    [0.10, 0.15, 0.05],
    [0.10, 0.25, 0.10],
    [0.05, 0.10, 0.10]
])

df = pd.DataFrame(pmf, index=['X=0', 'X=1', 'X=2'], columns=['Y=0', 'Y=1', 'Y=2'])
df['P(X=x)'] = pmf.sum(axis=1)
df.loc['P(Y=y)'] = pmf.sum(axis=0).tolist() + [1.0]
print(df)
```

### Continuous Joint PDF Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)

# f(x,y) = 6(1-y) for 0 <= x <= y <= 1
Z = np.where(X <= Y, 6 * (1 - Y), 0)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
fig.colorbar(c, ax=ax, label='f(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Joint PDF: f(x,y) = 6(1-y)')
plt.show()
```

### Bivariate Normal Sampling

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]
samples = np.random.multivariate_normal(mean, cov, 5000)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Key Takeaways

- Joint distributions describe the simultaneous behavior of multiple random variables.
- The joint PMF/PDF must be non-negative and sum/integrate to 1 over the entire support.
- Independence is equivalent to the joint distribution factoring into the product of marginals.
- Expectations of functions of multiple variables are computed by summing or integrating against the joint distribution.
- The variance of a sum depends on the covariance between variables; independence simplifies this to a simple sum of variances.
