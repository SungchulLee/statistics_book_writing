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

## Exercises

**Exercise 1.**
Joint PMF: $p(0,0)=0.10, p(0,1)=0.15, p(0,2)=0.05, p(1,0)=0.10, p(1,1)=0.20, p(1,2)=0.10, p(2,0)=0.05, p(2,1)=0.10, p(2,2)=0.15$. (a) Valid? (b) Marginals. (c) $\mathbb{E}[X], \mathbb{E}[Y], \mathbb{E}[XY]$. (d) $\mathrm{Cov}, \rho$. (e) Independent?

??? success "Solution to Exercise 1"
    (a) Sum = 1, all non-negative ✓.

    (b) Row sums: $p_X(0,1,2) = (0.30, 0.40, 0.30)$. Col sums: $p_Y(0,1,2) = (0.25, 0.45, 0.30)$.

    (c) $\mathbb{E}[X] = 1.0$, $\mathbb{E}[Y] = 1.05$, $\mathbb{E}[XY] = 0.20 + 0.20 + 0.20 + 0.60 = 1.20$.

    (d) $\mathrm{Cov} = 1.20 - 1.0 \cdot 1.05 = 0.15$. $\mathrm{Var}(X) = 0.60$, $\mathrm{Var}(Y) = 0.5475$. $\rho = 0.15/\sqrt{0.60 \cdot 0.5475} \approx 0.262$.

    (e) Not independent: $p(0,0) = 0.10 \ne p_X(0) p_Y(0) = 0.075$.

---

**Exercise 2.**
**Joint CDF.** For continuous $(X, Y)$ with joint PDF $f(x, y)$, define the joint CDF $F(x, y) = P(X \le x, Y \le y)$. Express $f$ in terms of $F$.

??? success "Solution to Exercise 2"
    Joint CDF is the integral of the joint PDF over the lower-left quadrant:

    $$
    F(x, y) = \int_{-\infty}^x \int_{-\infty}^y f(s, t) ds\, dt
    $$

    Differentiating with respect to both variables (assuming sufficient smoothness):

    $$
    f(x, y) = \frac{\partial^2 F(x, y)}{\partial x \partial y}
    $$

    This is the joint analog of the one-dimensional relationship $f = F'$.

    Properties of joint CDF: non-decreasing in both arguments; $F(-\infty, y) = F(x, -\infty) = 0$; $F(\infty, \infty) = 1$; marginal recovered as $F_X(x) = F(x, \infty)$ and $F_Y(y) = F(\infty, y)$.

---

**Exercise 3.**
**Joint distribution from conditionals.** $f_{X|Y}(x \mid y) = (1/y) \mathbf 1\{0 \le x \le y\}$ (uniform on $[0, y]$), and $f_Y(y) = e^{-y}$ for $y \ge 0$. Find the joint and marginal of $X$.

??? success "Solution to Exercise 3"
    Joint: $f_{X, Y}(x, y) = f_{X|Y}(x|y) f_Y(y) = (1/y) e^{-y} \mathbf 1\{0 \le x \le y\}$.

    Marginal of $X$: integrate over $y$ for $y \ge x$:

    $$
    f_X(x) = \int_x^\infty \frac{e^{-y}}{y} dy
    $$

    This integral has no elementary form (it equals $E_1(x)$, the **exponential integral**). For small $x$ it diverges like $-\ln x$; for large $x$ it behaves like $e^{-x}/x$. $X$ has a known but non-elementary distribution.

    **Lesson:** the conditional structure can be simple even when the marginal is complicated. This factorization view is the foundation of hierarchical Bayesian models.

---

**Exercise 4.**
**Sum and difference of independent normals.** $X_1, X_2 \sim N(\mu, \sigma^2)$ i.i.d. Show that $X_1 + X_2$ and $X_1 - X_2$ are independent.

??? success "Solution to Exercise 4"
    Both are normal (linear combinations of normals). Compute covariance:

    $$
    \mathrm{Cov}(X_1 + X_2, X_1 - X_2) = \mathrm{Var}(X_1) - \mathrm{Var}(X_2) + \mathrm{Cov}(X_1, X_2) - \mathrm{Cov}(X_2, X_1) = \sigma^2 - \sigma^2 + 0 - 0 = 0
    $$

    using $\mathrm{Cov}(X_1, X_2) = 0$ by independence and the bilinearity of covariance.

    For *jointly normal* random variables, zero covariance implies independence. Since $(X_1 + X_2, X_1 - X_2)$ is a linear transformation of $(X_1, X_2)$ — bivariate normal — they are jointly normal. Therefore independent.

    **Special property of normal:** this exemplifies how normality preserves linear independence. The sample mean $\bar X$ and the deviations $X_i - \bar X$ are uncorrelated and (because jointly normal) independent — a fact that underlies Student's $t$-distribution derivation.

---

**Exercise 5.**
**Multinomial distribution.** Generalize binomial to $k$ categories: $n$ trials, each independently producing outcome $j$ with probability $p_j$, $\sum_j p_j = 1$. Let $X_j$ = count of outcome $j$. Derive the joint PMF.

??? success "Solution to Exercise 5"
    The probability of any specific sequence of outcomes is $\prod_j p_j^{X_j}$. The number of sequences yielding the count vector $(X_1, \ldots, X_k)$ is the multinomial coefficient $\binom{n}{X_1, X_2, \ldots, X_k} = n!/(X_1! X_2! \cdots X_k!)$.

    Joint PMF:

    $$
    P(X_1 = x_1, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} \prod_{j=1}^k p_j^{x_j}
    $$

    valid for $x_j \ge 0$ and $\sum_j x_j = n$.

    **Marginal of $X_j$** is $\mathrm{Binomial}(n, p_j)$ (lumping categories other than $j$).

    **Covariance** between $X_i, X_j$ (for $i \ne j$): $\mathrm{Cov}(X_i, X_j) = -np_i p_j$. Negative — because increasing one count must come at the expense of others (constraint $\sum = n$).

---

**Exercise 6.**
**Bivariate normal.** State the formula for the joint PDF of $(X, Y)$ bivariate normal with means $\mu_X, \mu_Y$, variances $\sigma_X^2, \sigma_Y^2$, correlation $\rho$. What are the conditional distributions?

??? success "Solution to Exercise 6"
    PDF:

    $$
    f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2(1-\rho^2)}\!\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)
    $$

    **Conditional:** $Y \mid X = x$ is normal with

    $$
    \mathbb{E}[Y \mid X = x] = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X)
    $$

    $$
    \mathrm{Var}(Y \mid X = x) = \sigma_Y^2(1 - \rho^2)
    $$

    **Notable features:**

    - Conditional mean is **linear** in $x$ — this is the *regression line*. The slope $\rho\sigma_Y/\sigma_X$ is exactly the OLS regression coefficient.
    - Conditional variance does *not* depend on $x$ — homoscedasticity.
    - $\rho = 0$ implies independence (special property of jointly normal — not true in general).

    The bivariate normal is the foundation of correlation analysis and the simplest non-trivial example of a multivariate distribution with a closed-form regression structure.
