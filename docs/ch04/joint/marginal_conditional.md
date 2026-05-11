# Marginal and Conditional Distributions

## Overview

Given a joint distribution of two random variables, the **marginal distribution** recovers the distribution of each variable individually, while the **conditional distribution** describes one variable given a specific value of the other. These concepts are essential for Bayesian reasoning, regression, and understanding dependence.

---

## Marginal Distributions

### Discrete Case

From the joint PMF $p_{X,Y}(x,y)$, the marginal PMFs are obtained by summing over the other variable:

$$
p_X(x) = \sum_y p_{X,Y}(x, y), \qquad p_Y(y) = \sum_x p_{X,Y}(x, y)
$$

### Continuous Case

From the joint PDF $f_{X,Y}(x,y)$, the marginal PDFs are:

$$
f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dy, \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dx
$$

**Intuition:** Marginalizing "integrates out" the other variable, projecting the joint distribution onto a single axis.

---

## Conditional Distributions

### Discrete Case

The conditional PMF of $Y$ given $X = x$ is:

$$
p_{Y|X}(y \mid x) = \frac{p_{X,Y}(x, y)}{p_X(x)}, \qquad p_X(x) > 0
$$

### Continuous Case

The conditional PDF of $Y$ given $X = x$ is:

$$
f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x, y)}{f_X(x)}, \qquad f_X(x) > 0
$$

### Conditional Expectation

$$
E[Y \mid X = x] = \begin{cases} \sum_y y \cdot p_{Y|X}(y \mid x) & \text{(discrete)} \\ \int_{-\infty}^{\infty} y \cdot f_{Y|X}(y \mid x)\,dy & \text{(continuous)} \end{cases}
$$

### Conditional Variance

$$
\text{Var}(Y \mid X = x) = E[Y^2 \mid X = x] - (E[Y \mid X = x])^2
$$

---

## Fundamental Relationships

### Multiplication Rule

The joint distribution can always be factored as:

$$
f_{X,Y}(x, y) = f_{Y|X}(y \mid x) \cdot f_X(x) = f_{X|Y}(x \mid y) \cdot f_Y(y)
$$

### Law of Total Expectation

$$
E[Y] = E[E[Y \mid X]] = \begin{cases} \sum_x E[Y \mid X = x] \cdot p_X(x) & \text{(discrete)} \\ \int E[Y \mid X = x] \cdot f_X(x)\,dx & \text{(continuous)} \end{cases}
$$

### Law of Total Variance (Eve's Law)

$$
\text{Var}(Y) = E[\text{Var}(Y \mid X)] + \text{Var}(E[Y \mid X])
$$

The total variance decomposes into the mean of conditional variances (unexplained variance) plus the variance of conditional means (explained variance).

---

## Bayes' Theorem for Distributions

Combining the multiplication rule and marginal distributions yields Bayes' theorem:

$$
f_{X|Y}(x \mid y) = \frac{f_{Y|X}(y \mid x) \cdot f_X(x)}{f_Y(y)} = \frac{f_{Y|X}(y \mid x) \cdot f_X(x)}{\int f_{Y|X}(y \mid x) \cdot f_X(x)\,dx}
$$

This is the foundation of Bayesian inference: update the prior $f_X(x)$ with the likelihood $f_{Y|X}(y \mid x)$ to obtain the posterior $f_{X|Y}(x \mid y)$.

---

## Worked Example: Discrete

**Problem:** Using the joint PMF:

| | $Y=0$ | $Y=1$ | $Y=2$ | $p_X(x)$ |
|:---|:---:|:---:|:---:|:---:|
| $X=0$ | 0.10 | 0.15 | 0.05 | 0.30 |
| $X=1$ | 0.10 | 0.25 | 0.10 | 0.45 |
| $X=2$ | 0.05 | 0.10 | 0.10 | 0.25 |
| $p_Y(y)$ | 0.25 | 0.50 | 0.25 | 1.00 |

Find $P(Y = 1 \mid X = 1)$ and $E[Y \mid X = 1]$.

**Solution:**

$$
P(Y = 1 \mid X = 1) = \frac{p_{X,Y}(1,1)}{p_X(1)} = \frac{0.25}{0.45} = \frac{5}{9} \approx 0.556
$$

$$
E[Y \mid X = 1] = 0 \cdot \frac{0.10}{0.45} + 1 \cdot \frac{0.25}{0.45} + 2 \cdot \frac{0.10}{0.45} = \frac{0.45}{0.45} = 1.0
$$

---

## Worked Example: Continuous

**Problem:** Let $f_{X,Y}(x,y) = 2$ for $0 \leq x \leq y \leq 1$. Find $f_X(x)$, $f_{Y|X}(y \mid x)$, and $E[Y \mid X = x]$.

**Solution:**

**Marginal of $X$:**

$$
f_X(x) = \int_x^1 2\,dy = 2(1 - x), \quad 0 \leq x \leq 1
$$

**Conditional PDF of $Y$ given $X = x$:**

$$
f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x,y)}{f_X(x)} = \frac{2}{2(1-x)} = \frac{1}{1-x}, \quad x \leq y \leq 1
$$

This is $\text{Uniform}(x, 1)$.

**Conditional expectation:**

$$
E[Y \mid X = x] = \frac{x + 1}{2}
$$

**Verification via Law of Total Expectation:**

$$
E[Y] = \int_0^1 \frac{x+1}{2} \cdot 2(1-x)\,dx = \int_0^1 (x+1)(1-x)\,dx = \int_0^1 (1 - x^2)\,dx = \frac{2}{3}
$$

---

## Python: Marginal and Conditional Distributions

### Discrete Marginals and Conditionals

```python
import numpy as np
import pandas as pd

pmf = np.array([
    [0.10, 0.15, 0.05],
    [0.10, 0.25, 0.10],
    [0.05, 0.10, 0.10]
])

# Marginals
p_X = pmf.sum(axis=1)
p_Y = pmf.sum(axis=0)
print("Marginal of X:", p_X)
print("Marginal of Y:", p_Y)

# Conditional P(Y | X=1)
x_val = 1
cond_Y_given_X1 = pmf[x_val, :] / p_X[x_val]
print(f"\nP(Y|X={x_val}):", cond_Y_given_X1)

# Conditional expectation E[Y | X=1]
y_vals = np.array([0, 1, 2])
E_Y_given_X1 = np.sum(y_vals * cond_Y_given_X1)
print(f"E[Y|X={x_val}] = {E_Y_given_X1:.4f}")
```

### Continuous Marginals via Integration

```python
import numpy as np
from scipy import integrate

# f(x,y) = 2 for 0 <= x <= y <= 1
def joint_pdf(x, y):
    return 2.0 if 0 <= x <= y <= 1 else 0.0

# Marginal f_X(x) = integral of f(x,y) dy from x to 1
def marginal_X(x):
    result, _ = integrate.quad(lambda y: joint_pdf(x, y), x, 1)
    return result

# E[Y | X=x] via conditional
def E_Y_given_X(x):
    fx = marginal_X(x)
    if fx == 0:
        return 0
    result, _ = integrate.quad(lambda y: y * joint_pdf(x, y) / fx, x, 1)
    return result

# Verify Law of Total Expectation
E_Y, _ = integrate.quad(lambda x: E_Y_given_X(x) * marginal_X(x), 0, 1)
print(f"E[Y] via Law of Total Expectation: {E_Y:.4f}")  # Should be 2/3
```

### Visualizing Conditional Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
samples = np.random.multivariate_normal(mean, cov, 100_000)

fig, ax = plt.subplots(figsize=(12, 3))

# Conditional distribution of Y given X ≈ 1
for x_cond in [-1, 0, 1]:
    mask = np.abs(samples[:, 0] - x_cond) < 0.1
    ax.hist(samples[mask, 1], bins=50, density=True, alpha=0.4,
            label=f'Y | X≈{x_cond}')

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Y')
ax.legend()
plt.show()
```

---

## Key Takeaways

- Marginal distributions are obtained by summing or integrating the joint distribution over the other variable.
- Conditional distributions describe one variable given a known value of another, computed as the joint divided by the marginal.
- The Law of Total Expectation and Law of Total Variance connect marginal and conditional moments.
- The multiplication rule $f_{X,Y} = f_{Y|X} \cdot f_X$ provides the foundation for Bayes' theorem and Bayesian inference.
- Conditional expectation $E[Y \mid X]$ is itself a random variable (a function of $X$) and represents the best prediction of $Y$ given $X$.

## Exercises

**Exercise 1.**
Joint PDF $f(x, y) = 6(1 - y)$ for $0 \le x \le y \le 1$. (a) Verify $\int f = 1$. (b) Find $f_Y$. (c) Find $f_{X \mid Y}$. (d) Compute $\mathbb{E}[X \mid Y = y]$.

??? success "Solution to Exercise 1"
    (a) $\int_0^1 \int_0^y 6(1-y) dx \, dy = \int_0^1 6y(1-y) dy = 1$. ✓

    (b) $f_Y(y) = \int_0^y 6(1-y) dx = 6y(1-y)$ on $[0, 1]$. (This is $\mathrm{Beta}(2, 2)$.)

    (c) $f_{X \mid Y}(x \mid y) = 6(1-y)/[6y(1-y)] = 1/y$ on $[0, y]$. So $X \mid Y = y \sim \mathrm{Uniform}(0, y)$.

    (d) $\mathbb{E}[X \mid Y = y] = y/2$.

---

**Exercise 2.**
**Law of total expectation.** Use Exercise 1's distribution to compute $\mathbb{E}[X]$ via $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$ and verify directly.

??? success "Solution to Exercise 2"
    By iteration: $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[Y/2] = \mathbb{E}[Y]/2$.

    $\mathbb{E}[Y] = \int_0^1 y \cdot 6y(1-y) dy = 6\int_0^1(y^2 - y^3) dy = 6(1/3 - 1/4) = 1/2$.

    So $\mathbb{E}[X] = 1/4$.

    **Direct verification:** $\mathbb{E}[X] = \int_0^1 \int_0^y x \cdot 6(1-y) dx \, dy = \int_0^1 3 y^2 (1-y) dy = 3(1/3 - 1/4) = 1/4$. ✓

    The two methods agree, confirming the law of total expectation. The iterated approach is often computationally easier.

---

**Exercise 3.**
**Law of total variance.** Derive the formula $\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])$. Apply to Exercise 1.

??? success "Solution to Exercise 3"
    **Derivation:**

    $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.

    $\mathbb{E}[X^2] = \mathbb{E}[\mathbb{E}[X^2 \mid Y]] = \mathbb{E}[\mathrm{Var}(X \mid Y) + (\mathbb{E}[X \mid Y])^2]$.

    So $\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathbb{E}[(\mathbb{E}[X \mid Y])^2] - (\mathbb{E}[X])^2 = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])$. $\square$

    **Application:** $\mathrm{Var}(X \mid Y = y) = y^2/12$ (variance of Uniform(0, y)). So $\mathbb{E}[\mathrm{Var}(X \mid Y)] = \mathbb{E}[Y^2]/12$.

    $\mathbb{E}[Y^2] = \int_0^1 y^2 \cdot 6y(1-y) dy = 6\int_0^1(y^3 - y^4) dy = 6(1/4 - 1/5) = 3/10$.

    $\mathrm{Var}(\mathbb{E}[X \mid Y]) = \mathrm{Var}(Y/2) = \mathrm{Var}(Y)/4 = (3/10 - 1/4)/4 = (1/20)/4 = 1/80$.

    $\mathrm{Var}(X) = (3/10)/12 + 1/80 = 1/40 + 1/80 = 3/80$.

    The decomposition splits total variance into the "within-group" component $\mathbb{E}[\mathrm{Var}(X \mid Y)]$ and the "between-group" component $\mathrm{Var}(\mathbb{E}[X \mid Y])$ — the basis of ANOVA.

---

**Exercise 4.**
**Marginal can be misleading.** Construct an example where the marginal distribution of $X$ is symmetric, but the conditional $X \mid Y = y$ is asymmetric for every $y$.

??? success "Solution to Exercise 4"
    Take $Y \sim \mathrm{Bernoulli}(0.5)$ and:

    - $X \mid Y = 0 \sim \mathrm{Exp}(1)$ (right-skewed, support $[0, \infty)$).
    - $X \mid Y = 1 \sim -\mathrm{Exp}(1)$ (left-skewed, support $(-\infty, 0]$).

    Marginal of $X$: $f_X(x) = 0.5 \cdot \mathbf 1\{x \ge 0\} e^{-x} + 0.5 \cdot \mathbf 1\{x \le 0\} e^x$, which is the **Laplace distribution** — symmetric around 0.

    Yet conditional on either value of $Y$, $X$ is highly asymmetric. Mixture of two asymmetric distributions can produce symmetric marginal.

    **Lesson:** marginal distributions hide structure. In particular, modeling only the marginal of $X$ without considering the conditioning variable can give a misleading picture of the underlying mechanism. Always think about which variables to condition on.

---

**Exercise 5.**
**Bivariate normal marginals and conditionals.** $(X, Y)$ bivariate normal with means $(\mu_X, \mu_Y)$, variances $(\sigma_X^2, \sigma_Y^2)$, correlation $\rho$. State the marginal of $X$ and conditional $Y \mid X = x$.

??? success "Solution to Exercise 5"
    **Marginal:** $X \sim N(\mu_X, \sigma_X^2)$. Marginals of jointly normal are normal (a property of the multivariate normal).

    **Conditional:**

    $$
    Y \mid X = x \sim N\!\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X), \, \sigma_Y^2(1 - \rho^2)\right)
    $$

    Key observations:

    - Conditional mean is **linear** in $x$. This is the regression line $\mathbb{E}[Y \mid X = x] = \alpha + \beta x$ with $\beta = \rho \sigma_Y/\sigma_X$.
    - Conditional variance does *not* depend on $x$: **homoscedasticity**. A distinguishing feature of bivariate normal.
    - Conditional variance is reduced by factor $(1 - \rho^2)$: knowing $X$ explains fraction $\rho^2$ of the variance. This is exactly $R^2$.

    Linear regression theory in introductory statistics is essentially derived from these formulas under the bivariate-normal assumption.

---

**Exercise 6.**
**Bayes' theorem in continuous form.** State the version of Bayes for densities and derive the posterior $\pi(\theta \mid x)$ from prior $\pi(\theta)$ and likelihood $f(x \mid \theta)$.

??? success "Solution to Exercise 6"
    **Bayes for densities:**

    $$
    \pi(\theta \mid x) = \frac{f(x \mid \theta) \pi(\theta)}{\int f(x \mid \theta) \pi(\theta) d\theta} = \frac{f(x \mid \theta) \pi(\theta)}{f(x)}
    $$

    The denominator $f(x) = \int f(x \mid \theta) \pi(\theta) d\theta$ is the **marginal likelihood** or **evidence**, often denoted $Z$ in machine learning contexts.

    **In words:** posterior is proportional to likelihood times prior. The proportionality constant ensures normalization.

    **Common shorthand:** $\pi(\theta \mid x) \propto f(x \mid \theta) \pi(\theta)$. Computing $Z$ is often the hard part (typically requires numerical integration); proportionality is sometimes all that's needed (e.g., for MCMC sampling which doesn't require $Z$).

    Bayesian inference uses this formula iteratively as data arrives: prior → posterior (after data 1) → posterior (after data 1, 2) → ..., each time treating the previous posterior as the new prior.
