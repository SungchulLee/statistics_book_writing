# Kolmogorov-Smirnov Test

## Overview

The Kolmogorov-Smirnov (KS) test is a nonparametric test that compares the empirical distribution function of a sample to a specified theoretical distribution. For normality testing, it measures the maximum vertical distance between the empirical CDF and the normal CDF. A critical caveat is that the standard KS test requires the null-hypothesis parameters to be fully specified in advance; estimating them from the data invalidates the $p$-value.

## The Empirical Distribution Function

Given an i.i.d. sample $X_1, \ldots, X_n$, the empirical distribution function (EDF) is

$$
F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(X_i \leq x),
$$

where $\mathbf{1}(\cdot)$ is the indicator function. By the Glivenko-Cantelli theorem, $F_n(x) \to F(x)$ uniformly almost surely as $n \to \infty$.

## The KS Statistic

The one-sample KS statistic measures the supremum distance between $F_n$ and the hypothesised CDF $F_0$:

$$
D_n = \sup_x |F_n(x) - F_0(x)|.
$$

In practice, $D_n$ is computed as

$$
D_n = \max_{1 \leq i \leq n} \max\!\left(\left|\frac{i}{n} - F_0(X_{(i)})\right|,\; \left|F_0(X_{(i)}) - \frac{i-1}{n}\right|\right),
$$

where $X_{(1)} \leq \cdots \leq X_{(n)}$ are the order statistics.

## Hypotheses

$$
H_0: F = F_0 \quad (\text{data follow the specified distribution}), \qquad H_1: F \neq F_0.
$$

Under $H_0$ with $F_0$ completely specified, the distribution of $\sqrt{n}\, D_n$ converges to the Kolmogorov distribution, whose CDF is

$$
P(\sqrt{n}\, D_n \leq t) \to 1 - 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2 t^2}.
$$

### Code

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.normal(0.0, 1.0, size=250)

# Fully specified H0: Normal(0, 1)
D, p = stats.kstest(x, 'norm', args=(0.0, 1.0))

print(f"n = {x.size}")
print(f"KS one-sample vs N(0,1): D = {D:.4f}, p = {p:.4g}")
if p < 0.05:
    print("=> Reject H0: data may not follow N(0,1).")
else:
    print("=> Fail to reject H0 at alpha = 0.05.")
```

## The Lilliefors Problem

If $\mu$ and $\sigma$ are estimated from the data and plugged into $F_0$, the KS statistic becomes systematically smaller because the fitted CDF is optimised to be close to $F_n$. The standard KS critical values and $p$-values are then invalid (too conservative, leading to under-rejection). The Lilliefors test addresses this issue using simulation-based or tabulated critical values that account for parameter estimation.

## Interpretation

The KS test is a *consistent* test: as $n \to \infty$, it will eventually detect any departure from $F_0$. However, it has relatively low power compared to tests that exploit the specific structure of the normal distribution (e.g., Shapiro-Wilk, Anderson-Darling). The KS test treats all portions of the distribution equally, whereas the Anderson-Darling test gives extra weight to the tails, making it more sensitive to tail departures.

## Exercises

**Exercise 1.** Generate $n = 300$ observations from $\mathcal{N}(0,1)$. Run the KS test against $\mathcal{N}(0,1)$ and against $\mathcal{N}(0.5, 1)$. Compare the $p$-values.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=300)

    D1, p1 = stats.kstest(x, 'norm', args=(0.0, 1.0))
    D2, p2 = stats.kstest(x, 'norm', args=(0.5, 1.0))

    print(f"vs N(0,1):   D = {D1:.4f}, p = {p1:.4g}")
    print(f"vs N(0.5,1): D = {D2:.4f}, p = {p2:.4g}")
    ```

    Against $\mathcal{N}(0,1)$ (the true distribution), the test should not reject ($p > 0.05$). Against $\mathcal{N}(0.5, 1)$ (wrong mean), the ECDF is systematically shifted, producing a larger $D$ and a small $p$-value indicating rejection. $\square$

---

**Exercise 2.** Demonstrate the Lilliefors problem: generate $n = 200$ observations from $\mathcal{N}(0,1)$, estimate $\hat{\mu}$ and $\hat{\sigma}$, and run the KS test against $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$. Is the $p$-value trustworthy?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    mu_hat, sigma_hat = x.mean(), x.std(ddof=1)
    D, p = stats.kstest(x, 'norm', args=(mu_hat, sigma_hat))

    print(f"Estimated mu = {mu_hat:.4f}, sigma = {sigma_hat:.4f}")
    print(f"KS vs N(mu_hat, sigma_hat): D = {D:.4f}, p = {p:.4g}")
    ```

    The $p$-value will be inflated (too large) because the fitted normal is closer to the ECDF than a pre-specified normal would be. Repeated simulation would show that the empirical rejection rate at $\alpha = 0.05$ is well below 5%, demonstrating that the test is conservative and under-rejects. The Lilliefors correction is needed for valid inference. $\square$

---

**Exercise 3.** Derive the formula for computing $D_n$ from order statistics. Show that only the $n$ values $X_{(1)}, \ldots, X_{(n)}$ need to be checked.

??? success "Solution to Exercise 3"

    The ECDF $F_n(x)$ is a step function that jumps by $1/n$ at each $X_{(i)}$. Between consecutive order statistics, $F_n$ is constant while $F_0$ is monotonically increasing. Therefore the supremum $|F_n(x) - F_0(x)|$ is attained either just before or just after a jump:

    - Just after the $i$-th jump: $F_n(X_{(i)}) = i/n$, giving $|i/n - F_0(X_{(i)})|$.
    - Just before the $i$-th jump: $F_n(X_{(i)}^-) = (i-1)/n$, giving $|(i-1)/n - F_0(X_{(i)})|$.

    Hence

    $$
    D_n = \max_{1 \leq i \leq n} \max\!\left(\left|\frac{i}{n} - F_0(X_{(i)})\right|,\; \left|F_0(X_{(i)}) - \frac{i-1}{n}\right|\right).
    $$

    Only the $n$ order statistics need to be evaluated. $\square$

---

**Exercise 4.** Run a Monte Carlo simulation to estimate the power of the KS test at $\alpha = 0.05$ for $n = 200$ against a $t_5$ alternative (testing against $\mathcal{N}(0,1)$).

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 200, 5000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.standard_t(df=5, size=n)
        _, p = stats.kstest(x, 'norm', args=(0.0, 1.0))
        if p < alpha:
            rejections += 1

    print(f"KS power vs t(5): {rejections / reps:.4f}")
    ```

    The power is typically moderate (around 0.3--0.5). The $t_5$ distribution has the same mean and nearly the same variance as $\mathcal{N}(0,1)$ (variance is $5/3 \approx 1.67$, not 1, so the difference is detectable), but the KS test is not particularly sensitive to tail departures. The Anderson-Darling test would have higher power here. $\square$

---

**Exercise 5.** State and prove the Glivenko-Cantelli theorem in the one-dimensional case (outline of proof is acceptable).

??? success "Solution to Exercise 5"

    **Theorem (Glivenko-Cantelli).** Let $X_1, X_2, \ldots$ be i.i.d. with CDF $F$. Then

    $$
    \sup_x |F_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad \text{as } n \to \infty.
    $$

    **Proof outline.** Fix any $x \in \mathbb{R}$. By the strong law of large numbers, $F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(X_i \leq x) \to F(x)$ a.s. This gives pointwise convergence. To upgrade to uniform convergence, fix $\epsilon > 0$ and choose points $-\infty = t_0 < t_1 < \cdots < t_K = \infty$ such that $F(t_j) - F(t_{j-1}) < \epsilon$ for all $j$. For any $x \in [t_{j-1}, t_j]$:

    $$
    F_n(x) - F(x) \leq F_n(t_j) - F(t_{j-1}) = [F_n(t_j) - F(t_j)] + [F(t_j) - F(t_{j-1})] < [F_n(t_j) - F(t_j)] + \epsilon.
    $$

    Similarly, $F(x) - F_n(x) < [F(t_j) - F_n(t_{j-1})] + \epsilon$. Since there are finitely many $t_j$, pointwise convergence at each $t_j$ (by SLLN) implies that $\sup_x |F_n(x) - F(x)| < 2\epsilon$ eventually a.s. Since $\epsilon$ was arbitrary, the result follows. $\square$
