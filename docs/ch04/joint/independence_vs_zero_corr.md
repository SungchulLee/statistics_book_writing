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

### Uncorrelated but Dependent: Y = X-squared
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

## Exercises

**Exercise 1.**
$X$ symmetric with $\mathbb{E}[X] = 0$, $\mathbb{E}[X^2] = 1$, $\mathbb{E}[X^3] = 0$. $Y = X^2$. (a) $\mathrm{Cov}(X, Y)$. (b) $\rho(X, Y)$. (c) Are they independent?

??? success "Solution to Exercise 1"
    (a) $\mathrm{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] - 0 = 0$.

    (b) $\rho(X, Y) = 0/(\sigma_X \sigma_Y) = 0$.

    (c) **Not independent.** $Y = X^2$ is a deterministic function of $X$. Knowing $X = 3$ determines $Y = 9$. The correlation coefficient detects only the *linear* component of the relationship; the quadratic dependence has zero linear component because positive and negative $X$ values produce identical $Y$ values.

    **Lesson:** zero correlation is necessary but not sufficient for independence. Always plot the data.

---

**Exercise 2.**
**When zero correlation does imply independence.** State the case for which zero correlation guarantees independence, and prove it.

??? success "Solution to Exercise 2"
    **Special case:** if $(X, Y)$ is **jointly normal**, zero correlation implies independence.

    **Proof:** for bivariate normal,

    $$
    f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}\exp\!\left(-\frac{Q(x, y)}{2(1-\rho^2)}\right)
    $$

    where $Q$ contains a cross-term $-2\rho(x - \mu_X)(y - \mu_Y)/(\sigma_X \sigma_Y)$. When $\rho = 0$, this cross-term vanishes and $Q$ factors as a sum of terms involving only $x$ and only $y$. The density factors:

    $$
    f(x, y) = f_X(x) \cdot f_Y(y)
    $$

    which is the definition of independence. $\square$

    **Note:** the assumption of *joint* normality is crucial — both $X$ and $Y$ marginally normal is not enough. Counterexamples exist where $X, Y$ are each $N(0, 1)$ but they are not jointly normal and zero correlation does not imply independence.

---

**Exercise 3.**
**Distance correlation** addresses the limitation of Pearson by being zero if and only if independence holds. Define distance correlation conceptually and state its main advantage.

??? success "Solution to Exercise 3"
    **Distance correlation** (Székely, Rizzo, Bakirov 2007) is a measure of dependence that satisfies:

    $$
    \mathrm{dCor}(X, Y) = 0 \iff X \perp\!\!\!\perp Y
    $$

    **Definition (informal):** compute pairwise distances within $X$-samples and within $Y$-samples, then double-center each distance matrix, and compute a "correlation" between the centered distance matrices.

    **Advantages over Pearson:**

    - Detects **nonlinear** dependence (e.g., $Y = X^2$).
    - Detects dependence even when $X$ and $Y$ have different dimensions.
    - Always lies in $[0, 1]$ (like absolute Pearson, but always non-negative).
    - $\mathrm{dCor} = 0$ iff independent (full diagnostic, unlike Pearson).

    **Cost:** computational complexity $O(n^2)$ in sample size, vs. $O(n)$ for Pearson. For exploratory work with $n < 10^4$, distance correlation is increasingly the recommended dependence measure.

---

**Exercise 4.**
**Mutual information** $I(X; Y) = \mathbb{E}\!\left[\log \frac{p(X, Y)}{p(X) p(Y)}\right]$. Show $I(X; Y) \ge 0$ and $I(X; Y) = 0$ iff $X \perp\!\!\!\perp Y$.

??? success "Solution to Exercise 4"
    Mutual information is the **Kullback-Leibler divergence** between the joint distribution $p(X, Y)$ and the product of marginals $p(X) p(Y)$:

    $$
    I(X; Y) = D_{KL}(p(X, Y) \| p(X) p(Y))
    $$

    KL divergence is non-negative ($D_{KL}(p \| q) \ge 0$ by Jensen's inequality applied to the convex function $-\log$), with equality iff $p = q$ almost everywhere.

    So $I(X; Y) \ge 0$, with equality iff $p(X, Y) = p(X) p(Y)$ — which is exactly independence.

    $\square$

    **Use:** mutual information is the information-theoretic measure of dependence. Detects arbitrary nonlinear and high-order dependencies. Estimating $I$ from data is challenging (especially for continuous variables) but methods exist (k-nearest-neighbor estimators, kernel density estimation, mutual-information neural estimation).

---

**Exercise 5.**
**Rank correlation.** Spearman's $\rho_S$ is the Pearson correlation between ranks. Show that Spearman's $\rho_S$ captures *monotonic* dependence (unlike Pearson, which captures only linear), and is invariant to any monotonic transformation.

??? success "Solution to Exercise 5"
    Replace each $X_i$ with its rank $R_i^X$ (from 1 to $n$), and similarly for $Y_i$. Spearman's correlation is

    $$
    \rho_S = \frac{\mathrm{Cov}(R^X, R^Y)}{\sigma_{R^X} \sigma_{R^Y}}
    $$

    **Monotonic dependence:** if $Y = g(X)$ for monotonic increasing $g$, then $R^Y = R^X$ exactly (ranks preserved), so $\rho_S = 1$. If $g$ is monotonic decreasing, $R^Y = n + 1 - R^X$, so $\rho_S = -1$. Captures *any* monotonic relationship.

    **Pearson contrast:** Pearson's $\rho$ would only be 1 for linear $g$; for $Y = X^3$ with $X$ uniform on $[-1, 1]$, Pearson $\rho < 1$, but Spearman $\rho_S = 1$ because the relationship is perfectly monotonic.

    **Invariance:** applying any monotonic transformation $f$ to $X$ preserves $R^X$, so $\rho_S$ is invariant under such transformations. This makes Spearman a robust correlation measure when outliers may distort linear correlation.

    Spearman is preferred when (1) relationships may be nonlinear but monotonic, (2) outliers are present, or (3) variables are ordinal.

---

**Exercise 6.**
**Practical test for independence.** Given a sample $(X_i, Y_i)_{i=1}^n$, propose two complementary tests for independence and discuss when each is appropriate.

??? success "Solution to Exercise 6"
    **Test 1 — Pearson correlation test:** under $H_0$ (and joint normality), the statistic $t = \rho\sqrt{n-2}/\sqrt{1 - \rho^2}$ follows $t_{n-2}$. Reject if $|t|$ exceeds the critical value.

    *Appropriate when:* relationship is plausibly linear and data is approximately bivariate normal.

    **Test 2 — Distance correlation test:** compute $\mathrm{dCor}^2$ on the sample, scaled by $n$. Under $H_0$, asymptotic distribution involves a sum of weighted chi-squared variables; the test is calibrated by permutation (randomly permuting $Y$ values and recomputing $\mathrm{dCor}^2$ to build the null distribution).

    *Appropriate when:* relationship may be nonlinear, no assumption about marginals, or distance correlation makes sense (large enough $n$).

    **Combine for robustness:** Pearson catches strong linear signal cheaply; distance correlation catches subtler nonlinear signal but at higher computational cost. A standard workflow: scan many variable pairs with Pearson, then re-examine the apparently "uncorrelated" pairs with distance correlation or mutual information.

    Practical considerations: power for both tests scales with $n$; for $n < 30$, statistical significance is hard to achieve even with substantial dependence. Visualization (scatter plots) often diagnoses dependence faster than any formal test.
