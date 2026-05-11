# Financial Crisis Central Limit Theorem Failure

## Overview

The Central Limit Theorem requires independence (or at least weak dependence) among observations. When this condition is violated, the CLT can fail dramatically, and Gaussian-based risk models can vastly underestimate the probability of extreme events. This page demonstrates this through a model of correlated borrower defaults, showing how dependence produces heavy-tailed distributions that the normal approximation cannot capture. This phenomenon played a central role in the 2008 financial crisis.

## The Model

Consider $n = 100$ borrowers, each of whom either defaults or does not. Let $D = \sum_{i=1}^n X_i$ be the total number of defaults, where $X_i \in \{0, 1\}$.

### Independent Case

Each borrower defaults independently with fixed probability $\theta = 2/3$:

$$
X_i \overset{\text{iid}}{\sim} \text{Bernoulli}(2/3)
$$

$$
D \sim \text{Binomial}(100,\; 2/3)
$$

The CLT applies:

$$
D \;\dot{\sim}\; N\!\left(n\theta,\; n\theta(1 - \theta)\right) = N\!\left(\frac{200}{3},\; \frac{200}{9}\right)
$$

### Dependent Case (Shared Risk Factor)

The default probability is itself random, reflecting a shared economic factor:

$$
\theta \sim \text{Beta}(2, 1)
$$

Conditional on $\theta$, the defaults are independent:

$$
X_i \mid \theta \overset{\text{iid}}{\sim} \text{Bernoulli}(\theta)
$$

However, **marginally** (unconditionally), the defaults are positively correlated because they share the same $\theta$.

!!! warning "Key Point"
    The conditional independence structure $X_i \perp X_j \mid \theta$ does **not** imply marginal independence. The shared latent factor $\theta$ induces positive dependence: $\text{Cov}(X_i, X_j) > 0$ for $i \ne j$.

## Properties of the Beta Prior

The $\text{Beta}(2, 1)$ distribution has:

$$
E[\theta] = \frac{2}{3}, \qquad \text{Var}(\theta) = \frac{2}{18} = \frac{1}{9}
$$

The pdf is $f(\theta) = 2\theta$ for $\theta \in [0, 1]$, which places more weight on higher default probabilities.

## Marginal Distribution of Defaults

!!! abstract "Theoretical PMF"
    For $\theta \sim \text{Beta}(2, 1)$ and $D \mid \theta \sim \text{Binomial}(n, \theta)$, the marginal PMF of $D$ is:

    $$
    P(D = d) = \int_0^1 \binom{n}{d} \theta^d (1-\theta)^{n-d} \cdot 2\theta \, d\theta = \frac{2(d+1)}{(n+1)(n+2)}
    $$

    for $d = 0, 1, \ldots, n$.

This is nearly a **linearly increasing** function of $d$, which is dramatically different from the concentrated, bell-shaped Binomial distribution.

## Tail Risk Comparison

The probability of a catastrophic event (more than 90 defaults out of 100) under each model:

| Model | $P(D > 90)$ |
|---|---|
| Independent (Binomial) | $\approx 10^{-8}$ (negligible) |
| Gaussian approximation (CLT) | $\approx 10^{-8}$ (negligible) |
| Dependent (Beta mixing) | $\approx 0.016$ (significant) |

!!! danger "The Danger"
    The dependent model gives a tail probability roughly **six orders of magnitude** larger than what the Gaussian model predicts. Events that appear virtually impossible under the CLT-based model occur about 1.6% of the time under the dependent model. This is not a small correction -- it is a qualitative failure of the Gaussian framework.

## Simulation Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

n = 100
p = 2 / 3
n_sims = 500_000

# Independent defaults
d_indep = np.random.binomial(n, p, size=n_sims)

# Dependent defaults (shared risk factor)
thetas = np.random.beta(2, 1, size=n_sims)
d_dep = np.array([np.random.binomial(n, th) for th in thetas])

# Tail probabilities
threshold = 90
p_indep = np.mean(d_indep > threshold)
p_dep = np.mean(d_dep > threshold)
p_gauss = 1 - stats.norm.cdf(threshold, n * p, np.sqrt(n * p * (1 - p)))

print(f"P(D > {threshold}):")
print(f"  Independent:     {p_indep:.6f}")
print(f"  Gaussian approx: {p_gauss:.6f}")
print(f"  Dependent:       {p_dep:.6f}")
```

## Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Full distributions
ax = axes[0]
bins = np.arange(-0.5, n + 1.5, 1)
ax.hist(d_dep, bins=bins, density=True, alpha=0.5, color="tomato",
        label="Dependent (MC)")
d_vals = np.arange(0, n + 1)
ax.plot(d_vals, stats.binom.pmf(d_vals, n, p), "o", color="black",
        ms=3, label="Independent")
x_norm = np.linspace(0, n, 300)
ax.plot(x_norm, stats.norm.pdf(x_norm, n * p, np.sqrt(n * p * (1 - p))),
        "--", color="gray", lw=3, label="Gaussian approx")
ax.set_xlabel("Number of defaults")
ax.set_ylabel("Probability")
ax.set_title("Default Distributions")
ax.legend()

# Panel 2: Tail zoom
ax = axes[1]
tail = range(80, n + 1)
mc_tail = [np.mean(d_dep == d) for d in tail]
ax.bar(list(tail), mc_tail, color="tomato", alpha=0.7, label="Dependent")
ax.plot(list(tail), stats.binom.pmf(list(tail), n, p), "ko-",
        ms=4, label="Independent")
ax.set_xlabel("Number of defaults")
ax.set_title("Tail Risk (d >= 80)")
ax.legend()

# Panel 3: Beta prior
ax = axes[2]
theta_grid = np.linspace(0, 1, 300)
ax.plot(theta_grid, stats.beta.pdf(theta_grid, 2, 1), lw=2.5,
        label="Beta(2, 1)")
ax.axvline(p, color="red", linestyle="--", label=f"E[theta] = {p:.3f}")
ax.set_xlabel("theta")
ax.set_ylabel("Density")
ax.set_title("Prior on Shared Risk Factor")
ax.legend()

plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. **Independent model**: The distribution of $D$ is concentrated around the mean $n\theta = 200/3 \approx 66.7$, with negligible probability in the tails beyond 90.
    2. **Dependent model**: The distribution is roughly uniform (actually linearly increasing) across all values of $D$, with substantial probability mass in both tails.
    3. **Gaussian approximation**: Closely matches the independent model but completely fails to capture the heavy tails of the dependent model.
    4. **Beta prior**: Because $\text{Beta}(2, 1)$ places non-trivial probability on $\theta > 0.9$ (about 19%), there is a realistic chance that the shared risk factor drives nearly all borrowers to default.

### Why This Matters for Finance

Before the 2008 financial crisis, many risk models treated mortgage defaults as approximately independent events. The resulting Gaussian copula models dramatically underestimated tail risk. When housing markets declined simultaneously across regions (a shared risk factor), defaults became correlated, and "once-in-a-lifetime" losses occurred far more frequently than the models predicted.

## Exercises

**Exercise 1.** Compute $E[D]$ and $\text{Var}(D)$ for both the independent and dependent models.

??? success "Solution to Exercise 1"
    **Independent model**: $D \sim \text{Binomial}(100, 2/3)$.

    $$
    E[D] = n\theta = \frac{200}{3} \approx 66.67, \qquad \text{Var}(D) = n\theta(1-\theta) = \frac{200}{9} \approx 22.22
    $$

    **Dependent model**: Use the law of total expectation and total variance.

    $$
    E[D] = E[E[D \mid \theta]] = E[n\theta] = n \cdot E[\theta] = 100 \cdot \frac{2}{3} = \frac{200}{3}
    $$

    The means are identical. For the variance:

    $$
    \text{Var}(D) = E[\text{Var}(D \mid \theta)] + \text{Var}(E[D \mid \theta])
    $$

    $$
    = E[n\theta(1-\theta)] + \text{Var}(n\theta)
    $$

    $$
    = n(E[\theta] - E[\theta^2]) + n^2 \text{Var}(\theta)
    $$

    For $\text{Beta}(2,1)$: $E[\theta] = 2/3$, $E[\theta^2] = E[\theta]^2 + \text{Var}(\theta) = 4/9 + 1/18 = 1/2$, $\text{Var}(\theta) = 1/18$.

    $$
    \text{Var}(D) = 100\!\left(\frac{2}{3} - \frac{1}{2}\right) + 100^2 \cdot \frac{1}{18} = 100 \cdot \frac{1}{6} + \frac{10000}{18} \approx 16.67 + 555.56 = 572.22
    $$

    The dependent model has variance **25 times larger** than the independent model ($572$ vs. $22$), despite having the same mean. $\square$

---

**Exercise 2.** Derive the marginal PMF $P(D = d) = 2(d+1) / [(n+1)(n+2)]$ for $\theta \sim \text{Beta}(2, 1)$.

??? success "Solution to Exercise 2"
    $$
    P(D = d) = \int_0^1 \binom{n}{d}\theta^d(1-\theta)^{n-d} \cdot 2\theta \, d\theta
    $$

    $$
    = 2\binom{n}{d}\int_0^1 \theta^{d+1}(1-\theta)^{n-d} \, d\theta
    $$

    The integral is the Beta function $B(d+2, n-d+1) = \frac{\Gamma(d+2)\Gamma(n-d+1)}{\Gamma(n+3)} = \frac{(d+1)!(n-d)!}{(n+2)!}$.

    $$
    P(D = d) = 2 \cdot \frac{n!}{d!(n-d)!} \cdot \frac{(d+1)!(n-d)!}{(n+2)!} = 2 \cdot \frac{n! \cdot (d+1)}{(n+2)!}
    $$

    $$
    = 2 \cdot \frac{(d+1)}{(n+1)(n+2)}
    $$

    One can verify: $\sum_{d=0}^n \frac{2(d+1)}{(n+1)(n+2)} = \frac{2}{(n+1)(n+2)} \cdot \sum_{d=0}^n (d+1) = \frac{2}{(n+1)(n+2)} \cdot \frac{(n+1)(n+2)}{2} = 1$. $\square$

---

**Exercise 3.** Show that the marginal covariance between any two borrowers' defaults is $\text{Cov}(X_i, X_j) = \text{Var}(\theta)$ for $i \ne j$. Compute this for $\theta \sim \text{Beta}(2, 1)$.

??? success "Solution to Exercise 3"
    For $i \ne j$:

    $$
    E[X_i X_j] = E[E[X_i X_j \mid \theta]] = E[\theta \cdot \theta] = E[\theta^2]
    $$

    (using conditional independence: $E[X_i X_j \mid \theta] = E[X_i \mid \theta] \cdot E[X_j \mid \theta] = \theta^2$).

    $$
    E[X_i] \cdot E[X_j] = (E[\theta])^2
    $$

    Therefore:

    $$
    \text{Cov}(X_i, X_j) = E[\theta^2] - (E[\theta])^2 = \text{Var}(\theta)
    $$

    For $\theta \sim \text{Beta}(2, 1)$:

    $$
    \text{Cov}(X_i, X_j) = \text{Var}(\theta) = \frac{2 \cdot 1}{(2+1)^2(2+1+1)} = \frac{2}{36} = \frac{1}{18} \approx 0.0556
    $$

    This positive covariance is the source of the heavy tails. $\square$

---

**Exercise 4.** If instead $\theta \sim \text{Beta}(20, 10)$ (still with $E[\theta] = 2/3$ but much less variable), recompute $\text{Var}(\theta)$ and $\text{Var}(D)$. How does reducing the variability of the shared risk factor affect tail risk?

??? success "Solution to Exercise 4"
    For $\theta \sim \text{Beta}(20, 10)$: $\alpha = 20$, $\beta = 10$.

    $$
    E[\theta] = \frac{20}{30} = \frac{2}{3}, \qquad \text{Var}(\theta) = \frac{20 \cdot 10}{30^2 \cdot 31} = \frac{200}{27900} \approx 0.00717
    $$

    Using the total variance formula:

    $$
    \text{Var}(D) = n(E[\theta] - E[\theta^2]) + n^2\text{Var}(\theta)
    $$

    $E[\theta^2] = \text{Var}(\theta) + (E[\theta])^2 = 0.00717 + 4/9 \approx 0.4516$.

    $$
    \text{Var}(D) = 100(0.6667 - 0.4516) + 10000 \times 0.00717 = 21.51 + 71.68 = 93.19
    $$

    Compare: independent model gives $\text{Var}(D) = 22.2$, $\text{Beta}(2,1)$ gives 572, and $\text{Beta}(20,10)$ gives 93.

    With less variable $\theta$, the tail risk is greatly reduced (though still larger than the independent case). The $\text{Beta}(20, 10)$ distribution concentrates $\theta$ near $2/3$, so extreme scenarios ($\theta > 0.9$) become very rare. The model interpolates between the independent case ($\text{Var}(\theta) = 0$) and the highly dependent $\text{Beta}(2, 1)$ case. $\square$

---

**Exercise 5.** Explain in plain language why the Gaussian copula model used by rating agencies before 2008 failed. What assumption was most critically violated?

??? success "Solution to Exercise 5"
    The Gaussian copula model assumed that borrower defaults were driven by individual-specific factors with only weak correlations, modelled through a multivariate normal dependence structure. Under this framework, the probability of many simultaneous defaults (a tail event) was negligible because the multivariate normal distribution has light (exponentially decaying) tails.

    The most critically violated assumption was **the adequacy of the correlation structure**. Specifically:

    1. **Shared systemic risk**: The model underestimated how strongly borrower defaults were linked to common economic factors (housing prices, interest rates, employment). When housing prices declined nationwide, defaults became highly correlated.

    2. **Light tails of the normal distribution**: The Gaussian copula cannot capture "tail dependence" -- the phenomenon where extreme events (many defaults) tend to occur together. Real default dependence has much heavier tails than the normal distribution implies.

    3. **Static correlations**: The correlation parameters were estimated from benign economic periods and did not account for how correlations increase dramatically during crises (correlation breakdown or "correlation smile").

    The result was that events with a predicted probability of $10^{-8}$ (essentially impossible) actually occurred with probability closer to $10^{-2}$ -- a million-fold underestimation of risk. This led to catastrophic losses for institutions holding mortgage-backed securities that were rated as "safe" based on the Gaussian model. $\square$
