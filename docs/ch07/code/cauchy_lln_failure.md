# Cauchy Law of Large Numbers Failure

## Overview

The Cauchy distribution provides the most dramatic counterexample to the Law of Large Numbers. Because the Cauchy distribution has no finite mean ($E[|X|] = \infty$), the sample mean of iid Cauchy observations does not converge — it has the same distribution as a single observation regardless of sample size. This page contrasts Cauchy and normal sample mean behavior through trajectory plots, sampling distribution histograms, and Q-Q plots.

## The Cauchy Distribution

The standard Cauchy distribution has density:

$$f(x) = \frac{1}{\pi(1 + x^2)}, \quad x \in \mathbb{R}$$

Key properties:

- **No finite mean:** $E[|X|] = \int_0^\infty \frac{2x}{\pi(1+x^2)}dx = \frac{2}{\pi}\left[\frac{1}{2}\ln(1+x^2)\right]_0^\infty = \infty$
- **No finite variance** (since the mean does not exist)
- **Heavy tails:** $P(|X| > t) \sim 2/(\pi t)$ as $t \to \infty$ (polynomial decay, compared to exponential decay for the normal)
- **Characteristic function:** $\varphi(t) = e^{-|t|}$

!!! danger "LLN does not apply"
    The Law of Large Numbers requires $E[|X|] < \infty$. Since this fails for the Cauchy, the sample mean $\bar{X}_n$ does not converge to any value. In fact, $\bar{X}_n$ has exactly the same Cauchy distribution for every $n$.

## Sample Mean Trajectories

The running average $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ shows strikingly different behavior for normal vs Cauchy data.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sample_mean_trajectories(dist, n_max=10_000, n_tries=20):
    trajectories = []
    for _ in range(n_tries):
        if dist == "cauchy":
            data = np.random.standard_cauchy(n_max)
        else:
            data = np.random.standard_normal(n_max)
        running_mean = np.cumsum(data) / np.arange(1, n_max + 1)
        trajectories.append(running_mean)
    return trajectories

n_max = 10_000
ns = np.arange(1, n_max + 1)

cauchy_traj = sample_mean_trajectories("cauchy", n_max)
normal_traj = sample_mean_trajectories("normal", n_max)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for traj in cauchy_traj:
    clipped = np.clip(traj, -50, 50)
    ax.semilogx(ns, clipped, lw=0.7, alpha=0.6)
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("n"); ax.set_ylabel("Sample mean")
ax.set_title("Cauchy: Sample Mean Trajectories")
ax.set_ylim(-50, 50)

ax = axes[1]
for traj in normal_traj:
    ax.semilogx(ns, traj, lw=0.7, alpha=0.6)
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("n"); ax.set_ylabel("Sample mean")
ax.set_title("Normal: Sample Mean Trajectories")
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.show()
```

!!! note "Convergence vs non-convergence"
    For the normal distribution (right panel), all 20 trajectories visibly converge to 0 as $n$ grows. For the Cauchy (left panel), the trajectories continue to wander erratically — occasional extreme observations "reset" the running average even at large $n$.

## Distribution of Sample Means

For the normal distribution, the sampling distribution of $\bar{X}_n$ narrows as $n$ increases (by the CLT, its standard deviation is $1/\sqrt{n}$). For the Cauchy, the distribution of $\bar{X}_n$ does **not** narrow at all.

```python
from scipy import stats

def sample_mean_distributions(dist, n_vals, n_reps=10_000):
    results = {}
    for n in n_vals:
        if dist == "cauchy":
            data = np.random.standard_cauchy((n_reps, n))
        else:
            data = np.random.standard_normal((n_reps, n))
        results[n] = data.mean(axis=1)
    return results

n_vals = [100, 1000, 10_000]
cauchy_dists = sample_mean_distributions("cauchy", n_vals)
normal_dists = sample_mean_distributions("normal", n_vals)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for col, n in enumerate(n_vals):
    ax = axes[col]
    c_means = np.clip(cauchy_dists[n], -20, 20)
    n_means = normal_dists[n]
    ax.hist(c_means, bins=80, density=True, alpha=0.6, color="coral", label="Cauchy")
    ax.hist(n_means, bins=50, density=True, alpha=0.6, color="steelblue", label="Normal")
    ax.set_title(f"Sample Mean Distribution (n = {n})")
    ax.set_xlabel("Sample mean value")
    ax.set_xlim(-5, 5)
    ax.legend()
plt.tight_layout()
plt.show()
```

!!! warning "The Cauchy distribution does not concentrate"
    At $n = 10{,}000$, the normal sample mean distribution has collapsed to a spike at 0 (std $= 0.01$), while the Cauchy sample mean distribution looks virtually identical to the $n = 100$ case. Averaging more Cauchy data does not help.

## Q-Q Plot: Visualizing Heavy Tails

A **Q-Q plot** comparing Cauchy quantiles against normal quantiles reveals the extreme heaviness of the Cauchy tails.

```python
cauchy_sample = np.random.standard_cauchy(1000)
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(cauchy_sample, dist="norm", plot=ax)
ax.set_title("Cauchy vs Normal Q-Q Plot")
plt.tight_layout()
plt.show()
```

The characteristic S-shape (or hockey-stick shape) shows that the Cauchy generates values far more extreme than a normal distribution would.

## Why Averaging Fails: The Characteristic Function Proof

The characteristic function of the standard Cauchy is $\varphi_X(t) = e^{-|t|}$.

For the sample mean of $n$ iid Cauchy variables:

$$\varphi_{\bar{X}_n}(t) = \left[\varphi_X(t/n)\right]^n = \left[e^{-|t|/n}\right]^n = e^{-|t|}$$

This is the characteristic function of a single Cauchy observation. Therefore:

$$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{for all } n$$

!!! info "Stability property"
    The Cauchy distribution is a **stable distribution** with index $\alpha = 1$. For stable distributions, linear combinations of iid copies have the same distributional form (after rescaling). The Cauchy is the only symmetric stable distribution for which the sample mean has the same distribution as a single observation, because the scale parameter of the sum $S_n$ grows as $n$ rather than $\sqrt{n}$, exactly canceling the $1/n$ division.

## Alternative Estimators for Cauchy Data

Since the sample mean is useless for Cauchy data, what alternatives work?

| Estimator | Converges? | Rate |
|-----------|-----------|------|
| Sample mean | No | N/A |
| Sample median | Yes | $O(1/\sqrt{n})$ |
| MLE (location) | Yes | $O(1/\sqrt{n})$ |
| Trimmed mean ($\alpha > 0$) | Yes | $O(1/\sqrt{n})$ |

The **sample median** is consistent for the Cauchy location parameter and has asymptotic variance $\pi^2/(4n) \approx 2.47/n$, which is actually better than the MLE's asymptotic variance of $2/n$ by a factor of... wait, the MLE is *more* efficient: $\text{Var}(\hat{\mu}_{\text{MLE}}) \approx 2/n$ vs $\text{Var}(\text{median}) \approx \pi^2/(4n)$. The MLE's ARE relative to the median is $\pi^2/8 \approx 1.23$.

## Interpretation

- The Cauchy distribution is the canonical example of a distribution where the **LLN fails**. It demonstrates that finite mean is not just a mathematical technicality but a genuine requirement.
- The sample mean of Cauchy data has the **same distribution** as a single observation, so averaging provides zero improvement.
- **Heavy tails** are the root cause: occasional extreme observations dominate the sum, preventing concentration.
- The **median** and **MLE** are viable alternatives that do converge, at the usual $1/\sqrt{n}$ rate.
- In practice, the Cauchy serves as a warning: always verify that your estimator is appropriate for the data. If the data has extremely heavy tails, the sample mean may be misleading or meaningless.

## Exercises

**Exercise 1.**
Show that $E[|X|] = \infty$ for the standard Cauchy distribution by evaluating the integral directly.

??? success "Solution to Exercise 1"
    $$E[|X|] = \int_{-\infty}^{\infty} \frac{|x|}{\pi(1+x^2)}\,dx = \frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx$$

    Using the substitution $u = 1 + x^2$, $du = 2x\,dx$:

    $$\frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx = \frac{2}{\pi}\cdot\frac{1}{2}\int_1^{\infty}\frac{du}{u} = \frac{1}{\pi}\left[\ln u\right]_1^{\infty} = \frac{1}{\pi}\cdot\infty = \infty$$

    Since $E[|X|] = \infty$, the mean $E[X]$ does not exist. This is the fundamental reason the LLN fails for the Cauchy. $\square$

---

**Exercise 2.**
Using characteristic functions, prove that $\bar{X}_n$ of $n$ iid standard Cauchy random variables has the same distribution as a single standard Cauchy random variable.

??? success "Solution to Exercise 2"
    The characteristic function of the standard Cauchy is $\varphi_X(t) = e^{-|t|}$.

    For iid $X_1, \ldots, X_n$, the sum $S_n = \sum_{i=1}^n X_i$ has:

    $$\varphi_{S_n}(t) = \prod_{i=1}^n \varphi_{X_i}(t) = \left(e^{-|t|}\right)^n = e^{-n|t|}$$

    This is the characteristic function of $\text{Cauchy}(0, n)$ (Cauchy with scale parameter $n$).

    For the sample mean $\bar{X}_n = S_n/n$:

    $$\varphi_{\bar{X}_n}(t) = \varphi_{S_n}(t/n) = e^{-n|t/n|} = e^{-|t|}$$

    This is exactly $\varphi_X(t)$, the characteristic function of the standard Cauchy. Since the characteristic function uniquely determines the distribution:

    $$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{for every } n \geq 1$$

    $\square$

---

**Exercise 3.**
Compare the tails of the Cauchy and normal distributions. What is $P(|X| > 10)$ for each? What does this imply for the behavior of the sample mean?

??? success "Solution to Exercise 3"
    **Normal:** $P(|Z| > 10) = 2\mathcal{N}(-10) \approx 2 \times 7.62 \times 10^{-24} \approx 1.52 \times 10^{-23}$.

    **Cauchy:** $P(|X| > 10) = 2\left(\frac{1}{2} - \frac{1}{\pi}\arctan(10)\right) = 1 - \frac{2}{\pi}\arctan(10) \approx 1 - \frac{2}{\pi}(1.4711) \approx 1 - 0.9366 = 0.0634$.

    So $P(|X| > 10) \approx 6.3\%$ for the Cauchy — more than $10^{21}$ times larger than for the normal.

    **Implications for the sample mean:** In a sample of $n = 100$ Cauchy observations, we expect roughly 6 values exceeding 10 in absolute value. These extreme values contribute disproportionately to the sum, overwhelming the more moderate observations. Since such extreme values occur at a rate that does not decrease fast enough with $n$, the $1/n$ normalization in $\bar{X}_n$ cannot "tame" the sum. This is the intuitive reason the sample mean does not converge. $\square$

---

**Exercise 4.**
The sample median is consistent for the Cauchy location parameter. What is its asymptotic variance? Compare with the asymptotic variance of the MLE for the Cauchy location parameter.

??? success "Solution to Exercise 4"
    For the Cauchy density $f(x) = \frac{1}{\pi(1+x^2)}$, the value at the median (which is 0 for the standard Cauchy) is $f(0) = 1/\pi$.

    The asymptotic variance of the sample median is:

    $$\text{Var}(\text{median}) \approx \frac{1}{4nf(0)^2} = \frac{1}{4n(1/\pi)^2} = \frac{\pi^2}{4n} \approx \frac{2.467}{n}$$

    The Fisher information for the Cauchy location parameter $\mu$ is:

    $$I(\mu) = \int_{-\infty}^{\infty} \frac{[f'(x-\mu)]^2}{f(x-\mu)}\,dx = \frac{1}{2}$$

    So the CRLB (and asymptotic variance of the MLE) is:

    $$\text{Var}(\hat{\mu}_{\text{MLE}}) \approx \frac{1}{nI(\mu)} = \frac{2}{n}$$

    The asymptotic relative efficiency of the median relative to the MLE is:

    $$\text{ARE} = \frac{2/n}{\pi^2/(4n)} = \frac{8}{\pi^2} \approx 0.811$$

    So the median is about 81% as efficient as the MLE for Cauchy data — a reasonable tradeoff given the median's simplicity and computational ease. $\square$

---

**Exercise 5.**
Explain what a "stable distribution" is and why the Cauchy is one. What is the stability index of the Cauchy, and what does it determine about tail behavior?

??? success "Solution to Exercise 5"
    A random variable $X$ has a **stable distribution** with index $\alpha \in (0, 2]$ if for any $n$ iid copies $X_1, \ldots, X_n$:

    $$X_1 + X_2 + \cdots + X_n \overset{d}{=} n^{1/\alpha} X + c_n$$

    for some constant $c_n$. Equivalently, the family is closed under addition (up to location and scale). The characteristic function has the form $\varphi(t) = \exp(-c|t|^\alpha + i\delta t)$ for symmetric stable distributions.

    **The Cauchy has $\alpha = 1$:**

    - Characteristic function: $\varphi(t) = e^{-|t|}$, which is $e^{-|t|^1}$ (so $\alpha = 1$).
    - Sum property: $S_n = \sum X_i$ has $\varphi_{S_n}(t) = e^{-n|t|}$, which corresponds to $\text{Cauchy}(0, n)$. This matches $n^{1/\alpha}X = n^1 X \sim \text{Cauchy}(0, n)$.

    **The stability index $\alpha$ determines tail behavior:**

    - $P(|X| > t) \sim t^{-\alpha}$ as $t \to \infty$ (for $\alpha < 2$).
    - $\alpha = 2$: Gaussian (exponential tails, all moments finite).
    - $\alpha = 1$: Cauchy ($P(|X|>t) \sim 1/t$, no mean).
    - $0 < \alpha < 1$: Even heavier tails (no mean, $P(|X|>t) \sim t^{-\alpha}$).

    The smaller $\alpha$ is, the heavier the tails. For $\alpha < 2$, moments of order $\geq \alpha$ are infinite. For $\alpha \leq 1$, the mean does not exist, and the LLN fails for the sample mean. $\square$
