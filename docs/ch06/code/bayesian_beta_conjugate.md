# Bayesian Beta Conjugate Prior

## Overview

When modelling binary outcomes (success/failure), the Beta distribution serves as the conjugate prior for the Bernoulli and Binomial likelihoods. This means the posterior distribution is also Beta, making Bayesian updating a simple matter of adding counts to prior parameters. This page demonstrates how different prior choices lead to different posteriors, conducts a sensitivity analysis across informative and vague priors, and shows how to compute posterior probabilities for decision-making.

## The Beta-Binomial Conjugate Update

Suppose we observe $k$ successes out of $n$ independent Bernoulli trials with unknown success probability $\theta$. If the prior on $\theta$ is:

$$
\theta \sim \text{Beta}(a, b)
$$

then the posterior, after observing the data, is:

$$
\theta \mid k, n \sim \text{Beta}(a + k, \; b + n - k)
$$

The posterior mean is:

$$
E[\theta \mid k, n] = \frac{a + k}{a + b + n}
$$

This is a weighted average of the prior mean $a/(a+b)$ and the MLE $\hat{\theta} = k/n$:

$$
E[\theta \mid k, n] = \frac{a + b}{a + b + n} \cdot \frac{a}{a + b} + \frac{n}{a + b + n} \cdot \frac{k}{n}
$$

The quantity $a + b$ acts as a **prior effective sample size** -- the larger it is, the more influence the prior exerts relative to the data.

## The Bayesian Update Function

The core computation is remarkably simple: add the observed counts to the prior parameters.

```python
import numpy as np
from scipy.stats import beta

def bayesian_update(a_prior, b_prior, k, n):
    """Compute posterior Beta parameters after observing k/n."""
    a_post = a_prior + k
    b_post = b_prior + n - k
    return a_post, b_post
```

## Sensitivity Analysis Across Priors

Consider a polling scenario: $k = 281$ respondents support a candidate out of $n = 581$ total, giving an MLE of $\hat{\theta} = 281/581 \approx 0.4836$. We examine how six different priors affect the posterior.

```python
n = 581
k = 281

priors = [
    (1, 1, "Uniform (a=1, b=1)"),
    (5, 5, "Weakly informative (a=5, b=5)"),
    (50, 50, "Moderate prior centered at 0.5"),
    (2, 8, "Prior skewed toward low p"),
    (8, 2, "Prior skewed toward high p"),
    (100, 100, "Strong prior at 0.5"),
]

for a, b, label in priors:
    a_post, b_post = bayesian_update(a, b, k, n)
    post_mean = a_post / (a_post + b_post)
    p_less_half = beta.cdf(0.5, a_post, b_post)
    print(f"{label:<35s}  a_post={a_post:>4d}  b_post={b_post:>4d}  "
          f"mean={post_mean:.4f}  P(p<0.5)={p_less_half:.4f}")
```

## Visualising Prior-to-Posterior Updates

For each prior, we plot the prior density (dashed blue), the posterior density (solid red), the shaded region $P(\theta < 0.5)$, and the MLE as a vertical dotted line.

```python
import matplotlib.pyplot as plt

theta = np.linspace(0, 1, 1000)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, (a, b, label) in enumerate(priors):
    a_post, b_post = bayesian_update(a, b, k, n)
    p_less_half = beta.cdf(0.5, a_post, b_post)
    ax = axes.flatten()[idx]

    ax.plot(theta, beta.pdf(theta, a, b), "b--", lw=2, label="Prior")
    ax.plot(theta, beta.pdf(theta, a_post, b_post), "r-", lw=2.5,
            label="Posterior")

    mask = theta <= 0.5
    ax.fill_between(theta[mask],
                    beta.pdf(theta[mask], a_post, b_post),
                    alpha=0.2, color="blue",
                    label=f"P(p<0.5) = {p_less_half:.3f}")
    ax.axvline(k / n, color="green", linestyle=":", lw=1.5,
               label=f"MLE = {k/n:.3f}")
    ax.set_title(label)
    ax.set_xlabel("theta")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.set_xlim(0.35, 0.65)

plt.tight_layout()
plt.show()
```

## Interpretation

- **Uniform prior** $\text{Beta}(1,1)$: the posterior is driven entirely by the data. The posterior mean nearly equals the MLE, and $P(\theta < 0.5)$ reflects only the sampling evidence.
- **Weakly informative priors** (small $a + b$): the effective sample size of the prior is negligible compared to $n = 581$, so the posterior is barely distinguishable from the uniform-prior case.
- **Strong prior at 0.5** $\text{Beta}(100, 100)$: the prior effective sample size is 200, which is substantial relative to $n = 581$. The posterior mean is pulled noticeably toward 0.5, and the posterior is wider around the MLE.
- **Skewed priors**: the prior $\text{Beta}(2, 8)$ (mean 0.2) and $\text{Beta}(8, 2)$ (mean 0.8) have small effective sample sizes ($a + b = 10$) and are easily overwhelmed by the data.
- **Posterior probability** $P(\theta < 0.5)$ is a direct, interpretable quantity useful for decision-making -- for instance, determining whether a candidate likely holds less than majority support.

!!! info "Data Overwhelms the Prior"
    With $n = 581$ observations, even the strong $\text{Beta}(100, 100)$ prior is substantially updated by the data. This illustrates the Bayesian consistency property: as $n \to \infty$, the posterior concentrates at the true parameter value regardless of the prior.

## Exercises

**Exercise 1.** Starting from the Beta-Binomial conjugate model, show that the posterior mode (MAP estimate) is:

$$
\hat{\theta}_{\text{MAP}} = \frac{a + k - 1}{a + b + n - 2}
$$

for $a > 1$ and $b > 1$. What happens when $a = b = 1$ (uniform prior)?

??? success "Solution to Exercise 1"
    The posterior is $\text{Beta}(a + k, b + n - k)$. The mode of $\text{Beta}(\alpha, \beta)$ is:

    $$
    \frac{\alpha - 1}{\alpha + \beta - 2} \quad \text{for } \alpha > 1, \; \beta > 1
    $$

    Substituting $\alpha = a + k$ and $\beta = b + n - k$:

    $$
    \hat{\theta}_{\text{MAP}} = \frac{a + k - 1}{a + b + n - 2}
    $$

    When $a = b = 1$ (uniform prior), this becomes $(k)/(n) = k/n$, which is exactly the MLE. The uniform prior contributes no information, so the MAP and MLE coincide. $\square$

---

**Exercise 2.** A coin is flipped $n = 20$ times, yielding $k = 14$ heads. Compare the posterior mean, MAP, and MLE under three priors: $\text{Beta}(1,1)$, $\text{Beta}(10,10)$, and $\text{Beta}(2,5)$. Which prior pulls the posterior mean farthest from the MLE?

??? success "Solution to Exercise 2"
    The MLE is $\hat{p} = 14/20 = 0.70$.

    **Beta(1, 1):** Posterior $\text{Beta}(15, 7)$. Mean $= 15/22 \approx 0.6818$. MAP $= 14/20 = 0.70$.

    **Beta(10, 10):** Posterior $\text{Beta}(24, 16)$. Mean $= 24/40 = 0.60$. MAP $= 23/38 \approx 0.6053$.

    **Beta(2, 5):** Posterior $\text{Beta}(16, 11)$. Mean $= 16/27 \approx 0.5926$. MAP $= 15/25 = 0.60$.

    The $\text{Beta}(10, 10)$ prior pulls the posterior mean to 0.60 (a shift of 0.10 from the MLE), while $\text{Beta}(2, 5)$ pulls it to 0.5926 (a shift of 0.1074). The $\text{Beta}(2, 5)$ prior pulls the posterior mean farthest from the MLE because it concentrates mass near small values of $p$, and its effective sample size of 7 is comparable to the data size. $\square$

---

**Exercise 3.** A pollster wants to determine whether a candidate has majority support ($\theta > 0.5$). They observe $k = 281$ successes in $n = 581$ trials with a $\text{Beta}(1,1)$ prior. Compute $P(\theta > 0.5 \mid \text{data})$. Would you conclude majority support? What if the prior were $\text{Beta}(100, 100)$?

??? success "Solution to Exercise 3"
    **Uniform prior** $\text{Beta}(1,1)$: Posterior is $\text{Beta}(282, 301)$.

    $$
    P(\theta > 0.5 \mid \text{data}) = 1 - P(\theta \leq 0.5 \mid \text{data})
    $$

    Using `1 - beta.cdf(0.5, 282, 301)` gives approximately $0.2107$. Since this is well below 0.5, we would **not** conclude majority support -- in fact, the data suggest the candidate likely has less than 50% support.

    **Strong prior** $\text{Beta}(100, 100)$: Posterior is $\text{Beta}(381, 400)$.

    $$
    P(\theta > 0.5 \mid \text{data}) = 1 - \text{Beta-CDF}(0.5; 381, 400)
    $$

    This gives approximately $0.1802$. The strong prior centered at 0.5 slightly decreases the posterior probability of majority support because it adds weight at $\theta = 0.5$.

    In both cases, the evidence does not support the claim of majority support. $\square$

---

**Exercise 4.** Derive the posterior variance of $\theta$ in the Beta-Binomial model. Show that it decreases as $n$ increases and interpret the rate of decrease.

??? success "Solution to Exercise 4"
    The posterior is $\text{Beta}(a + k, b + n - k)$. Let $\alpha = a + k$ and $\beta = b + n - k$. The variance of a $\text{Beta}(\alpha, \beta)$ distribution is:

    $$
    \text{Var}(\theta \mid \text{data}) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}
    $$

    Since $\alpha + \beta = a + b + n$, the denominator includes the factor $(a + b + n)^2(a + b + n + 1)$. As $n \to \infty$:

    $$
    \text{Var}(\theta \mid \text{data}) \approx \frac{(a + k)(b + n - k)}{(a + b + n)^3} \approx \frac{\theta(1 - \theta)}{n}
    $$

    where $\theta$ is the true parameter. The posterior variance decreases at rate $O(1/n)$, matching the rate of the sampling variance of the MLE. This means the posterior concentrates around the true value at the same rate as the frequentist standard error shrinks. $\square$

---

**Exercise 5.** Suppose two analysts use different priors -- $\text{Beta}(1, 1)$ and $\text{Beta}(50, 50)$ -- for the same dataset with $n = 100$ and $k = 60$. Compute the posterior mean for each analyst. How large must $n$ be (with $k/n$ held at 0.6) for the difference between the two posterior means to be less than 0.005?

??? success "Solution to Exercise 5"
    **Analyst 1** (uniform prior): Posterior $\text{Beta}(61, 41)$. Mean $= 61/102 \approx 0.5980$.

    **Analyst 2** (informative prior): Posterior $\text{Beta}(110, 90)$. Mean $= 110/200 = 0.55$.

    Difference: $0.5980 - 0.55 = 0.048$.

    In general, with $k = 0.6n$, the means are:

    $$
    m_1 = \frac{1 + 0.6n}{2 + n}, \quad m_2 = \frac{50 + 0.6n}{100 + n}
    $$

    Setting $|m_1 - m_2| < 0.005$ and solving:

    $$
    m_1 - m_2 = \frac{1 + 0.6n}{2 + n} - \frac{50 + 0.6n}{100 + n}
    $$

    Cross-multiplying and simplifying (using a computer algebra system or numerical search), one finds $n \approx 1940$ is needed for the difference to fall below 0.005. This illustrates that even moderately informative priors require substantial data to become negligible. $\square$
