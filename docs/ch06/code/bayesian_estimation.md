# Bayesian Estimation Demonstrations

## Overview

Bayesian estimation treats the unknown parameter $\theta$ as a random variable with a **prior distribution** that encodes beliefs before observing data. After observing data, we update to a **posterior distribution** via Bayes' theorem. This page demonstrates two fundamental conjugate models -- Beta-Binomial and Normal-Normal -- showing how the posterior concentrates as data accumulate and how prior choice affects inference.

## Bayes' Theorem for Estimation

Given data $\mathbf{x} = (x_1, \ldots, x_n)$ and parameter $\theta$:

$$
p(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta)\, \pi(\theta)}{\int f(\mathbf{x} \mid \theta)\, \pi(\theta)\, d\theta}
$$

where:

- $\pi(\theta)$ is the **prior** distribution
- $f(\mathbf{x} \mid \theta)$ is the **likelihood**
- $p(\theta \mid \mathbf{x})$ is the **posterior** distribution

!!! info "Point Estimates from the Posterior"

    - **Posterior mean:** $E[\theta \mid \mathbf{x}]$ -- minimizes the Bayes risk under squared error loss.
    - **MAP (Maximum a Posteriori):** $\arg\max_\theta p(\theta \mid \mathbf{x})$ -- the mode of the posterior.
    - **Posterior median:** minimizes the Bayes risk under absolute error loss.

## Conjugate Priors

A prior $\pi(\theta)$ is **conjugate** to a likelihood if the posterior belongs to the same distributional family as the prior. Conjugate priors yield closed-form posteriors, making Bayesian updating analytically tractable.

| Likelihood | Conjugate Prior | Posterior |
|------------|----------------|-----------|
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known $\sigma^2$) | Normal | Normal |
| Normal (known $\mu$) | Inverse-Gamma | Inverse-Gamma |
| Exponential | Gamma | Gamma |

## Beta-Binomial Conjugate Model

### Setup

For estimating a proportion $p$ from binomial data:

- **Prior:** $p \sim \text{Beta}(\alpha_0, \beta_0)$
- **Data:** $k$ successes out of $n$ trials
- **Posterior:** $p \mid k \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

The posterior mean is:

$$
E[p \mid k] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}
$$

This is a weighted average of the prior mean $\alpha_0/(\alpha_0 + \beta_0)$ and the MLE $k/n$, with weights proportional to the prior's "effective sample size" $\alpha_0 + \beta_0$ and the actual sample size $n$.

### Demonstration

```python
import numpy as np
from scipy import stats

def demo_beta_binomial():
    """Beta-Binomial conjugate model for estimating a proportion."""
    alpha_0, beta_0 = 2, 2  # weakly informative prior

    n, k = 50, 32  # observed data

    # Posterior parameters
    alpha_post = alpha_0 + k
    beta_post = beta_0 + (n - k)

    # Point estimates
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    post_mean = alpha_post / (alpha_post + beta_post)
    mle = k / n

    print(f"Prior:     Beta({alpha_0}, {beta_0})")
    print(f"Data:      {k} successes in {n} trials")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")
    print(f"MAP        = {map_est:.4f}")
    print(f"Post. mean = {post_mean:.4f}")
    print(f"MLE        = {mle:.4f}")

    # 95% credible interval
    ci_low = stats.beta.ppf(0.025, alpha_post, beta_post)
    ci_high = stats.beta.ppf(0.975, alpha_post, beta_post)
    print(f"95% CI:    [{ci_low:.4f}, {ci_high:.4f}]")

demo_beta_binomial()
```

!!! note "Prior as Pseudo-Data"
    The Beta$(\alpha_0, \beta_0)$ prior acts as if we had already observed $\alpha_0 - 1$ successes and $\beta_0 - 1$ failures before seeing the actual data. With $\alpha_0 = \beta_0 = 2$, the prior contributes the equivalent of 2 total "pseudo-observations."

## Normal-Normal Conjugate Model

### Setup

For estimating a mean $\mu$ with known variance $\sigma^2$:

- **Prior:** $\mu \sim N(\mu_0, \tau_0^2)$
- **Data:** $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$
- **Posterior:** $\mu \mid \mathbf{x} \sim N(\mu_n, \tau_n^2)$

where:

$$
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
$$

$$
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}\right)
$$

The posterior mean is a **precision-weighted average** of the prior mean and the sample mean:

$$
\mu_n = \frac{\tau_0^{-2}\,\mu_0 + n\sigma^{-2}\,\bar{x}}{\tau_0^{-2} + n\sigma^{-2}}
$$

### Demonstration

```python
import numpy as np
from scipy import stats

def demo_normal_normal():
    """Normal-Normal conjugate model for estimating a mean."""
    sigma = 2.0        # known population std
    mu_true = 5.0      # true mean
    mu_0, tau_0 = 0, 10  # prior parameters

    rng = np.random.default_rng(42)
    n = 25
    data = rng.normal(mu_true, sigma, n)
    x_bar = data.mean()

    # Posterior parameters
    tau_n_sq = 1 / (1 / tau_0**2 + n / sigma**2)
    mu_n = tau_n_sq * (mu_0 / tau_0**2 + n * x_bar / sigma**2)
    tau_n = np.sqrt(tau_n_sq)

    print(f"Prior:      N({mu_0}, {tau_0}^2)")
    print(f"Data:       n={n}, x_bar={x_bar:.3f}")
    print(f"Posterior:  N({mu_n:.3f}, {tau_n:.3f}^2)")
    print(f"95% credible interval: [{mu_n - 1.96*tau_n:.3f}, {mu_n + 1.96*tau_n:.3f}]")

demo_normal_normal()
```

## How the Posterior Evolves with Data

As $n$ increases, the posterior mean converges to the MLE and the posterior variance shrinks to zero. The prior becomes irrelevant:

$$
\mu_n \to \bar{x} \quad \text{and} \quad \tau_n^2 \to \frac{\sigma^2}{n} \quad \text{as } n \to \infty
$$

This demonstrates the key asymptotic property: for large samples, the posterior is dominated by the likelihood, and Bayesian and frequentist estimates agree.

## Interpretation

- **Conjugate priors** provide analytical convenience. The posterior has a known distributional form, enabling exact computation of credible intervals and posterior probabilities.
- **The posterior mean** is a compromise between the prior belief and the data evidence, weighted by their respective precisions.
- **Prior sensitivity** is important for small samples. As $n$ grows, the influence of the prior vanishes.
- **Credible intervals** have a direct probability interpretation: a 95% credible interval contains the true parameter with posterior probability 0.95. This differs from a frequentist confidence interval.

## Exercises

**Exercise 1.** Suppose you observe $k = 7$ successes in $n = 10$ Bernoulli trials with a $\text{Beta}(1, 1)$ (uniform) prior. Compute the posterior distribution, the posterior mean, the MAP estimate, and a 95% credible interval.

??? success "Solution to Exercise 1"
    The posterior is $\text{Beta}(1 + 7, 1 + 3) = \text{Beta}(8, 4)$.

    Posterior mean: $\frac{8}{8 + 4} = \frac{2}{3} \approx 0.6667$.

    MAP: $\frac{8 - 1}{8 + 4 - 2} = \frac{7}{10} = 0.7$ (which equals the MLE).

    95% credible interval using `scipy.stats.beta.ppf`:

    ```python
    from scipy.stats import beta
    ci = (beta.ppf(0.025, 8, 4), beta.ppf(0.975, 8, 4))
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    ```

    This gives approximately $[0.3834, 0.9029]$. $\square$

---

**Exercise 2.** For the Normal-Normal model, show that as $\tau_0 \to \infty$ (vague prior), the posterior mean converges to the sample mean and the posterior variance converges to $\sigma^2/n$.

??? success "Solution to Exercise 2"
    As $\tau_0 \to \infty$, $1/\tau_0^2 \to 0$. Then:

    $$
    \frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2} \to \frac{n}{\sigma^2}
    $$

    So $\tau_n^2 \to \sigma^2/n$.

    For the posterior mean:

    $$
    \mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}\right)
    $$

    As $\tau_0 \to \infty$, the first term $\mu_0/\tau_0^2 \to 0$, so:

    $$
    \mu_n \to \frac{\sigma^2}{n} \cdot \frac{n\bar{x}}{\sigma^2} = \bar{x}
    $$

    With a vague (non-informative) prior, the Bayesian posterior reduces to the frequentist result. $\square$

---

**Exercise 3.** Prove that the posterior mean of the Beta-Binomial model can be written as a convex combination of the prior mean and the MLE. Identify the weights and interpret them.

??? success "Solution to Exercise 3"
    Let $\pi_0 = \alpha_0/(\alpha_0 + \beta_0)$ be the prior mean and $\hat{p} = k/n$ be the MLE. The posterior mean is:

    $$
    E[p \mid k] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n}
    $$

    Rewrite:

    $$
    = \frac{\alpha_0 + \beta_0}{\alpha_0 + \beta_0 + n}\cdot\frac{\alpha_0}{\alpha_0 + \beta_0} + \frac{n}{\alpha_0 + \beta_0 + n}\cdot\frac{k}{n}
    $$

    $$
    = w\,\pi_0 + (1 - w)\,\hat{p}
    $$

    where $w = (\alpha_0 + \beta_0)/(\alpha_0 + \beta_0 + n)$.

    The weight $w$ on the prior decreases as $n$ grows. The "effective sample size" of the prior is $\alpha_0 + \beta_0$, and the posterior mean allocates weight between prior and data in proportion to their respective sample sizes. $\square$

---

**Exercise 4.** You are estimating the probability of a rare disease with prevalence around 1%. A colleague suggests using a $\text{Beta}(1, 99)$ prior. After testing $n = 500$ people and finding $k = 8$ positives, compare the posterior mean to the MLE. Is the prior choice appropriate?

??? success "Solution to Exercise 4"
    Prior: $\text{Beta}(1, 99)$ with mean $1/100 = 0.01$.

    Posterior: $\text{Beta}(1 + 8, 99 + 492) = \text{Beta}(9, 591)$.

    Posterior mean: $9/600 = 0.015$.

    MLE: $8/500 = 0.016$.

    The prior has an effective sample size of $1 + 99 = 100$, which is modest compared to $n = 500$. The posterior mean (0.015) is pulled slightly toward the prior mean (0.01) from the MLE (0.016). The prior is reasonable because: (1) it encodes genuine domain knowledge about the disease being rare, (2) its effective sample size is much smaller than the actual sample size so it does not overwhelm the data, and (3) it keeps the posterior well-defined even with sparse data. $\square$

---

**Exercise 5.** Derive the posterior distribution for $\lambda$ in the Poisson-Gamma conjugate model: $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$ with prior $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$ (shape-rate parameterization). What is the posterior mean?

??? success "Solution to Exercise 5"
    The likelihood is:

    $$
    L(\lambda) \propto \lambda^{\sum x_i} e^{-n\lambda}
    $$

    The prior is:

    $$
    \pi(\lambda) \propto \lambda^{\alpha_0 - 1} e^{-\beta_0 \lambda}
    $$

    Multiplying:

    $$
    p(\lambda \mid \mathbf{x}) \propto \lambda^{\alpha_0 + \sum x_i - 1} e^{-(\beta_0 + n)\lambda}
    $$

    This is the kernel of $\text{Gamma}(\alpha_0 + \sum x_i, \beta_0 + n)$.

    The posterior mean is:

    $$
    E[\lambda \mid \mathbf{x}] = \frac{\alpha_0 + \sum x_i}{\beta_0 + n} = \frac{\beta_0}{\beta_0 + n}\cdot\frac{\alpha_0}{\beta_0} + \frac{n}{\beta_0 + n}\cdot\bar{x}
    $$

    Again, this is a weighted average of the prior mean $\alpha_0/\beta_0$ and the MLE $\bar{x}$, with weights proportional to the prior rate $\beta_0$ and sample size $n$. $\square$
