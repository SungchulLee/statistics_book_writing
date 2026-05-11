# Conjugate Priors

In Bayesian inference, computing the posterior distribution requires evaluating the integral $\int f(x \mid \theta) \pi(\theta) \, d\theta$, which is often intractable for arbitrary prior-likelihood combinations. Conjugate priors provide an elegant shortcut: when the prior belongs to a specific distributional family matched to the likelihood, the posterior is guaranteed to belong to that same family with updated parameters. This closed-form result eliminates the need for numerical integration or MCMC in simple models, making conjugate priors the foundation of tractable Bayesian analysis.

## Definition

Let $x = (x_1, \ldots, x_n)$ denote an i.i.d. sample from a distribution with likelihood $f(x \mid \theta)$. A family $\mathcal{F}$ of prior distributions is **conjugate** for the likelihood $f(x \mid \theta)$ if for every prior $\pi(\theta) \in \mathcal{F}$, the posterior $\pi(\theta \mid x) \in \mathcal{F}$.

In other words, observing data only changes the parameters of the prior distribution, not its functional form. This "closed under updating" property is what makes conjugate families so useful in practice.

## Common Conjugate Pairs

The following table lists the most commonly encountered conjugate pairs. In each case, the Gamma distribution uses the rate parameterization, where $\text{Gamma}(\alpha, \beta)$ has density proportional to $\theta^{\alpha - 1} e^{-\beta \theta}$.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli/Binomial | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) |
| Normal (known $\sigma^2$) | Normal($\mu_0, \sigma_0^2$) | Normal$\!\left(\frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2},\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}\right)$ |
| Exponential | Gamma($\alpha, \beta$) | Gamma($\alpha + n, \beta + \sum x_i$) |

Notice that in every row, the posterior has the same distributional form as the prior, with parameters updated by summary statistics of the data (sample sum, sample size, or sample mean).

## Derivation of the Beta-Binomial Conjugate Pair

To see why conjugacy works, consider the Beta-Binomial case. Suppose $X \mid p \sim \text{Binomial}(n, p)$ and $p \sim \text{Beta}(\alpha, \beta)$. The prior density is

$$
\pi(p) \propto p^{\alpha - 1}(1 - p)^{\beta - 1}
$$

and the likelihood given $k$ successes in $n$ trials is

$$
f(k \mid p) \propto p^{k}(1 - p)^{n - k}
$$

By Bayes' theorem, the posterior is proportional to the product:

$$
\pi(p \mid k) \propto p^{\alpha - 1}(1 - p)^{\beta - 1} \cdot p^{k}(1 - p)^{n - k} = p^{(\alpha + k) - 1}(1 - p)^{(\beta + n - k) - 1}
$$

This is the kernel of a $\text{Beta}(\alpha + k, \beta + n - k)$ distribution, confirming conjugacy.

!!! example "Beta-Binomial in Practice"
    Suppose a coin has an unknown probability $p$ of heads. We begin with a uniform prior $p \sim \text{Beta}(1, 1)$, which assigns equal weight to all values in $[0, 1]$. After observing $k = 7$ heads in $n = 10$ flips, the posterior is

    $$
    p \mid k = 7 \sim \text{Beta}(1 + 7, 1 + 3) = \text{Beta}(8, 4)
    $$

    The posterior mean is $8 / 12 \approx 0.667$, which lies between the prior mean of $0.5$ and the sample proportion of $0.7$. As more data accumulate, the posterior concentrates around the true value of $p$.

## When to Use Conjugate Priors

Conjugate priors are most useful when:

- **Analytical tractability** is needed, for example in sequential updating where new data arrive over time and the posterior must be recomputed quickly.
- **Interpretability** matters: the prior parameters often have a natural interpretation as "pseudo-observations" (e.g., $\alpha$ and $\beta$ in the Beta prior act like prior counts of successes and failures).

However, conjugate priors restrict the choice of prior family to match the likelihood, which may not always represent genuine prior beliefs. For complex models or when prior flexibility is important, non-conjugate priors combined with computational methods such as MCMC or variational inference are preferred.

## Exercises

**Exercise 1.**
The Beta distribution is the conjugate prior for the Binomial likelihood. If the prior is $\text{Beta}(2, 5)$ and we observe 3 successes in 10 trials, find the posterior distribution and the posterior mean.

??? success "Solution to Exercise 1"
    With a $\text{Beta}(\alpha, \beta)$ prior and $k$ successes in $n$ trials, the posterior is $\text{Beta}(\alpha + k, \beta + n - k)$.

    Here $\alpha = 2$, $\beta = 5$, $k = 3$, $n = 10$:

    $$
    \text{Posterior} = \text{Beta}(2 + 3, 5 + 7) = \text{Beta}(5, 12)
    $$

    The posterior mean is:

    $$
    E[p \mid \text{data}] = \frac{\alpha + k}{\alpha + k + \beta + n - k} = \frac{5}{5 + 12} = \frac{5}{17} \approx 0.294
    $$

    This is a compromise between the prior mean $2/7 \approx 0.286$ and the sample proportion $3/10 = 0.3$.

---

**Exercise 2.**
Show that the Gamma distribution is the conjugate prior for the Poisson likelihood. If $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$ and $\lambda \sim \text{Gamma}(\alpha, \beta)$, derive the posterior distribution of $\lambda$.

??? success "Solution to Exercise 2"
    The Poisson likelihood for $n$ observations is:

    $$
    L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} \propto \lambda^{\sum x_i} e^{-n\lambda}
    $$

    The Gamma$(\alpha, \beta)$ prior (with rate parameterization) is:

    $$
    \pi(\lambda) \propto \lambda^{\alpha - 1} e^{-\beta\lambda}
    $$

    The posterior is:

    $$
    \pi(\lambda \mid \mathbf{x}) \propto \lambda^{\sum x_i} e^{-n\lambda} \cdot \lambda^{\alpha - 1} e^{-\beta\lambda} = \lambda^{\alpha + \sum x_i - 1} e^{-(\beta + n)\lambda}
    $$

    This is the kernel of a $\text{Gamma}(\alpha + \sum x_i, \beta + n)$ distribution, confirming conjugacy. $\square$

---

**Exercise 3.**
Explain what it means for a prior to be "non-informative" or "weakly informative." Give an example of a weakly informative conjugate prior for a normal mean $\mu$ when $\sigma^2$ is known.

??? success "Solution to Exercise 3"
    A **non-informative** (or "vague") prior is intended to let the data dominate the posterior, encoding minimal prior knowledge. A **weakly informative** prior constrains the parameter to a reasonable range without strongly favoring any particular value.

    For a normal mean $\mu$ with known $\sigma^2$, the conjugate prior is $\mu \sim N(\mu_0, \tau^2)$. A weakly informative choice sets $\tau$ very large relative to the data scale -- e.g., if we expect $\mu$ to be between $-100$ and $100$, we might use $\mu_0 = 0$ and $\tau = 100$. This prior barely influences the posterior when $n$ is moderate but prevents extreme estimates.

    In the limit $\tau \to \infty$, we obtain the improper flat prior $\pi(\mu) \propto 1$, which is non-informative. The posterior mean then equals the MLE $\bar{x}$.

---

**Exercise 4.**
With a conjugate Normal-Gamma prior, the posterior for the normal mean and precision $(\mu, \tau)$ is also Normal-Gamma. Describe intuitively why conjugate priors are computationally convenient and name one limitation.

??? success "Solution to Exercise 4"
    **Convenience:** Conjugate priors are computationally convenient because the posterior has the same distributional family as the prior -- only the parameters change. This means:

    - The posterior can be written in closed form (no numerical integration needed).
    - Updating with new data simply updates the hyperparameters: $(\alpha, \beta) \to (\alpha', \beta')$.
    - Sequential updating is trivial: each new observation updates the hyperparameters incrementally.

    **Limitation:** Conjugate priors may not accurately represent the analyst's genuine prior beliefs. For example, a Beta prior for a binomial proportion is unimodal (or U-shaped), but the analyst might believe the true probability is bimodal (e.g., near 0.2 or 0.8). Forcing beliefs into the conjugate family sacrifices expressiveness for computational tractability. Modern MCMC methods allow arbitrary priors, reducing the need for conjugacy.
