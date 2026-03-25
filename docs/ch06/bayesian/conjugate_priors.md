# Conjugate Priors

In Bayesian inference, computing the posterior distribution requires evaluating the integral $\int f(x \mid \theta)\,\pi(\theta)\,d\theta$, which is often intractable for arbitrary prior-likelihood combinations. Conjugate priors provide an elegant shortcut: when the prior belongs to a specific distributional family matched to the likelihood, the posterior is guaranteed to belong to that same family with updated parameters. This closed-form result eliminates the need for numerical integration or MCMC in simple models, making conjugate priors the foundation of tractable Bayesian analysis.

## Definition

Let $x = (x_1, \ldots, x_n)$ denote an i.i.d. sample from a distribution with likelihood $f(x \mid \theta)$. A family $\mathcal{F}$ of prior distributions is **conjugate** for the likelihood $f(x \mid \theta)$ if for every prior $\pi(\theta) \in \mathcal{F}$, the posterior $\pi(\theta \mid x) \in \mathcal{F}$.

In other words, observing data only changes the parameters of the prior distribution, not its functional form. This "closed under updating" property is what makes conjugate families so useful in practice.

## Common Conjugate Pairs

The following table lists the most commonly encountered conjugate pairs. In each case, the Gamma distribution uses the rate parameterization, where $\text{Gamma}(\alpha, \beta)$ has density proportional to $\theta^{\alpha-1}e^{-\beta\theta}$.

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
