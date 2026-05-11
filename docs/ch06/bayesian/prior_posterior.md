# Prior, Likelihood, and Posterior

In frequentist statistics, the parameter $\theta$ is treated as a fixed but unknown constant, and inference relies entirely on the sampling distribution of the data. Bayesian inference takes a fundamentally different approach: it treats $\theta$ as a random variable with its own distribution. Before observing any data, our beliefs about $\theta$ are encoded in a **prior distribution**. After observing data, Bayes' theorem provides a principled mechanism for updating these beliefs into a **posterior distribution**. This section introduces the three core components of Bayesian inference and how they combine.

## Bayes' Theorem

The posterior distribution of $\theta$ given observed data $\mathbf{x} = (x_1, \ldots, x_n)$ is given by Bayes' theorem:

$$
\pi(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta)\,\pi(\theta)}{f(\mathbf{x})} \propto f(\mathbf{x} \mid \theta)\,\pi(\theta)
$$

The proportionality on the right drops the denominator $f(\mathbf{x})$, which does not depend on $\theta$. This proportional form is often sufficient because we can identify the posterior's distributional family from the $\theta$-dependent terms alone.

## Components

**Prior** $\pi(\theta)$: The prior encodes our beliefs about $\theta$ before seeing any data. It may reflect genuine domain knowledge (an informative prior) or express deliberate agnosticism (a noninformative or weakly informative prior). The choice of prior is one of the distinguishing features --- and frequent criticisms --- of Bayesian inference.

**Likelihood** $f(\mathbf{x} \mid \theta)$: The likelihood is the probability density (or probability mass, for discrete data) of the observed sample, viewed as a function of $\theta$. Although $f(\mathbf{x} \mid \theta)$ uses the same notation as a density in $\mathbf{x}$, in the Bayesian context we emphasize its dependence on $\theta$: the likelihood tells us how well each candidate parameter value explains the data.

**Posterior** $\pi(\theta \mid \mathbf{x})$: The posterior is the updated distribution of $\theta$ after incorporating the data. It combines the prior and the likelihood, and it is the central object of Bayesian inference --- all conclusions (point estimates, intervals, predictions) flow from the posterior.

**Marginal likelihood** $f(\mathbf{x})$: The marginal likelihood is the normalizing constant that ensures the posterior integrates to one:

$$
f(\mathbf{x}) = \int f(\mathbf{x} \mid \theta)\,\pi(\theta)\,d\theta
$$

This integral averages the likelihood over all possible parameter values weighted by the prior. Computing this integral is often intractable, which is why conjugate priors (which yield closed-form posteriors) and computational methods (such as MCMC) are essential tools in Bayesian practice.

## Bayesian Point Estimates

The full posterior distribution is the complete Bayesian answer to an inference problem. However, practitioners often need a single number to summarize the posterior. Each common point estimate corresponds to minimizing a different loss function.

| Estimate | Definition | Minimizes |
|---|---|---|
| Posterior mean | $E[\theta \mid \mathbf{x}]$ | Squared error loss |
| Posterior median | Median of $\pi(\theta \mid \mathbf{x})$ | Absolute error loss |
| MAP | Mode of $\pi(\theta \mid \mathbf{x})$ | 0-1 loss (in a limiting sense) |

!!! example "Normal Prior with Normal Likelihood"
    Suppose $X_1, \ldots, X_n \overset{iid}{\sim} N(\theta, \sigma^2)$ with known $\sigma^2$, and the prior is $\theta \sim N(\mu_0, \sigma_0^2)$. Applying Bayes' theorem and completing the square, the posterior is

    $$
    \theta \mid \mathbf{x} \sim N\!\left(\frac{\sigma^2\,\mu_0 + n\,\sigma_0^2\,\bar{x}}{\sigma^2 + n\,\sigma_0^2},\; \frac{\sigma^2\,\sigma_0^2}{\sigma^2 + n\,\sigma_0^2}\right)
    $$

    The posterior mean is a weighted average of the prior mean $\mu_0$ and the sample mean $\bar{x}$, with weights determined by their relative precisions (inverse variances). As $n$ increases, the data dominate and the posterior concentrates around $\bar{x}$, regardless of the prior. Because the posterior is symmetric and unimodal, the mean, median, and MAP all coincide in this case.

## Exercises

**Exercise 1.**
State Bayes' theorem for a parameter $\theta$ given data $\mathbf{x}$. Identify and name each component (prior, likelihood, posterior, marginal likelihood).

??? success "Solution to Exercise 1"
    Bayes' theorem states:

    $$
    \underbrace{\pi(\theta \mid \mathbf{x})}_{\text{posterior}} = \frac{\overbrace{L(\mathbf{x} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{\pi(\theta)}^{\text{prior}}}{\underbrace{m(\mathbf{x})}_{\text{marginal likelihood}}}
    $$

    - **Prior** $\pi(\theta)$: the distribution of $\theta$ before seeing data, encoding prior beliefs.
    - **Likelihood** $L(\mathbf{x} \mid \theta)$: the probability of the observed data given $\theta$.
    - **Marginal likelihood** $m(\mathbf{x}) = \int L(\mathbf{x} \mid \theta)\pi(\theta)\,d\theta$: a normalizing constant ensuring the posterior integrates to 1.
    - **Posterior** $\pi(\theta \mid \mathbf{x})$: the updated distribution of $\theta$ after observing data.

---

**Exercise 2.**
Suppose the prior for a coin's probability of heads is $p \sim \text{Uniform}(0, 1)$. You flip the coin once and observe heads. Compute the posterior distribution $\pi(p \mid H)$.

??? success "Solution to Exercise 2"
    The prior is $\pi(p) = 1$ for $p \in (0, 1)$ (which is $\text{Beta}(1, 1)$). The likelihood for one head is $L(H \mid p) = p$.

    By Bayes' theorem:

    $$
    \pi(p \mid H) = \frac{p \cdot 1}{\int_0^1 p \, dp} = \frac{p}{1/2} = 2p
    $$

    This is the $\text{Beta}(2, 1)$ density. The posterior mean is $E[p \mid H] = 2/3$, shifted upward from the prior mean of $1/2$, reflecting the evidence from observing heads.

---

**Exercise 3.**
Explain the relationship between the posterior mode (MAP estimate) and the MLE. Under what condition does the MAP estimate equal the MLE?

??? success "Solution to Exercise 3"
    The **MAP (Maximum A Posteriori)** estimate maximizes the posterior:

    $$
    \hat{\theta}_{\text{MAP}} = \arg\max_\theta \bigl[L(\mathbf{x} \mid \theta) \cdot \pi(\theta)\bigr] = \arg\max_\theta \bigl[\log L(\mathbf{x} \mid \theta) + \log \pi(\theta)\bigr]
    $$

    The **MLE** maximizes the likelihood alone: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\mathbf{x} \mid \theta)$.

    The MAP equals the MLE when the prior is flat (uniform/non-informative), i.e., $\pi(\theta) \propto c$ (constant), because $\log \pi(\theta)$ becomes a constant that does not affect the optimization. The MAP also approaches the MLE as $n \to \infty$, because the likelihood dominates the prior for large samples.

---

**Exercise 4.**
A prior $\pi(\theta)$ assigns probability 0.7 to $\theta = 0$ and 0.3 to $\theta = 1$. The likelihood satisfies $P(X = 1 \mid \theta = 0) = 0.2$ and $P(X = 1 \mid \theta = 1) = 0.9$. After observing $X = 1$, compute the posterior probabilities $P(\theta = 0 \mid X = 1)$ and $P(\theta = 1 \mid X = 1)$.

??? success "Solution to Exercise 4"
    By Bayes' theorem for discrete $\theta$:

    $$
    P(\theta = 0 \mid X = 1) = \frac{P(X = 1 \mid \theta = 0) \cdot P(\theta = 0)}{P(X = 1)}
    $$

    First compute the marginal:

    $$
    P(X = 1) = 0.2 \times 0.7 + 0.9 \times 0.3 = 0.14 + 0.27 = 0.41
    $$

    Then:

    $$
    P(\theta = 0 \mid X = 1) = \frac{0.14}{0.41} \approx 0.341
    $$

    $$
    P(\theta = 1 \mid X = 1) = \frac{0.27}{0.41} \approx 0.659
    $$

    The observation $X = 1$ shifted the posterior toward $\theta = 1$ (from prior 0.3 to posterior 0.659), because $X = 1$ is much more likely under $\theta = 1$.
