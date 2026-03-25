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
