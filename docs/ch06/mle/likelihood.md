# Likelihood Function

## From Probability to Likelihood

In probability, we fix a parameter $\theta$ and ask: what data are likely to arise? The likelihood function reverses this perspective. Given data that have already been observed, we ask: which values of $\theta$ make these observations most plausible? This shift from "parameters generate data" to "data inform us about parameters" is the foundation of likelihood-based inference and underpins maximum likelihood estimation, likelihood ratio tests, and information criteria.

## Intuition

Consider a simple experiment: we flip a coin 10 times and observe 7 heads. If the coin has probability $p = 0.5$ of landing heads, we can compute the probability of this outcome. But we can also compute the probability of this outcome for $p = 0.6$, $p = 0.7$, or any other value of $p$. Plotting these probabilities as a function of $p$ — with the data held fixed — gives the likelihood function. The value of $p$ that produces the tallest peak is the "most likely" parameter value, and finding it is the essence of maximum likelihood estimation.

## General Definition

The likelihood function for a parameter $\theta$ given observed data $\mathbf{x}$ is defined as the joint density (or joint mass function) evaluated at the observed data, viewed as a function of $\theta$:

$$
L(\theta; \mathbf{x}) = f(\mathbf{x}; \theta)
$$

Here $f(\mathbf{x}; \theta)$ is the probability density function (for continuous data) or probability mass function (for discrete data). The semicolon notation emphasizes that $\theta$ is a fixed parameter, not a random variable — we are not conditioning on $\theta$ in the probabilistic sense.

### The iid Case

When the data consist of $n$ independent and identically distributed observations $x_1, \ldots, x_n$, the joint density factors into a product, and the likelihood takes the form

$$
L(\theta; \mathbf{x}) = \prod_{i=1}^n f(x_i; \theta)
$$

This product structure is specific to the iid setting. For dependent data (e.g., time series), the likelihood takes a different form but the underlying idea remains the same.

!!! example "Bernoulli Likelihood"

    Suppose $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$ and we observe $k$ successes in $n$ trials. The likelihood function is

    $$
    L(p; \mathbf{x}) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^k (1-p)^{n-k}
    $$

    For $n = 10$ and $k = 7$, we can evaluate $L(0.5) = 0.5^{10} \approx 0.001$ and $L(0.7) = 0.7^7 \cdot 0.3^3 \approx 0.0022$. The value $p = 0.7$ makes the observed data about twice as plausible as $p = 0.5$.

## Log-Likelihood

Working with the likelihood directly is inconvenient because products of many small numbers cause numerical underflow. The **log-likelihood** transforms the product into a sum:

$$
\ell(\theta; \mathbf{x}) = \log L(\theta; \mathbf{x}) = \sum_{i=1}^n \log f(x_i; \theta)
$$

Since the logarithm is a strictly increasing function, maximizing $\ell$ is equivalent to maximizing $L$. The log-likelihood is the standard working tool in practice.

!!! example "Bernoulli Log-Likelihood"

    Continuing the Bernoulli example:

    $$
    \ell(p; \mathbf{x}) = k \log p + (n - k) \log(1-p)
    $$

    For $n = 10$ and $k = 7$: $\ell(0.5) = 10 \log(0.5) \approx -6.93$ and $\ell(0.7) = 7\log(0.7) + 3\log(0.3) \approx -6.12$. The higher log-likelihood at $p = 0.7$ confirms it is a more plausible parameter value.

## Key Properties

### Likelihood Is Not a Probability

The most important conceptual point about the likelihood function is that **it is not a probability distribution over $\theta$**. Specifically, there is no guarantee that

$$
\int L(\theta; \mathbf{x}) \, d\theta = 1
$$

In general, this integral may converge to any positive number, or even diverge. The likelihood tells us the relative plausibility of different parameter values, but it does not assign probabilities to parameters. Assigning probabilities to parameters requires Bayesian methods with a prior distribution.

### Relative Plausibility

The absolute value of $L(\theta; \mathbf{x})$ at any single point is not directly interpretable. What matters is the **likelihood ratio** between two parameter values:

$$
\frac{L(\theta_1; \mathbf{x})}{L(\theta_2; \mathbf{x})}
$$

A ratio of 10 means $\theta_1$ makes the data 10 times more plausible than $\theta_2$. This ratio is invariant to multiplicative constants, which is why the likelihood is often defined "up to a constant."

### Computational Convenience of Log-Likelihood

The log-likelihood converts products to sums, which offers three practical advantages:

- **Numerical stability**: products of many small probabilities cause underflow; sums of log-probabilities do not.
- **Differentiation**: sums are easier to differentiate than products, simplifying the search for the maximum.
- **Additivity**: for independent observations, the total log-likelihood is the sum of individual contributions, making it easy to add or remove data points.
