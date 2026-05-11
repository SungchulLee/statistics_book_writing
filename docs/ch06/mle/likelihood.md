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

## Exercises

**Exercise 1.**
Write the likelihood function and log-likelihood function for $n$ independent observations $x_1, \dots, x_n$ from a $N(\mu, \sigma^2)$ distribution, treating both $\mu$ and $\sigma^2$ as unknown.

??? success "Solution to Exercise 1"
    The likelihood is:

    $$
    L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2\right)
    $$

    The log-likelihood is:

    $$
    \ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2
    $$

---

**Exercise 2.**
Explain the difference between the likelihood function $L(\theta; \mathbf{x})$ and the probability function $P(\mathbf{x}; \theta)$. Why is the likelihood not a probability distribution over $\theta$?

??? success "Solution to Exercise 2"
    The **probability function** $P(\mathbf{x}; \theta)$ treats $\theta$ as fixed and $\mathbf{x}$ as the variable. For fixed $\theta$, it sums (or integrates) to 1 over all possible data outcomes.

    The **likelihood function** $L(\theta; \mathbf{x})$ treats $\mathbf{x}$ as fixed (the observed data) and $\theta$ as the variable. It uses the same formula as the probability function but reverses the roles.

    The likelihood is not a probability distribution over $\theta$ because it does not integrate to 1 over the parameter space. In fact, $\int L(\theta; \mathbf{x})\,d\theta$ can be any positive number (or even infinite). To obtain a proper distribution over $\theta$, one must multiply by a prior and normalize (Bayesian approach), yielding the posterior distribution.

---

**Exercise 3.**
For a sample of size $n = 3$ with observations $x_1 = 2, x_2 = 5, x_3 = 3$ from a $\text{Poisson}(\lambda)$ distribution, compute the likelihood and log-likelihood at $\lambda = 3$ and $\lambda = 4$. Which value is more likely?

??? success "Solution to Exercise 3"
    The Poisson likelihood is $L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}$.

    At $\lambda = 3$: $L(3) = \frac{3^2 e^{-3}}{2!} \cdot \frac{3^5 e^{-3}}{5!} \cdot \frac{3^3 e^{-3}}{3!} = \frac{3^{10} e^{-9}}{2! \cdot 5! \cdot 3!}$

    $$
    = \frac{59049 \times 0.0001234}{2 \times 120 \times 6} = \frac{7.2876}{1440} \approx 0.005061
    $$

    At $\lambda = 4$: $L(4) = \frac{4^{10} e^{-12}}{1440} = \frac{1048576 \times 6.144 \times 10^{-6}}{1440} \approx 0.004473$

    The log-likelihoods: $\ell(3) = 10\ln 3 - 9 - \ln 1440 \approx -5.287$ and $\ell(4) = 10\ln 4 - 12 - \ln 1440 \approx -5.411$.

    Since $\ell(3) > \ell(4)$, $\lambda = 3$ is more likely. Note: the MLE is $\hat{\lambda} = \bar{x} = 10/3 \approx 3.33$.

---

**Exercise 4.**
Explain why maximizing the log-likelihood is equivalent to maximizing the likelihood. State one practical advantage of working with the log-likelihood.

??? success "Solution to Exercise 4"
    Since the logarithm is a strictly increasing function, $L(\theta_1) > L(\theta_2)$ if and only if $\ell(\theta_1) > \ell(\theta_2)$. Therefore the value of $\theta$ that maximizes $L$ also maximizes $\ell$, and vice versa.

    **Practical advantage:** The log-likelihood converts products into sums:

    $$
    \ell(\theta) = \sum_{i=1}^n \log f(x_i; \theta)
    $$

    This is easier to differentiate (sum rule vs. product rule) and avoids numerical underflow. For large $n$, the likelihood $L(\theta) = \prod f(x_i; \theta)$ can be astronomically small (a product of $n$ numbers less than 1), causing floating-point underflow. The log-likelihood, being a sum, remains in a numerically tractable range.
