# Introduction to MLE

## Overview

Maximum Likelihood Estimation (MLE) is a method used to estimate the parameters of a statistical model by maximizing the likelihood function. The likelihood function measures how well the model, with certain parameters, explains the observed data. The MLE estimates are the parameters that make the observed data most probable.

In simpler terms, MLE finds the parameter values that make the observed data most "likely." Compared to other methods, such as the Method of Moments, MLE is often preferred because of its desirable properties, including consistency and asymptotic efficiency.

## Mathematical Formulation

The MLE is formally defined as:

$$
\hat{\theta}_{MLE} = \arg \max_{\theta} L(\theta \mid \mathbf{x}) = \arg \max_{\theta} \prod_{i=1}^n f(x_i \mid \theta)
$$

Here, $\theta$ represents the parameters we want to estimate, and $L(\theta \mid \mathbf{x})$ is the likelihood function, which is the product of the probability density function (or probability mass function for discrete data) evaluated at the observed data points $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.

## Log-Likelihood

For ease of computation, we often work with the log-likelihood function:

$$
\log L(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

This transformation simplifies maximization by turning the product of probabilities into a sum of log-probabilities. Since $\log$ is a monotonically increasing function, maximizing the log-likelihood is equivalent to maximizing the likelihood itself.

## Maximum Likelihood Principle

The key equivalences in the MLE framework:

$$
\text{argmax}_{\theta}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{\theta}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{\theta}\; J
$$

where $L$ is the likelihood, $\ell = \log L$ is the log-likelihood, and $J = -\ell$ is the cost function (negative log-likelihood).

## Properties of MLEs

| Property | Description |
|----------|-------------|
| **Consistency** | $\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$ as $n \to \infty$ |
| **Asymptotic normality** | $\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$ |
| **Asymptotic efficiency** | Achieves the Cramér–Rao lower bound asymptotically |
| **Invariance** | If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ |

## Summary

MLE provides a principled, general-purpose approach to parameter estimation. It connects naturally to:

- **Cost functions** in machine learning (cross-entropy loss = negative log-likelihood for classification)
- **Bayesian inference** (MLE is the MAP estimate with a flat prior)
- **Information theory** (minimizing KL divergence between the model and data)

## Exercises

**Exercise 1.**
Daily log-returns: $r_t \sim N(\mu, \sigma^2)$. With $n = 252$, $\bar r = 0.0004$, $s = 0.015$: (a) MLEs of $\mu_{\text{ann}} = 252\mu$, $\sigma_{\text{ann}} = \sigma\sqrt{252}$. (b) 95% CIs via asymptotic normality. (c) Why is $\hat\mu$ so much noisier than $\hat\sigma$?

??? success "Solution to Exercise 1"
    (a) $\hat\mu_{\text{ann}} = 252 \cdot 0.0004 = 0.1008$ (10.08%). $\hat\sigma_{\text{ann}} = 0.015 \sqrt{252} \approx 0.2381$ (23.81%).

    (b) $\mathrm{SE}(\hat\mu) = s/\sqrt n = 0.015/\sqrt{252} \approx 0.000945$. $\mathrm{SE}(\hat\mu_{\text{ann}}) = 252 \cdot 0.000945 \approx 0.238$. 95% CI: $(-0.366, 0.568)$.

    For $\sigma$: asymptotic SE is $\sigma/\sqrt{2n}$, so $\mathrm{SE}(\hat\sigma_{\text{ann}}) \approx 0.0106$. 95% CI: $(0.217, 0.259)$.

    (c) SE of $\hat\mu_{\text{ann}}$ ≈ 0.238 — *larger* than the point estimate 0.101. CI spans $-37\%$ to $+57\%$. Volatility CI is tight (≈ ±2 pp around 24%). **The drift is hopelessly noisy at 1-year horizons; the volatility is well-estimated.** A classical finance pitfall: relying on historical sample mean to predict future returns.

---

**Exercise 2.**
**Derive MLE for normal.** Given i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$, derive $\hat\mu$ and $\hat\sigma^2$.

??? success "Solution to Exercise 2"
    Log-likelihood: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(X_i - \mu)^2$.

    $\partial\ell/\partial\mu = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial\ell/\partial\sigma^2 = -n/(2\sigma^2) + \sum(X_i - \mu)^2/(2\sigma^4) = 0 \Rightarrow \hat\sigma^2_{\text{MLE}} = (1/n)\sum(X_i - \bar X)^2$.

    Note MLE divides by $n$, not $n - 1$: biased ($\mathbb{E}[\hat\sigma^2] = (n-1)\sigma^2/n$). For unbiased estimation use Bessel's correction.

---

**Exercise 3.**
**MLE for Bernoulli/Binomial.** Sample $X \sim \mathrm{Binomial}(n, p)$. Derive $\hat p_{\text{MLE}}$.

??? success "Solution to Exercise 3"
    Likelihood: $L(p) = \binom{n}{X} p^X (1-p)^{n-X} \propto p^X(1-p)^{n-X}$.

    Log-likelihood: $\ell(p) = X\ln p + (n-X)\ln(1-p)$.

    $\ell'(p) = X/p - (n-X)/(1-p) = 0$.

    $X(1-p) = (n-X)p \Rightarrow X = np \Rightarrow \hat p = X/n$.

    The MLE is the sample proportion. Unbiased: $\mathbb{E}[\hat p] = p$.

---

**Exercise 4.**
**Likelihood vs. probability.** Distinguish "likelihood" from "probability" in MLE language. Why is the likelihood a function of $\theta$ but not a probability density in $\theta$?

??? success "Solution to Exercise 4"
    **Probability:** $P(X = x \mid \theta)$ — function of $x$ for fixed $\theta$. Sums/integrates to 1 over $x$.

    **Likelihood:** $L(\theta) = P(X = x \mid \theta)$ — same mathematical formula, but viewed as a function of $\theta$ for *fixed observed* $x$. Does not integrate to 1 over $\theta$.

    **Why not a density in $\theta$:** $\theta$ is a *parameter*, not a random variable in frequentist statistics. The likelihood ranks values of $\theta$ by how well they explain the data, not by how probable they are. MLE picks the $\theta$ that maximizes likelihood, but this is not "the most probable parameter."

    In Bayesian inference, $\theta$ becomes a random variable and we compute the *posterior* $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$, which **is** a density in $\theta$. The likelihood remains the same; what changes is the framework.

---

**Exercise 5.**
**Asymptotic normality of MLE.** State the result and verify for the Bernoulli case.

??? success "Solution to Exercise 5"
    **Asymptotic normality of MLE:**

    $$
    \sqrt n(\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))
    $$

    where $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$ is the Fisher information per observation.

    **Bernoulli verification:** $\log f = x\log p + (1-x)\log(1-p)$. Second derivative: $-x/p^2 - (1-x)/(1-p)^2$. Expectation: $-p/p^2 - (1-p)/(1-p)^2 = -1/p - 1/(1-p) = -1/[p(1-p)]$.

    So $I(p) = 1/[p(1-p)]$, and $\mathrm{Var}(\hat p) = p(1-p)/n$. Matches the CLT for sample proportion directly.

    Implication: MLEs achieve the **Cramér-Rao lower bound** asymptotically — they are asymptotically efficient. No unbiased estimator can have smaller asymptotic variance.

---

**Exercise 6.**
**MLE failure modes.** Give one example each: (a) MLE doesn't exist; (b) MLE on boundary; (c) MLE inconsistent.

??? success "Solution to Exercise 6"
    **(a) MLE doesn't exist:** for $X \sim \mathrm{Uniform}(0, \theta)$, the likelihood is $1/\theta^n$ on $\theta \ge \max X_i$, 0 below. As $\theta \to \infty$, $L \to 0$. As $\theta \to \max X_i$, $L \to (1/\max X_i)^n$, a *supremum* but at boundary. Strictly, no interior maximum; MLE is the boundary $\hat\theta = \max X_i$.

    **(b) MLE on boundary:** the uniform example above. Boundary MLE has non-standard asymptotic distribution (not $N(0, 1/I)$); convergence rate is $n$ rather than $\sqrt n$.

    **(c) MLE inconsistent:** Neyman-Scott problem. Observations $X_{ij} \sim N(\mu_i, \sigma^2)$ for $i = 1, \ldots, n$, $j = 1, 2$. Number of parameters ($\mu_i$'s + $\sigma^2$) grows with sample size. $\hat\sigma^2_{\text{MLE}} \to \sigma^2/2$, not $\sigma^2$. Inconsistent because the dimension of the parameter space scales with $n$.

    These failure modes motivate refinements: bias-corrected MLE, penalized likelihood, profile likelihood, marginal likelihood (Bayesian).
