# MAP Estimation

After computing the posterior distribution $\pi(\theta \mid \mathbf{x})$, we often want a single point estimate that summarizes our updated beliefs. While the posterior mean and median are common choices, the posterior mode --- known as the **Maximum A Posteriori (MAP)** estimate --- has a special appeal: it connects Bayesian inference directly to penalized optimization, bridging the gap between Bayesian and frequentist thinking.

## Definition

The MAP estimator selects the parameter value that maximizes the posterior density:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; \pi(\theta \mid \mathbf{x})
$$

Since Bayes' theorem gives $\pi(\theta \mid \mathbf{x}) = f(\mathbf{x} \mid \theta)\,\pi(\theta) / f(\mathbf{x})$, and the marginal likelihood $f(\mathbf{x})$ does not depend on $\theta$, maximizing the posterior is equivalent to maximizing the numerator:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; f(\mathbf{x} \mid \theta)\,\pi(\theta)
$$

Taking logarithms (a monotone transformation that preserves the maximizer), this becomes:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; \bigl[\log f(\mathbf{x} \mid \theta) + \log \pi(\theta)\bigr]
$$

The MAP estimate therefore maximizes the log-likelihood plus a log-prior term. This additive structure is the key to understanding MAP's connection to both MLE and regularization.

## Relationship to MLE

The MAP objective differs from MLE only by the addition of $\log \pi(\theta)$. When the sample size $n$ is large, the log-likelihood $\log f(\mathbf{x} \mid \theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$ grows proportionally to $n$, while the log-prior remains a fixed function of $\theta$. As a result, under standard regularity conditions and provided the prior is positive in a neighborhood of the true parameter value, the prior's influence vanishes and the MAP estimate converges to the MLE:

$$
\hat{\theta}_{\text{MAP}} \to \hat{\theta}_{\text{MLE}} \quad \text{as } n \to \infty
$$

!!! example "MAP vs MLE for a Normal Mean"
    Suppose $X_1, \ldots, X_n \overset{iid}{\sim} N(\mu, 1)$ with prior $\mu \sim N(0, \sigma_0^2)$. The MAP estimate is

    $$
    \hat{\mu}_{\text{MAP}} = \frac{n\sigma_0^2}{n\sigma_0^2 + 1}\,\bar{X}
    $$

    When $n = 1$ and $\sigma_0^2 = 1$, the MAP estimate is $\bar{X}/2$, a compromise between the prior mean $0$ and the data. When $n = 100$, the MAP estimate is approximately $0.99\,\bar{X}$, nearly identical to the MLE $\hat{\mu}_{\text{MLE}} = \bar{X}$.

## Relationship to Regularization

The log-prior term $\log \pi(\theta)$ acts as a penalty that discourages certain parameter values. Different prior families produce different penalty structures.

**Gaussian prior and L2 regularization.** If $\theta_j \overset{iid}{\sim} N(0, \sigma_0^2)$, the log-prior is

$$
\log \pi(\theta) = \text{const} - \frac{1}{2\sigma_0^2}\sum_j \theta_j^2
$$

Maximizing $\log f(\mathbf{x} \mid \theta) + \log \pi(\theta)$ is therefore equivalent to minimizing the negative log-likelihood plus an L2 penalty $\lambda \|\theta\|_2^2$ with $\lambda = 1/(2\sigma_0^2)$. This is exactly Ridge regression.

**Laplace prior and L1 regularization.** If $\theta_j \overset{iid}{\sim} \text{Laplace}(0, b)$, the log-prior is

$$
\log \pi(\theta) = \text{const} - \frac{1}{b}\sum_j |\theta_j|
$$

Maximizing this is equivalent to minimizing the negative log-likelihood plus an L1 penalty $\lambda \|\theta\|_1$ with $\lambda = 1/b$. This is exactly Lasso regression, which promotes sparsity because the L1 penalty drives some coefficients to exactly zero.

| Prior | Log-Prior Penalty | Regularization |
|---|---|---|
| $N(0, \sigma_0^2)$ | $-\frac{1}{2\sigma_0^2}\|\theta\|_2^2$ | Ridge (L2) |
| Laplace$(0, b)$ | $-\frac{1}{b}\|\theta\|_1$ | Lasso (L1) |

This correspondence reveals that regularized estimation, often motivated by purely frequentist arguments about overfitting, has a natural Bayesian interpretation: the penalty encodes prior beliefs about the parameter magnitudes.
