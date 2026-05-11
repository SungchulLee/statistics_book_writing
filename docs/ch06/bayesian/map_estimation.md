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

## Exercises

**Exercise 1.**
Bayesian-inspired estimator for Bernoulli: $\hat\theta_B = (\sum X_i + a)/(n + a + b)$. (a) Bias and variance. (b) For $a = b = \sqrt n/2$, show biased but consistent. (c) Compare MSE with MLE at $\theta = 0.5, n = 10$.

??? success "Solution to Exercise 1"
    (a) Let $S = \sum X_i$, $\mathbb{E}[S] = n\theta$, $\mathrm{Var}(S) = n\theta(1-\theta)$.

    $\mathbb{E}[\hat\theta_B] = (n\theta + a)/(n + a + b)$. $\mathrm{Bias}(\hat\theta_B) = (a - (a+b)\theta)/(n + a + b)$.

    $\mathrm{Var}(\hat\theta_B) = n\theta(1-\theta)/(n + a + b)^2$.

    (b) With $a = b = \sqrt n/2$: $\mathrm{Bias} = \sqrt n(0.5 - \theta)/(n + \sqrt n) \to 0$. Variance $\to 0$. Consistent.

    (c) At $\theta = 0.5, n = 10$: bias = 0 (the prior centers at $0.5$, where $\theta$ actually is). $\mathrm{Var}(\hat\theta_B) \approx 0.0144$. MLE: $\mathrm{Var}(\hat p) = 0.025$. The Bayesian estimator has lower MSE at $\theta = 0.5$ — shrinkage toward the prior reduces variance at the cost of bias elsewhere.

---

**Exercise 2.**
**Derive MAP estimator** for $X \sim \mathrm{Binomial}(n, p)$ with $\mathrm{Beta}(\alpha, \beta)$ prior.

??? success "Solution to Exercise 2"
    Posterior: $\pi(p \mid x) \propto p^x(1-p)^{n-x} \cdot p^{\alpha-1}(1-p)^{\beta-1} = p^{x+\alpha-1}(1-p)^{n-x+\beta-1}$.

    This is $\mathrm{Beta}(x + \alpha, n - x + \beta)$ (conjugacy).

    Mode of Beta$(\alpha', \beta')$: $(\alpha' - 1)/(\alpha' + \beta' - 2)$ when both $> 1$.

    $\hat p_{\mathrm{MAP}} = (x + \alpha - 1)/(n + \alpha + \beta - 2)$.

    **Special cases:**

    - $\alpha = \beta = 1$ (uniform prior): $\hat p_{\mathrm{MAP}} = x/n = \hat p_{\mathrm{MLE}}$. Flat prior recovers MLE.
    - $\alpha = \beta = 0.5$ (Jeffreys prior): mild shrinkage toward 0.5.
    - $\alpha = \beta = $ large: heavy shrinkage toward $\alpha/(\alpha + \beta) = 0.5$.

    The MAP point estimator differs from the posterior mean: $\hat p_{\mathrm{Bayes,mean}} = (x + \alpha)/(n + \alpha + \beta)$. Mean is preferred when symmetric loss (squared error); mode is preferred when 0-1 loss.

---

**Exercise 3.**
**MAP vs. MLE.** When does MAP equal MLE? When does it not?

??? success "Solution to Exercise 3"
    MAP maximizes $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$. MLE maximizes $L(\theta)$ alone.

    **MAP = MLE** iff $\pi(\theta)$ is constant over the support — i.e., a flat (improper) prior. Equivalently, the prior is uninformative.

    **MAP $\ne$ MLE** when the prior has shape — e.g., a Beta(2, 2) puts more mass near 0.5 than at the boundaries, pulling MAP toward 0.5 relative to MLE.

    With informative priors, MAP introduces **shrinkage toward the prior mean/mode**. The strength of shrinkage scales with $1/n$ — informative priors dominate for small samples; the data dominates for large samples.

    **Practical use:** for very small $n$ (or extreme observed counts), MAP with weakly informative priors is more stable than MLE. For example, MLE of $p$ from $X = 0$ heads in $n$ flips is $\hat p = 0$ (impossible to be exactly 0); MAP with Beta(1.5, 1.5) gives $\hat p = 0.5/(n + 1)$ — small but nonzero.

---

**Exercise 4.**
**Conjugate priors.** Why are conjugate priors convenient computationally? Give one example beyond Beta-Bernoulli.

??? success "Solution to Exercise 4"
    **Conjugate prior** for likelihood family $L(\theta; X)$: prior $\pi(\theta)$ such that the posterior $\pi(\theta \mid X)$ is in the same family.

    **Convenience:**

    - Posterior is computed analytically — no numerical integration.
    - Updating with new data simply updates hyperparameters (e.g., Beta $\to$ Beta with shifted parameters).
    - Sequential / online updating is trivial.

    **Examples:**

    | Likelihood | Conjugate prior | Posterior |
    |---|---|---|
    | Bernoulli$(p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + x, \beta + n - x)$ |
    | Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum X_i, \beta + n)$ |
    | Normal$(\mu, \sigma^2)$, $\sigma$ known | Normal$(\mu_0, \tau_0^2)$ | Normal (updated) |
    | Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + n, \beta + \sum X_i)$ |

    **Modern caveat:** with MCMC and HMC available, conjugacy is no longer essential. Models can use arbitrary priors and likelihoods, sampling the posterior numerically. Conjugacy remains valuable for educational tractability and as a baseline.

---

**Exercise 5.**
**Posterior mean vs. MAP.** For $\pi(\theta \mid x) = \mathrm{Beta}(20, 5)$, compute both. Why might they differ?

??? success "Solution to Exercise 5"
    $\mathrm{Beta}(\alpha, \beta)$:

    - Mean: $\alpha/(\alpha + \beta) = 20/25 = 0.80$.
    - Mode: $(\alpha - 1)/(\alpha + \beta - 2) = 19/23 \approx 0.826$.

    They differ because $\mathrm{Beta}(20, 5)$ is asymmetric (right-skewed toward 1). For symmetric posteriors (e.g., normal), mean = mode = median. For skewed posteriors, the mode lies on the "heavier" side, between median and mean.

    **Which to use?**

    - **Squared error loss:** posterior mean is optimal (minimizes expected squared error).
    - **0-1 loss:** posterior mode (MAP) is optimal.
    - **Absolute error loss:** posterior median.

    Choose the point estimator based on the loss function. In practice, the posterior **distribution** is more informative than any point summary; full posterior should be reported when possible.

---

**Exercise 6.**
**Improper priors.** A prior with $\int \pi(\theta) d\theta = \infty$ (e.g., $\pi(\mu) = 1$ on $\mathbb{R}$) is **improper**. Can we still compute a valid posterior? When?

??? success "Solution to Exercise 6"
    Yes — if the *posterior* is proper. The condition is that the numerator $L(\theta) \pi(\theta)$ has finite integral over $\theta$:

    $$
    \int L(\theta) \pi(\theta) d\theta < \infty
    $$

    Then $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$ can be normalized.

    **Common improper priors:**

    - $\pi(\mu) = 1$ (Lebesgue measure on $\mathbb{R}$) for location parameters.
    - $\pi(\sigma) = 1/\sigma$ (Jeffreys for scale) for variance/scale parameters.
    - $\pi(p) \propto 1/\sqrt{p(1-p)}$ (Jeffreys for Bernoulli).

    **Concerns with improper priors:**

    - Posterior may be improper if data is uninformative.
    - Bayes factors and model comparison can break (the marginal likelihood is undefined).
    - Counterintuitive paradoxes can arise (Lindley's paradox).

    **Modern practice:** use weakly informative *proper* priors (e.g., $N(0, 10^4)$ for a location parameter) that approximate improper priors but stay proper. Avoids the pitfalls while retaining the "minimal prior information" intent.
