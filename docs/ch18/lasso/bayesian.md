# Bayesian Interpretation (Laplace Prior)

Just as ridge regression arises from a Gaussian prior on the coefficients, the lasso arises from a Laplace (double-exponential) prior. The shape of the Laplace density, with its sharp peak at zero and heavier tails than the Gaussian, provides the Bayesian mechanism behind sparsity: the prior concentrates substantial probability mass at exactly zero while still allowing large coefficients when the data provide strong evidence.

## The Laplace Distribution

The Laplace distribution with location 0 and scale $b > 0$ has density:

$$
p(\beta_j) = \frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right)
$$

Key properties of this distribution compared to the Gaussian:

- **Cusp at zero.** The density has a non-differentiable peak at $\beta_j = 0$, unlike the smooth Gaussian peak. This cusp concentrates more probability mass near zero.
- **Heavier tails.** The Laplace density decays exponentially (linearly in the log scale), while the Gaussian decays as $\exp(-\beta_j^2)$ (quadratically in the log scale). The Laplace therefore assigns more probability to large values of $|\beta_j|$.
- **Variance.** The variance of the Laplace$(0, b)$ distribution is $2b^2$.

## MAP Derivation

Place independent Laplace priors on each coefficient:

$$
\beta_j \overset{\text{iid}}{\sim} \text{Laplace}(0, b)
$$

Combined with the Gaussian likelihood $\mathbf{y} \mid \boldsymbol{\beta} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})$, the joint prior is:

$$
p(\boldsymbol{\beta}) = \prod_{j=1}^p \frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right) = \frac{1}{(2b)^p}\exp\left(-\frac{\|\boldsymbol{\beta}\|_1}{b}\right)
$$

The negative log-posterior is:

$$
-\log p(\boldsymbol{\beta} \mid \mathbf{y}) = \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{b}\|\boldsymbol{\beta}\|_1 + \text{const}
$$

Minimizing this expression is equivalent to solving the lasso problem with:

$$
\lambda = \frac{\sigma^2}{nb}
$$

where the factor of $n$ arises from the $1/(2n)$ normalization convention in the lasso objective. The MAP estimate under a Laplace prior is the lasso estimator.

## Why the Laplace Prior Induces Sparsity

The MAP estimate corresponds to finding the mode of the posterior distribution. The cusp of the Laplace prior at zero creates a "ridge" in the posterior along the coordinate hyperplanes $\beta_j = 0$. For coefficients with weak data support, the posterior mode lies exactly at zero because the prior's cusp pulls the mode to $\beta_j = 0$ unless the likelihood pulls sufficiently hard in another direction.

!!! note "MAP versus Posterior Mean"
    The sparsity of the lasso is a property of the MAP estimate (the posterior mode), not of the full posterior. The posterior mean under a Laplace prior is generally not sparse. If full posterior inference is desired (e.g., credible intervals), the Bayesian lasso requires MCMC methods because the Laplace prior is not conjugate to the Gaussian likelihood.

## Comparing Gaussian and Laplace Priors

The following table summarizes how the choice of prior shapes the resulting estimator:

| Property | Gaussian prior $N(0, \tau^2)$ | Laplace prior Laplace$(0, b)$ |
|---|---|---|
| Density at 0 | Smooth, finite | Cusp, finite |
| Tail behavior | Light (sub-Gaussian) | Heavier (exponential decay) |
| Resulting estimator | Ridge | Lasso |
| Penalty | $\lambda\|\boldsymbol{\beta}\|_2^2$ | $\lambda\|\boldsymbol{\beta}\|_1$ |
| Sparsity of MAP | No | Yes |
| Conjugacy | Yes (posterior is Gaussian) | No |
| Posterior computation | Closed-form | Requires MCMC |
| Regularization parameter | $\lambda = \sigma^2/\tau^2$ | $\lambda = \sigma^2/(nb)$ |

The Gaussian prior penalizes large coefficients quadratically, resulting in smooth, proportional shrinkage. The Laplace prior penalizes them linearly, resulting in the soft-thresholding behavior that produces exact zeros.

## The Bayesian Lasso

Park and Casella (2008) developed a full Bayesian treatment of the lasso, called the **Bayesian lasso**. The key idea is to represent the Laplace prior as a scale mixture of normals:

$$
\text{Laplace}(0, b) = \int_0^\infty N(0, s^2)\,\frac{1}{2b^2}\exp\left(-\frac{s^2}{2b^2}\right)\, ds^2
$$

This representation introduces latent variance parameters $s_j^2$ for each coefficient, enabling a Gibbs sampler that alternates between:

1. Sampling $\boldsymbol{\beta} \mid s_1^2, \ldots, s_p^2, \mathbf{y}$ from a Gaussian conditional.
2. Sampling each $s_j^2 \mid \beta_j$ from an inverse-Gaussian distribution.

!!! tip "When to Use the Bayesian Lasso"
    Use the Bayesian lasso when uncertainty quantification (credible intervals) for sparse models is needed. The frequentist lasso provides point estimates and variable selection but does not directly provide valid confidence intervals for the selected coefficients.

## Spike-and-Slab Priors

For even stronger sparsity enforcement, **spike-and-slab** priors place a point mass at zero (the "spike") and a diffuse distribution on nonzero values (the "slab"):

$$
p(\beta_j) = (1 - \pi)\,\delta_0(\beta_j) + \pi\, g(\beta_j)
$$

where $\pi$ is the prior probability that $\beta_j \neq 0$ and $g$ is a continuous density (e.g., Gaussian). The MAP estimate under this prior corresponds to best subset selection with penalty $\lambda\|\boldsymbol{\beta}\|_0$ (the number of nonzero coefficients). The Laplace prior can be viewed as a continuous relaxation of the spike-and-slab, sitting between the Gaussian (no sparsity) and the spike-and-slab (strongest sparsity).

| Prior | Sparsity strength | MAP penalty | Computation |
|---|---|---|---|
| Gaussian $N(0, \tau^2)$ | None | $\lambda\|\boldsymbol{\beta}\|_2^2$ | Closed-form |
| Laplace$(0, b)$ | Moderate | $\lambda\|\boldsymbol{\beta}\|_1$ | Convex optimization |
| Spike-and-slab | Strongest | $\lambda\|\boldsymbol{\beta}\|_0$ | NP-hard (combinatorial) |

## Summary

The lasso is the MAP estimate under independent Laplace priors on the regression coefficients. The Laplace density's cusp at zero concentrates prior mass near zero, causing the posterior mode to sit exactly at $\beta_j = 0$ for weakly supported coefficients. The regularization parameter satisfies $\lambda = \sigma^2/(nb)$, linking the penalty strength to the prior scale. Unlike the Gaussian prior (which yields ridge), the Laplace prior is not conjugate, so full posterior inference requires MCMC. The Laplace prior occupies a natural middle ground between the Gaussian prior (no sparsity) and the spike-and-slab prior (maximum sparsity).
