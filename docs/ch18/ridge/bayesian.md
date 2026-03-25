# Bayesian Interpretation (Gaussian Prior)

Ridge regression can be derived purely from a penalized optimization perspective, but it also arises naturally as the maximum a posteriori (MAP) estimate in a Bayesian linear model with a Gaussian prior on the coefficients. This connection provides both a principled interpretation of the regularization parameter $\lambda$ and a pathway to full posterior inference, including uncertainty quantification through credible intervals.

## The Bayesian Linear Model

Assume the standard linear model with Gaussian errors:

$$
\mathbf{y} \mid \boldsymbol{\beta} \sim N(\mathbf{X}\boldsymbol{\beta},\; \sigma^2\mathbf{I}_n)
$$

Place a Gaussian prior on the coefficient vector, expressing the belief that coefficients are likely to be small:

$$
\boldsymbol{\beta} \sim N(\mathbf{0},\; \tau^2\mathbf{I}_p)
$$

The parameter $\tau^2$ controls the prior variance: a small $\tau^2$ expresses strong belief that coefficients are near zero, while a large $\tau^2$ represents a diffuse prior with minimal constraint.

## Deriving the MAP Estimate

The posterior distribution is proportional to the product of the likelihood and the prior:

$$
p(\boldsymbol{\beta} \mid \mathbf{y}) \propto p(\mathbf{y} \mid \boldsymbol{\beta})\, p(\boldsymbol{\beta})
$$

Taking the negative log-posterior:

$$
-\log p(\boldsymbol{\beta} \mid \mathbf{y}) = \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{2\tau^2}\|\boldsymbol{\beta}\|^2 + \text{const}
$$

Minimizing the negative log-posterior is equivalent to minimizing:

$$
\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\beta}\|^2
$$

This is the ridge regression objective with:

$$
\lambda = \frac{\sigma^2}{\tau^2}
$$

The MAP estimate (the mode of the posterior) is therefore the ridge estimate:

$$
\hat{\boldsymbol{\beta}}_{\text{MAP}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

!!! note "Interpreting the Regularization Parameter"
    The relationship $\lambda = \sigma^2/\tau^2$ reveals that the regularization strength reflects the **signal-to-noise ratio** of the prior. When prior variance $\tau^2$ is small relative to noise variance $\sigma^2$, we have large $\lambda$ and strong regularization. When the prior is diffuse ($\tau^2 \to \infty$), $\lambda \to 0$ and the MAP estimate approaches OLS.

## The Full Posterior Distribution

Because both the likelihood and the prior are Gaussian, the posterior is also Gaussian. Using standard results for conjugate Gaussian models:

$$
\boldsymbol{\beta} \mid \mathbf{y} \sim N\bigl(\boldsymbol{\mu}_{\text{post}},\; \boldsymbol{\Sigma}_{\text{post}}\bigr)
$$

where the posterior mean and covariance are:

$$
\boldsymbol{\mu}_{\text{post}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{ridge}}
$$

$$
\boldsymbol{\Sigma}_{\text{post}} = \sigma^2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
$$

The posterior mean coincides with the ridge estimate, and the posterior covariance provides a natural measure of uncertainty. Bayesian credible intervals for each coefficient $\beta_j$ follow directly from the marginal posterior:

$$
\beta_j \mid \mathbf{y} \sim N\bigl([\boldsymbol{\mu}_{\text{post}}]_j,\; [\boldsymbol{\Sigma}_{\text{post}}]_{jj}\bigr)
$$

A $95\%$ credible interval for $\beta_j$ is:

$$
[\boldsymbol{\mu}_{\text{post}}]_j \pm 1.96\,\sqrt{[\boldsymbol{\Sigma}_{\text{post}}]_{jj}}
$$

## Prior Strength and Its Effect

The prior $\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2\mathbf{I})$ assumes:

1. **Zero mean.** The prior expects coefficients to be near zero, which is reasonable when there is no prior information favoring specific directions.
2. **Isotropic variance.** All coefficients are given the same prior variance $\tau^2$, which is why standardizing predictors before applying ridge is important: it ensures the prior treats all coefficients symmetrically in their natural scales.
3. **Independence.** The prior assumes coefficients are a priori independent, encoded by the identity covariance structure.

| Prior variance $\tau^2$ | $\lambda = \sigma^2/\tau^2$ | Effect |
|---|---|---|
| Large (diffuse prior) | Small | Weak regularization, close to OLS |
| Moderate | Moderate | Balanced shrinkage |
| Small (tight prior) | Large | Strong shrinkage toward zero |
| $\tau^2 \to 0$ | $\lambda \to \infty$ | All coefficients shrunk to zero |

## Empirical Bayes and Hyperparameter Estimation

In practice, $\sigma^2$ and $\tau^2$ (and hence $\lambda$) are unknown. The **empirical Bayes** approach estimates these from the data by maximizing the marginal likelihood:

$$
p(\mathbf{y}) = \int p(\mathbf{y} \mid \boldsymbol{\beta})\, p(\boldsymbol{\beta})\, d\boldsymbol{\beta}
$$

For the Gaussian model, this integral is tractable:

$$
\mathbf{y} \sim N(\mathbf{0},\; \sigma^2\mathbf{I} + \tau^2\mathbf{X}\mathbf{X}^\top)
$$

Maximizing $p(\mathbf{y})$ over $\sigma^2$ and $\tau^2$ provides data-driven estimates of the regularization strength, connecting the Bayesian and cross-validation approaches to selecting $\lambda$.

!!! tip "Bayesian vs Frequentist Uncertainty"
    The posterior covariance $\boldsymbol{\Sigma}_{\text{post}}$ is smaller than the frequentist covariance of OLS ($\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$) because it incorporates the prior information. Bayesian credible intervals for ridge coefficients are therefore narrower than OLS confidence intervals, reflecting the reduced uncertainty from regularization.

## Connection to Other Priors

The Gaussian prior is just one choice. Different priors lead to different regularization methods:

| Prior on $\boldsymbol{\beta}$ | Regularization | Penalty |
|---|---|---|
| $N(\mathbf{0}, \tau^2\mathbf{I})$ (Gaussian) | Ridge | $\lambda\|\boldsymbol{\beta}\|_2^2$ |
| Laplace$(0, b)$ | Lasso | $\lambda\|\boldsymbol{\beta}\|_1$ |
| Spike-and-slab | Best subset selection | $\lambda\|\boldsymbol{\beta}\|_0$ |

The Gaussian prior is smooth at zero with light tails, which explains why ridge shrinks coefficients smoothly without producing exact zeros. The Laplace prior, with its cusp at zero and heavier tails, encourages sparsity and corresponds to lasso regularization, as discussed in the next section.

## Summary

Ridge regression is the MAP estimate under a Bayesian linear model with Gaussian prior $\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2\mathbf{I})$. The regularization parameter is $\lambda = \sigma^2/\tau^2$, linking prior belief about coefficient magnitude to the penalty strength. The Gaussian conjugacy yields a closed-form posterior that provides both point estimates (the ridge solution) and uncertainty quantification (credible intervals). The choice of a Gaussian prior, with its smooth density at zero, is the Bayesian explanation for why ridge shrinks but does not eliminate coefficients.
