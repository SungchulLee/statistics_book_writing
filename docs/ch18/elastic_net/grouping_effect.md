# Grouping Effect for Correlated Features

When two predictors are highly correlated, an ideal estimator should assign them similar coefficients, reflecting their comparable relationship to the response. Ridge regression does this naturally, but without sparsity. Lasso selects one and discards the other. The elastic net achieves both: it encourages correlated predictors to receive similar coefficients while still performing variable selection. This behavior is called the **grouping effect**, and it is one of the elastic net's most important theoretical properties.

## The Grouping Theorem

Zou and Hastie (2005) proved a bound on how different the elastic net coefficients of two correlated predictors can be. Assume the response $\mathbf{y}$ is centered and the predictors $\mathbf{x}_1, \ldots, \mathbf{x}_p$ are standardized to have unit norm ($\|\mathbf{x}_j\| = 1$).

Let $\hat{\beta}_i$ and $\hat{\beta}_j$ be two elastic net coefficients with the same sign (both positive or both negative). Define the sample correlation between the corresponding predictors as:

$$
\rho_{ij} = \mathbf{x}_i^\top\mathbf{x}_j
$$

Then the grouping theorem states:

$$
|\hat{\beta}_i - \hat{\beta}_j| \leq \frac{1}{\lambda(1-\alpha)}\sqrt{2(1 - \rho_{ij})}\,\|\mathbf{y}\|
$$

## Interpretation of the Bound

The bound reveals three key relationships:

**Correlation effect.** When $\rho_{ij} \to 1$ (predictors become identical), the right-hand side approaches zero, forcing $\hat{\beta}_i \approx \hat{\beta}_j$. Highly correlated predictors receive nearly identical coefficients.

**L2 penalty effect.** The factor $1/[\lambda(1-\alpha)]$ shows that the grouping effect strengthens as the L2 component increases (either through larger $\lambda$ or smaller $\alpha$). When $\alpha = 1$ (pure lasso), the denominator is zero and there is no grouping guarantee.

**Scale effect.** The factor $\|\mathbf{y}\|$ provides an overall scale. For standardized data, this is $\sqrt{n \cdot \text{Var}(y)}$.

!!! note "Same-Sign Requirement"
    The grouping theorem applies when $\hat{\beta}_i$ and $\hat{\beta}_j$ have the same sign. If they have opposite signs, the result does not hold. In practice, when two predictors are positively correlated and both positively related to the response, they will typically receive coefficients of the same sign.

## Proof Sketch

The elastic net KKT conditions for predictor $i$ (assuming $\hat{\beta}_i > 0$) are:

$$
-\frac{1}{n}\mathbf{x}_i^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) + \lambda\alpha + \lambda(1-\alpha)\hat{\beta}_i = 0
$$

Writing the analogous condition for predictor $j$ (also assuming $\hat{\beta}_j > 0$) and subtracting:

$$
\frac{1}{n}(\mathbf{x}_i - \mathbf{x}_j)^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \lambda(1-\alpha)(\hat{\beta}_i - \hat{\beta}_j)
$$

Applying the Cauchy-Schwarz inequality:

$$
\lambda(1-\alpha)|\hat{\beta}_i - \hat{\beta}_j| \leq \frac{1}{n}\|\mathbf{x}_i - \mathbf{x}_j\|\cdot\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|
$$

Since $\|\mathbf{x}_i - \mathbf{x}_j\|^2 = 2(1 - \rho_{ij})$ (for unit-norm predictors) and $\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\| \leq \|\mathbf{y}\|$, the bound follows.

## Contrast with Lasso and Ridge

The behavior of each method on a pair of correlated predictors ($\rho_{ij}$ close to 1) illustrates the fundamental differences:

| Method | Coefficients of correlated pair | Grouping effect |
|---|---|---|
| Ridge | Both nonzero, similar magnitude | Strong (inherent in L2) |
| Lasso | One nonzero, other zero (typically) | None |
| Elastic net | Both nonzero, similar magnitude | Controlled by $\alpha$ |

Ridge regression naturally exhibits grouping because the L2 penalty penalizes the squared difference $(\beta_i - \beta_j)^2$ implicitly. Lasso has no grouping mechanism and treats each coefficient independently in its soft-thresholding update.

## Grouping Effect Strength and Alpha

The strength of the grouping effect depends on $\alpha$:

| $\alpha$ | L2 weight $(1-\alpha)$ | Grouping strength | Sparsity |
|---|---|---|---|
| 0 (ridge) | 1 | Strongest | None |
| 0.1 | 0.9 | Very strong | Mild |
| 0.5 | 0.5 | Moderate | Moderate |
| 0.9 | 0.1 | Weak | Strong |
| 1 (lasso) | 0 | None (no bound) | Strongest |

Choosing $\alpha$ involves a tradeoff between grouping strength and sparsity. Applications where group structure is important (e.g., genomics, where correlated genes in a pathway should be selected together) benefit from smaller $\alpha$ values.

## Practical Implications

### Genomics and Biological Pathways

In gene expression studies, genes within the same biological pathway often show correlated expression patterns. The elastic net's grouping effect selects pathway members together, providing more biologically interpretable results than the lasso, which might select a single representative gene from each pathway.

### Economics and Finance

Economic indicators (e.g., GDP growth, unemployment rate, consumer confidence) are often correlated. The elastic net assigns similar coefficients to correlated indicators, producing stable predictions. The lasso might select GDP growth in one dataset and unemployment rate in another, making the model sensitive to minor data variations.

### Feature Engineering

When features are constructed from a common source (e.g., polynomial terms, interaction terms, or derived ratios), they are inherently correlated. The elastic net handles these gracefully, while the lasso may behave erratically.

!!! tip "Checking for Grouping"
    After fitting an elastic net, examine the coefficients of predictors known to be correlated. If they have similar magnitudes and signs, the grouping effect is working as expected. If they differ substantially, consider decreasing $\alpha$ to strengthen the L2 component.

## Summary

The elastic net's grouping effect ensures that highly correlated predictors receive similar coefficients, with the difference bounded by a quantity that decreases with correlation strength and L2 penalty weight. This property, formalized by the Zou-Hastie grouping theorem, distinguishes the elastic net from the lasso, which has no grouping guarantee. The grouping effect is controlled by the mixing parameter $\alpha$: smaller $\alpha$ strengthens grouping at the expense of sparsity. In applications where correlated predictors represent related concepts (biological pathways, economic indicators), the grouping effect produces more stable and interpretable models.

## Exercises

**Exercise 1.**
Explain the "grouping effect" of Elastic Net. Why does Lasso fail with highly correlated predictors? Give an example where $X_1 = X_2 + \varepsilon$ (with small $\varepsilon$) and describe what Lasso, Ridge, and Elastic Net would do.

---

**Exercise 2.**
Create a dataset where predictors come in correlated groups: $X_1, X_2, X_3$ are highly correlated ($\rho = 0.95$) and all have nonzero effects. $X_4, X_5, X_6$ are another correlated group with nonzero effects. The remaining 14 predictors are noise.

(a) Fit Lasso. Does it select one predictor per group or multiple?

(b) Fit Elastic Net with $\alpha = 0.5$. Compare the selected features.

(c) Plot the coefficient paths for both methods.
