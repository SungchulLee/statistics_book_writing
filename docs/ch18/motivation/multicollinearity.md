# Multicollinearity and Regularization

When two or more predictors are highly correlated, OLS coefficient estimates become unreliable: their signs may flip, their magnitudes may be implausibly large, and small changes in the data can drastically alter the fitted model. This section defines multicollinearity precisely, introduces the variance inflation factor as a diagnostic, and explains how regularization stabilizes estimation in the presence of correlated predictors.

## What Is Multicollinearity

Multicollinearity exists when one predictor can be approximately expressed as a linear combination of others. Formally, if the design matrix $\mathbf{X}$ has columns $\mathbf{x}_1, \ldots, \mathbf{x}_p$, then multicollinearity means there exist constants $c_1, \ldots, c_p$ (not all zero) such that:

$$
c_1\mathbf{x}_1 + c_2\mathbf{x}_2 + \cdots + c_p\mathbf{x}_p \approx \mathbf{0}
$$

**Perfect multicollinearity** (exact equality) makes $\mathbf{X}^\top\mathbf{X}$ singular and OLS impossible. **Near-multicollinearity** (approximate equality) makes $\mathbf{X}^\top\mathbf{X}$ invertible but ill-conditioned, inflating the variance of coefficient estimates.

## The Variance Inflation Factor

The **variance inflation factor** (VIF) for the $j$-th predictor quantifies how much the variance of $\hat{\beta}_j$ is inflated due to correlations with other predictors.

To compute $\text{VIF}_j$, regress $x_j$ on all other predictors and obtain the $R^2$ from that regression, denoted $R_j^2$. Then:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

The variance of the $j$-th OLS coefficient is:

$$
\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{n\, s_j^2} \cdot \text{VIF}_j
$$

where $s_j^2$ is the sample variance of $x_j$ (after centering). Without multicollinearity, $R_j^2 = 0$ and $\text{VIF}_j = 1$. As predictors become more collinear, $R_j^2 \to 1$ and $\text{VIF}_j \to \infty$.

!!! warning "VIF Interpretation Guidelines"
    A common rule of thumb is:

    - $\text{VIF}_j < 5$: low concern
    - $5 \leq \text{VIF}_j < 10$: moderate collinearity, investigate further
    - $\text{VIF}_j \geq 10$: serious collinearity, coefficient estimates are unreliable

    These thresholds are guidelines, not rigid cutoffs. The impact depends on the sample size and the goals of the analysis.

## How Multicollinearity Inflates Variance

Consider the simplest case: two predictors $x_1$ and $x_2$ with sample correlation $r$. The variance of the OLS slope estimates is:

$$
\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{n\, s_1^2(1 - r^2)}, \quad \text{Var}(\hat{\beta}_2) = \frac{\sigma^2}{n\, s_2^2(1 - r^2)}
$$

As $|r| \to 1$, both variances diverge. The OLS estimates remain unbiased but become so noisy that they are practically useless. The coefficient estimates for $\hat{\beta}_1$ and $\hat{\beta}_2$ become highly anti-correlated: a positive fluctuation in one is compensated by a negative fluctuation in the other.

In the general $p$-predictor case, the eigenvalue decomposition of $\mathbf{X}^\top\mathbf{X}$ reveals the same phenomenon. Let $\mathbf{X}^\top\mathbf{X} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_p$. The OLS coefficient variance along the $j$-th eigenvector direction is $\sigma^2/\lambda_j$. When collinearity makes $\lambda_p$ small, the variance along that direction explodes.

## Ridge Regression and Correlated Predictors

Ridge regression addresses multicollinearity by replacing $(\mathbf{X}^\top\mathbf{X})^{-1}$ with $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}$. This has a specific effect on correlated predictors.

Consider two highly correlated predictors $x_1 \approx x_2$. OLS may assign a large positive coefficient to one and a large negative coefficient to the other (they nearly cancel in prediction). Ridge penalizes this by shrinking both coefficients toward zero, distributing the coefficient mass more evenly among correlated predictors.

In the SVD framework, the ridge shrinkage factor for the $j$-th principal component is:

$$
\frac{d_j^2}{d_j^2 + \lambda}
$$

Components with small singular values $d_j$ (the collinear directions) receive the strongest shrinkage, which is precisely where OLS variance is largest. Ridge regression therefore targets its regularization where it is most needed.

!!! note "Ridge Distributes, Lasso Selects"
    Ridge regression tends to assign similar coefficients to correlated predictors, distributing their effect. Lasso, by contrast, tends to select one predictor from a correlated group and set the others to zero. Elastic net offers a middle ground, as discussed in later sections.

## Multicollinearity versus Ill-Conditioning

Multicollinearity and ill-conditioning are related but distinct concepts:

| Concept | Focus | Diagnostic |
|---|---|---|
| Multicollinearity | Statistical: variance of coefficient estimates | VIF, pairwise correlations |
| Ill-conditioning | Numerical: sensitivity to floating-point errors | Condition number $\kappa(\mathbf{X}^\top\mathbf{X})$ |

Multicollinearity always causes ill-conditioning, but ill-conditioning can also arise from scale disparity without true collinearity. Standardizing predictors addresses scale-induced ill-conditioning but not collinearity-induced ill-conditioning, which requires regularization or predictor removal.

## Detecting and Diagnosing Multicollinearity

A systematic diagnostic procedure includes:

1. **Correlation matrix.** Compute pairwise correlations among predictors. Values above 0.8 in absolute value suggest collinearity, though multicollinearity can exist among groups of variables even when no pairwise correlation is high.

2. **VIF computation.** Calculate $\text{VIF}_j$ for each predictor. This detects both pairwise and multiway collinearity.

3. **Eigenvalue analysis.** Examine the eigenvalues of $\mathbf{X}^\top\mathbf{X}$. A large ratio $\lambda_1/\lambda_p$ (the condition number) and eigenvalues near zero indicate collinear directions.

4. **Coefficient stability.** Fit the model on bootstrap samples or perturbed data. If coefficients change dramatically, collinearity is likely driving instability.

## Summary

Multicollinearity inflates the variance of OLS coefficients, making them unreliable even though they remain unbiased. The VIF provides a per-predictor diagnostic, while the eigenvalue spectrum reveals the geometry of collinear directions. Ridge regression directly addresses multicollinearity by shrinking coefficients along the poorly determined directions, distributing coefficient mass among correlated predictors rather than allowing wild fluctuations. This statistical motivation for regularization complements the numerical motivation from ill-conditioning and the predictive motivation from the bias-variance tradeoff.


## Exercises

**Exercise 1.**
Describe the main concept of Multicollinearity and Regularization and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Multicollinearity and Regularization is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
