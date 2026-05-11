# Overfitting and the Bias-Variance Tradeoff

A model that memorizes its training data performs perfectly in-sample but poorly on new observations. This phenomenon, called **overfitting**, is the central motivation for regularization. Before introducing specific penalties, we need a precise framework for understanding why overfitting happens and how adding bias can actually improve predictions.

## Training Error versus Test Error

Consider a regression model $\hat{f}$ fitted on training data $\{(x_i, y_i)\}_{i=1}^n$. Define two error measures.

**Training error** is the average loss on the data used to fit the model:

$$
\text{Err}_{\text{train}} = \frac{1}{n}\sum_{i=1}^n \bigl(y_i - \hat{f}(x_i)\bigr)^2
$$

**Test error** (or generalization error) is the expected loss on a new, independent observation $(x_0, y_0)$:

$$
\text{Err}_{\text{test}} = E\bigl[(y_0 - \hat{f}(x_0))^2\bigr]
$$

As model complexity increases, training error monotonically decreases because the model gains more freedom to fit the observed data. Test error, however, follows a characteristic **U-shaped curve**: it first decreases as the model captures genuine signal, then increases as the model begins fitting noise.

The gap $\text{Err}_{\text{test}} - \text{Err}_{\text{train}}$ grows with model complexity. Regularization controls this gap by constraining the model, keeping test error near its minimum.

## The Bias-Variance Decomposition

The test error at a fixed input $x_0$ admits a fundamental decomposition. Assume the true data-generating process is $y = f(x) + \varepsilon$, where $\varepsilon$ has mean zero and variance $\sigma^2$, and the expectation below is over training sets of size $n$.

$$
E\bigl[(y_0 - \hat{f}(x_0))^2\bigr] = \underbrace{\bigl(f(x_0) - E[\hat{f}(x_0)]\bigr)^2}_{\text{Bias}^2} + \underbrace{E\bigl[(\hat{f}(x_0) - E[\hat{f}(x_0)])^2\bigr]}_{\text{Variance}} + \sigma^2
$$

Each term has a clear interpretation:

- **Bias squared** measures how far the average prediction is from the truth. A model that is too simple (e.g., fitting a line to a curved relationship) has high bias.
- **Variance** measures how much predictions fluctuate across different training sets. A model with too many parameters relative to the sample size has high variance.
- **Irreducible error** $\sigma^2$ is the noise floor that no model can eliminate.

!!! note "The Tradeoff"
    Bias and variance pull in opposite directions. Simple models have high bias but low variance. Complex models have low bias but high variance. The optimal model complexity minimizes their sum, not either term alone.

## Overfitting in Linear Regression

In the linear regression setting with $p$ predictors and $n$ observations, the OLS estimator $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ has well-known bias-variance properties.

The bias of OLS is zero (it is unbiased), but its variance is:

$$
\text{Var}(\hat{\boldsymbol{\beta}}_{\text{OLS}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}
$$

When $p$ is large relative to $n$, or when the columns of $\mathbf{X}$ are nearly collinear, the matrix $\mathbf{X}^\top\mathbf{X}$ has small eigenvalues and $(\mathbf{X}^\top\mathbf{X})^{-1}$ has large eigenvalues. This inflates the variance of every coefficient estimate.

The expected in-sample prediction error for OLS can be expressed as:

$$
E\bigl[\text{Err}_{\text{train}}\bigr] = \sigma^2\Bigl(1 - \frac{p}{n}\Bigr)
$$

while the expected test error is:

$$
E\bigl[\text{Err}_{\text{test}}\bigr] = \sigma^2\Bigl(1 + \frac{p}{n}\Bigr)
$$

The gap between these two quantities is $2\sigma^2 p/n$, which grows linearly with the number of parameters. When $p$ approaches $n$, training error approaches zero while test error diverges.

## Regularization as Variance Reduction

Regularization introduces a penalty $P(\boldsymbol{\beta})$ into the objective:

$$
\hat{\boldsymbol{\beta}}_{\text{reg}} = \arg\min_{\boldsymbol{\beta}} \left\{\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\, P(\boldsymbol{\beta})\right\}
$$

The penalty $\lambda\, P(\boldsymbol{\beta})$ deliberately introduces bias by shrinking coefficients toward zero. In return, it reduces variance, often substantially. The total MSE of the regularized estimator can be written as:

$$
\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{reg}}) = \text{Bias}^2(\hat{\boldsymbol{\beta}}_{\text{reg}}) + \text{Var}(\hat{\boldsymbol{\beta}}_{\text{reg}})
$$

For any $\boldsymbol{\beta} \neq \mathbf{0}$, there exists a value $\lambda > 0$ such that the regularized estimator has lower MSE than OLS. This is the Hoerl-Kennard theorem for ridge regression, and similar results hold for other penalties.

!!! tip "Intuition for Why Bias Helps"
    Imagine estimating a parameter near zero. OLS gives an unbiased but noisy estimate. A regularized estimator that shrinks toward zero has a small bias but dramatically reduced variance. The net effect is a lower total error whenever the true parameter is not too far from zero.

## The Role of the Tuning Parameter

The regularization parameter $\lambda \geq 0$ controls the bias-variance tradeoff:

| $\lambda$ | Bias | Variance | Model |
|---|---|---|---|
| $\lambda = 0$ | Zero (OLS) | High | Most complex |
| Small $\lambda$ | Low | Moderate | Slightly constrained |
| Large $\lambda$ | High | Low | Heavily constrained |
| $\lambda \to \infty$ | Maximum | Zero | Null model ($\hat{\boldsymbol{\beta}} = \mathbf{0}$) |

Selecting the optimal $\lambda$ requires data-driven methods such as cross-validation, which are covered in detail in the tuning section of this chapter.

## Summary

Overfitting occurs when a model captures noise rather than signal, leading to a gap between training and test performance. The bias-variance decomposition provides the theoretical framework: total prediction error equals bias squared plus variance plus irreducible noise. OLS is unbiased but can have excessive variance, especially when $p/n$ is large or predictors are correlated. Regularization deliberately introduces a small amount of bias to achieve a much larger reduction in variance, lowering overall prediction error. The remaining sections of this chapter develop specific forms of the penalty $P(\boldsymbol{\beta})$ and methods for choosing $\lambda$.


## Exercises

**Exercise 1.**
Describe the main concept of Overfitting and the Bias-Variance Tradeoff and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Overfitting and the Bias-Variance Tradeoff is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
