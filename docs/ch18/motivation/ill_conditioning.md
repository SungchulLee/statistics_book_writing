# Ill-Conditioned Design Matrices

The OLS estimator requires inverting the matrix $\mathbf{X}^\top\mathbf{X}$. When this matrix is close to singular, the inversion becomes numerically unstable, and small perturbations in the data lead to wild swings in the estimated coefficients. Understanding this instability motivates regularization as a tool for numerical stabilization, independent of any statistical bias-variance argument.

## The Condition Number

The **condition number** of a matrix $\mathbf{A}$ quantifies how sensitive the solution of $\mathbf{A}\mathbf{x} = \mathbf{b}$ is to perturbations in $\mathbf{b}$. For a symmetric positive semi-definite matrix like $\mathbf{X}^\top\mathbf{X}$, the condition number is:

$$
\kappa(\mathbf{X}^\top\mathbf{X}) = \frac{\lambda_{\max}(\mathbf{X}^\top\mathbf{X})}{\lambda_{\min}(\mathbf{X}^\top\mathbf{X})}
$$

where $\lambda_{\max}$ and $\lambda_{\min}$ denote the largest and smallest eigenvalues, respectively.

If the singular values of $\mathbf{X}$ are $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$, then $\mathbf{X}^\top\mathbf{X}$ has eigenvalues $d_1^2, d_2^2, \ldots, d_p^2$, and:

$$
\kappa(\mathbf{X}^\top\mathbf{X}) = \frac{d_1^2}{d_p^2} = \left(\frac{d_1}{d_p}\right)^2
$$

A condition number near 1 indicates a well-conditioned problem. A condition number of $10^k$ means that roughly $k$ digits of precision are lost when solving the system. When $\kappa(\mathbf{X}^\top\mathbf{X}) > 10^{10}$, double-precision floating-point arithmetic may produce meaningless results.

## Sources of Ill-Conditioning

Several common data configurations lead to ill-conditioned design matrices.

**Near-collinearity.** When two or more predictors are nearly linearly dependent, the smallest eigenvalue of $\mathbf{X}^\top\mathbf{X}$ approaches zero, driving the condition number toward infinity.

**High dimensionality.** When $p$ is close to $n$, the matrix $\mathbf{X}^\top\mathbf{X}$ tends to have very small eigenvalues even if no exact collinearity exists. When $p > n$, the matrix is rank-deficient and the OLS solution is not unique.

**Scale disparity.** If predictors have vastly different scales (e.g., one measured in meters and another in kilometers), the eigenvalue spread of $\mathbf{X}^\top\mathbf{X}$ increases. Standardizing predictors before regression mitigates this source of ill-conditioning.

## Sensitivity of OLS to Perturbations

Consider the OLS system $(\mathbf{X}^\top\mathbf{X})\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$. If the data vector $\mathbf{y}$ is perturbed by a small amount $\delta\mathbf{y}$, the resulting change in the solution satisfies the bound:

$$
\frac{\|\delta\hat{\boldsymbol{\beta}}\|}{\|\hat{\boldsymbol{\beta}}\|} \leq \kappa(\mathbf{X}^\top\mathbf{X})\,\frac{\|\delta(\mathbf{X}^\top\mathbf{y})\|}{\|\mathbf{X}^\top\mathbf{y}\|}
$$

This means the relative error in $\hat{\boldsymbol{\beta}}$ can be amplified by a factor of $\kappa(\mathbf{X}^\top\mathbf{X})$ relative to the perturbation. When the condition number is $10^{6}$, a perturbation at the sixth decimal place in the data can change the leading digit of a coefficient estimate.

!!! warning "Numerical Instability in Practice"
    Ill-conditioning is not merely a theoretical concern. With real-world data, rounding errors during data entry, floating-point arithmetic, and measurement noise all act as perturbations. If $\kappa(\mathbf{X}^\top\mathbf{X})$ is large, these unavoidable perturbations corrupt the OLS solution.

## Regularization as a Stabilizer

Ridge regression adds $\lambda\mathbf{I}$ to $\mathbf{X}^\top\mathbf{X}$ before inverting:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

The eigenvalues of $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ are $d_j^2 + \lambda$, so the condition number becomes:

$$
\kappa(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}) = \frac{d_1^2 + \lambda}{d_p^2 + \lambda}
$$

Since $\lambda > 0$ raises the floor of the eigenvalue spectrum, the condition number decreases. For any $\lambda > 0$:

$$
\kappa(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}) < \kappa(\mathbf{X}^\top\mathbf{X})
$$

In the extreme case where $\lambda \gg d_1^2$, the condition number approaches 1 (the identity matrix). Even a modest $\lambda$ can reduce the condition number by orders of magnitude when $d_p^2$ is near zero.

!!! note "Ridge as Tikhonov Regularization"
    In numerical analysis, adding $\lambda\mathbf{I}$ to a nearly singular system is known as **Tikhonov regularization**. Ridge regression is the statistical instantiation of this classical technique, motivating it from both a numerical and a statistical perspective.

## Eigenvalue Spectrum and Effective Rank

The eigenvalue spectrum of $\mathbf{X}^\top\mathbf{X}$ reveals the severity of ill-conditioning. Define the **effective rank** at threshold $\tau$ as:

$$
\text{rank}_\tau(\mathbf{X}) = \#\{j : d_j > \tau \cdot d_1\}
$$

When the effective rank is much smaller than $p$, many directions in coefficient space are poorly determined by the data. Ridge regression shrinks coefficients most strongly along these poorly determined directions, as described by the SVD shrinkage factors $d_j^2/(d_j^2 + \lambda)$.

## Detecting Ill-Conditioning in Practice

Several diagnostic tools help identify ill-conditioning before fitting a model:

| Diagnostic | Threshold | Interpretation |
|---|---|---|
| $\kappa(\mathbf{X}^\top\mathbf{X})$ | $> 10^4$ | Moderate ill-conditioning |
| $\kappa(\mathbf{X}^\top\mathbf{X})$ | $> 10^8$ | Severe ill-conditioning |
| Smallest eigenvalue | Near machine epsilon | Numerically singular |
| VIF for predictor $j$ | $> 10$ | Predictor involved in collinearity |

The variance inflation factor (VIF) is explored in detail in the next section on multicollinearity.

## Summary

The condition number $\kappa(\mathbf{X}^\top\mathbf{X})$ measures how sensitively OLS coefficients respond to small data perturbations. Near-collinearity, high dimensionality, and scale disparity all inflate the condition number, making OLS numerically unstable. Ridge regularization adds $\lambda\mathbf{I}$ to $\mathbf{X}^\top\mathbf{X}$, raising every eigenvalue by $\lambda$ and reducing the condition number. This stabilization makes regularization valuable even in settings where the statistical bias-variance argument alone would not justify it.


## Exercises

**Exercise 1.**
Describe the main concept of Ill-Conditioned Design Matrices and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Ill-Conditioned Design Matrices is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
