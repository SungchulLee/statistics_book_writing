# Geometric Interpretation (L2 Penalty)

The algebraic formulation of ridge regression, while precise, does not immediately reveal why ridge shrinks coefficients uniformly toward zero or why it never produces exact zeros. A geometric perspective makes both properties visually intuitive. By viewing the ridge problem as constrained optimization, we can understand the solution as the point where elliptical contours of the residual sum of squares first touch a spherical constraint region.

## RSS Contours in Coefficient Space

Consider a regression problem with two predictors for visualization. The residual sum of squares is:

$$
\text{RSS}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

This is a convex quadratic function of $\boldsymbol{\beta}$, and its level sets (contours of constant RSS) are **ellipses** centered at the OLS estimate $\hat{\boldsymbol{\beta}}_{\text{OLS}}$. The shape and orientation of these ellipses are determined by the eigenstructure of $\mathbf{X}^\top\mathbf{X}$:

- The axes of the ellipses align with the eigenvectors of $\mathbf{X}^\top\mathbf{X}$.
- The axis lengths are inversely proportional to the square roots of the eigenvalues.
- When eigenvalues differ substantially (near-collinearity), the ellipses are elongated.

## The L2 Constraint Region

The constrained form of ridge regression restricts $\boldsymbol{\beta}$ to the L2 ball:

$$
\|\boldsymbol{\beta}\|_2^2 = \beta_1^2 + \beta_2^2 + \cdots + \beta_p^2 \leq t
$$

In two dimensions, this constraint region is a **disk** of radius $\sqrt{t}$ centered at the origin. In $p$ dimensions, it is a hypersphere. The key geometric properties are:

- The boundary $\|\boldsymbol{\beta}\|_2^2 = t$ is **smooth everywhere** (no corners, edges, or vertices).
- The boundary is **strictly convex**: every boundary point has a unique supporting hyperplane.
- The constraint region is **symmetric** about all coordinate axes and all rotations.

## The Ridge Solution as a Tangency Point

The ridge estimate is the point inside (or on the boundary of) the L2 ball that minimizes RSS. Geometrically, we shrink the RSS ellipse from its center $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ until it first touches the ball. The touching point is $\hat{\boldsymbol{\beta}}_{\text{ridge}}$.

At this tangency point, the KKT conditions require:

$$
\nabla \text{RSS}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda\,\nabla\|\boldsymbol{\beta}\|_2^2\big|_{\hat{\boldsymbol{\beta}}_{\text{ridge}}}
$$

which simplifies to the familiar normal equations:

$$
-2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -2\lambda\hat{\boldsymbol{\beta}}_{\text{ridge}}
$$

The gradient of the RSS (pointing away from the OLS center) must be parallel to the gradient of the constraint (pointing radially outward from the origin). This parallelism condition is what produces the ridge normal equations.

## Why Ridge Never Produces Exact Zeros

The sphere boundary has no corners. In two dimensions, a corner at $(t, 0)$ would be a point where the boundary changes direction abruptly, creating a region where the tangent is not unique. The sphere has no such points.

For an RSS contour to be tangent to the sphere at a coordinate axis (say at $(\beta_1, 0)$), the gradient of RSS at that point would need to point exactly along the $\beta_1$-axis. This requires the off-diagonal structure of $\mathbf{X}^\top\mathbf{X}$ to satisfy a measure-zero condition. In practice, the tangency point generically lies in the interior of a face of the positive orthant, meaning all coordinates are nonzero.

!!! note "Contrast with the L1 Geometry"
    The L1 constraint set (a diamond or cross-polytope) has corners exactly on the coordinate axes. Because RSS contours are smooth ellipses, the first contact with a diamond-shaped region generically occurs at a corner, producing exact zeros. This geometric difference is the fundamental reason lasso achieves sparsity while ridge does not.

## Shrinkage Toward the Origin

The geometric picture also explains **why** ridge shrinks toward the origin (zero vector) rather than toward some other point. The L2 constraint $\|\boldsymbol{\beta}\|_2^2 \leq t$ is centered at the origin. The ridge solution lies on the line segment from the origin to $\hat{\boldsymbol{\beta}}_{\text{OLS}}$, pulled toward the center of the ball.

More precisely, in the SVD parameterization, ridge applies the shrinkage factor $d_j^2/(d_j^2 + \lambda)$ to each principal component. Each factor is less than 1, so the ridge solution is a shrunk version of OLS, moved toward $\mathbf{0}$ along every direction.

## Effect of Eigenvalue Disparity

When $\mathbf{X}^\top\mathbf{X}$ has one large and one small eigenvalue (a common situation with correlated predictors), the RSS contours are highly elongated. The OLS estimate can lie far from the origin along the elongated direction.

The ridge constraint sphere "clips" this elongation. The tangency point lies much closer to the origin along the elongated axis (where the eigenvalue is small) but only slightly closer along the short axis (where the eigenvalue is large). This differential shrinkage is the geometric manifestation of ridge regression's ability to stabilize ill-conditioned problems.

| Eigenvalue of $\mathbf{X}^\top\mathbf{X}$ | RSS contour axis | Shrinkage factor | Effect |
|---|---|---|---|
| Large ($d_j^2 \gg \lambda$) | Short axis | $\approx 1$ | Minimal shrinkage |
| Small ($d_j^2 \ll \lambda$) | Long axis | $\approx 0$ | Heavy shrinkage |

## The Budget Interpretation

The parameter $t$ in the constrained formulation acts as a **budget** for the total squared magnitude of coefficients. A small budget forces the model to distribute its coefficient mass carefully among predictors, favoring directions where the data provides strong signal over directions dominated by noise.

As $t$ increases (equivalently, $\lambda$ decreases), the constraint relaxes, the feasible region grows, and the ridge solution approaches OLS. As $t$ decreases ($\lambda$ increases), the feasible region shrinks, and the ridge solution approaches the origin.

## Summary

The ridge solution is the point where the smallest RSS contour ellipse is tangent to the L2 ball. Because the sphere boundary is smooth and strictly convex, this tangency point generically has all coordinates nonzero, explaining why ridge regression shrinks but does not eliminate coefficients. Components aligned with small eigenvalues of $\mathbf{X}^\top\mathbf{X}$ are shrunk the most, providing the strongest regularization where OLS is most unstable. The geometric contrast between the L2 sphere and the L1 diamond foreshadows why lasso, but not ridge, achieves sparsity.


## Exercises

**Exercise 1.**
Describe the main concept of Geometric Interpretation (L2 Penalty) and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Geometric Interpretation (L2 Penalty) is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
