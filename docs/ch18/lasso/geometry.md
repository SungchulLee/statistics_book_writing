# Geometric Interpretation (L1 Penalty)

The geometric picture for lasso reveals why the L1 penalty produces exact zeros while the L2 penalty does not. The key difference lies in the shape of the constraint region: the L1 ball has corners on the coordinate axes, and these corners are exactly the points where one or more coefficients equal zero. Because the RSS contours are smooth ellipses, first contact with a cornered constraint set generically occurs at a corner.

## The L1 Constraint Region

The lasso constraint restricts $\boldsymbol{\beta}$ to the L1 ball:

$$
\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j| \leq t
$$

In two dimensions, this region is a **diamond** (square rotated 45 degrees) with vertices at $(\pm t, 0)$ and $(0, \pm t)$. In $p$ dimensions, it is a **cross-polytope** (hyperoctahedron).

The geometric properties that distinguish it from the L2 ball are:

- **Corners on coordinate axes.** The vertices of the cross-polytope lie at $\pm t\,\mathbf{e}_j$ for each coordinate direction $\mathbf{e}_j$. At a vertex, all coordinates except one are exactly zero.
- **Edges and faces on coordinate hyperplanes.** The edges of the diamond (in 2D) lie along lines where one coordinate is zero. Higher-dimensional faces similarly correspond to subsets of coordinates being zero.
- **Non-smooth boundary.** The boundary has corners and edges where the normal direction is not unique.

## RSS Contours Meet the Diamond

As with ridge, the residual sum of squares defines elliptical contours centered at $\hat{\boldsymbol{\beta}}_{\text{OLS}}$. The lasso solution is the point on or inside the L1 ball that minimizes RSS, which is where the smallest RSS contour touches the diamond.

In two dimensions, picture an ellipse shrinking from its center $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ (which typically lies outside the diamond for sufficiently large $\lambda$). As the ellipse shrinks, its first contact with the diamond is the lasso solution.

## Why Sparsity Occurs

The critical geometric observation is that **smooth ellipses generically touch cornered polytopes at corners**. Here is the precise argument.

The boundary of the L1 ball consists of flat faces (in 2D, the four edges of the diamond). Each face lies in a subspace where one coordinate has a fixed sign and the others are constrained. The corners are intersections of faces.

For the tangency to occur on a flat face (away from a corner), the RSS gradient at that point must be perpendicular to the face. This requires the gradient to satisfy a specific alignment condition. In contrast, at a corner, the subdifferential of the constraint is a cone of directions, making it easier to satisfy the KKT conditions.

!!! note "Generic Sparsity"
    In a precise mathematical sense, for "most" data configurations (i.e., for a set of full Lebesgue measure), the lasso solution lies at a vertex or on a lower-dimensional face of the L1 ball, producing at least one zero coordinate. The solution lies in the interior of a face (all coordinates nonzero) only for a measure-zero set of data configurations.

## The Two-Dimensional Picture

Consider $p = 2$ predictors. The geometry has four distinct cases depending on where $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ lies relative to the diamond:

**Case 1: Tangency at a corner.** If $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ lies roughly along a coordinate axis, the ellipse touches the diamond at a vertex. One coefficient is nonzero; the other is exactly zero. This is the lasso performing feature selection.

**Case 2: Tangency on an edge.** If $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ lies between the coordinate axes, the ellipse may touch an edge of the diamond. Both coefficients are nonzero but constrained to lie on the edge $|\beta_1| + |\beta_2| = t$. Both coefficients are shrunk relative to OLS.

**Case 3: Interior solution.** If $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ lies inside the diamond (small OLS coefficients or large $t$), the constraint is not active and the lasso solution equals OLS. This occurs when $\lambda = 0$ or is very small.

**Case 4: Both coefficients zero.** If $t$ is very small (large $\lambda$), the diamond shrinks to a point at the origin, and $\hat{\boldsymbol{\beta}}_{\text{lasso}} = \mathbf{0}$.

## Contrast with L2 Geometry

The L2 constraint region is a sphere (disk in 2D). Its key property is that the boundary is smooth everywhere, with a unique outward normal at every point. This means:

- The tangency condition requires the RSS gradient to be parallel to the sphere's radial direction.
- This tangency generically occurs at a point where no coordinate is exactly zero.
- The sphere has no corners to "catch" the ellipse at a coordinate axis.

| Property | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Constraint shape | Diamond / cross-polytope | Sphere / hypersphere |
| Boundary smoothness | Corners and edges | Smooth everywhere |
| Generic tangency location | At a corner or edge | Interior of the boundary |
| Coordinates at tangency | Some exactly zero | All nonzero |
| Sparsity | Yes | No |

## Higher Dimensions

In $p$ dimensions, the L1 ball has $2p$ vertices (one pair per coordinate axis), $2^p$ faces of dimension $p-1$, and a rich combinatorial structure of lower-dimensional faces. The probability of the tangency occurring at a vertex (only one nonzero coefficient) increases when the ellipse is nearly spherical (predictors are weakly correlated) and decreases when it is highly elongated (strong correlations).

The number of nonzero coefficients in the lasso solution at any given $\lambda$ corresponds to the dimension of the face where the tangency occurs. This dimension is at most $\min(n, p)$, reflecting the constraint that the lasso can select at most $n$ features when $p > n$.

## The Solution Path as Lambda Varies

As $\lambda$ decreases from $\lambda_{\max}$ to 0, the constraint region expands. Geometrically:

- At $\lambda = \lambda_{\max}$: the diamond is tiny, and $\hat{\boldsymbol{\beta}}_{\text{lasso}} = \mathbf{0}$.
- As $\lambda$ decreases: the diamond grows, and the tangency point moves to faces of increasing dimension (more nonzero coefficients).
- At $\lambda = 0$: the constraint is inactive, and $\hat{\boldsymbol{\beta}}_{\text{lasso}} = \hat{\boldsymbol{\beta}}_{\text{OLS}}$.

The path $\hat{\boldsymbol{\beta}}_{\text{lasso}}(\lambda)$ is piecewise linear in $\lambda$, with breakpoints where the tangency point moves to a different face of the polytope.

## Summary

The lasso constraint region is a diamond (cross-polytope) whose corners lie on the coordinate axes. Because RSS contours are smooth ellipses, their first contact with the diamond generically occurs at a corner, where one or more coordinates are zero. This is the geometric mechanism behind lasso sparsity. In contrast, the smooth sphere of the L2 penalty has no corners, so ridge regression generically produces solutions with all coordinates nonzero. The shape of the constraint region, not the nature of the objective function, determines whether the estimator is sparse.


## Exercises

**Exercise 1.**
Describe the main concept of Geometric Interpretation (L1 Penalty) and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Geometric Interpretation (L1 Penalty) is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
