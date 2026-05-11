# Advantages over Pure Ridge and Lasso

The elastic net was designed to address specific limitations of ridge regression and the lasso. Ridge regression cannot perform feature selection, and the lasso has well-known difficulties with correlated predictors and the $p > n$ setting. By combining both penalties, the elastic net overcomes these limitations while retaining the strengths of each method. This section details the specific advantages.

## Overcoming the Lasso's Saturation Limit

When $p > n$, the lasso can select **at most $n$ features**. This is because the lasso solution lies in the column space of $\mathbf{X}$, which has dimension at most $n$. If the true model involves more than $n$ relevant predictors, the lasso cannot recover all of them.

The elastic net does not have this limitation. The L2 component of the penalty ensures that the objective is strictly convex, and the elastic net can select more than $n$ features. This makes the elastic net suitable for applications such as genomics, where the number of relevant genes may exceed the sample size.

| Setting | Lasso | Elastic Net |
|---|---|---|
| $p < n$ | Selects up to $p$ features | Selects up to $p$ features |
| $p > n$ | Selects at most $n$ features | No upper limit on selected features |
| $p \gg n$ | Severely limited | Performs well |

## Handling Groups of Correlated Predictors

Consider a group of predictors that are highly correlated with each other and with the response. The lasso tends to select one predictor from the group (essentially at random) and set the rest to zero. Which predictor is selected can change with small perturbations of the data, making the lasso solution unstable.

Ridge regression distributes the coefficient mass evenly across the correlated group but retains all predictors, including irrelevant ones.

The elastic net achieves a middle ground: it tends to **select or exclude correlated predictors as a group** while still maintaining sparsity. When one predictor in a correlated group is selected, the others in the group are more likely to be selected as well (the grouping effect, discussed in the next section).

!!! note "Practical Consequence for Interpretation"
    In scientific applications, knowing that an entire group of correlated variables is relevant is often more informative than knowing which single variable within the group was arbitrarily selected. The elastic net's group selection behavior produces more scientifically meaningful results.

## Solution Uniqueness

The pure lasso ($\alpha = 1$) has a non-unique solution when $p > n$ or when some predictors are exactly collinear. Different implementations or starting points may produce different solutions that achieve the same objective value.

The elastic net with any $\alpha < 1$ has a **strictly convex** objective, guaranteeing a unique solution regardless of $p$ and $n$. This uniqueness provides:

- **Reproducibility.** The same data always produces the same solution.
- **Stability.** Small perturbations in the data produce small changes in the solution.
- **Path continuity.** The solution path $\hat{\boldsymbol{\beta}}(\lambda)$ is continuous in $\lambda$.

## Improved Prediction Accuracy

Zou and Hastie (2005) showed through simulation studies that the elastic net often achieves better prediction accuracy than either ridge or lasso alone, particularly when:

1. The true model has a moderate number of nonzero coefficients (some but not extreme sparsity).
2. Predictors are correlated in groups.
3. $p$ is comparable to or larger than $n$.

The mechanism is that the L2 component reduces the variance of the estimator (as in ridge), while the L1 component provides the appropriate inductive bias toward sparsity.

## Comparison of Limitations

| Limitation | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| No feature selection | Yes | No | No |
| Arbitrary selection in correlated groups | N/A | Yes | No (grouping effect) |
| At most $n$ features when $p > n$ | N/A | Yes | No |
| Non-unique solution ($p > n$) | No | Yes | No ($\alpha < 1$) |
| Excessive bias on large coefficients | Moderate | High | Moderate |
| Computationally expensive | No (closed form) | Moderate | Moderate |

## The Cost of Flexibility

The elastic net introduces an additional hyperparameter $\alpha$ that must be chosen. This increases the complexity of the model selection process:

- **One-dimensional search** for ridge or lasso: optimize over $\lambda$ only.
- **Two-dimensional search** for elastic net: optimize over $(\lambda, \alpha)$.

However, in practice, fixing $\alpha$ at a reasonable value (e.g., $\alpha = 0.5$) and optimizing only over $\lambda$ often works well and adds minimal computational cost compared to the pure lasso.

!!! tip "Default Recommendation"
    When unsure whether ridge or lasso is more appropriate, the elastic net with $\alpha = 0.5$ is a safe default. It provides a reasonable balance of sparsity and stability, and its performance is rarely much worse than the better of ridge and lasso.

## Naive Elastic Net versus Corrected Elastic Net

The original elastic net estimator suffers from a double shrinkage problem: the L1 penalty shrinks coefficients and the L2 penalty shrinks them further. Zou and Hastie (2005) proposed a correction that rescales the elastic net coefficients:

$$
\hat{\boldsymbol{\beta}}_{\text{corrected}} = (1 + \lambda(1-\alpha))\,\hat{\boldsymbol{\beta}}_{\text{EN}}
$$

This correction undoes the extra shrinkage from the L2 penalty, improving prediction accuracy. Most modern implementations apply this correction automatically.

## Summary

The elastic net addresses the lasso's limitation of selecting at most $n$ features when $p > n$, its instability with correlated predictors, and its non-unique solutions. It achieves group selection of correlated variables while maintaining sparsity, and its strictly convex objective ensures a unique, stable solution. These advantages come at the cost of an additional hyperparameter $\alpha$, but the improved robustness and prediction accuracy in the presence of correlated predictors typically justify the extra tuning effort.


## Exercises

**Exercise 1.**
Describe the main concept of Advantages over Pure Ridge and Lasso and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Advantages over Pure Ridge and Lasso is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
