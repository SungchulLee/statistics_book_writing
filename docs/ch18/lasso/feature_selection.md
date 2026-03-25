# Lasso for Feature Selection

Feature selection identifies a subset of predictors that are most relevant for predicting the response. Traditional approaches such as forward selection, backward elimination, and best subset selection treat model fitting and variable selection as separate steps. The lasso performs both simultaneously: by setting some coefficients to exactly zero, it produces a fitted model that uses only a subset of predictors. This "embedded" approach to feature selection has both theoretical guarantees and practical advantages.

## Lasso as Embedded Selection

Feature selection methods fall into three categories:

- **Filter methods** rank features by a univariate criterion (e.g., correlation with the response) before fitting any model.
- **Wrapper methods** evaluate subsets of features by fitting a model (e.g., forward stepwise) and measuring performance.
- **Embedded methods** perform selection as part of the model fitting process.

The lasso is an embedded method. At any fixed $\lambda$, the set of predictors with nonzero coefficients defines the selected model:

$$
\hat{S}(\lambda) = \{j : \hat{\beta}_j(\lambda) \neq 0\}
$$

As $\lambda$ decreases from $\lambda_{\max}$, features enter the selected set one by one (approximately), creating a nested sequence of models.

## Model Selection Consistency

A natural question is whether the lasso selects the "correct" set of features as the sample size grows. Let $S^* = \{j : \beta_j^* \neq 0\}$ denote the true support (the set of truly nonzero coefficients).

**Definition.** The lasso is **model selection consistent** if there exists a sequence $\lambda_n$ such that:

$$
P\bigl(\hat{S}(\lambda_n) = S^*\bigr) \to 1 \quad \text{as } n \to \infty
$$

Zhao and Yu (2006) showed that model selection consistency requires the **irrepresentable condition**. Let $S = S^*$ and $S^c$ be its complement. Partition $\mathbf{X}^\top\mathbf{X}/n$ conformally:

$$
\frac{1}{n}\mathbf{X}^\top\mathbf{X} = \begin{pmatrix} \mathbf{C}_{SS} & \mathbf{C}_{SS^c} \\ \mathbf{C}_{S^cS} & \mathbf{C}_{S^cS^c} \end{pmatrix}
$$

The irrepresentable condition requires:

$$
\|\mathbf{C}_{S^cS}\mathbf{C}_{SS}^{-1}\text{sign}(\boldsymbol{\beta}_S^*)\|_\infty < 1
$$

!!! warning "When the Irrepresentable Condition Fails"
    The irrepresentable condition can fail when irrelevant predictors are highly correlated with relevant ones. In such cases, the lasso may include irrelevant predictors or exclude relevant ones, even asymptotically. This is a fundamental limitation, not a small-sample issue.

## Bias of Selected Coefficients

The lasso applies soft-thresholding, which shrinks all retained coefficients toward zero. This means the nonzero lasso coefficients are biased estimates of the true parameters. The bias is:

$$
E[\hat{\beta}_j^{\text{lasso}}] \neq \beta_j^* \quad \text{for } j \in \hat{S}
$$

even asymptotically for the selected variables.

**Post-lasso OLS** (also called the "relaxed lasso") addresses this bias:

1. Fit the lasso to select the active set $\hat{S}(\lambda)$.
2. Refit OLS using only the predictors in $\hat{S}(\lambda)$.

The post-lasso estimator combines the selection capability of the lasso with the unbiasedness of OLS on the selected subset.

## Stability Selection

A single lasso fit at one $\lambda$ may select a somewhat different set of features if the data were slightly perturbed. **Stability selection** (Meinshausen and Buhlmann, 2010) addresses this sensitivity:

1. For $b = 1, \ldots, B$, draw a random subsample of size $\lfloor n/2 \rfloor$.
2. Fit the lasso on each subsample for a range of $\lambda$ values.
3. For each feature $j$, compute the **selection probability** $\hat{\pi}_j$: the fraction of subsamples in which feature $j$ was selected.
4. Select features with $\hat{\pi}_j$ above a threshold (e.g., 0.6 or 0.9).

Stability selection controls the expected number of false selections (features incorrectly included) and provides more robust variable selection than a single lasso fit.

!!! note "Error Control in Stability Selection"
    Under mild conditions, the expected number of falsely selected variables is bounded by:

    $$
    E[V] \leq \frac{q^2}{(2\pi_{\text{thr}} - 1)p}
    $$

    where $V$ is the number of false positives, $q$ is the average number of selected variables across subsamples, $\pi_{\text{thr}}$ is the selection threshold, and $p$ is the total number of predictors.

## Comparison with Other Selection Methods

| Method | Type | Handles $p > n$ | Computation | Selection stability |
|---|---|---|---|---|
| Best subset | Wrapper | No (NP-hard) | Exponential in $p$ | High (deterministic) |
| Forward stepwise | Wrapper | Limited | $O(p^2 n)$ | Moderate |
| Lasso | Embedded | Yes | $O(np)$ per path | Moderate |
| Stability selection | Embedded + resampling | Yes | $O(Bnp)$ | High |
| Elastic net | Embedded | Yes | $O(np)$ per path | Higher than lasso |

Best subset selection is optimal in theory but computationally infeasible for large $p$. Forward stepwise is greedy and can miss important variables. The lasso provides a computationally efficient middle ground with theoretical guarantees under appropriate conditions.

## Practical Recommendations

1. **Use the lasso regularization path** to identify candidate feature sets at different levels of sparsity.

2. **Apply the 1-SE rule** for $\lambda$ selection when parsimony is valued. This tends to select fewer features than $\lambda_{\min}$.

3. **Consider post-lasso OLS** to remove the shrinkage bias from retained coefficients.

4. **Use stability selection** when the reliability of individual feature selections is important, such as in scientific discovery applications.

5. **Prefer elastic net over pure lasso** when predictors are correlated, as the lasso may arbitrarily select one predictor from a correlated group.

## Summary

The lasso performs feature selection by setting coefficients to zero through the L1 penalty. Model selection consistency requires the irrepresentable condition, which may fail when irrelevant predictors correlate with relevant ones. The lasso's selected coefficients are biased toward zero, which post-lasso OLS can correct. Stability selection improves the reliability of lasso-based feature selection by aggregating results across subsamples. In practice, the lasso's combination of computational efficiency, automatic variable selection, and theoretical guarantees makes it a leading tool for feature selection in high-dimensional settings.
