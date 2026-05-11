# Guidelines for Choosing a Method

Selecting among ridge, lasso, and elastic net requires assessing the characteristics of the data and the goals of the analysis. This section provides a decision framework organized around the key factors that influence the choice, along with practical recommendations for common scenarios.

## Decision Factors

Five factors drive the choice of regularization method:

1. **Sparsity.** Do you expect most predictors to be irrelevant (true coefficients near or at zero)?
2. **Correlation structure.** Are predictors highly correlated, forming natural groups?
3. **Dimensionality.** Is $p$ close to or larger than $n$?
4. **Goal.** Is the primary goal prediction, interpretation, or feature selection?
5. **Computational constraints.** Is the model fit computationally expensive?

## Decision Framework

### Step 1: Assess Expected Sparsity

**If most predictors are expected to contribute** (dense true model), ridge regression is likely the best choice. Ridge retains all predictors and distributes shrinkage optimally when many coefficients are small but nonzero.

**If only a few predictors are expected to matter** (sparse true model), lasso or elastic net is preferred. Both perform automatic feature selection through the L1 penalty.

**If uncertain about sparsity**, the elastic net with moderate $\alpha$ (e.g., $\alpha = 0.5$) provides a reasonable compromise.

### Step 2: Assess Predictor Correlation

**If predictors are weakly correlated** (correlation matrix is approximately diagonal), lasso is appropriate. In the absence of strong correlations, the lasso selects features stably and its limitations with correlated groups are not relevant.

**If predictors are highly correlated**, ridge or elastic net is preferred:

- Choose **ridge** if feature selection is not needed (e.g., prediction-only applications).
- Choose **elastic net** if feature selection is needed alongside handling correlations. The grouping effect ensures correlated predictors are selected together.

!!! warning "Lasso with Correlated Predictors"
    Avoid the pure lasso when predictors are strongly correlated. The lasso may select one predictor from a correlated group while excluding the others, and the selection may change erratically across different data splits or small perturbations.

### Step 3: Check Dimensionality

**If $p < n$ (more observations than predictors)**, all three methods are viable. The choice depends primarily on the other factors (sparsity, correlation, goal).

**If $p \approx n$ or $p > n$**, consider:

- **Ridge** works well for prediction but retains all $p$ predictors.
- **Lasso** can select at most $n$ features, which may be limiting.
- **Elastic net** has no such limitation and is generally preferred in high-dimensional settings.

### Step 4: Define the Goal

**Prediction.** If the sole goal is minimizing prediction error, all three methods are candidates. Use cross-validation to select among them empirically. Often the differences in prediction performance are small.

**Feature selection.** If the goal is to identify which predictors are relevant, lasso or elastic net is required. Ridge does not perform feature selection.

**Interpretation.** If the goal is understanding the relationship between predictors and response, the elastic net's grouping effect produces more interpretable results when predictors are correlated. Post-lasso OLS can improve interpretability by removing the shrinkage bias from selected coefficients.

## Quick Reference Table

| Scenario | Recommended method | $\alpha$ | Key reason |
|---|---|---|---|
| Dense model, correlated predictors | Ridge | N/A | Distributes coefficients, no feature selection needed |
| Sparse model, uncorrelated predictors | Lasso | 1.0 | Clean feature selection |
| Sparse model, correlated predictors | Elastic net | 0.5 | Grouping effect + sparsity |
| $p > n$, moderate sparsity | Elastic net | 0.5-0.9 | No $n$-feature limit |
| $p > n$, strong sparsity | Elastic net | 0.9 | Near-lasso sparsity + stability |
| Prediction only, any structure | Cross-validate all three | -- | Let data decide |
| Unsure about structure | Elastic net | 0.5 | Safe default |

## Practical Workflow

A systematic approach to regularized regression:

1. **Standardize predictors.** Center and scale all predictors to have mean zero and unit variance. This ensures the penalty treats all coefficients equally.

2. **Examine the correlation structure.** Compute the correlation matrix and identify groups of correlated predictors. If pairwise correlations above 0.7 are common, avoid the pure lasso.

3. **Check $p/n$ ratio.** If $p > n$, plan to use elastic net or ridge.

4. **Choose candidate methods.** Based on the decision framework above, select one to three candidate methods.

5. **Tune hyperparameters.** For each candidate method, use $K$-fold cross-validation to select $\lambda$ (and $\alpha$ for elastic net).

6. **Compare methods.** If multiple methods were considered, compare their CV errors. Select the method with the lowest CV error, or use the 1-SE rule for a more parsimonious model.

7. **Validate.** Evaluate the final model on a held-out test set (if available) or report CV error as the estimate of generalization performance.

!!! tip "When in Doubt, Use Elastic Net"
    If the analysis is exploratory and the data structure is unknown, the elastic net with $\alpha = 0.5$ is a robust default. It performs well across a wide range of scenarios and avoids the worst-case behaviors of both ridge (no sparsity) and lasso (instability with correlations).

## Common Pitfalls

**Selecting features, then fitting ridge.** Running lasso to select features and then fitting ridge on the selected subset is a reasonable strategy but should be compared against elastic net, which performs both steps jointly.

**Ignoring the intercept.** Always exclude the intercept from the penalty. Most implementations handle this automatically, but verify.

**Forgetting to standardize.** If predictors are on different scales, the penalty disproportionately shrinks coefficients of predictors with small variance. Always standardize before regularization.

**Overfitting the tuning process.** Performing extensive hyperparameter search increases the risk of overfitting the validation set. Use nested cross-validation if reporting performance estimates from the same data used for tuning.

**Reporting lasso coefficients as effect sizes.** Lasso coefficients are biased toward zero. For effect size estimates, use post-lasso OLS or confidence intervals from the debiased lasso.

## Summary

The choice among ridge, lasso, and elastic net depends on the expected sparsity of the true model, the correlation structure of predictors, the dimensionality of the problem, and the goal of the analysis. Ridge suits dense models with correlated predictors. Lasso suits sparse models with uncorrelated predictors. The elastic net provides a robust default when the data structure is uncertain, combining sparsity with grouping of correlated features. A systematic workflow of standardization, correlation analysis, candidate method selection, cross-validated tuning, and final validation provides a principled approach to regularized regression.

## Exercises

**Exercise 1.**
Using the Boston Housing or California Housing dataset:

(a) Fit all three regularized models and OLS. Compare test MSE.

(b) Plot the regularization paths. At what $\lambda$ values do coefficients become zero for Lasso?

(c) Use the one-standard-error rule for model selection. How does the selected model compare to the minimum-CV-error model?
