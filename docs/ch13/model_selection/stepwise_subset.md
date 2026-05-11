# Stepwise and Best-Subset Selection

When a regression problem involves many candidate predictors, we face the question of which subset to include in the final model. Trying every possible combination is the most thorough approach but becomes computationally infeasible as the number of predictors grows. Stepwise methods provide greedy alternatives that explore a manageable portion of the model space.

---

## 1. Best-Subset Selection

**Best-subset selection** evaluates all possible subsets of the $p$ candidate predictors and selects the one that optimizes a given criterion.

### Algorithm

1. For each $k = 0, 1, \ldots, p$, fit all $\binom{p}{k}$ models that contain exactly $k$ predictors.
2. Among the models of size $k$, identify the one with the smallest SSE (or equivalently, the largest $R^2$). Call this $\mathcal{M}_k$.
3. Select the overall best model from $\mathcal{M}_0, \mathcal{M}_1, \ldots, \mathcal{M}_p$ using a criterion that penalizes complexity: adjusted $R^2$, AIC, BIC, or cross-validation error.

### Computational Cost

The total number of models to fit is:

$$
\sum_{k=0}^{p} \binom{p}{k} = 2^p
$$

This grows exponentially. For $p = 10$ there are $2^{10} = 1{,}024$ models, which is manageable. For $p = 20$ there are over one million. For $p = 40$ there are over one trillion, making exhaustive search impractical.

!!! warning "Best subset is infeasible for large $p$"
    Best-subset selection is generally limited to problems with $p \leq 20$ or so. For larger predictor sets, stepwise methods or regularization approaches (ridge, lasso, elastic net) are necessary.

---

## 2. Forward Stepwise Selection

**Forward stepwise selection** builds a model by starting from the intercept-only model and adding one predictor at a time.

### Algorithm

1. Let $\mathcal{M}_0$ be the intercept-only model (no predictors).
2. For $k = 0, 1, \ldots, p-1$:
    - Consider all $p - k$ predictors not yet in the model.
    - Add the predictor that produces the greatest reduction in SSE (or equivalently, the highest partial F-statistic or lowest p-value).
    - Call the resulting model $\mathcal{M}_{k+1}$.
3. Select the best model from $\mathcal{M}_0, \mathcal{M}_1, \ldots, \mathcal{M}_p$ using adjusted $R^2$, AIC, BIC, or cross-validation.

### Computational Cost

At step $k$, forward selection fits $p - k$ models. The total is:

$$
\sum_{k=0}^{p-1} (p - k) = \frac{p(p+1)}{2}
$$

This is $O(p^2)$, far smaller than the $2^p$ models in best-subset selection. For $p = 20$, forward selection fits only 210 models versus over one million for best subset.

### Limitation

Forward selection is a greedy algorithm. Once a predictor enters the model, it stays forever. This means forward selection cannot find the best two-predictor model if neither of those predictors is the single best predictor. It explores only a path through the model space, not the full space.

---

## 3. Backward Stepwise Selection

**Backward stepwise selection** starts from the full model and removes one predictor at a time.

### Algorithm

1. Let $\mathcal{M}_p$ be the full model containing all $p$ predictors.
2. For $k = p, p-1, \ldots, 1$:
    - Consider removing each of the $k$ predictors currently in the model.
    - Remove the predictor whose removal causes the smallest increase in SSE (or equivalently, the lowest partial F-statistic or highest p-value).
    - Call the resulting model $\mathcal{M}_{k-1}$.
3. Select the best model from $\mathcal{M}_0, \mathcal{M}_1, \ldots, \mathcal{M}_p$ using a complexity-penalized criterion.

### Computational Cost

The total number of models fit is the same as forward selection: $p(p+1)/2$.

### Limitation

Backward selection requires $n > p$ because the full model cannot be fit when there are more predictors than observations. Forward selection does not have this restriction, making it applicable even in high-dimensional settings ($p > n$) as long as the algorithm stops before exceeding $n$ predictors.

---

## 4. Hybrid Approaches

Some implementations combine forward and backward steps:

- **Stepwise regression** (bidirectional): At each step, the algorithm considers both adding a new predictor and removing an existing one. A predictor that was added in an earlier step can be removed later if it becomes redundant in the presence of other predictors. This increases flexibility relative to pure forward or backward selection.

- **Sequential replacement**: After completing forward selection, the algorithm attempts to swap each included predictor with each excluded predictor, keeping any swap that improves the criterion.

---

## 5. Selection Criteria

The choice of criterion in step 3 of each algorithm matters significantly:

| Criterion | Formula | Tendency |
|-----------|---------|----------|
| Adjusted $R^2$ | $1 - \frac{\text{SSE}/(n-p-1)}{\text{SST}/(n-1)}$ | Moderate complexity |
| AIC | $2k + n\ln(\text{SSE}/n)$ | Prediction-oriented |
| BIC | $k\ln n + n\ln(\text{SSE}/n)$ | Parsimonious |
| CV error | $\text{CV}_{(K)}$ | Direct estimate of prediction error |

Using SSE or $R^2$ alone (without a complexity penalty) always selects the largest model, which defeats the purpose of variable selection.

---

## 6. Critique of Stepwise Methods

Stepwise methods remain widely used in practice, but they have well-known drawbacks.

**P-value inflation**: When many predictors are tested sequentially, the chance of including a spurious predictor by chance increases. The p-values from the final stepwise model are overly optimistic because they do not account for the search process that led to that model.

**Instability**: Small changes in the data can lead to very different selected models. A predictor that narrowly enters the model in one dataset may be excluded in a slightly perturbed version of the same data.

**Biased coefficients**: Coefficients in the selected model are biased away from zero because the selection process preferentially retains predictors with large estimated effects, some of which are large by chance.

**Ignoring multicollinearity**: Stepwise methods do not explicitly handle correlated predictors. When two predictors are highly correlated, the method may arbitrarily include one and exclude the other, even though both carry similar information.

!!! note "Modern alternatives"
    Regularization methods (ridge, lasso, elastic net) address many of these criticisms by shrinking coefficients toward zero rather than performing hard inclusion/exclusion decisions. Lasso in particular performs variable selection as a byproduct of its $\ell_1$ penalty, providing a principled alternative to stepwise methods.

## Exercises

**Exercise 1.**
Describe the difference between forward selection, backward elimination, and best-subset selection. Which is computationally most expensive?

??? success "Solution to Exercise 1"
    **Forward selection** starts with no predictors and adds one at a time, choosing at each step the predictor that most improves the model (e.g., largest reduction in AIC). It stops when no addition improves the criterion.

    **Backward elimination** starts with all predictors and removes one at a time, choosing the predictor whose removal least degrades the model. It stops when every remaining predictor is significant or contributes to the criterion.

    **Best-subset selection** evaluates all $2^p$ possible subsets of $p$ predictors and selects the best model of each size, then uses a criterion (AIC, BIC, adjusted $R^2$) to choose among sizes.

    Best-subset is by far the most expensive: $2^p$ models must be fitted. With $p = 20$, that is over 1 million models. Forward and backward selection fit at most $O(p^2)$ models, making them feasible for larger $p$. However, stepwise methods may miss the globally best subset because they are greedy.

---

**Exercise 2.**
Explain why stepwise selection can inflate Type I error rates and produce overfit models if p-values are not adjusted.

??? success "Solution to Exercise 2"
    At each step, stepwise selection tests multiple candidate predictors and selects the one with the smallest p-value (or largest improvement). This is a form of multiple testing: even if all predictors are truly unrelated to the response, the best among $p$ candidates is likely to have a small p-value by chance.

    The resulting p-values are not valid for inference because they do not account for the search process. A predictor that enters the model with $p = 0.03$ may have been selected from 20 candidates, making the true significance level much higher. Additionally, the model selected by stepwise methods tends to overfit training data because it was optimized to that specific dataset.

    Remedies include: using information criteria (AIC, BIC) instead of p-values, cross-validation for final model assessment, and regularization methods (LASSO) that simultaneously select variables and shrink coefficients.

---

**Exercise 3.**
With $p = 5$ predictors, how many models must best-subset selection evaluate? List all possible model sizes.

??? success "Solution to Exercise 3"
    With $p = 5$ predictors, the total number of subsets is $2^5 = 32$ (including the null model with no predictors).

    Models by size:

    - Size 0 (intercept only): $\binom{5}{0} = 1$ model
    - Size 1: $\binom{5}{1} = 5$ models
    - Size 2: $\binom{5}{2} = 10$ models
    - Size 3: $\binom{5}{3} = 10$ models
    - Size 4: $\binom{5}{4} = 5$ models
    - Size 5 (full model): $\binom{5}{5} = 1$ model

    Total: $1 + 5 + 10 + 10 + 5 + 1 = 32$ models. Best-subset selection finds the best model at each size (6 candidates), then uses AIC/BIC/CV to select among the 6.

---

**Exercise 4.**
Compare LASSO variable selection with stepwise selection. Why is LASSO generally preferred in modern practice?

??? success "Solution to Exercise 4"
    **LASSO** adds an $L_1$ penalty to the regression objective, shrinking some coefficients exactly to zero and thereby performing variable selection and estimation simultaneously. The regularization parameter $\lambda$ controls the trade-off between fit and sparsity.

    **Advantages of LASSO over stepwise:**

    1. **Continuous path:** LASSO produces a continuous path of models as $\lambda$ varies, avoiding the discrete, greedy decisions of stepwise methods.
    2. **Shrinkage:** Non-selected coefficients are shrunk toward zero, reducing overfitting even for included predictors.
    3. **Valid inference:** Post-selection inference methods exist for LASSO (e.g., selective inference). Stepwise p-values are invalid without correction.
    4. **Scalability:** LASSO handles $p > n$ (more predictors than observations), where stepwise methods fail.
    5. **Cross-validation integration:** $\lambda$ is chosen by CV, providing an honest estimate of test error.

    Stepwise methods remain useful when interpretability of each selection step is desired or when computational resources for regularization paths are limited.
