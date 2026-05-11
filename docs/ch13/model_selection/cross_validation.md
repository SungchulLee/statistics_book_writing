# Cross-Validation for Model Selection

Information criteria like AIC and BIC estimate out-of-sample performance using mathematical approximations. Cross-validation takes a more direct approach: it repeatedly splits the data into training and validation subsets, fits the model on training data, and evaluates predictions on held-out data. This provides a concrete estimate of how well the model generalizes to new observations.

---

## 1. The Validation Set Approach

The simplest strategy is to split the data into two disjoint subsets: a **training set** used to fit the model and a **validation set** (or hold-out set) used to evaluate it.

Given $n$ observations, randomly assign a fraction (typically 50--80%) to training and the rest to validation. Fit the model on the training set and compute the prediction error on the validation set:

$$
\text{CV}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i \in \text{val}} (y_i - \hat{y}_i)^2
$$

where $\hat{y}_i$ is the prediction for observation $i$ using the model fit to the training set only.

### Limitations

- **High variance**: The estimate depends heavily on which observations end up in training versus validation. Different random splits can produce substantially different error estimates.
- **Reduced training data**: The model is trained on fewer observations than are available, leading to a pessimistic bias in the error estimate.

---

## 2. K-Fold Cross-Validation

**K-fold cross-validation** addresses the limitations of the validation set approach by using every observation for both training and validation.

### Procedure

1. Randomly partition the $n$ observations into $K$ roughly equal-sized groups (folds) $C_1, C_2, \ldots, C_K$.
2. For each fold $k = 1, \ldots, K$:
    - Fit the model using all observations except those in fold $C_k$.
    - Predict the held-out observations in $C_k$ and record the errors.
3. Average the prediction errors across all folds.

### Formula

Let $n_k = |C_k|$ be the number of observations in fold $k$, and let $\hat{y}_i^{(-k)}$ denote the prediction for observation $i$ from the model trained without fold $k$. The K-fold CV estimate of prediction error is:

$$
\text{CV}_{(K)} = \frac{1}{n} \sum_{k=1}^{K} \sum_{i \in C_k} (y_i - \hat{y}_i^{(-k)})^2
$$

This is equivalent to:

$$
\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^{K} \text{MSE}_k
$$

where $\text{MSE}_k = \frac{1}{n_k} \sum_{i \in C_k} (y_i - \hat{y}_i^{(-k)})^2$ is the mean squared error on fold $k$, provided all folds have the same size.

Common choices are $K = 5$ and $K = 10$. Both have been shown empirically to provide a good tradeoff between bias and variance of the CV estimate.

---

## 3. Leave-One-Out Cross-Validation

**Leave-one-out cross-validation (LOOCV)** is the special case of K-fold CV with $K = n$: each fold contains exactly one observation.

$$
\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i^{(-i)})^2
$$

where $\hat{y}_i^{(-i)}$ is the prediction for observation $i$ from the model trained on all observations except $i$.

### Shortcut for Linear Regression

For linear regression, LOOCV has a remarkable computational shortcut. Rather than fitting $n$ separate models, the LOOCV error can be computed from a single fit using the **hat matrix** $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$:

$$
\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{e_i}{1 - h_{ii}}\right)^2
$$

where $e_i = y_i - \hat{y}_i$ is the ordinary residual and $h_{ii}$ is the $i$-th diagonal element of $\mathbf{H}$. This formula shows that the leave-one-out prediction error for observation $i$ is simply the ordinary residual divided by $1 - h_{ii}$, where $h_{ii}$ measures the leverage of observation $i$.

??? note "Derivation of the LOOCV shortcut"
    When observation $i$ is removed, the Sherman-Morrison-Woodbury formula gives the change in $\hat{\boldsymbol{\beta}}$, and the prediction at $x_i$ changes by exactly $e_i \cdot h_{ii} / (1 - h_{ii})$. The leave-one-out residual is therefore $y_i - \hat{y}_i^{(-i)} = e_i / (1 - h_{ii})$.

---

## 4. Bias-Variance Tradeoff in Cross-Validation

The choice of $K$ involves a tradeoff:

**Bias**: Each training set in K-fold CV has size approximately $n(K-1)/K$. When $K$ is small (e.g., $K = 2$), each training set is much smaller than the full dataset, leading to models that underperform relative to the model trained on all data. This introduces an upward bias in the CV error estimate. LOOCV ($K = n$) minimizes this bias because each training set has $n - 1$ observations.

**Variance**: The $K$ training sets in LOOCV overlap almost completely (they share $n - 2$ observations), so the $K$ fitted models are highly correlated. Averaging correlated quantities reduces variance less effectively than averaging uncorrelated quantities. As a result, LOOCV can have high variance. Smaller $K$ (e.g., $K = 5$ or $K = 10$) produces less correlated estimates and typically lower variance.

| Choice of $K$ | Bias of CV estimate | Variance of CV estimate | Computation |
|----------------|---------------------|-------------------------|-------------|
| $K = 5$       | Moderate (upward)   | Low                     | 5 model fits |
| $K = 10$      | Small               | Moderate                | 10 model fits |
| $K = n$ (LOOCV) | Approximately unbiased | Can be high          | 1 fit (with shortcut) |

!!! tip "Default recommendation"
    In practice, $K = 5$ or $K = 10$ is recommended. James et al. (2013) note that these values have been empirically shown to yield CV error estimates that suffer from neither excessively high bias nor excessively high variance.

---

## 5. Cross-Validation for Model Selection

To select among $M$ candidate models using K-fold CV:

1. For each model $j = 1, \ldots, M$, compute $\text{CV}_{(K)}^{(j)}$.
2. Select the model with the smallest CV error: $\hat{j} = \arg\min_j \text{CV}_{(K)}^{(j)}$.

### One-Standard-Error Rule

Rather than selecting the model with the absolute lowest CV error, the **one-standard-error rule** selects the simplest model whose CV error is within one standard error of the minimum. This provides an additional guard against overfitting.

Let $\text{SE}(\text{CV}_{(K)}^{(j)})$ be the standard error of the CV estimate for model $j$, computed as the standard deviation of the $K$ fold-specific MSE values divided by $\sqrt{K}$. The one-SE rule selects the simplest model $j$ satisfying:

$$
\text{CV}_{(K)}^{(j)} \leq \text{CV}_{(K)}^{(\hat{j})} + \text{SE}(\text{CV}_{(K)}^{(\hat{j})})
$$

---

## 6. Connection to AIC

Stone (1977) proved that, for linear regression, AIC model selection is asymptotically equivalent to LOOCV. This means that in large samples, AIC and LOOCV tend to select the same model. K-fold CV with small $K$ is not equivalent to AIC; it is closer to BIC in its tendency to select simpler models due to the upward bias in the CV error estimate from using smaller training sets.

!!! warning "Cross-validation is not free of assumptions"
    Cross-validation assumes that observations are exchangeable (roughly, that their order does not matter). For time series data, standard random-fold CV violates the temporal structure and produces misleadingly optimistic estimates. Time series data requires specialized CV strategies such as rolling-window or expanding-window cross-validation.

## Exercises

**Exercise 1.**
Explain the difference between k-fold cross-validation and leave-one-out cross-validation (LOOCV). What are the trade-offs in bias and variance of the estimated test error?

??? success "Solution to Exercise 1"
    **k-fold CV** splits the data into $k$ roughly equal folds, trains on $k-1$ folds, and tests on the held-out fold, rotating through all $k$ folds. The CV estimate is the average test error across folds. Common choice: $k = 5$ or $10$.

    **LOOCV** is the special case $k = n$: each observation serves as its own test set. The model is trained $n$ times, each time on $n-1$ observations.

    **Bias-variance trade-off:**

    - LOOCV has low bias (training sets are nearly the full dataset) but high variance (the $n$ training sets overlap heavily, making the $n$ error estimates highly correlated).
    - 5- or 10-fold CV has slightly higher bias (training sets are smaller) but lower variance (less overlap between training sets reduces correlation among estimates).

    In practice, 5- or 10-fold CV tends to give a better estimate of test error because the variance reduction outweighs the small increase in bias.

---

**Exercise 2.**
A data scientist fits a polynomial regression and uses the training data to select the degree that minimizes training error. They find degree 15 is best. Why is this problematic, and how would cross-validation help?

??? success "Solution to Exercise 2"
    Using training error to select model complexity always favors the most complex model because training error decreases monotonically as the model becomes more flexible. A degree-15 polynomial will likely overfit the training data, memorizing noise and performing poorly on new data.

    Cross-validation addresses this by estimating test error (performance on unseen data). For each candidate degree $d$:

    1. Split data into $k$ folds.
    2. For each fold, fit a degree-$d$ polynomial on the remaining data and compute the prediction error on the held-out fold.
    3. Average the errors across folds.

    The degree that minimizes the CV error balances fit and complexity. Typically, CV would select a much lower degree (e.g., 2-4) that generalizes better.

---

**Exercise 3.**
In stratified k-fold cross-validation, what is the stratification based on, and when is it important?

??? success "Solution to Exercise 3"
    In stratified k-fold CV, the folds are constructed so that each fold has approximately the same distribution of the response variable as the full dataset. For classification, this means each fold has roughly the same proportion of each class.

    Stratification is important when:

    1. **Class imbalance:** If only 5% of observations belong to the minority class, a random fold might contain zero minority examples, making the test error estimate unreliable.
    2. **Small datasets:** With limited data, each fold must be representative to avoid high-variance CV estimates.
    3. **Ordinal or grouped responses:** Ensuring each fold spans the range of the outcome variable.

    For regression, stratification can be based on binned values of $Y$. Most CV implementations in scikit-learn support stratified splitting via `StratifiedKFold`.

---

**Exercise 4.**
Explain the "one-standard-error rule" for model selection via cross-validation and its rationale.

??? success "Solution to Exercise 4"
    The one-standard-error rule selects the simplest model whose CV error is within one standard error of the minimum CV error. That is:

    1. Compute the mean CV error $\overline{\text{CV}}_d$ and its standard error $\text{SE}_d$ for each model $d$.
    2. Find the model $d^*$ with the minimum $\overline{\text{CV}}_{d^*}$.
    3. Select the simplest model $d$ such that $\overline{\text{CV}}_d \leq \overline{\text{CV}}_{d^*} + \text{SE}_{d^*}$.

    **Rationale:** The CV error estimate is noisy, and the model with the absolute minimum may be more complex than necessary. Models within one SE of the minimum are statistically indistinguishable in predictive performance. Among these, the simplest model is preferred for parsimony, interpretability, and robustness.

    This rule was popularized by Breiman et al. (1984) in the context of CART and is widely used in regularization (e.g., LASSO cross-validation).
