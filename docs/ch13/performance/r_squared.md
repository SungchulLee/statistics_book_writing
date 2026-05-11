# R-Squared and Adjusted R-Squared

After fitting a regression model, the natural first question is: how much of the variation in the response does this model actually explain? A model that captures nearly all the variability in $Y$ is more useful than one that explains very little. R-squared provides a single number that answers this question by comparing the model's residual variation to the total variation in the data.

---

## 1. Decomposition of Variability

To measure how well a regression model fits the data, we decompose the total variability of the response variable $Y$ into two components.

**Total Sum of Squares (SST)** measures the total variability of $Y$ around its mean:

$$
\text{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

**Regression Sum of Squares (SSR)** measures the variability explained by the model:

$$
\text{SSR} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2
$$

**Residual Sum of Squares (SSE)** measures the variability left unexplained:

$$
\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

When the model includes an intercept, these three quantities satisfy the fundamental identity:

$$
\text{SST} = \text{SSR} + \text{SSE}
$$

??? note "Why the decomposition requires an intercept"
    The identity $\text{SST} = \text{SSR} + \text{SSE}$ relies on the cross-term $\sum (y_i - \hat{y}_i)(\hat{y}_i - \bar{y})$ vanishing. This follows from the normal equations, which guarantee that $\sum e_i = 0$ and $\sum e_i \hat{y}_i = 0$ when the model includes an intercept. Without an intercept, these orthogonality conditions may fail, and the decomposition no longer holds.

---

## 2. Coefficient of Determination

The **coefficient of determination** $R^2$ is the proportion of total variability in $Y$ that is explained by the regression model:

$$
R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}
$$

Since $\text{SSR} = \text{SST} - \text{SSE}$ and all sums of squares are non-negative, $R^2$ is bounded between 0 and 1 for models with an intercept:

- $R^2 = 0$: the model explains none of the variability (the fitted values equal $\bar{y}$ for every observation).
- $R^2 = 1$: the model explains all the variability (every observation lies exactly on the fitted line).

### Connection to Correlation

In simple linear regression with a single predictor, $R^2$ equals the square of the Pearson correlation coefficient between $X$ and $Y$:

$$
R^2 = r_{XY}^2
$$

In multiple regression, $R^2$ equals the square of the correlation between the observed values $y_i$ and the fitted values $\hat{y}_i$:

$$
R^2 = r_{y, \hat{y}}^2
$$

---

## 3. Limitations of R-Squared

Although $R^2$ is widely used, it has a critical flaw for model comparison: **it never decreases when additional predictors are added to the model**, even if those predictors have no genuine relationship with $Y$.

To see why, consider two nested models where Model 2 includes all predictors from Model 1 plus an additional variable. The least squares procedure minimizes SSE, so Model 2's SSE is at most equal to Model 1's SSE (it can always set the new coefficient to zero). Since SST is unchanged, $R^2$ for Model 2 is at least as large as for Model 1.

This means a model with $p$ predictors will always have $R^2$ at least as large as a model with fewer predictors, regardless of whether the additional predictors are useful. Taken to the extreme, a model with $n$ parameters (one per observation) achieves $R^2 = 1$ by perfectly interpolating the data, despite having no predictive value.

---

## 4. Adjusted R-Squared

To correct for the automatic inflation of $R^2$ with additional predictors, we use **adjusted R-squared**, which penalizes model complexity by accounting for the number of parameters:

$$
R^2_{\text{adj}} = 1 - \frac{\text{SSE} / (n - p - 1)}{\text{SST} / (n - 1)}
$$

where $n$ is the number of observations and $p$ is the number of predictors (not counting the intercept).

The ratio $\text{SSE} / (n - p - 1)$ is the unbiased estimate of the error variance $\sigma^2$, and $\text{SST} / (n - 1)$ is the sample variance of $Y$. Thus, adjusted $R^2$ compares variance estimates rather than raw sums of squares.

### Relationship to R-Squared

Adjusted $R^2$ can be expressed directly in terms of $R^2$:

$$
R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}
$$

Since $\dfrac{n - 1}{n - p - 1} > 1$ whenever $p \geq 1$, we have $R^2_{\text{adj}} \leq R^2$. The gap between $R^2$ and $R^2_{\text{adj}}$ grows as $p$ increases relative to $n$.

### Key Properties

- **Can decrease** when a useless predictor is added, because the penalty for increasing $p$ may outweigh the small reduction in SSE.
- **Can be negative** when the model fits worse than the intercept-only model (i.e., when $\text{SSE}/(n - p - 1)$ exceeds $\text{SST}/(n - 1)$).
- **Equals $R^2$** when $p = 0$ (intercept-only model), since the penalty factor becomes $\frac{n-1}{n-1} = 1$.

---

## 5. Interpreting R-Squared in Practice

There is no universal threshold for what constitutes a "good" $R^2$. The acceptable range depends heavily on the field and the nature of the data:

| Context | Typical $R^2$ range |
|---|---|
| Physical sciences (controlled experiments) | 0.90 -- 0.99 |
| Engineering models | 0.70 -- 0.95 |
| Social sciences | 0.30 -- 0.70 |
| Financial returns (daily) | 0.01 -- 0.10 |

!!! warning "R-squared does not validate model assumptions"
    A high $R^2$ does not mean the model is correctly specified. A model can achieve a high $R^2$ while violating linearity, independence, or homoscedasticity assumptions. Always supplement $R^2$ with residual diagnostics to verify model adequacy.

---

## 6. Numerical Example

Consider a dataset with $n = 5$ observations where the observed and fitted values are:

| $i$ | $y_i$ | $\hat{y}_i$ | $y_i - \bar{y}$ | $\hat{y}_i - \bar{y}$ | $y_i - \hat{y}_i$ |
|-----|--------|--------------|------------------|------------------------|--------------------|
| 1   | 2      | 2.2          | $-4$             | $-3.8$                 | $-0.2$             |
| 2   | 4      | 3.8          | $-2$             | $-2.2$                 | 0.2                |
| 3   | 5      | 5.4          | $-1$             | $-0.6$                 | $-0.4$             |
| 4   | 8      | 7.0          | 2                | 1.0                    | 1.0                |
| 5   | 11     | 11.6         | 5                | 5.6                    | $-0.6$             |

The sample mean is $\bar{y} = 6$. Computing the sums of squares:

$$
\text{SST} = 16 + 4 + 1 + 4 + 25 = 50
$$

$$
\text{SSE} = 0.04 + 0.04 + 0.16 + 1.00 + 0.36 = 1.60
$$

$$
R^2 = 1 - \frac{1.60}{50} = 1 - 0.032 = 0.968
$$

For a simple linear regression ($p = 1$):

$$
R^2_{\text{adj}} = 1 - (1 - 0.968)\frac{5 - 1}{5 - 1 - 1} = 1 - 0.032 \times \frac{4}{3} = 1 - 0.0427 = 0.957
$$

The model explains about 96.8% of the total variability in $Y$, with the adjusted value of 95.7% reflecting the penalty for one predictor.

## Exercises

**Exercise 1.**
A model has SSE $= 200$ and SST $= 1000$. Compute $R^2$. If we add a useless predictor (random noise), what happens to $R^2$ and why is adjusted $R^2$ needed?

??? success "Solution to Exercise 1"
    $$
    R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{200}{1000} = 0.80
    $$

    Adding any predictor (even random noise) can only decrease SSE (or leave it unchanged) because OLS minimizes SSE over a larger parameter space. So $R^2$ increases (or stays the same) regardless of whether the predictor is useful. This means $R^2$ always favors more complex models.

    **Adjusted $R^2$** corrects for this by penalizing additional parameters:

    $$
    R^2_{\text{adj}} = 1 - \frac{\text{SSE}/(n-p)}{\text{SST}/(n-1)} = 1 - \frac{n-1}{n-p}(1 - R^2)
    $$

    Adding a useless predictor increases $p$ without meaningfully reducing SSE, so the penalty $\frac{n-1}{n-p}$ grows and $R^2_{\text{adj}}$ decreases. This makes adjusted $R^2$ a better criterion for model comparison.

---

**Exercise 2.**
Can $R^2$ be negative? Under what circumstances?

??? success "Solution to Exercise 2"
    For the training data with an intercept, $R^2 \geq 0$ by construction (OLS with an intercept guarantees SSE $\leq$ SST). However, $R^2$ can be negative in two scenarios:

    1. **Test data:** When applying a model fitted on training data to test data, the predictions may be worse than simply predicting the test-set mean. This gives SSE $>$ SST (computed with the test-set mean), so $R^2 < 0$.

    2. **No intercept:** If the model is fitted without an intercept, the prediction does not pass through $\bar{y}$, and SSE can exceed SST.

    A negative $R^2$ on test data indicates the model's predictions are worse than a constant baseline (just predicting the mean). This is a sign of severe overfitting or model misspecification.

---

**Exercise 3.**
Derive the relationship between $R^2$ and the Pearson correlation $r$ between $y$ and $\hat{y}$ in simple linear regression.

??? success "Solution to Exercise 3"
    In simple linear regression (one predictor), the Pearson correlation between $y$ and $\hat{y}$ equals $|r_{xy}|$ (the absolute correlation between $x$ and $y$). The $R^2$ is:

    $$
    R^2 = r_{xy}^2
    $$

    More generally (multiple regression), $R^2 = r_{y\hat{y}}^2$ -- the squared correlation between observed and fitted values. This holds because:

    $$
    R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = \frac{\text{SSR}}{\text{SST}} = \frac{\lVert\hat{\mathbf{y}} - \bar{y}\mathbf{1}\rVert^2}{\lVert\mathbf{y} - \bar{y}\mathbf{1}\rVert^2} = r_{y\hat{y}}^2
    $$

    This relationship provides the interpretation: $R^2$ is the proportion of variance in $y$ that is linearly explained by the model. $\square$

---

**Exercise 4.**
A model has $R^2 = 0.95$ on training data and $R^2 = 0.60$ on test data. Diagnose the likely problem and suggest remedies.

??? success "Solution to Exercise 4"
    The large gap between training $R^2$ (0.95) and test $R^2$ (0.60) is a classic sign of **overfitting**. The model fits the training data very well (including its noise) but fails to generalize.

    Remedies:

    1. **Regularization:** Apply ridge or LASSO regression to shrink coefficients and reduce variance.
    2. **Feature selection:** Remove predictors that contribute noise rather than signal (use cross-validation or information criteria).
    3. **More data:** Increasing $n$ reduces overfitting by providing more information relative to model complexity.
    4. **Simpler model:** Reduce the number of predictors or the polynomial degree.
    5. **Cross-validation:** Use CV during model development instead of relying on training $R^2$, which always overstates performance.
