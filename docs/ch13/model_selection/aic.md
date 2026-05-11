# Akaike Information Criterion

When comparing regression models with different numbers of predictors, goodness-of-fit measures like $R^2$ always favor the more complex model. We need a criterion that balances fit against complexity — rewarding models that explain the data well while penalizing unnecessary parameters. The Akaike Information Criterion (AIC) achieves this by estimating the information lost when a model approximates the true data-generating process.

---

## 1. Motivation from Information Theory

Suppose the true distribution of the data is $f$ and a candidate model has distribution $g$. The **Kullback-Leibler (KL) divergence** measures the information lost when $g$ is used to approximate $f$:

$$
D_{\text{KL}}(f \| g) = \int f(x) \ln \frac{f(x)}{g(x)} \, dx
$$

We cannot compute $D_{\text{KL}}$ directly because $f$ is unknown, but Akaike (1973) showed that the expected log-likelihood, evaluated at the maximum likelihood estimate, provides an asymptotically unbiased estimate of the relative KL divergence — up to a bias correction of $k$ parameters. This insight leads to the AIC formula.

---

## 2. Definition

The **Akaike Information Criterion** for a model with $k$ estimated parameters and maximized log-likelihood $\ln \hat{L}$ is:

$$
\text{AIC} = 2k - 2 \ln \hat{L}
$$

where:

- $k$ is the total number of estimated parameters (including the intercept and the error variance $\sigma^2$ if applicable).
- $\hat{L}$ is the value of the likelihood function evaluated at the maximum likelihood estimates.

The term $2k$ penalizes model complexity, while $-2 \ln \hat{L}$ rewards goodness of fit. A **lower AIC indicates a better model** — one that achieves a favorable tradeoff between fit and parsimony.

---

## 3. AIC for Linear Regression

For a linear regression model with Gaussian errors, the maximized log-likelihood takes the form:

$$
\ln \hat{L} = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln \hat{\sigma}^2 - \frac{n}{2}
$$

where $\hat{\sigma}^2 = \text{SSE}/n$ is the maximum likelihood estimate of the error variance. Substituting into the AIC formula:

$$
\text{AIC} = 2k + n \ln\!\left(\frac{\text{SSE}}{n}\right) + n \ln(2\pi) + n
$$

Since the terms $n \ln(2\pi) + n$ are constant across models with the same data, model comparisons can use the simplified form:

$$
\text{AIC} = 2k + n \ln\!\left(\frac{\text{SSE}}{n}\right)
$$

Here $k = p + 2$ (the $p$ regression coefficients, the intercept, and $\sigma^2$).

---

## 4. Small-Sample Correction (AICc)

When the sample size $n$ is small relative to the number of parameters $k$, AIC tends to select overly complex models. Hurvich and Tsai (1989) proposed a corrected version:

$$
\text{AIC}_c = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

The correction term $\frac{2k(k+1)}{n-k-1}$ is negligible when $n \gg k$ but substantial when $n/k < 40$. A common rule of thumb is to use AICc whenever $n / k < 40$.

!!! tip "When in doubt, use AICc"
    AICc converges to AIC as $n \to \infty$, so using AICc is always at least as good as AIC for model selection. Many practitioners default to AICc regardless of sample size.

---

## 5. Using AIC for Model Comparison

AIC is meaningful only in relative terms — the absolute value of AIC has no interpretation. To compare $M$ candidate models, compute $\text{AIC}_j$ for each model $j = 1, \ldots, M$ and select the model with the smallest AIC.

### Delta AIC

The **delta AIC** for model $j$ is:

$$
\Delta_j = \text{AIC}_j - \text{AIC}_{\min}
$$

where $\text{AIC}_{\min}$ is the smallest AIC among all candidates. Burnham and Anderson (2002) suggest the following interpretation:

| $\Delta_j$ | Interpretation |
|-------------|----------------|
| 0 -- 2     | Substantial support; model is competitive |
| 4 -- 7     | Considerably less support |
| > 10       | Essentially no support |

### Akaike Weights

Akaike weights provide a probability-like measure of each model's relative support:

$$
w_j = \frac{\exp(-\Delta_j / 2)}{\sum_{m=1}^{M} \exp(-\Delta_m / 2)}
$$

The weights sum to 1 and can be interpreted as the approximate probability that model $j$ is the best model among the candidates, given the data.

---

## 6. Important Properties

- **Not a hypothesis test**: AIC does not test whether a model is "significantly" better. It ranks models by estimated predictive accuracy.
- **Relative, not absolute**: AIC values are only meaningful when compared within the same dataset. Comparing AIC across different datasets is invalid.
- **Favors prediction**: AIC is asymptotically equivalent to leave-one-out cross-validation for model selection, making it prediction-oriented rather than oriented toward identifying the "true" model.
- **No consistency**: If the true model is among the candidates and $n \to \infty$, AIC does not necessarily select the true model. It tends to select slightly more complex models. BIC has this consistency property instead.

!!! warning "Same data required"
    When comparing models via AIC, all models must be fit to exactly the same dataset (same observations). Comparing AIC values across models fit to datasets of different sizes or with different observations is meaningless.

---

## 7. Numerical Example

Consider three candidate models for a dataset with $n = 50$ observations:

| Model | Predictors ($p$) | $k$ | SSE | $\text{AIC}$ |
|-------|-------------------|------|------|---------------|
| A     | 1                 | 3    | 120  | $50\ln(120/50) + 2(3) = 50(0.875) + 6 = 49.8$ |
| B     | 3                 | 5    | 90   | $50\ln(90/50) + 2(5) = 50(0.588) + 10 = 39.4$ |
| C     | 6                 | 8    | 85   | $50\ln(85/50) + 2(8) = 50(0.531) + 16 = 42.5$ |

Model B has the lowest AIC (39.4), suggesting the best balance between fit and complexity. Model C achieves a slightly lower SSE than Model B, but its three additional parameters are not justified by the modest improvement in fit — the complexity penalty of $2 \times 8 = 16$ versus $2 \times 5 = 10$ outweighs the gain.

The delta values are $\Delta_A = 10.4$, $\Delta_B = 0$, and $\Delta_C = 3.1$. Under Burnham and Anderson's guidelines, Model A has essentially no support, Model B is the best, and Model C has noticeably less support but cannot be dismissed entirely.

## Exercises

**Exercise 1.**
Two linear regression models, **Model A** and **Model B**, were fitted to the same dataset. The results show that:

- Model A has a higher $R^2$ value.
- Model B has a lower AIC (Akaike Information Criterion).

**(a)** Between Model A (higher $R^2$) and Model B (lower AIC), which model should be selected?

**(b)** Why is AIC preferred over $R^2$ for model selection?

??? success "Solution to Exercise 1"

    **(a)** It is generally recommended to prioritize the model with the **lower AIC** (Model B in this case).

    **(b)** Three key reasons:

    1. **Limitations of $R^2$**: $R^2$ measures the proportion of variance explained but does not penalize for model complexity. A model with more predictors can artificially increase $R^2$, even if those predictors do not improve true predictive performance, leading to overfitting.

    2. **Strengths of AIC**: AIC balances model fit (how well the model explains the data) and model simplicity (penalizing additional predictors). This helps select the model likely to have better predictive performance on unseen data.

    3. **Key Difference**: $R^2$ focuses solely on explanatory power, while AIC accounts for both explanation and complexity. AIC is therefore a more reliable criterion for comparing models in terms of prediction accuracy.
