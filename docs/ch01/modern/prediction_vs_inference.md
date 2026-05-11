# Prediction vs Inference

Data analysis serves two fundamentally different goals: **predicting outcomes** as accurately as possible, and **understanding relationships** between variables. The distinction drives every methodological decision — what model to use, how to evaluate it, what assumptions matter, and how to interpret the result. Conflating the two is the single most common source of methodological confusion in applied work.

## Definition

**Prediction** aims to estimate $\hat Y = \hat f(X)$ with minimum error on unseen data; the internal structure of $\hat f$ is secondary to its out-of-sample loss.

**Inference** aims to understand how $X$ relates to $Y$: which variables matter, the direction and magnitude of effects, whether relationships are causal, and how confident we are in our conclusions.

| Aspect | Prediction | Inference |
|---|---|---|
| Goal | Minimize forecast loss | Understand relationships |
| Model choice | Whichever predicts best | Interpretable model preferred |
| Evaluation | Out-of-sample MSE / AUC | $p$-values, CIs, effect sizes |
| Complexity | High complexity welcome | Simpler models preferred |
| Validity threat | Overfitting | Model misspecification, confounding |

## Explanation

### Two error decompositions

For **prediction** at a fixed input $x_0$,

$$
\mathbb{E}[(Y - \hat f(x_0))^2] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{(\mathbb{E}[\hat f(x_0)] - f^*(x_0))^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{variance}}
$$

Total prediction error is what the analyst minimizes — a more complex model trades bias for variance until the sum is minimized.

For **inference** on a parameter $\theta$,

$$
\mathrm{MSE}(\hat \theta) = \mathrm{bias}(\hat \theta)^2 + \mathrm{Var}(\hat \theta)
$$

but the *interpretation* of $\hat\theta$ as estimating a specific population quantity also matters. A regression coefficient is a meaningful estimator of $\beta_j$ only if the model is correctly specified; a complex flexible estimator can have low MSE but no interpretable target parameter at all.

### Different evaluation criteria

- **Prediction**: evaluate on data the model has not seen. A simple held-out test set, $k$-fold cross-validation, or time-based splits for temporal data.
- **Inference**: evaluate the validity of confidence intervals and $p$-values via coverage and Type I/II error rates, under explicit assumptions about the data-generating process. There is no equivalent of cross-validation for a $p$-value.

### Different complexity sweet spots

For prediction, the bias-variance trade-off allows complexity up to the point where adding parameters increases test MSE. Modern deep models work because at sufficient scale (data and parameters), they can be very complex without overfitting.

For inference, model complexity directly inflates variance of each individual parameter (multicollinearity, weak instruments) and threatens the interpretability of the resulting coefficients. Simpler models are usually preferred — even at the cost of some predictive accuracy — because their parameters can be defended as estimands of specific population quantities.

### Bridging methods

The boundary is not sharp:

- **LASSO** selects variables (inference-like sparsity) while optimizing prediction.
- **SHAP / LIME** provide post-hoc interpretability for black-box predictions.
- **Causal ML** (double/debiased ML, causal forests) targets unbiased causal estimands using flexible nuisance estimators.
- **Conformal prediction** gives valid prediction intervals around arbitrary predictors without distributional assumptions.

These tools attempt to keep prediction's flexibility while restoring some of inference's interpretability or uncertainty quantification.

## Examples

```python
"""Same data, two different analytical goals."""

import numpy as np

rng = np.random.default_rng(42)
n = 300
x1 = rng.standard_normal(n)
x2 = rng.standard_normal(n)
x3 = 0.8 * x1 + rng.normal(0, 0.5, n)  # correlated with x1
y = 3 * x1 - 2 * x2 + rng.standard_normal(n)  # x3 has no causal effect

X = np.column_stack([np.ones(n), x1, x2, x3])

# === Inference: estimate coefficients with standard errors ===
beta = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ beta
s2 = ((y - y_hat) ** 2).sum() / (n - 4)
se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
print("Inference:")
for name, b, s in zip(["intercept", "x1", "x2", "x3"], beta, se):
    print(f"  {name:>10s}: beta = {b:+.3f}, SE = {s:.3f}, t = {b/s:+.2f}")

# === Prediction: out-of-sample MSE ===
train_idx = rng.choice(n, 200, replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
b_train = np.linalg.solve(X[train_idx].T @ X[train_idx], X[train_idx].T @ y[train_idx])
mse_test = ((y[test_idx] - X[test_idx] @ b_train) ** 2).mean()
print(f"\nPrediction: test MSE = {mse_test:.3f}")
```

Notice: $x_3$ is correlated with $x_1$ but has no causal effect on $y$. The inference table will show $x_3$'s coefficient near zero with a wide standard error (correctly identifying it as not adding signal beyond $x_1$). The prediction model would be hurt by removing $x_3$ slightly (it carries some redundant information), but the *causal* conclusion is unchanged.

## Exercises

**Exercise 1.**
For each research question, state whether the primary goal is **prediction** or **inference**, and explain the implications for method choice.

**(a)** Does a college degree cause higher lifetime earnings, controlling for ability and family background?
**(b)** Which customers are most likely to churn in the next 30 days?
**(c)** What is the effect of class size on student test scores?
**(d)** How accurately can we forecast next month's regional electricity demand?
**(e)** Does a particular gene variant increase Alzheimer's disease risk?

??? success "Solution to Exercise 1"
    (a) **Inference** — estimate and test a causal effect. Methods: regression with controls, instrumental variables, or natural experiments. Interpretability is essential.
    (b) **Prediction** — identify at-risk customers accurately. Flexible algorithms (gradient-boosted trees, neural nets) are appropriate.
    (c) **Inference** — causal estimand. Quasi-experimental methods (regression discontinuity, randomized class-size experiments) are needed.
    (d) **Prediction** — minimize forecast error. ARIMA, gradient boosting on engineered time-series features, or neural forecasting models.
    (e) **Inference** — estimate the effect size and its significance, controlling for population structure and other confounders. GWAS methodology with Bonferroni or FDR control.

---

**Exercise 2.**
A team trains an XGBoost model on observational health data and reports that its "feature importance" identifies blood pressure as the strongest predictor of stroke. Explain why this does *not* imply that lowering blood pressure causally reduces stroke risk.

??? success "Solution to Exercise 2"
    Feature importance in tree ensembles is a *predictive* notion — it measures how much a feature contributes to reducing in-sample loss across the trees. A feature can be highly predictive without being causal:

    - **Confounding**: blood pressure and stroke share many common causes (age, obesity, diabetes); the model uses BP as a *proxy* for these underlying risks, not as a causal driver.
    - **Reverse causation**: ongoing cardiovascular damage can raise BP, so BP is correlated with stroke partly through being a downstream marker.
    - **Mediator**: even if there is a true causal pathway, BP may be partly mediating the effect of (say) sodium intake — interventions on BP may not reproduce the natural-history correlation.

    The causal effect of BP-lowering interventions is established through *randomized trials* (e.g., SPRINT trial), not through observational feature importance. The trials confirm a real effect, but smaller than the observational association.

---

**Exercise 3.**
You build two models on the same training data: a linear regression with $R^2_{\text{train}} = 0.45$ and a random forest with $R^2_{\text{train}} = 0.95$. Why is it inappropriate to compare them on $R^2_{\text{train}}$? What is the appropriate comparison?

??? success "Solution to Exercise 3"
    $R^2_{\text{train}}$ is a measure of *in-sample* fit. A sufficiently flexible model (random forest, deep tree) can fit the training data arbitrarily well by memorizing it — pushing $R^2_{\text{train}}$ toward 1 without any genuine signal. Linear regression cannot do this because of its strict parametric form, so its $R^2_{\text{train}}$ is bounded by the true signal.

    The appropriate comparison is **$R^2$ on a held-out test set** (or equivalently, $k$-fold cross-validation $R^2$). If the random forest's test $R^2$ is also 0.95, it has genuinely captured real signal; if its test $R^2$ falls to 0.30 while training is still 0.95, it has overfit and is actually *worse* than the linear model out of sample.

    This is the same lesson as Exercise 6 of Chapter 1's `supervised.md`: the held-out test set is non-negotiable for honest evaluation.

---

**Exercise 4.**
Explain why a 95% confidence interval for a regression coefficient is meaningful only under explicit assumptions (correct functional form, no omitted confounders, correct error structure), while a 95% conformal prediction interval requires almost no assumptions. What is the price for this difference in assumptions?

??? success "Solution to Exercise 4"
    A regression coefficient's CI estimates the uncertainty of a specific population parameter (the slope, holding other variables fixed). If the linear model is wrong (omitted confounder, nonlinearity, heteroscedasticity), the CI may have wrong coverage — not because of sample variability but because of misspecification. Validity rests on the model.

    A conformal prediction interval estimates "given $X = x$, where will $Y$ likely lie?" using only the assumption that training and test data are exchangeable. It is valid for *any* underlying model — even a black-box predictor.

    **The price:** conformal intervals refer to the *distribution of $Y$ given $X$*, not to any structural parameter. They are wider than parametric CIs at any given coverage level because they make weaker assumptions. They cannot tell you the *effect* of an intervention — only the range of likely outcomes if the world continues to behave like the training data.

    Use parametric CIs when you trust the model and want interpretation of structural parameters. Use conformal intervals when you only want valid prediction intervals from a flexible predictor.

---

**Exercise 5.**
A retailer A/B-tests a recommendation algorithm and finds a 2% lift in revenue per user. They want to know (a) how much extra revenue to expect when they roll out, and (b) why the new algorithm works better. Discuss which question is prediction, which is inference, and the analyses each requires.

??? success "Solution to Exercise 5"
    **(a) Prediction:** what revenue uplift will result from rollout? This is a forecasting question. The A/B test gives an unbiased point estimate (+2%) and a confidence interval that already addresses sampling uncertainty. Additional considerations include **scale effects** (the test population may not be the same as the rollout population), **novelty effects** (a short test may overstate steady-state lift), and **compositional shifts** (the test was during a specific season). Bringing in time-series forecasting models that incorporate these dynamics is appropriate; the goal is accurate next-quarter revenue.

    **(b) Inference (specifically, causal mechanism):** why does the new algorithm work better? The A/B test confirms the *effect* but not the *mechanism*. Possible explanations: better personalization, surfacing more long-tail products, exposing users to higher-margin items, reducing decision fatigue. To distinguish among these, ablation studies (turn off one component at a time, re-A/B test) and analysis of behavioral pathways (e.g., did the recommended-then-purchased rate increase or did unrelated browsing increase) are needed. This is closer to scientific inference about mechanism than to forecasting.

---

**Exercise 6.**
Why are **assumptions about the data-generating process** central to inference but secondary to prediction? Use this to explain the practical paradox: complex flexible models often *fail* in inference settings even when they succeed in prediction.

??? success "Solution to Exercise 6"
    Inference targets a quantity (a parameter $\theta$, a treatment effect, a marginal effect) that is defined *only with reference to a data-generating process*. To say "the effect of $X$ on $Y$ controlling for $Z$" presupposes a model in which such an effect exists. The estimator's validity rests on the model being approximately correct.

    Prediction targets future $Y$ values directly, without defining a structural parameter. A model that produces accurate forecasts is useful even if it has no interpretable structure.

    **The paradox:** complex flexible models can predict well because they capture rich patterns in $(X, Y)$, regardless of whether those patterns are causal. But "rich pattern" includes everything — confounded associations, reverse causality, sample-selection artifacts — that *invalidates* inference. A neural network might predict heart disease perfectly from a thousand observational features and yet provide zero guidance on which feature, when intervened upon, would reduce disease. The model has high prediction accuracy and zero causal validity.

    The lesson: choose method based on the *question*, not on what's currently fashionable. Prediction tools and inference tools answer different questions.
