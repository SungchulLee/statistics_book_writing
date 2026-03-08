# AIC and BIC: Model Selection Criteria

When building statistical models, a core challenge is balancing model complexity against model fit. **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** are two widely used techniques that address this trade-off by penalizing models for the number of parameters they use.

## Akaike Information Criterion (AIC)

AIC measures the trade-off between model accuracy and complexity:

$$
AIC = -2 \ln(L) + 2k
$$

where:

- $L$ is the likelihood of the model given the data (how well the model fits).
- $k$ is the number of parameters in the model.

A lower AIC indicates a better model. AIC rewards goodness of fit but penalizes complexity, meaning adding parameters is only beneficial if it significantly improves the fit.

AIC emphasizes **predictive accuracy** and is often used when the main objective is finding a model that generalizes well to unseen data. It aims to balance bias and variance, which is crucial for making accurate predictions. AIC does not explicitly consider sample size in its penalty term, which can lead to different results compared to BIC for larger datasets.

## Bayesian Information Criterion (BIC)

BIC is similar to AIC but applies a stronger penalty for model complexity:

$$
BIC = -2 \ln(L) + k \ln(n)
$$

where:

- $L$ is the likelihood of the model.
- $k$ is the number of parameters.
- $n$ is the number of observations.

BIC penalizes complexity more heavily than AIC by incorporating $\ln(n)$. As the sample size increases, the penalty for additional parameters grows, making BIC more **conservative** and more likely to select simpler models.

BIC is rooted in Bayesian probability theory and focuses on finding the **true model** among candidates. It is particularly useful when interpretability is a priority and when overfitting is a significant concern. Unlike AIC, which focuses on predictive accuracy, BIC prioritizes explaining the data with the fewest parameters.

## Comparison

| Aspect | AIC | BIC |
|---|---|---|
| **Penalty** | $2k$ | $k \ln(n)$ |
| **Complexity Preference** | Favors more complex models | Favors simpler models |
| **Sample Size Sensitivity** | Not sensitive to $n$ | More conservative as $n$ grows |
| **Primary Goal** | Predictive accuracy | Model parsimony and truth |
| **Best Use Case** | Prediction and forecasting | Interpretation and explanation |

Key distinctions:

- **Complexity vs. Parsimony**: AIC tends to favor more flexible models; BIC chooses simpler, more interpretable models.
- **Sample Size**: BIC explicitly accounts for sample size, making it more suitable for large datasets.
- **Model Interpretation**: Use AIC when prediction is the primary goal; use BIC when the focus is on identifying the simplest adequate model.

## Practical Example

Consider fitting several linear regression models to predict house prices. Starting with a simple model with one or two predictors, you gradually add features. As complexity increases, the likelihood improves, but overfitting risk grows.

Suppose you have three candidate models: a simple model with two predictors, a medium model with five, and a complex model with ten.

- **AIC** might select the medium or complex model if the improvement in fit justifies the added parameters.
- **BIC**, with its stronger penalty, might prefer the simple model, particularly with a large dataset.

In the Advertising dataset example from the previous sections:

| Model | AIC | BIC |
|---|---|---|
| TV, Radio, Newspaper | 555.8 | 567.5 |
| TV, Radio | 554.0 | 562.8 |
| TV, Radio, TV:Radio | **399.6** | **411.4** |

Both AIC and BIC agree that the interaction model is preferred, and both show that removing the insignificant Newspaper predictor slightly improves the criteria.

## Summary

AIC and BIC are essential tools for model selection that help avoid overfitting by incorporating penalties for complexity. AIC focuses on predictive accuracy, while BIC emphasizes simplicity. The choice between them depends on the analysis goals:

- Use **AIC** when minimizing prediction error is the priority.
- Use **BIC** when model simplicity, interpretability, or large sample sizes are priorities.

Neither should be used in isolation. They are best complemented with other methods such as cross-validation, which provides additional validation by assessing generalization to new data. By considering multiple criteria, analysts can make more informed decisions that balance fit, complexity, and predictive performance.
