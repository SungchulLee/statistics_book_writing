# Prediction vs. Inference

## Overview

Data analysis serves two fundamentally different goals: **prediction** (forecasting an outcome as accurately as possible) and **inference** (understanding the relationship between variables). While both use models and data, they prioritize different qualities and lead to different methodological choices.

## Prediction

In a **prediction** task, the goal is to estimate the value of an outcome variable $Y$ for new observations as accurately as possible, given input features $X$. The emphasis is on minimizing prediction error, not on understanding *why* the prediction works.

$$
\hat{Y} = \hat{f}(X)
$$

The quality of $\hat{f}$ is judged by a loss function evaluated on **unseen data**, such as mean squared error (MSE) or classification accuracy. The internal structure of $\hat{f}$—whether it is a linear formula or a deep neural network—is secondary to its predictive performance.

**Examples in finance:**

- Predicting next-day stock returns from historical data.
- Estimating the probability that a loan applicant will default.
- Forecasting portfolio volatility.

## Inference

In an **inference** task, the goal is to understand the **relationship** between $X$ and $Y$: which variables matter, how they are related, and whether the relationship is causal. The emphasis is on interpretability, uncertainty quantification, and testing hypotheses.

**Key questions in inference:**

- Which predictors are associated with the outcome? (variable selection)
- What is the direction and magnitude of each association? (coefficient estimation)
- Is the association statistically significant? (hypothesis testing)
- Is the relationship causal or merely correlational? (causal inference)

**Examples in finance:**

- Does a company's ESG score affect its stock returns, controlling for size and sector?
- What is the marginal effect of an additional year of education on earnings?
- Did a new regulation cause a reduction in market volatility?

## Comparison

| Aspect | Prediction | Inference |
|---|---|---|
| **Goal** | Minimize forecast error | Understand relationships |
| **Model choice** | Whichever model predicts best | Interpretable model preferred |
| **Evaluation** | Out-of-sample accuracy | Coefficient significance, causal validity |
| **Complexity** | Complex models welcome | Simpler models preferred for clarity |
| **Key metric** | MSE, accuracy, AUC | p-values, confidence intervals, effect sizes |
| **Overfitting concern** | Major (manage via cross-validation) | Moderate (manage via assumptions) |

## The Bias–Variance Perspective

The distinction maps naturally onto the **bias–variance tradeoff**:

- **Inference** favors lower-variance, interpretable models (even at the cost of some bias) because stable, interpretable coefficients are the goal.
- **Prediction** favors the model complexity that minimizes total error (bias² + variance), which often means more flexible, higher-variance models paired with regularization.

## Can We Do Both?

In practice, many analyses require elements of both prediction and inference. Some approaches that bridge the gap:

- **Regularized regression** (LASSO): selects variables (inference-like) while optimizing prediction.
- **SHAP values and feature importance**: provide post-hoc interpretability for complex predictive models.
- **Causal machine learning** (e.g., double/debiased ML, causal forests): combines flexible prediction with valid causal inference.

## Key Takeaways

- Prediction asks "what will happen?" while inference asks "why does it happen?"
- The best model for prediction is not always the best model for inference, and vice versa.
- Modern methods increasingly aim to deliver both accurate predictions and interpretable insights, but the primary goal should always guide methodology.
