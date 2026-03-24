# Bayesian Information Criterion

AIC estimates the relative predictive accuracy of competing models, but it does not aim to identify the true data-generating model. When the goal is to select the correct model from a set of candidates — and the true model is believed to be among them — the Bayesian Information Criterion (BIC) provides a criterion with stronger theoretical guarantees. BIC applies a heavier penalty for model complexity, making it more conservative than AIC.

---

## 1. Bayesian Motivation

The BIC was derived by Schwarz (1978) from a Bayesian perspective. Consider $M$ candidate models $\mathcal{M}_1, \ldots, \mathcal{M}_M$, each with prior probability $P(\mathcal{M}_j)$. By Bayes' theorem, the posterior probability of model $j$ given data $\mathbf{y}$ is:

$$
P(\mathcal{M}_j \mid \mathbf{y}) \propto P(\mathbf{y} \mid \mathcal{M}_j) \cdot P(\mathcal{M}_j)
$$

The marginal likelihood $P(\mathbf{y} \mid \mathcal{M}_j)$ integrates over all parameter values:

$$
P(\mathbf{y} \mid \mathcal{M}_j) = \int P(\mathbf{y} \mid \boldsymbol{\theta}_j, \mathcal{M}_j) \, P(\boldsymbol{\theta}_j \mid \mathcal{M}_j) \, d\boldsymbol{\theta}_j
$$

Schwarz showed that, under regularity conditions, the log marginal likelihood can be approximated as:

$$
\ln P(\mathbf{y} \mid \mathcal{M}_j) \approx \ln \hat{L}_j - \frac{k_j}{2} \ln n
$$

where $\hat{L}_j$ is the maximized likelihood, $k_j$ is the number of parameters, and $n$ is the sample size. Multiplying by $-2$ yields the BIC.

---

## 2. Definition

The **Bayesian Information Criterion** for a model with $k$ estimated parameters and maximized log-likelihood $\ln \hat{L}$ is:

$$
\text{BIC} = k \ln n - 2 \ln \hat{L}
$$

where:

- $k$ is the total number of estimated parameters.
- $n$ is the number of observations.
- $\hat{L}$ is the maximized likelihood.

Like AIC, a **lower BIC indicates a better model**. The key difference is the penalty term: BIC uses $k \ln n$ instead of $2k$.

---

## 3. BIC for Linear Regression

For a linear regression model with Gaussian errors, substituting the maximized log-likelihood gives:

$$
\text{BIC} = k \ln n + n \ln\!\left(\frac{\text{SSE}}{n}\right) + n \ln(2\pi) + n
$$

Since the constant terms do not affect model comparison, the working formula is:

$$
\text{BIC} = k \ln n + n \ln\!\left(\frac{\text{SSE}}{n}\right)
$$

where $k = p + 2$ counts the $p$ regression coefficients, the intercept, and the error variance.

---

## 4. BIC Penalty vs AIC Penalty

The complexity penalties of AIC and BIC are:

| Criterion | Penalty per parameter |
|-----------|----------------------|
| AIC       | $2$                  |
| BIC       | $\ln n$              |

Since $\ln n > 2$ when $n > e^2 \approx 7.39$, the BIC penalty is stricter than the AIC penalty for any sample size $n \geq 8$. In practice, virtually all datasets satisfy this condition, so BIC favors simpler models than AIC.

The growing penalty means that as $n$ increases, BIC requires progressively stronger evidence in the likelihood to justify adding parameters. This is the mechanism behind BIC's consistency property.

---

## 5. Consistency

BIC is **consistent** for model selection: if the true data-generating model is among the candidates, BIC selects the true model with probability approaching 1 as $n \to \infty$.

Formally, let $\mathcal{M}^*$ be the true model and $\hat{\mathcal{M}}_{\text{BIC}}$ be the model selected by BIC. Then:

$$
P(\hat{\mathcal{M}}_{\text{BIC}} = \mathcal{M}^*) \to 1 \quad \text{as } n \to \infty
$$

This property is not shared by AIC. AIC is asymptotically efficient (it minimizes the expected prediction error), but it does not converge to the true model. Instead, AIC tends to select slightly overfit models in large samples.

!!! note "Consistency requires the true model to be a candidate"
    BIC's consistency guarantee holds only when the true model is among the candidates. If the true model is not in the candidate set — which is arguably always the case for real data — BIC's consistency is irrelevant, and AIC's focus on predictive accuracy may be more appropriate.

---

## 6. AIC vs BIC Summary

| Property | AIC | BIC |
|---|---|---|
| Penalty | $2k$ | $k \ln n$ |
| Goal | Minimize prediction error | Identify the true model |
| Consistency | No | Yes |
| Efficiency | Yes (asymptotically) | No |
| Tendency | Slightly overfit | Slightly underfit |
| Small samples | AICc corrects for finite $n$ | No standard small-sample correction |

!!! tip "Practical guidance"
    When the goal is prediction, prefer AIC (or AICc). When the goal is identifying a parsimonious explanatory model and you believe the true model is among the candidates, prefer BIC. When AIC and BIC agree, the choice is clear. When they disagree, the discrepancy often involves a single marginal predictor — consider the scientific context to decide.

---

## 7. Numerical Example

Consider the same three models from the AIC page, with $n = 50$:

| Model | $p$ | $k$ | SSE | AIC | BIC |
|-------|-----|------|------|------|------|
| A     | 1   | 3    | 120  | 49.8 | $3 \ln(50) + 50 \ln(120/50) = 11.7 + 43.8 = 55.5$ |
| B     | 3   | 5    | 90   | 39.4 | $5 \ln(50) + 50 \ln(90/50) = 19.6 + 29.4 = 49.0$ |
| C     | 6   | 8    | 85   | 42.5 | $8 \ln(50) + 50 \ln(85/50) = 31.3 + 26.5 = 57.8$ |

Both AIC and BIC select Model B as the best. However, BIC penalizes Model C more heavily: the BIC gap between B and C ($57.8 - 49.0 = 8.8$) is larger than the AIC gap ($42.5 - 39.4 = 3.1$), reflecting BIC's stronger penalty against the three additional parameters in Model C.

In this case AIC and BIC agree, but with a larger sample or a smaller improvement from Model C's extra predictors, BIC would be even more decisive in favoring Model B.
