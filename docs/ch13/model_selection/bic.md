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

## Exercises

**Exercise 1.**
A linear regression with $p = 3$ predictors and $n = 100$ observations has a maximized log-likelihood of $\ell_1 = -150$. A nested model with $p = 5$ predictors has $\ell_2 = -145$. Compute BIC for both models and determine which is preferred.

??? success "Solution to Exercise 1"
    BIC $= -2\ell + k\ln(n)$, where $k$ is the number of estimated parameters.

    Model 1 ($k = 3 + 1 = 4$ including intercept, plus $\sigma^2$, so $k = 5$):

    $$
    \text{BIC}_1 = -2(-150) + 5\ln(100) = 300 + 5(4.605) = 300 + 23.03 = 323.03
    $$

    Model 2 ($k = 5 + 1 + 1 = 7$):

    $$
    \text{BIC}_2 = -2(-145) + 7\ln(100) = 290 + 7(4.605) = 290 + 32.24 = 322.24
    $$

    Model 2 has slightly lower BIC (322.24 vs 323.03), so it is marginally preferred. The improvement in fit ($\Delta\ell = 5$) barely justifies the added complexity.

---

**Exercise 2.**
Compare the penalty terms of AIC ($2k$) and BIC ($k\ln n$). For what sample size $n$ does BIC penalize complexity more heavily than AIC?

??? success "Solution to Exercise 2"
    BIC penalizes more heavily than AIC when $k\ln n > 2k$, i.e., $\ln n > 2$, which gives $n > e^2 \approx 7.39$.

    For any sample size $n \geq 8$, BIC imposes a stricter penalty per parameter than AIC. In practice, since $n$ is almost always much larger than 8, BIC consistently selects simpler (more parsimonious) models than AIC.

    As $n$ grows, BIC's penalty grows without bound ($\ln n \to \infty$), while AIC's penalty remains constant at 2 per parameter. This means BIC increasingly favors simpler models for larger datasets. BIC is consistent (selects the true model as $n \to \infty$ if it is among the candidates), while AIC is efficient (minimizes prediction error) but may overfit.

---

**Exercise 3.**
Explain the Bayesian justification for BIC. In what sense does BIC approximate a Bayesian model comparison?

??? success "Solution to Exercise 3"
    BIC approximates the log marginal likelihood $\log m(\mathbf{y} \mid M)$, which is the key quantity in Bayesian model comparison:

    $$
    \log m(\mathbf{y} \mid M) \approx \ell(\hat{\theta}) - \frac{k}{2}\ln n + O(1)
    $$

    Since $\text{BIC} = -2\ell(\hat{\theta}) + k\ln n$, we have $\text{BIC} \approx -2\log m(\mathbf{y} \mid M) + \text{constant}$.

    Minimizing BIC is approximately equivalent to maximizing the marginal likelihood, which integrates over the parameter space with respect to the prior. This integration naturally penalizes models with more parameters because a complex model "spreads" its prior probability over a larger parameter space (Occam's razor). The $k\ln n$ penalty is the leading-order term in the Laplace approximation to this integral.

---

**Exercise 4.**
When would you prefer AIC over BIC for model selection, and vice versa?

??? success "Solution to Exercise 4"
    **Prefer AIC when:**

    - The goal is **prediction**: AIC minimizes the expected Kullback-Leibler divergence and is asymptotically equivalent to leave-one-out cross-validation. It selects models that predict well.
    - The true model is not among the candidates (AIC performs better in misspecified settings).
    - You want to avoid underfitting at the cost of slight overfitting.

    **Prefer BIC when:**

    - The goal is **model identification**: finding the true data-generating process. BIC is consistent -- it selects the true model with probability approaching 1 as $n \to \infty$.
    - Parsimony is important (e.g., scientific interpretation requires the simplest adequate model).
    - The sample size is large and you want to guard against overfitting.

    In practice, reporting both criteria and noting any disagreements provides the most informative analysis.
