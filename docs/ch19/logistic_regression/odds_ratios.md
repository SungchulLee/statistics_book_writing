# Odds Ratios and Coefficient Interpretation

## The Odds and Odds Ratio

In logistic regression, the coefficient $\theta_j$ has a direct interpretation through the **odds ratio**. Recall that the logit model relates the log-odds linearly to the features:

$$
\operatorname{logit}\bigl(P(Y=1\mid\mathbf{x})\bigr) = \mathbf{x}^T\boldsymbol{\theta}
$$

which is equivalent to

$$
\log\left(\frac{p}{1-p}\right) = \mathbf{x}^T\boldsymbol{\theta}
$$

where $p = P(Y=1\mid\mathbf{x})$.

## Multiplicative Effect of Coefficients

When we increase feature $x_j$ by one unit while holding all other features constant, the log-odds increases by $\theta_j$. Therefore, the **odds multiply by $e^{\theta_j}$**:

$$
\frac{\text{odds}_{\text{new}}}{\text{odds}_{\text{old}}} = e^{\theta_j}
$$

### Example

If $\theta_j = 0.5$ for a feature representing the borrower's credit score, then a one-unit increase in the score multiplies the odds of default by $e^{0.5} \approx 1.649$. This means the odds increase by about 64.9%.

Conversely, if $\theta_j = -0.5$, the odds multiply by $e^{-0.5} \approx 0.606$, indicating a 39.4% decrease in odds.

## Interpreting Coefficients

The **odds ratio** $OR_j = e^{\theta_j}$ has an intuitive interpretation:

| $\theta_j$ | $OR_j = e^{\theta_j}$ | Interpretation |
|---|---|---|
| $-1.0$ | $\approx 0.368$ | Odds decrease by 63.2% per unit increase |
| $-0.5$ | $\approx 0.606$ | Odds decrease by 39.4% per unit increase |
| $0.0$ | $1.0$ | No effect on odds |
| $0.5$ | $\approx 1.649$ | Odds increase by 64.9% per unit increase |
| $1.0$ | $\approx 2.718$ | Odds increase by 171.8% per unit increase |

## Example: Loan Default Prediction

In a loan default study, the estimated coefficients might be:

| Feature | Coefficient | Odds Ratio | Interpretation |
|---|---|---|---|
| payment_inc_ratio | $0.0797$ | $e^{0.0797} \approx 1.083$ | 8.3% increase in odds per unit |
| borrower_score | $-4.6126$ | $e^{-4.6126} \approx 0.0098$ | 99% decrease in odds per unit increase |
| small_business | $1.2153$ | $e^{1.2153} \approx 3.373$ | 237% increase in odds (vs. baseline) |

Higher payment-to-income ratios increase default risk, while higher borrower scores dramatically reduce it. Loans for small business purposes carry much higher default risk compared to credit card purposes (the baseline).

## Confidence Intervals for Odds Ratios

When conducting inference via Maximum Likelihood Estimation, we obtain standard errors and confidence intervals for the coefficients $\theta_j$. These can be transformed to confidence intervals for the odds ratios:

If a 95% CI for $\theta_j$ is $[\theta_j^L, \theta_j^U]$, then the 95% CI for $OR_j = e^{\theta_j}$ is:

$$
[e^{\theta_j^L}, e^{\theta_j^U}]
$$

**Important:** A confidence interval for $OR_j$ that excludes 1.0 indicates that the coefficient $\theta_j$ is statistically significantly different from zero at the corresponding confidence level.

## Categorical Features and Baseline Coding

When using one-hot or reference coding for categorical variables (e.g., home ownership type), the coefficient represents the **change relative to a baseline category**. The baseline category (omitted to avoid multicollinearity) has an implicit coefficient of 0 and odds ratio of 1.

For example, if "MORTGAGE" is the baseline and the coefficient for "RENT" is $0.157$, then renting (vs. owning with a mortgage) increases the odds of default by $e^{0.157} - 1 \approx 17\%$.

## Statistical Significance

To test whether a coefficient is significantly different from zero, we use:

- **Wald test:** $Z = \theta_j / \text{SE}(\theta_j) \sim N(0,1)$
- **Likelihood ratio test:** Compares log-likelihoods of nested models

Both methods are implemented in statistical packages like `statsmodels` and provide p-values for hypothesis testing.
