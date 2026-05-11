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


## Exercises

**Exercise 1.**
In a logistic regression, $\hat{\beta}_1 = 0.693$ for a binary predictor. Compute and interpret the odds ratio.

??? success "Solution to Exercise 1"
    The odds ratio is $\text{OR} = e^{\hat{\beta}_1} = e^{0.693} = 2.0$.

    Interpretation: the odds of the outcome (e.g., disease) are 2 times higher for the group with $X = 1$ compared to $X = 0$, holding other variables constant. Equivalently, the presence of the factor doubles the odds.

    Note: $\text{OR} = 2$ does not mean the probability doubles. If the baseline probability is 10% (odds = 0.111), the new odds are 0.222, giving a probability of $0.222/1.222 = 18.2\%$ -- less than double.

---

**Exercise 2.**
A logistic regression for heart disease includes age (continuous) with $\hat{\beta}_{\text{age}} = 0.05$. Interpret this coefficient in terms of odds ratios.

??? success "Solution to Exercise 2"
    The odds ratio per one-year increase in age is $e^{0.05} = 1.051$. Each additional year of age increases the odds of heart disease by about 5.1%.

    For a 10-year increase: $\text{OR}_{10} = e^{10 \times 0.05} = e^{0.5} = 1.649$. A person 10 years older has about 65% higher odds of heart disease.

    This multiplicative interpretation is key: each year multiplies the odds by 1.051, so effects compound over larger age differences.

---

**Exercise 3.**
Explain the difference between an odds ratio and a relative risk. When are they approximately equal?

??? success "Solution to Exercise 3"
    The **odds ratio** is $\text{OR} = \frac{p_1/(1-p_1)}{p_0/(1-p_0)}$. The **relative risk** is $\text{RR} = p_1/p_0$.

    They are approximately equal when the outcome is rare ($p_0$ and $p_1$ are both small). When $p \ll 1$, odds $\approx p$, so $\text{OR} \approx \text{RR}$.

    For common outcomes ($p > 10\%$), OR exaggerates the association compared to RR. For example, if $p_0 = 0.3$ and $p_1 = 0.5$: $\text{RR} = 1.67$ but $\text{OR} = (0.5/0.5)/(0.3/0.7) = 2.33$. The OR overstates the relative increase.

    Logistic regression directly estimates OR (not RR). For rare outcomes, this distinction is minor; for common outcomes, log-binomial regression or Poisson regression with robust standard errors can estimate RR directly.

---

**Exercise 4.**
A 95% confidence interval for an odds ratio is $(0.85, 1.42)$. What does this imply about the statistical significance of the predictor?

??? success "Solution to Exercise 4"
    Since the CI for the odds ratio includes 1.0, the effect is **not statistically significant** at $\alpha = 0.05$. An OR of 1.0 corresponds to no association ($\beta = 0$), and its inclusion in the CI means we cannot reject $H_0: \text{OR} = 1$.

    Equivalently, the CI for $\beta$ is $(\ln 0.85, \ln 1.42) = (-0.163, 0.351)$, which includes 0.

    The result suggests the predictor has a small, non-significant effect. The point estimate OR $= \sqrt{0.85 \times 1.42} \approx 1.10$ suggests a modest positive association, but the data are consistent with no effect or even a slight negative effect.
