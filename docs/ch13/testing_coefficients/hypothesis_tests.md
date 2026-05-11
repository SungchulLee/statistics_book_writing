# Hypothesis Tests for Regression Coefficients (t-tests)


In linear regression, a key objective is to evaluate the impact of each predictor on the dependent variable. Hypothesis tests for regression coefficients, typically conducted using **t-tests**, assess whether the predictors significantly contribute to the model.

---

## Formulation of the t-test

A t-test in linear regression evaluates the null hypothesis that a regression coefficient equals zero, indicating that the corresponding predictor variable has no effect on the dependent variable.

The hypotheses are:

- **Null Hypothesis ($H_0$):** $\beta_i = 0$ — the predictor has no effect.
- **Alternative Hypothesis ($H_1$):** $\beta_i \neq 0$ — the predictor has a significant effect.

The **t-statistic** is calculated for each coefficient:

$$
t = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)}
$$

where:

- $\hat{\beta}_i$ is the estimated coefficient for predictor $i$,
- $SE(\hat{\beta}_i)$ is the standard error of the estimated coefficient, representing the variability of the estimate.

The t-statistic measures how many standard errors the estimated coefficient is away from zero. A larger absolute value indicates stronger evidence against the null hypothesis.

---

## p-values in t-tests

The **p-value** is the probability of observing a t-statistic as extreme as (or more extreme than) the one computed, assuming the null hypothesis is true.

- **Low p-value (< 0.05):** The coefficient is significantly different from zero — reject $H_0$. The predictor has a significant impact on the dependent variable.
- **High p-value ($\geq$ 0.05):** We fail to reject $H_0$. The predictor might not significantly affect the dependent variable, and its inclusion may not improve predictive performance.

!!! example "Salary Prediction"
    In a linear regression model predicting salary based on years of experience, a t-test assesses whether "years of experience" has a significant effect on salary. If the p-value for its coefficient is 0.001, we conclude that years of experience is a significant predictor of salary.

---

## Confidence Intervals for Coefficients

While p-values provide a binary significance decision, **confidence intervals** (CIs) give a range of plausible values for the coefficient. A 95% CI is computed as:

$$
CI = \hat{\beta}_i \pm \left( t_{\alpha/2} \times SE(\hat{\beta}_i) \right)
$$

where $t_{\alpha/2}$ is the critical value from the t-distribution at the desired confidence level.

**Interpretation:**

- If the CI **includes 0**, the coefficient is not significantly different from zero — the predictor may not have a significant impact.
- If the CI **excludes 0**, the predictor is significant, providing stronger evidence that it contributes meaningfully to the model.

!!! example "Advertising Budget"
    For a predictor "advertising budget" in a regression model predicting sales, a 95% CI of $(0.03, 0.12)$ does not contain 0, so we conclude that advertising budget significantly influences sales.

---

## Interpreting Significance

The significance of a predictor is determined by the t-test outcome and its associated p-value.

**Statistically Significant Coefficient** ($p < 0.05$):

- The predictor explains some variability in the dependent variable and improves the model.
- Changes in this variable are associated with changes in the outcome.
- For example, if "education level" is significant in predicting job performance, education level is an important determinant.

**Statistically Insignificant Coefficient** ($p \geq 0.05$):

- There is not enough evidence to claim this predictor impacts the dependent variable.
- The variable may be removed from the model if it does not improve overall fit.

---

## The Role of Multicollinearity in t-tests

**Multicollinearity** occurs when two or more predictors are highly correlated, making it difficult to distinguish their individual effects. This can:

- Inflate standard errors of coefficients, reducing t-statistics.
- Inflate p-values, leading to incorrect conclusions about significance.
- Widen confidence intervals, making it harder to conclude significance.

**Addressing Multicollinearity:**

- **Variance Inflation Factor (VIF):** A VIF greater than 5 or 10 suggests multicollinearity. Corrective actions include removing or combining correlated predictors.
- **Principal Component Analysis (PCA):** Transforms predictors into uncorrelated components.
- **Ridge Regression:** Regularizes coefficients to reduce the impact of multicollinearity.

---

## Summary

Hypothesis tests for regression coefficients (t-tests) are critical for determining the significance of individual predictors. By evaluating p-values and confidence intervals, we determine which predictors significantly contribute to explaining variability in the dependent variable. It is also important to check for multicollinearity, which can obscure the true significance of predictors.

## Exercises

**Exercise 1.**
A researcher studying education in Italy and France wanted to compare how many years, on average, men in each country spent in school. The researcher obtained a random sample of men from each country:

| | Italy | France |
|:---:|:---:|:---:|
| Mean | 10.7 | 10.4 |
| Standard Deviation | 2.3 | 2.5 |
| Number of Samples | 46 | 58 |

Test whether there is a significant difference between the two countries' mean school years with significance level $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    Two Sample $t$-Test (pooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 10.7, 10.4
        s_1, s_2 = 2.3, 2.5
        n_1, n_2 = 46, 58

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

**Exercise 2.**
An economist studying income in Norway and the United States wanted to compare the average annual income between the two countries. The economist obtained incomes for a random sample of people from each country denominated in thousands of US dollars:

| | Norway | US |
|:---:|:---:|:---:|
| Mean | 64.3 | 53.4 |
| Standard Deviation | 18.2 | 23.9 |
| Number of Samples | 65 | 75 |

Test whether there is a significant difference between the two countries' mean annual income with significance level $\alpha = 0.05$.

??? success "Solution to Exercise 2"

    Two Sample $t$-Test (Welch's, unpooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 64.3, 53.4
        s_1, s_2 = 18.2, 23.9
        n_1, n_2 = 65, 75

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / n_1 + (s_2**2 / n_2)**2 / n_2
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

**Exercise 3.**
A sociologist studying marriages in the United States and Canada wanted to compare how old, on average, women in each country were when they first got married. The sociologist obtained a random sample of married women from each country:

| | US | Canada |
|:---:|:---:|:---:|
| Mean | 25.5 | 26.3 |
| Standard Deviation | 3.8 | 3.2 |
| Number of Samples | 108 | 102 |

Test whether there is a significant difference between the two countries' mean age at first marriage with significance level $\alpha = 0.05$.

??? success "Solution to Exercise 3"

    Two Sample $t$-Test (pooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 25.5, 26.3
        s_1, s_2 = 3.8, 3.2
        n_1, n_2 = 108, 102

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

**Exercise 4.**
Julie was testing how far two new electric cars — models A and B — could drive on a full charge. She obtained a sample of 5 new cars of each model, charged them fully, and drove them as far as she could along a controlled route:

| | Model A | Model B |
|:---:|:---:|:---:|
| Mean | 168 km | 172 km |
| Standard Deviation | 5.4 km | 7.5 km |
| Number of Samples | 5 | 5 |

Test whether there is a significant difference between the two models' mean distance with significance level $\alpha = 0.05$.

??? success "Solution to Exercise 4"

    Two Sample $t$-Test (Welch's, unpooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 168, 172
        s_1, s_2 = 5.4, 7.5
        n_1, n_2 = 5, 5

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / n_1 + (s_2**2 / n_2)**2 / n_2
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```
