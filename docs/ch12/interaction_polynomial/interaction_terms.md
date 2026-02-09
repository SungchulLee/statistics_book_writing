# Interaction Terms

## Why Use Interaction Terms?

In many real-world scenarios, the effect of one variable on the outcome is not constant but changes depending on another variable. **Interaction terms** model this combined effect of two or more independent variables on the dependent variable.

Examples where interactions arise naturally:

- In a study on the effect of exercise and diet on weight loss, the impact of exercise might be greater for individuals with a specific type of diet.
- In sales forecasting, the effectiveness of a marketing campaign might vary depending on the economic conditions or time of year.
- In finance, the effect of interest rate changes on asset prices may depend on the current volatility regime.

When interactions are present but not modeled, the resulting regression may be misleading because it assumes the effect of each predictor is the same regardless of the values of other predictors.

---

## Mathematical Formulation

An interaction between two variables $X_1$ and $X_2$ is added to the regression model as a product term:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \times X_2) + \epsilon
$$

where:

- $Y$ is the dependent variable,
- $\beta_1$ and $\beta_2$ are the **main effects** of $X_1$ and $X_2$,
- $\beta_3$ is the **interaction effect** between $X_1$ and $X_2$,
- $\epsilon$ is the error term.

The term $X_1 \times X_2$ captures the interaction: the partial effect of $X_1$ on $Y$ is no longer a constant $\beta_1$ but instead becomes $\beta_1 + \beta_3 X_2$, which depends on the level of $X_2$.

!!! note "Including Main Effects"
    When an interaction term is included, both corresponding main effects ($X_1$ and $X_2$) should generally be retained in the model. Omitting a main effect while including the interaction can lead to biased and uninterpretable coefficients.

---

## Interpreting Interaction Terms

The sign and magnitude of $\beta_3$ determine the nature of the interaction:

- **Positive interaction coefficient ($\beta_3 > 0$)**: When both predictors increase together, the combined effect on $Y$ is **greater** than the sum of their individual effects. The predictors reinforce each other.

- **Negative interaction coefficient ($\beta_3 < 0$)**: The combined effect of both predictors is **less** than the sum of their individual effects. One predictor dampens the effect of the other.

- **Zero interaction ($\beta_3 = 0$)**: The effect of $X_1$ on $Y$ does not depend on $X_2$. The model reduces to an additive model with no interaction.

A statistically significant interaction term indicates that the relationship between one variable and the outcome changes depending on the other variable.

---

## Example: Study Hours, Sleep, and Performance

Consider a study on the effect of study hours and sleep on student exam performance. Without an interaction term, the model assumes that additional study hours have the same benefit regardless of how much a student sleeps.

An interaction term between study hours and sleep might reveal that the benefit of additional study hours **diminishes** when sleep is inadequate. Formally:

$$
\text{Score} = \beta_0 + \beta_1 \cdot \text{StudyHours} + \beta_2 \cdot \text{Sleep} + \beta_3 \cdot (\text{StudyHours} \times \text{Sleep}) + \epsilon
$$

If $\beta_3 > 0$, more sleep amplifies the benefit of studying. If $\beta_3 < 0$, studying more provides diminishing returns for students who sleep less.

---

## Higher-Order and Multi-Way Interactions

Interactions are not limited to pairs of variables. A **three-way interaction** among $X_1$, $X_2$, and $X_3$ takes the form:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_1 X_2 + \beta_5 X_1 X_3 + \beta_6 X_2 X_3 + \beta_7 X_1 X_2 X_3 + \epsilon
$$

In practice, three-way and higher-order interactions are difficult to interpret and are used sparingly. Most applied work focuses on two-way interactions.

---

## Summary

Interaction terms extend the multiple regression framework by allowing the effect of one predictor to depend on the level of another. They are essential for accurately modeling many real-world relationships and should be tested whenever theory or domain knowledge suggests that predictor effects are not purely additive.
