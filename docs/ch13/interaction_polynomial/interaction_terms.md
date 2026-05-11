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

## Example 1: Study Hours, Sleep, and Performance

Consider a study on the effect of study hours and sleep on student exam performance. Without an interaction term, the model assumes that additional study hours have the same benefit regardless of how much a student sleeps.

An interaction term between study hours and sleep might reveal that the benefit of additional study hours **diminishes** when sleep is inadequate. Formally:

$$
\text{Score} = \beta_0 + \beta_1 \cdot \text{StudyHours} + \beta_2 \cdot \text{Sleep} + \beta_3 \cdot (\text{StudyHours} \times \text{Sleep}) + \epsilon
$$

If $\beta_3 > 0$, more sleep amplifies the benefit of studying. If $\beta_3 < 0$, studying more provides diminishing returns for students who sleep less.

---

## Example 2: Marketing Effectiveness (TV and Radio)

A classic example from advertising analytics examines how TV and Radio advertising spending affect Sales. The main effects model assumes each medium has an independent effect:

$$
\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \epsilon
$$

However, there may be a **synergistic effect**: advertising on both TV and Radio together might be more effective than the sum of their individual effects. Adding an interaction term captures this:

$$
\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot (\text{TV} \times \text{Radio}) + \epsilon
$$

**Interpretation**:

- If $\beta_3 > 0$: Combining TV and Radio advertising creates a synergistic boost in sales beyond what each medium provides independently.
- If $\beta_3 < 0$: There is a diminishing effect—spending heavily on both media simultaneously may be less efficient.

This model can be fit using statsmodels' formula syntax for convenience:

```python
import statsmodels.api as sm

# Using formula syntax (R-like)
# The * operator includes main effects and the interaction
model = sm.ols('Sales ~ TV + Radio + TV*Radio', data=advertising).fit()
print(model.summary())
```

---

## Example 3: Income and Student Status Interaction

Consider a model examining how balance (credit card debt) depends on income and student status. A **qualitative variable** (student: yes/no) can interact with a continuous variable (income):

$$
\text{Balance} = \beta_0 + \beta_1 \cdot \text{Income} + \beta_2 \cdot \text{Student} + \beta_3 \cdot (\text{Income} \times \text{Student}) + \epsilon
$$

Where Student is coded as 1 (yes) or 0 (no).

**Interpretation**:

- $\beta_1$: The effect of Income on Balance for **non-students** is $\beta_1$.
- $\beta_1 + \beta_3$: The effect of Income on Balance for **students** is $\beta_1 + \beta_3$.
- If $\beta_3 \neq 0$: The relationship between Income and Balance **differs by student status**.

This type of interaction reveals whether different groups respond differently to the same predictor—a crucial insight for segmentation and targeted analysis.

```python
# Example with categorical variable
# statsmodels automatically encodes categorical variables
model = sm.ols('Balance ~ Income + C(Student) + Income:C(Student)',
                data=credit).fit()
print(model.summary())
```

Visual representation of such interactions often shows two separate regression lines (one for each group) with different slopes, demonstrating the differential effect of Income on Balance.

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
## Exercises

**Exercise 1.**
In the model $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$, interpret $\beta_3$ in terms of how the effect of $X_1$ on $Y$ changes with $X_2$.

??? success "Solution to Exercise 1"
    The partial effect of $X_1$ on $Y$ is:

    $$
    \frac{\partial E[Y]}{\partial X_1} = \beta_1 + \beta_3 X_2
    $$

    So $\beta_3$ represents the **change in the slope of $X_1$** for each one-unit increase in $X_2$. If $\beta_3 > 0$, the effect of $X_1$ on $Y$ becomes stronger as $X_2$ increases. If $\beta_3 < 0$, the effect weakens. If $\beta_3 = 0$, the effect of $X_1$ is the same regardless of $X_2$ (no interaction).

---

**Exercise 2.**
A model predicting salary includes experience, education (binary: 0 = no degree, 1 = degree), and their interaction. The estimated equation is $\hat{Y} = 30000 + 2000 X_1 + 10000 X_2 + 1500 X_1 X_2$. Write separate regression equations for individuals with and without a degree and interpret the difference.

??? success "Solution to Exercise 2"
    **Without degree** ($X_2 = 0$): $\hat{Y} = 30000 + 2000 X_1$. Each additional year of experience increases salary by \$2,000.

    **With degree** ($X_2 = 1$): $\hat{Y} = (30000 + 10000) + (2000 + 1500) X_1 = 40000 + 3500 X_1$. Each additional year of experience increases salary by \$3,500.

    The interaction term (\$1,500) means that having a degree amplifies the return to experience by \$1,500 per year. Degree holders start higher (\$40,000 vs. \$30,000) and gain more from each year of experience (\$3,500 vs. \$2,000).

---

**Exercise 3.**
Explain why it is generally inappropriate to remove a main effect ($X_1$ or $X_2$) from a model while retaining their interaction term ($X_1 X_2$). What statistical principle does this violate?

??? success "Solution to Exercise 3"
    This violates the **hierarchy principle** (or marginality principle), which states that if an interaction term is included, all lower-order terms that compose it must also be present.

    Removing a main effect while keeping the interaction changes the interpretation of the remaining terms. For example, dropping $X_1$ from $Y = \beta_0 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$ forces the model to assume that $X_1$ has zero effect when $X_2 = 0$, which is a strong and usually unjustified constraint. It also makes the interaction coefficient dependent on the coding of $X_2$, destroying interpretability.
