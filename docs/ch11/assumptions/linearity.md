# Checking Linearity


## Why Linearity Is Relevant in ANOVA

Although linearity is not always explicitly stated as a requirement for ANOVA, it becomes relevant when ANOVA is viewed through the lens of the general linear model. In the one-way ANOVA framework, the model is:

$$
Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}
$$

where $\mu$ is the overall mean, $\alpha_i$ is the effect of group $i$, and $\varepsilon_{ij}$ is the random error. This model is inherently linear in the parameters. Linearity becomes more explicitly important in two-way ANOVA, ANCOVA (analysis of covariance), and when ANOVA is extended to include continuous covariates.

The linearity assumption states that the relationship between any continuous independent variable and the dependent variable is linear within each group. Nonlinear relationships can lead to systematic patterns in residuals and model misspecification.

## How to Check

### Scatter Plots

When continuous covariates are present (e.g., in ANCOVA), plot the dependent variable against each covariate, colored by group.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=data, x='covariate', y='response', hue='group', alpha=0.6)
plt.title("Response vs. Covariate by Group")
plt.show()
```

Look for:

- **Linear trends** within each group confirm the linearity assumption.
- **Curved patterns** suggest a nonlinear relationship that may require polynomial terms or a different model.

### Residual Plots

A plot of residuals against the independent variable (or fitted values) should show no systematic patterns. Random scatter around zero confirms linearity.

```python
import matplotlib.pyplot as plt

plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()
```

- **Random scatter:** Linearity is satisfied.
- **Curvature:** A curved pattern suggests the need for polynomial terms or a nonlinear model.
- **Distinct clusters:** May indicate the need for additional grouping variables.

## What to Do If Linearity Is Violated

- **Polynomial terms:** Add quadratic or higher-order terms to the model.
- **Nonlinear transformations:** Transform the dependent variable or covariates (e.g., log, square root).
- **Generalized additive models (GAMs):** Use smooth functions of covariates instead of assuming a linear relationship.
- **Nonparametric methods:** If the nonlinearity is severe, consider non-parametric alternatives that make no assumptions about functional form.
## Exercises

**Exercise 1.**
In a one-way ANOVA with purely categorical factors and no continuous covariates, explain why the linearity assumption is automatically satisfied.

??? success "Solution to Exercise 1"
    The one-way ANOVA model $Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}$ uses indicator variables for group membership. These indicator variables enter the model linearly by definition. Since there is no continuous predictor, there is no functional relationship that could be nonlinear. Linearity becomes a concern only when continuous covariates are included, as in ANCOVA or regression-based formulations.

---

**Exercise 2.**
An ANCOVA model includes a continuous covariate (pre-test score) along with group indicators. The residual-vs-fitted plot shows a clear U-shaped pattern. Explain what this indicates and propose two remedies.

??? success "Solution to Exercise 2"
    A U-shaped pattern in the residual-vs-fitted plot indicates that the relationship between the covariate and the response is **nonlinear** -- the linear model systematically underpredicts at the extremes and overpredicts in the middle.

    Two remedies:

    1. **Add a polynomial term.** Include $X^2$ (and possibly $X^3$) for the covariate in the model to capture the curvature within the linear regression framework.

    2. **Apply a nonlinear transformation.** Transform the covariate (e.g., $\log(X)$ or $\sqrt{X}$) so that its relationship with the response becomes approximately linear.

---

**Exercise 3.**
Explain the difference between detecting nonlinearity via a scatterplot of $Y$ versus $X$ and detecting it via a residual plot. In which scenario might one method succeed where the other fails?

??? success "Solution to Exercise 3"
    A **scatterplot** of $Y$ versus $X$ shows the raw relationship between the variables, which may be obscured by the influence of other predictors in a multiple regression or ANOVA setting. A **residual plot** shows the relationship after removing the effects of all other predictors, isolating the contribution of the variable in question.

    In multiple regression or ANCOVA, the scatterplot of $Y$ versus a single covariate can appear linear even when the partial relationship (after adjusting for other predictors) is nonlinear. The residual plot would detect this. Conversely, in simple regression the two methods are largely equivalent.
