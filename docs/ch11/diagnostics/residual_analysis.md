# Residual Analysis


## Overview

Residual analysis is a critical diagnostic tool in ANOVA. Residuals are the differences between the observed data points and the predicted values from the model:

$$
e_{ij} = Y_{ij} - \hat{Y}_{ij} = Y_{ij} - \bar{Y}_{i\cdot}
$$

where $Y_{ij}$ is the observed value for observation $j$ in group $i$ and $\bar{Y}_{i\cdot}$ is the mean of group $i$. In a correctly specified model, residuals should be randomly distributed around zero with constant variance and no systematic patterns.

## Residual vs. Fitted Value Plots

The most informative diagnostic plot displays residuals against fitted values. In one-way ANOVA, the fitted values are simply the group means, so the plot shows vertical strips of residuals at each group mean.

```python
import matplotlib.pyplot as plt

plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()
```

### Identifying Patterns

The following patterns in residual plots signal specific assumption violations:

**Funnel Shape (Heteroscedasticity):**
A widening or narrowing pattern indicates that the variance of residuals differs across levels of the independent variable. This violates the homoscedasticity assumption and can distort the F-test.

**Curvature (Nonlinearity):**
A curved pattern suggests that the relationship between the independent and dependent variables is not linear. This may indicate the need for polynomial terms, a nonlinear model, or data transformation.

**Clustering (Non-Independence):**
Clusters of residuals can indicate that observations within clusters are correlated, violating the independence assumption. This often arises in hierarchical or nested data structures.

**Outliers:**
Individual points far from the zero line may be outliers that exert disproportionate influence on the analysis.

## Standardized Residuals

Standardized residuals divide each residual by an estimate of its standard deviation, placing all residuals on a common scale:

$$
r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
$$

where $\hat{\sigma}$ is the estimated standard deviation and $h_{ii}$ is the leverage of observation $i$. Under the model assumptions, standardized residuals should approximately follow a standard normal distribution.

```python
import numpy as np

influence = model.get_influence()
standardized_resid = influence.resid_studentized_internal

plt.scatter(model.fittedvalues, standardized_resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
plt.xlabel("Fitted Values")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs. Fitted Values")
plt.show()
```

Observations with $|r_i| > 2$ deserve closer inspection, and those with $|r_i| > 3$ are strong candidates for outliers.

## Scale-Location Plot

The scale-location plot displays $\sqrt{|r_i|}$ against fitted values and is useful for assessing homoscedasticity. A horizontal trend line with evenly spread points indicates constant variance.

```python
plt.scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
plt.xlabel("Fitted Values")
plt.ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
plt.title("Scale-Location Plot")
plt.show()
```

## Summary

Residual analysis provides a comprehensive visual diagnostic framework for ANOVA. By examining residual plots, researchers can detect violations of normality, homoscedasticity, independence, and linearity, and take appropriate corrective actions before interpreting the ANOVA results.
## Exercises

**Exercise 1.**
In a one-way ANOVA with four groups, the residual-vs-fitted plot shows a clear funnel shape (residuals fan out as fitted values increase). Identify which ANOVA assumption is violated and describe two approaches to address the problem.

??? success "Solution to Exercise 1"
    The funnel shape indicates **heteroscedasticity** -- the variance of residuals increases with the fitted values (group means). Two approaches:

    1. **Welch's ANOVA:** Does not assume equal variances and adjusts the degrees of freedom to account for the different group variances.

    2. **Variance-stabilizing transformation:** Apply a log or square root transformation to the response variable. If the variance is proportional to the mean (common in count or positive continuous data), $\log(Y)$ often stabilizes it.

---

**Exercise 2.**
An ANOVA is performed on student test scores across five schools. The histogram of residuals shows two distinct peaks rather than a single bell curve. What might this indicate, and how should the researcher proceed?

??? success "Solution to Exercise 2"
    A **bimodal** histogram of residuals suggests that there is a **missing grouping variable** or an important covariate that has not been included in the model. For example, within each school there may be two distinct subpopulations (e.g., students in different programs or grade levels) whose scores differ systematically.

    The researcher should investigate potential additional factors and consider fitting a two-way ANOVA or ANCOVA that includes the additional variable. If a natural grouping is identified, including it will reduce the residual variance and improve the model.

---

**Exercise 3.**
Explain why standardized residuals are preferred over raw residuals when assessing normality in ANOVA. How is a standardized residual computed?

??? success "Solution to Exercise 3"
    Raw residuals $e_i = Y_i - \hat{Y}_i$ have different variances depending on the leverage of each observation: $\text{Var}(e_i) = \sigma^2(1 - h_{ii})$, where $h_{ii}$ is the leverage. This means direct comparison of raw residual magnitudes across observations is misleading.

    **Standardized (internally studentized) residuals** divide each raw residual by its estimated standard deviation:

    $$
    r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
    $$

    These should follow approximately a standard normal distribution under the model assumptions, making them directly comparable across observations and easier to assess using Q-Q plots and the $\pm 2$ rule of thumb.
