# Checking Independence of Observations

## Why Independence Matters

The independence assumption states that each observation should be unrelated to every other observation, both within and across groups. This is arguably the most critical assumption in ANOVA because violations of independence cannot be corrected by transformations or alternative test statistics—they require fundamentally different modeling approaches (e.g., mixed-effects models, repeated measures ANOVA).

When observations are correlated, the effective sample size is smaller than the nominal sample size, which means:

- Standard errors are underestimated.
- The F-statistic is inflated.
- Type I error rates increase dramatically.

## How to Check

### Study Design Review

The most effective way to ensure independence is through proper experimental design:

- **Random sampling** from the population ensures that one observation does not influence another.
- **Random assignment** to treatment groups prevents systematic dependencies.
- **No repeated measures** on the same subject within a one-way ANOVA framework. If the same subjects are measured under multiple conditions, a repeated-measures ANOVA or mixed-effects model is required.

Common scenarios that violate independence:

- Students nested within classrooms (clustered data).
- Repeated measurements on the same patient over time.
- Spatial or temporal proximity of observations (e.g., adjacent plots in an agricultural experiment).

### Durbin-Watson Test

The Durbin-Watson test is primarily used in regression to detect autocorrelation in residuals, but it can be applied to ANOVA residuals when observations have a natural ordering (e.g., time series data).

$$
d = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n}e_i^2}
$$

where $e_i$ are the residuals ordered by time or sequence.

- $d \approx 2$: No autocorrelation.
- $d < 2$: Positive autocorrelation (adjacent residuals tend to be similar).
- $d > 2$: Negative autocorrelation (adjacent residuals tend to alternate in sign).

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(model.resid)
print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
```

### Residual Plots Against Order

If data has a natural ordering (e.g., time of collection), plotting residuals against this order can reveal patterns suggesting dependence.

```python
import matplotlib.pyplot as plt

plt.scatter(range(len(model.resid)), model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Observation Order")
plt.ylabel("Residuals")
plt.title("Residuals vs. Observation Order")
plt.show()
```

Look for:

- **Trends:** A systematic increase or decrease suggests a time effect.
- **Cycles:** Periodic patterns indicate autocorrelation.
- **Clusters:** Groups of similar residuals suggest block effects.

## What to Do If Independence Is Violated

- **Mixed-effects models:** Account for hierarchical or clustered data structures by modeling both fixed effects (treatment) and random effects (cluster).
- **Repeated-measures ANOVA:** If the same subjects appear in multiple groups, use a design that accounts for within-subject correlation.
- **Generalized estimating equations (GEE):** Provide population-averaged estimates while accounting for correlation structures.
- **Time-series methods:** If data are collected over time with autocorrelation, specialized time-series ANOVA approaches may be needed.
