# Confounding and Association vs Causation

Two variables may move together without one causing the other. Understanding confounding is essential for interpreting any statistical association correctly.

## Definition

A **confounding variable** is a third variable associated with both the independent and dependent variables but not on the causal pathway between them. When present but unaccounted for, confounders create spurious associations. The principle **"correlation does not imply causation"** follows directly: observational associations may be entirely driven by confounders.

## Explanation

The confounding structure is: Confounder influences both Variable A and Variable B, creating a statistical association between A and B even though neither causes the other.

Classic examples: ice cream sales and drowning both increase in summer (confounder: temperature); coffee drinkers appear to have higher lung cancer risk (confounder: smoking); children with larger shoe sizes read better (confounder: age).

To move from association to causation requires either a randomized controlled experiment (which distributes confounders evenly) or advanced causal-inference techniques (instrumental variables, difference-in-differences, regression discontinuity).

Mitigation in observational studies: stratification (analyze within confounder subgroups), matching (pair treated/untreated by confounder values), multivariable regression (include confounders as covariates), and propensity score methods.

**Simpson's paradox** illustrates confounding dramatically: an association within every subgroup can reverse when subgroups are combined, because the confounder determines subgroup membership.

## Examples

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
dept = np.random.choice(["A", "B"], size=n, p=[0.5, 0.5])

# Dept A: harder grading, more studious applicants
study_hours = np.where(dept == "A",
                       np.random.normal(8, 1, n),
                       np.random.normal(4, 1, n))
grade = np.where(dept == "A",
                 50 + 3 * study_hours + np.random.normal(0, 5, n),
                 70 + 3 * study_hours + np.random.normal(0, 5, n))

df = pd.DataFrame({"dept": dept, "study_hours": study_hours, "grade": grade})

overall_corr = df["study_hours"].corr(df["grade"])
print(f"Overall correlation: {overall_corr:.3f}")
for d in ["A", "B"]:
    sub = df[df["dept"] == d]
    print(f"  Dept {d}: corr = {sub['study_hours'].corr(sub['grade']):.3f}, "
          f"mean grade = {sub['grade'].mean():.1f}")
```
