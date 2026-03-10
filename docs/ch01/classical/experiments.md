# Controlled Experiments

A controlled experiment is the gold standard for establishing causal relationships, using random assignment and manipulation of variables to isolate treatment effects from confounders.

## Definition

A **controlled experiment** is a research design in which the investigator manipulates one or more independent variables (treatments), randomly assigns subjects to treatment and control groups, and measures the effect on a dependent variable. The **control group** receives a placebo or standard treatment; the **treatment group** receives the intervention. Random assignment ensures confounders are distributed evenly across groups.

## Explanation

The key advantage of controlled experiments over observational studies is the ability to establish **causation**, not merely association. Randomization ensures that any systematic differences between groups (age, health, socioeconomic status) balance out on average, so observed differences in outcomes can be attributed to the treatment.

The **placebo effect** -- genuine improvement from believing one is treated -- is controlled by giving the control group an inert treatment identical in appearance. Without a placebo control, treatment effects and psychological effects are confounded.

| | Controlled Experiment | Observational Study |
|---|---|---|
| Group assignment | Investigator (random) | Subject (self-selected) |
| Confounding | Minimized by randomization | Many potential confounders |
| Causal claims | Yes | Association only |

Limitations: ethical constraints (cannot assign harmful treatments), artificiality of lab settings, high cost and time, and limited generalizability from tightly controlled conditions.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n_per_group = 100
true_effect = 5.0

# Simulate a randomized experiment
control = np.random.normal(50, 10, n_per_group)
treatment = np.random.normal(50 + true_effect, 10, n_per_group)

t_stat, p_value = stats.ttest_ind(treatment, control)
diff = treatment.mean() - control.mean()
print(f"Control mean:   {control.mean():.2f}")
print(f"Treatment mean: {treatment.mean():.2f}")
print(f"Difference:     {diff:.2f} (true effect = {true_effect})")
print(f"t-statistic:    {t_stat:.3f}")
print(f"p-value:        {p_value:.4f}")
```
