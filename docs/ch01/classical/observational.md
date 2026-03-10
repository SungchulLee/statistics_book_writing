# Observational Studies

An observational study records data without the investigator intervening or assigning treatments, making it ideal for studying associations when experiments are impractical or unethical.

## Definition

An **observational study** is a research design in which the investigator observes subjects and measures variables without manipulating any conditions or randomly assigning groups. The four main types are:

- **Cross-sectional**: Snapshot of a population at a single point in time.
- **Cohort (longitudinal)**: A group followed over time (prospective or retrospective).
- **Case-control**: Subjects with a condition (cases) compared to those without (controls).
- **Ecological**: Data analyzed at the population/group level rather than individual level.

## Explanation

The fundamental limitation is that observational studies cannot establish causation because subjects self-select into groups. Unmeasured confounders may drive apparent associations. For example, the Framingham Heart Study identified that high blood pressure, cholesterol, and smoking are associated with cardiovascular disease, but these associations alone do not prove causation without further evidence.

The advantages are real-world relevance (results generalize to natural settings) and ethical feasibility (e.g., studying smoking effects without forcing people to smoke).

Confounding is mitigated, though not eliminated, by stratification, matching, multivariable regression, and propensity score methods.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 200

# Simulate an observational study with a confounder (age)
age = np.random.uniform(20, 70, n)
# Older people exercise less AND have higher blood pressure
exercise = 10 - 0.1 * age + np.random.normal(0, 1, n)
bp = 80 + 0.5 * age - 0.3 * exercise + np.random.normal(0, 5, n)

# Naive correlation: exercise vs blood pressure
r_naive, p_naive = stats.pearsonr(exercise, bp)
print(f"Naive correlation (exercise, BP): r={r_naive:.3f}, p={p_naive:.4f}")
print("Appears exercise RAISES BP -- but age confounds!")

# Partial out age via residuals
from numpy.polynomial.polynomial import polyfit
age_on_ex = np.polyval(np.polyfit(age, exercise, 1), age)
age_on_bp = np.polyval(np.polyfit(age, bp, 1), age)
r_partial, _ = stats.pearsonr(exercise - age_on_ex, bp - age_on_bp)
print(f"Partial correlation (controlling age): r={r_partial:.3f}")
print("After removing age effect, exercise lowers BP as expected")
```
