# Randomization and Blinding

Randomization and blinding are the two pillars of experimental design that guard against confounding and human expectation bias, making the double-blind RCT the gold standard for causal evidence.

## Definition

**Randomization** assigns subjects to treatment or control groups by chance, ensuring that both known and unknown confounders are distributed evenly. **Blinding** conceals group assignments to prevent bias: single-blind (subjects unaware), double-blind (subjects and administrators unaware), triple-blind (analysts also unaware).

## Explanation

Randomization eliminates selection bias in group assignment and provides the probabilistic foundation for valid p-values and confidence intervals. Without it, observed treatment effects may reflect pre-existing group differences rather than the treatment itself.

Blinding prevents the placebo effect (subjects improve from believing they are treated) and researcher assessment bias (investigators unconsciously favoring the treatment group). The double-blind RCT combines both protections.

Historical example: The 1954 Salk polio vaccine trial randomly assigned 400,000 children to vaccine or placebo in a double-blind design. The vaccine group had 28 cases per 100,000 versus 71 in controls, establishing efficacy. The non-randomized NFIP design, which used grade-level controls, found a similar effect but with weaker internal validity due to potential confounders.

The FDA requires Phase III double-blind RCTs before drug approval, embedding these principles into the regulatory process.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 200

# Simulate a double-blind RCT
treatment_effect = 3.0
control = np.random.normal(50, 8, n)
treatment = np.random.normal(50 + treatment_effect, 8, n)

t_stat, p_val = stats.ttest_ind(treatment, control)
print(f"Control mean:   {control.mean():.2f}")
print(f"Treatment mean: {treatment.mean():.2f}")
print(f"Estimated effect: {treatment.mean() - control.mean():.2f} (true: {treatment_effect})")
print(f"p-value: {p_val:.4f}")

# Without randomization: confounder biases assignment
severity = np.random.uniform(0, 10, 2 * n)
# Sicker patients more likely to get treatment (confounded)
prob_treat = 1 / (1 + np.exp(-(severity - 5)))
assigned_treat = np.random.binomial(1, prob_treat).astype(bool)
outcome = 50 - 2 * severity + treatment_effect * assigned_treat + np.random.normal(0, 5, 2 * n)
t_conf, p_conf = stats.ttest_ind(outcome[assigned_treat], outcome[~assigned_treat])
print(f"\nConfounded design: estimated effect = "
      f"{outcome[assigned_treat].mean() - outcome[~assigned_treat].mean():.2f}")
print(f"p-value: {p_conf:.4f} (misleading due to confounding)")
```
