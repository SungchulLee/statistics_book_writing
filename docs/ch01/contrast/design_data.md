# Design Your Data Collection

The classical approach to data analysis begins with a research question and designs the collection process to answer it. Data does not yet exist when the study is planned -- the researcher controls how, from whom, and under what conditions it is gathered.

## Definition

The classical philosophy is **"design first, collect second, analyze third."** The three study types are:

- **Observational study**: Observe without intervention; identifies associations.
- **Controlled experiment**: Manipulate variables with random assignment; establishes causation.
- **Sample survey**: Select a representative sample via probabilistic methods; enables population inference.

## Explanation

By controlling the data-generation process, the researcher gains three advantages that the modern approach lacks: the ability to make causal claims (via randomization), built-in uncertainty quantification (standard errors and confidence intervals have clear probabilistic meaning), and explicit bias control (random sampling and blinding).

The limitations are cost, time, ethical constraints (cannot randomly assign harmful treatments), and restricted scope (works best for structured, well-defined problems).

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Classical approach: designed experiment
# Randomly assign 100 patients to drug vs placebo
n = 100
effect = 5.0
control = np.random.normal(120, 15, n)  # systolic BP
treatment = np.random.normal(120 - effect, 15, n)

t_stat, p_val = stats.ttest_ind(control, treatment)
print(f"Control BP mean:   {control.mean():.1f}")
print(f"Treatment BP mean: {treatment.mean():.1f}")
print(f"Estimated effect:  {control.mean() - treatment.mean():.1f} (true: {effect})")
print(f"p-value: {p_val:.4f}")
```
