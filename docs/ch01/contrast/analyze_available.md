# Analyze the Data You Have

The modern approach starts with data that already exists -- transaction logs, sensor readings, financial data -- and applies algorithms to discover patterns, make predictions, and extract insights.

## Definition

The modern philosophy is **"data first, algorithm second, insight third."** It leverages abundant digital data and computational power, applying supervised, unsupervised, and reinforcement learning to extract knowledge without designing the data collection process.

## Explanation

Strengths: scalability to billions of observations, flexibility to capture nonlinear relationships, speed (no data collection delay), ability to handle unstructured data (images, text, audio), and discovery of unanticipated patterns.

Limitations: difficulty establishing causation without designed experiments, no control over data quality (biases, missing values, measurement error), low interpretability of complex models, overfitting risk with flexible methods, and ethical/privacy concerns with existing personal data.

The most effective practice combines both approaches: modern algorithms for prediction and classical principles for causal reasoning and uncertainty quantification.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 1000

# Modern approach: analyze existing observational data
# Transaction data with a lurking confounder
age = np.random.uniform(20, 65, n)
income = 20_000 + 800 * age + np.random.normal(0, 5000, n)
spending = 500 + 0.05 * income + 10 * age + np.random.normal(0, 2000, n)

# Naive correlation
r, p = stats.pearsonr(income, spending)
print(f"Income-Spending correlation: r={r:.3f}, p={p:.2e}")
print("Strong association, but is it causal?")
print("Age confounds: older people earn more AND spend more")
```
