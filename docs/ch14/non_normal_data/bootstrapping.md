# Bootstrapping as an Alternative

When data is non-normal, or when assumptions of normality cannot be met, **bootstrapping** provides a powerful alternative. Bootstrapping is a resampling technique that generates multiple samples from the observed data by sampling with replacement. This method allows for the estimation of sampling distributions, confidence intervals, and hypothesis testing without the need for parametric assumptions.

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Original data sample
data = np.random.exponential(scale=2, size=100)

# Bootstrapping: Resample the data 1000 times and compute the mean of each sample
bootstrap_means = [np.mean(resample(data)) for _ in range(1000)]

# Plot the distribution of bootstrap means
plt.hist(bootstrap_means, bins=30)
plt.title('Bootstrap Sampling Distribution of the Mean')
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.show()

# Confidence interval for the mean
conf_interval = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"95% Confidence Interval: {conf_interval}")
```

Bootstrapping is a powerful tool when the sample size is small or the data deviates significantly from normality, as it allows for the estimation of uncertainty without relying on normal distribution assumptions.
