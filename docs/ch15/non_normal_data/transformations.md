# Transformations to Achieve Normality

When data deviates significantly from normality, certain statistical methods that rely on normality assumptions may no longer be appropriate. One common approach is to apply a transformation to make the data more normal.

## Common Transformations

Popular transformations include:

**Log Transformation**: Suitable for positively skewed data.

$$
X' = \log(X)
$$

**Square Root Transformation**: Also used for right-skewed data, particularly when there are small values.

$$
X' = \sqrt{X}
$$

**Box-Cox Transformation**: A more flexible transformation that finds an optimal power parameter $\lambda$ to transform the data.

$$
X' = \frac{X^\lambda - 1}{\lambda}, \quad \lambda \neq 0
$$

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# Generate positively skewed data
skewed_data = np.random.exponential(scale=2, size=1000)

# Log transformation
log_transformed_data = np.log(skewed_data + 1)  # Adding 1 to avoid log(0)

# Box-Cox transformation
boxcox_transformed_data, best_lambda = boxcox(skewed_data + 1)

# Plot the original and transformed data
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].hist(skewed_data, bins=30)
axs[0].set_title('Original Data')

axs[1].hist(log_transformed_data, bins=30)
axs[1].set_title('Log Transformed Data')

axs[2].hist(boxcox_transformed_data, bins=30)
axs[2].set_title(f'Box-Cox Transformed Data (λ={best_lambda:.2f})')

plt.show()
```

Both log and Box-Cox transformations are applied to skewed data. These transformations often make data more symmetric and closer to normality, making it suitable for parametric tests.
