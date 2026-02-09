# Standard Error

## Overview

> **References:** [YouTube — Standard Error](https://www.youtube.com/watch?v=A82brFpdr9g) | [Blog — SD vs SE](https://statisticsbyjim.com/basics/difference-standard-deviation-vs-standard-error/)

The **standard error** (SE) quantifies how much a sample statistic varies from sample to sample. It is the standard deviation of the **sampling distribution** of that statistic.

## Standard Deviation vs Standard Error

### Standard Deviation (SD)

The SD measures how spread out **individual observations** are around the population mean:

$$
\text{SD} = \sqrt{\text{Var}(X)}
$$

### Standard Error (SE)

The SE measures how spread out a **sample statistic** is around the true parameter:

$$
\text{SE} = \sqrt{\text{Var}(\hat{\theta}(X_1, \dots, X_n))}
$$

### Key Distinction

- **Standard deviation** measures the spread of individual data points around the mean.
$$
\text{SD} = \sqrt{\text{Var}(X)}
$$
- **Standard error** measures the spread of sample statistics (e.g., means) around the population parameter.
$$
\text{SE} = \sqrt{\text{Var}(\hat{\theta}(X_1, \dots, X_n))}
$$

| | Standard Deviation | Standard Error |
|---|---|---|
| **Measures** | Spread of individual data | Spread of a sample statistic |
| **Depends on** | Population variability | Population variability **and** sample size |
| **Formula (for $\bar{X}$)** | $\sigma$ | $\sigma / \sqrt{n}$ |
| **Decreases with $n$?** | No | Yes |

## The Standardization Pattern

> **Reference:** [Khan Academy — Standard Error of the Mean](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/sampling-distribution-mean/v/standard-error-of-the-mean)

A unifying pattern in inferential statistics:

$$
\begin{array}{lllllll}
\displaystyle
\frac{\text{unbiased\_estimator} - \text{parameter}}{\text{standard\_error}}
&=&
\displaystyle
\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}
&\approx&
\displaystyle
\frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}}
&\approx&
z \;\text{ or }\; t_{n-1} \\[16pt]
\displaystyle
\frac{\text{unbiased\_estimator} - \text{parameter}}{\text{standard\_error}}
&=&
\displaystyle
\frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}}
&\approx&
\displaystyle
\frac{\hat{p} - p}{\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}}
&\approx&
z
\end{array}
$$

## Example: Running Out of Water

> **Reference:** [Khan Academy — Sampling Distribution Example Problem](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/sampling-distribution-mean/v/sampling-distribution-example-problem)

**Problem.** On average, a male drinks 2 liters of water when active outdoors, with a standard deviation of 0.7 liters. For a full-day nature trip of 50 men, we will bring 110 liters of water along. Determine the probability of running out of water during the trip.

**Solution.** Let $X_i$ be the water consumption of the $i$-th person. Assuming independence, by the CLT the sample mean $\bar{X}$ is approximately normally distributed with mean 2 and standard deviation $0.7/\sqrt{50} \approx 0.0990$.

$$
\begin{array}{lll}
\displaystyle
P\!\left(\bar{X} > \frac{110}{50}\right)
&=&
\displaystyle
P\!\left(\frac{\bar{X} - 2}{0.0990} > \frac{2.2 - 2}{0.0990}\right) \\[12pt]
&\approx&
\displaystyle
P(Z > 2.020) \\[8pt]
&\approx&
0.0217
\end{array}
$$

## Python: Standard Error of $\bar{X}$

### Standalone Version

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def main():
    X_bar = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        x_bar = x.mean()
        X_bar.append(x_bar)

    average = np.array(X_bar).mean()  # very good estimate of mu
    standard_error = np.array(X_bar).std()

    print(f'(Estimated) Mean of X_bar : {average:.4}')
    print(f'Standard Error   of X_bar : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of X_bar", fontsize=20)

    ax.hist(X_bar, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=5, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')

    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 5),
                xytext=(average + standard_error, 5),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 5.5),
                xytext=(average, 5.5),
                fontsize=15)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```

### Modular Version: `global_name_space.py`

```python
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
```

### Modular Version: `standard_error_of_x_bar.py`

```python
import matplotlib.pyplot as plt
import numpy as np

from global_name_space import ARGS

def main():
    X_bar = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        x_bar = x.mean()
        X_bar.append(x_bar)

    average = np.array(X_bar).mean()
    standard_error = np.array(X_bar).std()

    print(f'(Estimated) Mean of X_bar : {average:.4}')
    print(f'Standard Error   of X_bar : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of X_bar", fontsize=20)

    ax.hist(X_bar, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=5, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')

    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 5),
                xytext=(average + standard_error, 5),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 5.5),
                xytext=(average, 5.5),
                fontsize=15)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```

## Python: Standard Error of $S^2$

### `standard_error_of_s_square.py`

```python
import matplotlib.pyplot as plt
import numpy as np

from global_name_space import ARGS

def main():
    S_square = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        sigma = x.std()
        S_square.append(sigma**2)

    average = np.array(S_square).mean()
    standard_error = np.array(S_square).std()

    print(f'(Estimated) Mean of S^2 : {average:.4}')
    print(f'Standard Error   of S^2 : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of S^2", fontsize=20)

    ax.hist(S_square, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=12, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=12, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=12, alpha=0.7, color='k', ls='--')

    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 12),
                xytext=(average + standard_error, 12),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 13),
                xytext=(average, 13),
                fontsize=15)

    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(-0.1, 15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```
