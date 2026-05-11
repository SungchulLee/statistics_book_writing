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

## Python: Standard Error of X-bar
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

## Python: Standard Error of S-squared
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

## Exercises

**Exercise 1.**
$\sigma = 50$. (a) Compute $\mathrm{SE}$ for $n = 25, 100$. (b) For $\mathrm{SE} = 5, n = 16$, infer $\sigma$ and compute $\mathrm{SE}$ at $n = 64$.

??? success "Solution to Exercise 1"
    (a) $\mathrm{SE}_{25} = 50/\sqrt{25} = 10$. $\mathrm{SE}_{100} = 50/\sqrt{100} = 5$. SE halved when $n$ quadruples.

    (b) From $\sigma/\sqrt{16} = 5$: $\sigma = 20$. At $n = 64$: $\mathrm{SE} = 20/\sqrt{64} = 2.5$. Quadrupling $n$ halves SE.

---

**Exercise 2.**
**SE vs. SD.** A researcher reports "sample mean $= 50$, SD $= 8$" for $n = 100$. (a) What is the SE of the sample mean? (b) Explain to a non-technical reader the difference between the two.

??? success "Solution to Exercise 2"
    (a) Estimated SE (using $s$ instead of $\sigma$): $\mathrm{SE} = 8/\sqrt{100} = 0.8$.

    (b) **SD = 8:** describes how much individual observations vary in the data set. A typical individual is about 8 units from the mean.

    **SE = 0.8:** describes how much the sample mean varies across different samples. The true population mean is likely within about 1.6 units (≈ 2 SEs) of 50.

    The SD doesn't change as you collect more data; the SE shrinks at rate $1/\sqrt n$. Reporting an SD when an SE is meant — or vice versa — is a common error in scientific writing.

---

**Exercise 3.**
**Bootstrap SE.** When the population is not normal and $\sigma$ is unknown, the **bootstrap** provides an SE estimate. Describe the procedure for computing SE($\bar X$) via bootstrap.

??? success "Solution to Exercise 3"
    Given an i.i.d. sample $X_1, \ldots, X_n$:

    1. Draw a bootstrap sample $X_1^*, \ldots, X_n^*$ by sampling with replacement from the original sample.
    2. Compute $\bar X^*$ from the bootstrap sample.
    3. Repeat steps 1-2 $B$ times (typically $B = 1000$ to 10000), yielding $\bar X^*_1, \ldots, \bar X^*_B$.
    4. Estimate SE as the sample SD of the bootstrap replicates: $\hat{\mathrm{SE}}_{\text{boot}} = \sqrt{(1/(B-1))\sum(\bar X^*_b - \bar X^*_\cdot)^2}$.

    **Why it works:** the bootstrap distribution approximates the sampling distribution under repeated sampling. Asymptotically, $\hat{\mathrm{SE}}_{\text{boot}} \to \sigma/\sqrt n$, but the bootstrap captures distributional shape (skew, heavy tails) better than the normal approximation.

    Especially valuable when no closed-form SE exists (medians, ratios, regression coefficients in complex models).

---

**Exercise 4.**
**SE of a function.** Use the **delta method** to compute $\mathrm{SE}(g(\hat\theta))$ when $\hat\theta$ has $\mathrm{SE}(\hat\theta)$ and $g$ is differentiable.

??? success "Solution to Exercise 4"
    **Delta method:** if $\sqrt n(\hat\theta - \theta) \xrightarrow{d} N(0, \sigma^2)$, then for differentiable $g$ with $g'(\theta) \ne 0$:

    $$
    \sqrt n(g(\hat\theta) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)
    $$

    In SE form: $\mathrm{SE}(g(\hat\theta)) \approx |g'(\hat\theta)| \cdot \mathrm{SE}(\hat\theta)$.

    **Example:** $g(\hat p) = \log(\hat p/(1 - \hat p))$ (logit). $g'(\hat p) = 1/(\hat p(1 - \hat p))$. So $\mathrm{SE}(\hat\eta) = \mathrm{SE}(\hat p)/(\hat p(1 - \hat p))$.

    The delta method is the workhorse for SE computation when the estimator is a transformation of a simpler estimator with known SE.

---

**Exercise 5.**
**Pooled SE for two samples.** Independent samples from two populations with means $\mu_1, \mu_2$ and (unknown) variances. Derive SE of $\bar X_1 - \bar X_2$ under (a) equal variance assumption (pooled), (b) unequal variances (Welch).

??? success "Solution to Exercise 5"
    By independence: $\mathrm{Var}(\bar X_1 - \bar X_2) = \sigma_1^2/n_1 + \sigma_2^2/n_2$.

    **(a) Pooled (assume $\sigma_1 = \sigma_2 = \sigma$):**

    Pool: $s_p^2 = ((n_1 - 1)s_1^2 + (n_2 - 1)s_2^2)/(n_1 + n_2 - 2)$.

    $\mathrm{SE}_{\text{pool}} = s_p \sqrt{1/n_1 + 1/n_2}$.

    Used in the standard two-sample $t$-test when variances are believed equal. Slightly more efficient when this assumption holds.

    **(b) Welch (unequal variances):**

    $\mathrm{SE}_{\text{Welch}} = \sqrt{s_1^2/n_1 + s_2^2/n_2}$.

    No pooling, each sample contributes its own variance. Welch's degrees of freedom (non-integer) used for the $t$ critical value.

    Modern recommendation: prefer Welch by default — it doesn't require the strong equal-variance assumption and works nearly as well even when variances are equal.

---

**Exercise 6.**
**SE under sampling without replacement.** A population of size $N$, sample of size $n$ drawn without replacement. Compute SE($\bar X$) and identify the **finite-population correction**.

??? success "Solution to Exercise 6"
    For sampling without replacement from a finite population:

    $$
    \mathrm{Var}(\bar X) = \frac{\sigma^2}{n}\!\left(1 - \frac{n}{N}\right)
    $$

    So $\mathrm{SE}(\bar X) = (\sigma/\sqrt n) \sqrt{1 - n/N}$. The factor $\sqrt{1 - n/N}$ is the **finite-population correction (FPC)**.

    **Limits:**

    - $n/N \to 0$ (sampling fraction tiny): FPC $\to 1$, recovers standard $\sigma/\sqrt n$. Use for national surveys ($n = 1000, N \approx 10^8$).
    - $n/N \to 1$ (census): FPC $\to 0$, no sampling variability. Census produces deterministic estimates.

    **When FPC matters:** auditing (sampling 100 of 500 invoices, $n/N = 0.2$, FPC $\approx 0.89$). The FPC narrows confidence intervals by about 11% — non-trivial.

    Most introductory statistics formulas ignore FPC because typical scientific samples have small $n/N$.
