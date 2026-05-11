# Power Analysis and Sample Size

## Overview

Power analysis determines the sample size needed to detect a meaningful effect with a specified probability. The **power** of a test is $1-\beta$, where $\beta$ is the probability of a Type II error (failing to reject a false null hypothesis). A well-designed study balances the significance level $\alpha$, the desired power, and the minimum effect size of interest to compute the required number of observations.

## Key Formulas

For a **two-sample t-test** (equal group sizes, known $\sigma$), the approximate power at sample size $n$ per group is

$$
\text{Power} = 1 - \mathcal{N}\!\left(z_{\alpha/2} - \frac{\delta}{\sigma\sqrt{2/n}}\right) + \mathcal{N}\!\left(-z_{\alpha/2} - \frac{\delta}{\sigma\sqrt{2/n}}\right),
$$

where $\delta = \mu_1 - \mu_2$ is the true difference and $\mathcal{N}$ is the standard normal CDF.

The required sample size per group to achieve power $1-\beta$ is

$$
n = \frac{2\,(z_{\alpha/2} + z_\beta)^2\,\sigma^2}{\delta^2}.
$$

For a **two-proportion z-test** comparing $p_1$ and $p_2$ with $\bar{p} = (p_1+p_2)/2$:

$$
n = \frac{\left(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_\beta\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_1 - p_2)^2}.
$$

## Code

### Power of a Two-Sample t-Test

```python
import numpy as np
from scipy import stats

def power_ttest(n, delta, sigma=1.0, alpha=0.05):
    """Compute power of a two-sided two-sample t-test."""
    se = sigma * np.sqrt(2 / n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_effect = delta / se
    power = (1 - stats.norm.cdf(z_crit - z_effect)
             + stats.norm.cdf(-z_crit - z_effect))
    return power
```

### Required Sample Size

```python
def sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80):
    """Compute minimum n per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(np.ceil(n))

def sample_size_proportion(p1, p2, alpha=0.05, power=0.80):
    """Compute minimum n per group for a two-proportion z-test."""
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    numer = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
             + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numer / (p1 - p2) ** 2
    return int(np.ceil(n))
```

### Example Computations

```python
# Two-sample t-test: medium effect size (Cohen's d = 0.5)
delta = 0.5
n_req = sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80)
print(f"Required n per group: {n_req}")  # ~64

# Proportion test (A/B test)
p1, p2 = 0.0121, 0.011
n_prop = sample_size_proportion(p1, p2, alpha=0.05, power=0.80)
print(f"Required n per group: {n_prop:,}")
```

### Power Curves

```python
import matplotlib.pyplot as plt

ns = np.arange(10, 500)
fig, ax = plt.subplots(figsize=(10, 5))
for d, ls in [(0.2, '--'), (0.5, '-'), (0.8, ':')]:
    powers = [power_ttest(n, d) for n in ns]
    ax.plot(ns, powers, ls, label=f'd = {d}')

ax.axhline(0.80, color='grey', linestyle='-.', alpha=0.5, label='Power = 0.80')
ax.set_xlabel('Sample size per group (n)')
ax.set_ylabel('Power')
ax.set_title('Power Curves for Two-Sample t-Test')
ax.legend()
plt.tight_layout()
plt.show()
```

### Interpretation

- A **small effect** ($d=0.2$) requires hundreds of subjects per group to achieve 80% power.
- A **medium effect** ($d=0.5$) needs roughly 64 per group.
- A **large effect** ($d=0.8$) needs only about 26 per group.
- For proportion tests, when the difference $|p_1-p_2|$ is very small, tens of thousands of observations may be required.

## Exercises

**Exercise 1.** A researcher wants 90% power to detect a difference of $\delta=0.3$ standard deviations between two groups at $\alpha=0.01$. How many subjects per group are needed?

??? success "Solution to Exercise 1"

    Using the sample size formula:

    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2}{\delta^2}.
    $$

    Here $z_{0.005} = 2.576$, $z_{0.10} = 1.282$, and $\delta = 0.3$:

    $$
    n = \frac{2(2.576 + 1.282)^2}{0.3^2} = \frac{2(3.858)^2}{0.09} = \frac{2 \times 14.884}{0.09} = \frac{29.768}{0.09} \approx 331.
    $$

    The researcher needs at least 331 subjects per group. $\square$

---

**Exercise 2.** Show that the power of a one-sample z-test (two-sided, known $\sigma$) for detecting $\mu = \mu_0 + \delta$ can be written as

$$
1 - \beta = \mathcal{N}\!\left(\frac{\delta\sqrt{n}}{\sigma} - z_{\alpha/2}\right) + \mathcal{N}\!\left(-\frac{\delta\sqrt{n}}{\sigma} - z_{\alpha/2}\right).
$$

??? success "Solution to Exercise 2"

    Under $H_1\colon \mu = \mu_0 + \delta$, the test statistic $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n})$ has distribution $N(\delta\sqrt{n}/\sigma,\,1)$. Let $\lambda = \delta\sqrt{n}/\sigma$. The test rejects when $|Z| > z_{\alpha/2}$, so

    $$
    1 - \beta = P(Z > z_{\alpha/2}) + P(Z < -z_{\alpha/2}).
    $$

    Since $Z \sim N(\lambda, 1)$:

    $$
    P(Z > z_{\alpha/2}) = P(Z - \lambda > z_{\alpha/2} - \lambda) = \mathcal{N}(\lambda - z_{\alpha/2}),
    $$

    $$
    P(Z < -z_{\alpha/2}) = P(Z - \lambda < -z_{\alpha/2} - \lambda) = \mathcal{N}(-z_{\alpha/2} - \lambda).
    $$

    Summing gives the desired result. $\square$

---

**Exercise 3.** An A/B test compares conversion rates $p_1 = 0.05$ and $p_2 = 0.04$. Compute the required sample size per group for 80% power at $\alpha = 0.05$.

??? success "Solution to Exercise 3"

    We have $\bar{p} = (0.05 + 0.04)/2 = 0.045$, $z_{0.025}=1.96$, $z_{0.20}=0.842$:

    $$
    n = \frac{\left(1.96\sqrt{2(0.045)(0.955)} + 0.842\sqrt{0.05(0.95) + 0.04(0.96)}\right)^2}{(0.05 - 0.04)^2}.
    $$

    Computing the pieces: $2(0.045)(0.955) = 0.08595$, so $\sqrt{0.08595} \approx 0.2932$. Also $0.0475 + 0.0384 = 0.0859$, so $\sqrt{0.0859}\approx 0.2931$.

    $$
    n = \frac{(1.96 \times 0.2932 + 0.842 \times 0.2931)^2}{0.0001} = \frac{(0.5747 + 0.2468)^2}{0.0001} = \frac{0.6741}{0.0001} \approx 6741.
    $$

    Approximately 6741 subjects are needed per group. $\square$

---

**Exercise 4.** Using Python, plot the power of a one-sample t-test as a function of $n$ (from 5 to 200) for Cohen's $d \in \{0.2, 0.5, 0.8\}$ at $\alpha = 0.05$. Use `statsmodels.stats.power.TTestPower`.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.stats.power import TTestPower

    analysis = TTestPower()
    ns = np.arange(5, 201)

    fig, ax = plt.subplots(figsize=(9, 5))
    for d in [0.2, 0.5, 0.8]:
        powers = [analysis.power(effect_size=d, nobs=n, alpha=0.05) for n in ns]
        ax.plot(ns, powers, label=f'd = {d}')

    ax.axhline(0.80, color='grey', linestyle='--', label='80% power')
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Power')
    ax.set_title('Power Curves (One-Sample t-Test)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    The plot shows that larger effect sizes require fewer subjects, and the curves are S-shaped, increasing steeply near the sample size that achieves the target power. $\square$

---

**Exercise 5.** Prove that for a fixed significance level $\alpha$ and effect size $\delta > 0$, the power of the two-sample z-test approaches 1 as $n \to \infty$.

??? success "Solution to Exercise 5"

    The power is

    $$
    1 - \beta = \mathcal{N}\!\left(\frac{\delta}{\sigma\sqrt{2/n}} - z_{\alpha/2}\right) + \mathcal{N}\!\left(-\frac{\delta}{\sigma\sqrt{2/n}} - z_{\alpha/2}\right).
    $$

    As $n \to \infty$, the term $\delta / (\sigma\sqrt{2/n}) = \delta\sqrt{n}/({\sigma\sqrt{2}}) \to \infty$. Therefore:

    - The first term: $\mathcal{N}(\infty - z_{\alpha/2}) = \mathcal{N}(\infty) = 1$.
    - The second term: $\mathcal{N}(-\infty - z_{\alpha/2}) = \mathcal{N}(-\infty) = 0$.

    Hence $1-\beta \to 1$. This confirms that with enough data, any fixed nonzero effect will eventually be detected. $\square$
