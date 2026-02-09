# Confidence Level and Coverage

## What Is a Confidence Interval?

Confidence intervals are a fundamental concept in inferential statistics, providing a way to quantify the uncertainty associated with estimating a population parameter. When working with sample data, there is always some sampling variability — differences between the sample and the population — since samples represent only a portion of the population. A point estimate, such as the sample mean, gives a single best guess for the population parameter (e.g., the population mean). However, this estimate does not reflect the uncertainty inherent in using a sample to make inferences about the entire population.

Confidence intervals address this by offering a range of plausible values for the population parameter, with an associated confidence level (usually 90%, 95%, or 99%). The general form of a confidence interval is given by

$$
\text{Point Estimate} \pm \text{Margin of Error}
$$

The margin of error is determined by the variability in the data (e.g., the standard deviation), the sample size, and the desired confidence level. The margin of error tends to be smaller for larger sample sizes because the sample provides a more precise estimate of the population parameter.

The wider the interval, the greater the uncertainty acknowledged in the estimate; conversely, a narrower interval reflects more precision.

---

## Formal Definition

A confidence interval for a population parameter is an interval computed from sample data that is likely to cover the true value of the parameter with a specified probability, known as the confidence level. Mathematically, a confidence interval for a parameter $\theta$ is given by:

$$
\hat{\theta} \pm \text{Margin of Error}
$$

where $\hat{\theta}$ is the point estimate of the parameter (e.g., the sample mean $\bar{X}$), and the margin of error is a function of the standard error and the desired confidence level.

The confidence level (typically 90%, 95%, or 99%) reflects the degree of certainty that the interval contains the true population parameter. For example, a 95% confidence interval means that if we were to take many random samples and compute a confidence interval for each, approximately 95% of these intervals would contain the true population parameter.

---

## Population and Parameter

In statistics, the terms **population** and **parameter** are essential when discussing data analysis, as they help us understand the distinction between the entire group we are interested in and the specific measures that summarize its characteristics.

The **population** refers to the entire data set or all possible observations we want to study. It is the complete collection of individuals, items, or data points that share a common characteristic or set of characteristics. For example, if we are studying the average height of adult men in a country, the population would be every adult man in that country. Populations can be finite (e.g., all students at a particular university) or infinite (e.g., all possible outcomes of rolling a fair die). Since populations are often large or inaccessible, collecting data from every individual in the population is usually impractical.

A **parameter** is a numerical value that summarizes some characteristic of the entire population. It is a fixed, usually unknown, value that describes a particular feature, such as the population mean ($\mu$), population proportion ($p$), population variance ($\sigma^2$), or population standard deviation ($\sigma$). Parameters represent the true value of the population but are often unknown because obtaining data from every member of the population is infeasible. Instead, we collect a **sample** (a population subset) and use this sample to compute statistics to estimate the population parameter.

$$
\begin{array}{ccc}
\textbf{Population} & \longrightarrow & \textbf{Parameter} \\
\text{All adult men in a country} & & \text{Population mean height } (\mu) \\
\text{All registered voters} & & \text{Population proportion of voters supporting a candidate } (p) \\
\text{All manufactured products from a factory} & & \text{Population defect rate } (\theta) \\
\end{array}
$$

In most real-world situations, we cannot directly measure the population parameter, so we rely on **sample statistics** to estimate these unknown parameters.

---

## Sample, Statistic, Estimator, and Estimate

A **sample** is a subset of individuals or observations selected from the larger population. Collecting data from the entire population can be impractical, time-consuming, or costly. **Random samples** are particularly valuable because they help ensure that the sample accurately reflects the population, minimizing bias. **Larger samples** tend to provide more reliable information about the population, as they reduce the sampling error.

A **statistic** is any numerical value that summarizes or describes some aspect of the sample data. Statistics include the sample mean $\bar{x}$, the sample proportion $\hat{p}$, and the sample variance $s^2$. We use these statistics to infer or estimate unknown population parameters.

An **estimator** is a formula or method used to estimate a population parameter from sample data. For example, the estimator for the population mean $\mu$ is the sample mean:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i
$$

The estimator for the population proportion $p$ is the sample proportion:

$$
\hat{p} = \frac{x}{n}
$$

An **estimate** is the specific numerical value calculated from sample data using the estimator. For instance, if we compute $\bar{x} = 10$ from a specific sample, then 10 is the estimate of $\mu$.

$$
\begin{array}{cccc}
\textbf{Population} & \longrightarrow & \textbf{Parameter} & (\mu, p, \sigma^2, \text{etc.}) \\
\downarrow & & \uparrow & \\
\textbf{Sample} & \longrightarrow & \textbf{Estimate} & (\bar{x}, \hat{p}, s^2, \text{etc.}) \\
\end{array}
$$

---

## Structure of a Confidence Interval

A **confidence interval** provides a range of values likely to contain the true population parameter. The general form is

$$
\text{estimate} \pm \text{margin of error}
$$

The **estimate** is the sample statistic (e.g., sample mean or sample proportion), and the **margin of error** captures the uncertainty in that estimate. The margin of error consists of two components: the **standard error** (which reflects the variability in the sample) and a **critical value** from a probability distribution, typically either the standard normal distribution or the $t$-distribution.

- For large samples (usually when the sample size $n \geq 30$), we use the **standard normal distribution** ($z$-scores) to find the critical value.
- For smaller samples (typically $n < 30$), we use the **$t$-distribution**, which adjusts for the additional uncertainty due to small sample sizes. When we use the $t$-distribution, we must ensure the original population follows the normal distribution.

$$
\begin{array}{cccc}
\textbf{Population} & \longrightarrow & \textbf{Parameter} \\
\downarrow & & \uparrow \\
\textbf{Sample} & \longrightarrow & \textbf{Estimate} \\
& & \textbf{Estimate} & \pm & \textbf{Margin of Error} \\
\end{array}
$$

---

## Correct Interpretation of Confidence Level

The confidence level represents the probability that the interval contains the true population parameter if we were to sample from the population under the same conditions repeatedly. For example, a 95% confidence interval suggests that in 95 out of 100 such samples, the interval would contain the true population parameter.

!!! warning "Common Misconception"
    A confidence interval does **not** give the probability that the parameter lies within the interval for a given sample. If we have a specific confidence interval, there are only two possibilities: either it contains the true parameter or it does not. Instead, the interval construction **procedure** has a level of confidence. When the procedure constructs many confidence intervals using different random samples from the population, some intervals contain the true parameter, and others do not. However, the ratio of the number of intervals containing the true parameter over all the generated intervals will be the confidence level in the long run. The confidence level describes the credential of the construction procedure, not a particular interval.

---

## Importance of Confidence Intervals in Statistical Inference

In practice, confidence intervals serve several vital functions in statistical analysis:

**Estimating Population Parameters.** We typically use confidence intervals when we want to estimate unknown population parameters, such as the population mean ($\mu$), population proportion ($p$), or population variance ($\sigma^2$). Since it is often impractical or impossible to collect data from the entire population, we rely on sample data to provide estimates of these parameters. For example, in a survey estimating the average income in a city, we could take a random sample of individuals, calculate the sample mean, and construct a confidence interval around that mean to estimate the true population average.

**Quantifying Uncertainty.** Due to the natural variability in data, every sample provides only an approximation of the true population parameter. A confidence interval gives us a way to quantify the uncertainty associated with an estimate. It accounts for the fact that different samples would yield slightly different estimates, and it provides a range within which we expect the true population parameter to fall with a specified level of confidence (e.g., 90%, 95%, 99%).

**Providing a Range of Supported Values.** Instead of giving a single-point estimate, which might be misleading by itself, a confidence interval offers a range of values that are consistent with the data. For instance, a point estimate of the average height of students might be 170 cm, but this single value does not tell us how precise the estimate is. A confidence interval might say, "We are 95% confident that the true average height lies between 168 cm and 172 cm." This range allows for more informed decision-making.

---

## Relationship Between Confidence Intervals and Hypothesis Testing

There is a close relationship between confidence intervals and hypothesis testing. Consider the case where we are testing the null hypothesis $H_0: \mu = \mu_0$ against the alternative hypothesis $H_1: \mu \neq \mu_0$.

One way to make this determination is by constructing a confidence interval for $\mu$ and then seeing whether the hypothesized value $\mu_0$ lies within this interval:

- If the value of $\mu_0$ **falls outside** the confidence interval for $\mu$, we would **reject** the null hypothesis at the corresponding significance level, since values outside the confidence interval are considered unlikely given the observed data.
- If $\mu_0$ **falls within** the confidence interval, we **fail to reject** the null hypothesis, as the data do not provide sufficient evidence to conclude that $\mu$ is different from $\mu_0$.

The **significance level** of a hypothesis test ($\alpha$) corresponds to the **confidence level** of a confidence interval:

$$
\textbf{significance level} = 1 - \textbf{confidence level}
$$

---

## Confidence Interval Simulation

The following simulation scripts help visualize coverage properties of confidence intervals. Each script repeatedly draws random samples from a known population, constructs confidence intervals, and tracks what fraction of intervals capture the true parameter.

### Summary of Simulation Parameters

$$
\begin{array}{llll}
\text{Parameter} & \text{Estimate} & \text{Sampling Distribution} & \text{CI Formula} \\
\hline
\mu & \bar{x} & \displaystyle\frac{\bar{x}-\mu}{\sigma/\sqrt{n}}\approx z & \displaystyle\bar{x}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \\[8pt]
& & \displaystyle\frac{\bar{x}-\mu}{s/\sqrt{n}}\approx t_{n-1} & \displaystyle\bar{x}\pm t_{\alpha/2,n-1}\frac{s}{\sqrt{n}} \\[8pt]
p & \hat{p} & \displaystyle\frac{\hat{p}-p}{\sqrt{\hat{p}(1-\hat{p})/n}}\approx z & \displaystyle\hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \\[8pt]
\sigma^2 & s^2 & \displaystyle\frac{(n-1)s^2}{\sigma^2}\sim \chi^2_{n-1} & \displaystyle\left[\frac{(n-1)s^2}{\chi^2_{\alpha/2,n-1}},\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,n-1}}\right] \\[8pt]
\frac{\sigma_1^2}{\sigma_2^2} & \frac{s_1^2}{s_2^2} & \displaystyle\frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2}\sim F_{n_1-1,n_2-1} & \displaystyle\left[\frac{s_1^2/s_2^2}{F_{\alpha/2}},\frac{s_1^2/s_2^2}{F_{1-\alpha/2}}\right] \\[8pt]
\mu_d & \bar{x}_d & \displaystyle\frac{\bar{x}_d-\mu_d}{s_d/\sqrt{n}}\approx t_{n-1} & \displaystyle\bar{x}_d\pm t_{\alpha/2,n-1}\frac{s_d}{\sqrt{n}} \\[8pt]
\mu_1-\mu_2 & \bar{x}_1-\bar{x}_2 & \text{Welch or pooled } t & \displaystyle(\bar{x}_1-\bar{x}_2)\pm t_{\alpha/2,\text{df}}\cdot\text{SE} \\[8pt]
p_1-p_2 & \hat{p}_1-\hat{p}_2 & z & \displaystyle(\hat{p}_1-\hat{p}_2)\pm z_{\alpha/2}\cdot\text{SE}
\end{array}
$$

### Finite Population Correction (FPC)

If the data are sampled *without replacement* from a finite population of size $N$, the **finite population correction (FPC)** may be applied to the standard error:

$$
\text{SE}_\text{FPC} = \frac{s}{\sqrt{n}} \sqrt{\frac{N-n}{N-1}}, \qquad \text{when } n > 0.1N \text{ is not negligible.}
$$

### Mean CI Simulation

```python
#!/usr/bin/env python3
"""
Mean CI simulation: Z (known σ), Z (plug-in s), and t (σ unknown).

Usage:
    python mean_ci_simulation.py --method t --n-sim 100 --n 10 --alpha 0.05
    python mean_ci_simulation.py --method z_known --sigma 1.0
    python mean_ci_simulation.py --method z_plugin --N 500  # with FPC
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t


def finite_population_correction(n: int, N: int | None) -> float:
    """Return FPC factor sqrt((N−n)/(N−1)) if N is provided; else 1.0."""
    if N is None:
        return 1.0
    if N <= 1 or n >= N:
        raise ValueError("FPC requires N > 1 and n < N.")
    return float(np.sqrt((N - n) / (N - 1)))


def simulate_data(n_sim: int, n: int, mu: float, sigma: float, rng) -> np.ndarray:
    """Return X with shape (n_sim, n): each row ~ N(mu, sigma^2)."""
    return rng.normal(loc=mu, scale=sigma, size=(n_sim, n))


def compute_intervals(xbar, s, n, alpha, method, sigma_known=None, N=None):
    fpc = finite_population_correction(n, N)
    if method == "z_known":
        if sigma_known is None:
            raise ValueError("z_known requires sigma_known.")
        z_star = norm.ppf(1 - alpha / 2.0)
        se = sigma_known / np.sqrt(n) * fpc
        moe = z_star * se
    elif method == "z_plugin":
        z_star = norm.ppf(1 - alpha / 2.0)
        se = (s / np.sqrt(n)) * fpc
        moe = z_star * se
    elif method == "t":
        df = n - 1
        t_star = t.ppf(1 - alpha / 2.0, df=df)
        se = (s / np.sqrt(n)) * fpc
        moe = t_star * se
    else:
        raise ValueError(f"Unknown method: {method}")
    return xbar - moe, xbar + moe


def plot_intervals(ax, lower, upper, xbar, covered, mu, title):
    for i in range(len(xbar)):
        color = "k" if covered[i] else "r"
        ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)
        ax.plot(xbar[i], i, marker="o", ms=3, color=color)
    ax.axvline(mu, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(title, fontsize=12)
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Mean value")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["z_known", "z_plugin", "t"], default="t")
    p.add_argument("--rng-seed", type=int, default=None)
    p.add_argument("--n-sim", type=int, default=100)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--N", type=int, default=None)
    args, _ = p.parse_known_args()

    rng = np.random.default_rng(args.rng_seed)
    X = simulate_data(args.n_sim, args.n, args.mu, args.sigma, rng)
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)

    lower, upper = compute_intervals(
        xbar, s, args.n, args.alpha, args.method,
        sigma_known=args.sigma if args.method == "z_known" else None, N=args.N
    )
    covered = (lower <= args.mu) & (args.mu <= upper)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    title = (f"{args.n_sim} {args.method} CIs | n={args.n}, "
             f"CL={int((1 - args.alpha) * 100)}% | "
             f"Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    plot_intervals(ax, lower, upper, xbar, covered, args.mu, title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

### Proportion CI Simulation

```python
#!/usr/bin/env python3
"""
Proportion CI simulation: Wald, Wilson, Agresti-Coull, Clopper-Pearson.

Usage:
    python proportion_ci_simulation.py  # defaults to Wald
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

rng_seed = None
n_simulations = 100
n = 20
p_true = 0.20
alpha = 0.05
method = "wald"  # 'wald' | 'wilson' | 'ac' | 'cp'


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    k = np.random.binomial(n=n, p=p_true, size=n_simulations)
    phat = k / n
    lower = np.empty(n_simulations)
    upper = np.empty(n_simulations)
    z = norm.ppf(1 - alpha / 2.0)

    for i, ki in enumerate(k):
        p = ki / n
        if method == "wald":
            se = np.sqrt(p * (1 - p) / n)
            lo, hi = p - z * se, p + z * se
        elif method == "wilson":
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            lo, hi = center - half, center + half
        elif method == "ac":
            n_tilde = n + z**2
            p_tilde = (ki + 0.5 * z**2) / n_tilde
            se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
            lo, hi = p_tilde - z * se_tilde, p_tilde + z * se_tilde
        elif method == "cp":
            lo = 0.0 if ki == 0 else beta.ppf(alpha / 2.0, ki, n - ki + 1)
            hi = 1.0 if ki == n else beta.ppf(1 - alpha / 2.0, ki + 1, n - ki)
        lower[i] = max(0.0, lo)
        upper[i] = min(1.0, hi)

    covered = (lower <= p_true) & (p_true <= upper)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)
        ax.plot(phat[i], i, marker="o", ms=3, color=color)
    ax.axvline(p_true, linestyle="--", linewidth=1.5)
    ax.set_title(
        f"{n_simulations} {method.upper()} Proportion CIs | n={n}, p={p_true:.3f}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Proportion value")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```
