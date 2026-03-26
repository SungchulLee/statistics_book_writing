# Skewed Distributions

## Why Skewness Matters for Estimation

Many real-world data sets exhibit asymmetry: household incomes tend to be right-skewed with a long upper tail, while failure-time data and insurance claims often follow exponential or log-normal distributions. When the underlying distribution is skewed, the usual estimation methods — built on the assumption of symmetry or normality — may perform poorly. The sample mean can be unduly influenced by extreme values in the tail, confidence intervals based on the Central Limit Theorem converge more slowly, and the choice between mean and median as a measure of center becomes a substantive question. This page examines how skewness affects estimation and what tools are available to address it.

## Measuring Skewness

Before discussing its effects, we need a precise definition. The **skewness coefficient** of a random variable $X$ with mean $\mu$ and standard deviation $\sigma$ is

$$
\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}
$$

- $\gamma_1 > 0$: the distribution is **right-skewed** (the right tail is longer or heavier).
- $\gamma_1 < 0$: the distribution is **left-skewed** (the left tail is longer or heavier).
- $\gamma_1 = 0$: the distribution is symmetric about its mean.

The sample skewness, which estimates $\gamma_1$, is

$$
g_1 = \frac{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^3}{\left(\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right)^{3/2}}
$$

## Effects of Skewness on Location Measures

For many common unimodal distributions, skewness pulls the mean toward the longer tail relative to the median:

- **Right-skewed** ($\gamma_1 > 0$): the mean tends to exceed the median (e.g., income data, exponential waiting times).
- **Left-skewed** ($\gamma_1 < 0$): the mean tends to fall below the median.

!!! warning "This Is a Heuristic, Not a Theorem"

    The ordering Mean > Median > Mode for right-skewed distributions is widely cited but is not universally true. Counterexamples exist among multimodal or discrete distributions. The relationship holds reliably for many standard unimodal continuous distributions (exponential, log-normal, gamma), but should not be applied blindly to arbitrary data.

!!! example "Exponential Distribution"

    Let $X \sim \text{Exp}(\lambda)$ with rate $\lambda > 0$. The mean is $1/\lambda$, the median is $\ln 2 / \lambda \approx 0.693/\lambda$, and the mode is 0. We have

    $$
    \text{Mean} = \frac{1}{\lambda} > \text{Median} = \frac{\ln 2}{\lambda} > \text{Mode} = 0
    $$

    The skewness coefficient is $\gamma_1 = 2$ regardless of $\lambda$, confirming substantial right skewness.

## Estimation Considerations

### Unbiasedness vs Efficiency

The sample mean $\bar{X}$ is an unbiased estimator of the population mean $\mu = E[X]$ regardless of the shape of the distribution — this follows directly from linearity of expectation and does not require normality or symmetry. However, unbiasedness does not address the precision of the estimate.

For skewed distributions, the sample mean can have high variance because extreme observations in the long tail contribute disproportionately to $\bar{X}$. The sample median, while potentially biased for the mean, can be a more efficient estimator of the center of the distribution in terms of MSE when the data are heavily skewed.

### CLT Convergence Rate

The Central Limit Theorem guarantees that $\bar{X}$ is asymptotically normal, but the speed of convergence depends on the distribution's shape. The Berry-Esseen theorem provides a bound on the approximation error:

$$
\sup_z \left|P\left(\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z\right) - \Phi(z)\right| \leq \frac{C \cdot E[|X - \mu|^3]}{\sigma^3 \sqrt{n}}
$$

where $C \leq 0.4748$. For highly skewed distributions, $E[|X - \mu|^3]/\sigma^3$ is large, so larger sample sizes are needed before the normal approximation becomes reliable. A common rule of thumb is that $n \geq 30$ suffices for mildly skewed data, but $n \geq 100$ or more may be needed for strongly skewed distributions.

## Transformations to Reduce Skewness

When working with right-skewed data, applying a concave transformation can pull in the long right tail and produce a distribution closer to symmetry.

### Log Transformation

The simplest approach for right-skewed positive data is the **log transformation**: replace each observation $x_i$ with $\log x_i$. If $X$ follows a log-normal distribution, then $\log X$ is exactly normal, and standard methods apply directly to the transformed data.

!!! example "Log-Normal Data"

    If $X \sim \text{LogNormal}(\mu, \sigma^2)$, then $\gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$, which can be very large. After the log transformation, $Y = \log X \sim N(\mu, \sigma^2)$ with $\gamma_1 = 0$.

### Box-Cox Transformation

The **Box-Cox family** generalizes the log transformation by introducing a parameter $\lambda$ that controls the strength of the transformation. For strictly positive data ($y > 0$):

$$
y^{(\lambda)} = \begin{cases} \dfrac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\[6pt] \log y & \text{if } \lambda = 0 \end{cases}
$$

The parameter $\lambda$ is typically chosen by maximum likelihood: for each candidate $\lambda$, transform the data, fit a normal model to the transformed data, and select the $\lambda$ that maximizes the profile log-likelihood. Common special cases include $\lambda = 1$ (no transformation), $\lambda = 0.5$ (square root), $\lambda = 0$ (log), and $\lambda = -1$ (reciprocal).

!!! warning "Interpretation After Transformation"

    After applying a transformation, inferences are on the transformed scale. Back-transforming point estimates to the original scale requires care: the mean of $\log X$ is **not** the log of the mean of $X$. For the log-normal case, the back-transformed mean is $\exp(\hat{\mu} + \hat{\sigma}^2/2)$, not $\exp(\hat{\mu})$.

## When to Use the Mean vs the Median

The choice between mean and median as a summary of center depends on the purpose of the analysis:

- **Mean**: appropriate when the goal is to estimate the expected value (e.g., average cost, total revenue projection). Sensitive to outliers and tail behavior.
- **Median**: appropriate when the goal is to describe the "typical" observation. Robust to outliers and invariant to monotone transformations of the data.

For policy or business decisions that depend on totals or averages, the mean is the natural target. For describing what a "typical" individual experiences, the median is often more informative in the presence of skewness.
