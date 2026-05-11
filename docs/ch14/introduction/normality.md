# Introduction to Normality

## What Is Normality?

We say a dataset is **normally distributed** if it follows a bell-shaped curve known as the **normal distribution**. The normal distribution is a continuous probability distribution defined by the probability density function (PDF):

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

where

- $\mu$ is the mean of the distribution,
- $\sigma > 0$ is the standard deviation,
- $x \in \mathbb{R}$ is a variable that can take any real value.

The normal distribution is symmetric around its mean $\mu$. The **empirical rule** summarizes the concentration of probability:

| Interval | Approximate probability |
|----------|------------------------|
| $\mu \pm \sigma$ | 68.3% |
| $\mu \pm 2\sigma$ | 95.4% |
| $\mu \pm 3\sigma$ | 99.7% |

!!! tip "Standard Normal"
    When $\mu = 0$ and $\sigma = 1$, we obtain the **standard normal distribution** $Z \sim N(0,1)$. Any normal variable $X \sim N(\mu, \sigma^2)$ can be standardized via $Z = (X - \mu)/\sigma$.

## Importance of Normality in Statistical Inference

Many statistical methods assume that the underlying data is normally distributed. Common techniques such as $t$-tests, Analysis of Variance (ANOVA), and linear regression rely on this assumption to ensure valid results. If the normality assumption is violated, these methods may produce inaccurate conclusions.

The **Central Limit Theorem (CLT)** tells us that, given a sufficiently large sample size, the sampling distribution of the sample mean tends to be normal, regardless of the population distribution. Specifically, if $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty
$$

This asymptotic normality is why normality plays such a crucial role in inferential statistics, even when the underlying population is non-normal.

## Situations Where Normality Is Assumed

We often assume normality in the following scenarios:

- **Confidence Intervals**: When constructing confidence intervals for the mean, it is assumed that the underlying data or the sample means are normally distributed.
- **Hypothesis Testing**: Many hypothesis tests, including the $t$-test, assume normality in the population to compute $p$-values accurately.
- **Linear Models**: Assuming that residuals are normally distributed in linear regression is critical for valid hypothesis testing and confidence intervals for the regression coefficients.

!!! warning "Assumption vs. Requirement"
    Normality is an assumption, not a guaranteed property of real data. Always verify normality before applying methods that depend on it. The tools for doing so---graphical methods (Q-Q plots, histograms) and formal tests (Shapiro-Wilk, Anderson-Darling)---are covered in the subsequent sections of this chapter.

## Examples of Non-Normal Data

Not all data follows a normal distribution. Some common examples of non-normal data include:

- **Skewed Distributions**: Income data is typically right-skewed, with a long tail of high earners pulling the mean above the median.
- **Heavy-Tailed Distributions**: Financial returns often exhibit heavier tails than the normal distribution, meaning extreme events occur more frequently than a Gaussian model would predict.
- **Bimodal or Multimodal Distributions**: Data with more than one peak---for example, the heights of a mixed population of adult men and women---deviates significantly from normality.
- **Bounded or Discrete Data**: Proportions, counts, and Likert-scale responses are inherently non-normal due to their restricted range or discrete nature.

Understanding when data deviates from normality is essential for choosing the appropriate statistical tools and methods for analysis.


## Exercises

**Exercise 1.**
State the probability density function of the normal distribution $N(\mu, \sigma^2)$ and identify the roles of the two parameters.

??? success "Solution to Exercise 1"
    The PDF is:

    $$
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}
    $$

    - $\mu$ (mean) is the center of the distribution -- it determines the location of the peak and the axis of symmetry.
    - $\sigma^2$ (variance) controls the spread. Larger $\sigma^2$ produces a wider, flatter bell curve; smaller $\sigma^2$ produces a taller, narrower one.
    - $\sigma$ (standard deviation) is in the same units as $x$, making it more interpretable than $\sigma^2$.

---

**Exercise 2.**
State the 68-95-99.7 rule for the normal distribution and verify the 95% figure using the standard normal CDF.

??? success "Solution to Exercise 2"
    The 68-95-99.7 rule states that for $X \sim N(\mu, \sigma^2)$:

    - $P(\mu - \sigma < X < \mu + \sigma) \approx 0.683$
    - $P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.954$
    - $P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 0.997$

    For the 95% figure with $Z \sim N(0,1)$: $P(-2 < Z < 2) = \mathcal{N}(2) - \mathcal{N}(-2) = 0.9772 - 0.0228 = 0.9544 \approx 95.4\%$.

    The exact 95% interval uses $z = 1.96$: $P(-1.96 < Z < 1.96) = 0.95$.

---

**Exercise 3.**
If $X \sim N(70, 100)$ (mean 70, variance 100), find $P(X > 85)$.

??? success "Solution to Exercise 3"
    Standardize: $Z = (X - \mu)/\sigma = (85 - 70)/10 = 1.5$.

    $$
    P(X > 85) = P(Z > 1.5) = 1 - \mathcal{N}(1.5) = 1 - 0.9332 = 0.0668
    $$

    There is approximately a 6.7% probability of exceeding 85.

---

**Exercise 4.**
Explain why the normal distribution arises so frequently in practice, referencing the Central Limit Theorem.

??? success "Solution to Exercise 4"
    The Central Limit Theorem (CLT) states that the sum (or average) of $n$ independent, identically distributed random variables with finite mean and variance converges in distribution to a normal distribution as $n \to \infty$, regardless of the original distribution.

    Many real-world measurements are the aggregate effect of numerous small, independent factors. Height is influenced by many genes and environmental factors; measurement errors are the sum of many small instrumental imprecisions; financial returns (approximately) aggregate many independent information shocks. In each case, the CLT drives the aggregate toward normality.

    The CLT also explains why normal-based methods (t-tests, regression) work well even for non-normal data when $n$ is large: the sampling distribution of the mean is approximately normal regardless.
