# Introduction to Normality

## What Is Normality?

We say a dataset is **normally distributed** if it follows a bell-shaped curve known as the **normal distribution**. The normal distribution is a continuous probability distribution defined by the probability density function (PDF):

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

where

- $\mu$ is the mean of the distribution,
- $\sigma$ is the standard deviation,
- $x$ is a variable that can take any real value.

The normal distribution is symmetric around its mean $\mu$, and about 68% of the data lies within one standard deviation ($\sigma$) from the mean, while approximately 95% lies within two standard deviations.

## Importance of Normality in Statistical Inference

Many statistical methods assume that the underlying data is normally distributed. Common techniques such as $t$-tests, Analysis of Variance (ANOVA), and linear regression rely on this assumption to ensure valid results. If the normality assumption is violated, these methods may produce inaccurate conclusions.

The **Central Limit Theorem (CLT)** tells us that, given a sufficiently large sample size, the sampling distribution of the sample mean tends to be normal, regardless of the population distribution. This asymptotic normality is why normality plays such a crucial role in inferential statistics.

## Situations Where Normality Is Assumed

We often assume normality in the following scenarios:

- **Confidence Intervals**: When constructing confidence intervals for the mean, it is assumed that the underlying data or the sample means are normally distributed.
- **Hypothesis Testing**: Many hypothesis tests, including the $t$-test, assume normality in the population to compute $p$-values accurately.
- **Linear Models**: Assuming that residuals are normally distributed in linear regression is critical for valid hypothesis testing and confidence intervals for the regression coefficients.

## Examples of Non-Normal Data

Not all data follows a normal distribution. Some common examples of non-normal data include:

- **Skewed Distributions**: Asymmetric data (positively or negatively skewed) deviates from normality.
- **Heavy-Tailed Distributions**: Distributions with heavier tails than the normal distribution, such as the $t$-distribution with low degrees of freedom, are considered non-normal.
- **Bimodal or Multimodal Distributions**: Data with more than one peak, where there is more than one mode in the dataset, deviates significantly from normality.

Understanding when data deviates from normality is essential for choosing the appropriate statistical tools and methods for analysis.
