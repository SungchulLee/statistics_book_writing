# Exercises

## Exercise 1: Jarque-Bera Test Statistic for Normality

The test statistic $JB$ for testing the normality of data using the Jarque-Bera test is defined as:

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2_2
$$

where:

- $n$: Sample size
- $S$: Sample skewness
- $K$: Sample kurtosis
- The denominator $6$ is a scaling constant that normalizes the contributions of skewness and kurtosis.

**(a)** Find the theoretical value of $S$ for a normal distribution.

**(b)** Evaluate the integral $\int_{-\infty}^\infty x^4 e^{-x^2} dx$.

**(c)** Why is $3$ subtracted from $K$ when calculating $JB$?

**(d)** If $JB = 3.2189$, express the $p$-value using the cumulative distribution function (CDF) $F$ of $\chi^2_2$.

**(e)** If the $p$-value is 0.2 and the significance level ($\alpha$) is 5%, what is the conclusion of the test?

### Solution

**(a)** For a normal distribution, the theoretical skewness $S$ is:

$$
S = \mathbb{E}\left(\frac{X - \mu}{\sigma}\right)^3 = \mathbb{E}Z^3 = \int_{-\infty}^\infty x^3 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 0
$$

**(b)**

$$
I
= \int_{-\infty}^\infty x^4 e^{-x^2} dx
= 2 \int_0^\infty x^4 e^{-x^2} dx
$$

Let $u = x^2$, so that $x = u^{1/2}$ and $dx = \frac{1}{2} u^{-1/2} du$. The integral becomes:

$$
I
= 2 \int_0^\infty (u^{1/2})^4 e^{-u} \cdot \frac{1}{2} u^{-1/2} du
= \int_0^\infty u^{\frac{5}{2}-1} e^{-u} du
= \Gamma\left(\frac{5}{2}\right)
= \frac{3}{2}\cdot\Gamma\left(\frac{3}{2}\right)
= \frac{3}{2}\cdot\frac{1}{2}\cdot\Gamma\left(\frac{1}{2}\right)
= \frac{3}{2}\cdot\frac{1}{2}\cdot\sqrt{\pi}
$$

where the Gamma function is defined as:

$$
\Gamma(n) = \int_0^\infty x^{n-1} e^{-x} dx
$$

The Gamma function satisfies:

$$
\Gamma(n+1) = n \cdot \Gamma(n),\quad
\Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}
$$

**(c)** For a normal distribution, the theoretical kurtosis $K$ is $3$. Subtracting $3$ ensures that for normal data, $K - 3 = 0$. This simplifies the test statistic under the null hypothesis of normality.

**(d)** The $p$-value is:

$$
p = 1 - F(3.2189)
$$

**(e)** Since $p = 0.2 > \alpha = 0.05$, there is insufficient evidence to reject the null hypothesis. Therefore, we conclude that the data does not violate normality and maintain the assumption of normality at the given significance level.

---

## Exercise 2: Normality Test with Outlier

A normality test was performed on a dataset, and one outlier was detected. When the outlier is included, the $p$-value is 0.01, but after removing the outlier, the $p$-value increases to 0.20. How should this be interpreted and addressed?

### Solution

- The remaining data, excluding the outlier, is judged to follow a normal distribution.
- Including the outlier indicates that the data does not follow a normal distribution, so the outlier should be investigated in detail.
- If the outlier is found to be contaminated or incorrectly recorded, correct the error or remove the data point and proceed with the analysis.
- If the outlier is neither contaminated nor incorrectly recorded, report the results both with and without the outlier if necessary.
