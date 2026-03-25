# Sample Size for Desired Margin of Error

## Overview

In previous sections we constructed confidence intervals and saw that their width depends on the sample size $n$. A natural planning question arises: how large must $n$ be to guarantee a desired level of precision? This section derives the required sample size by inverting the margin-of-error formula for both means and proportions.

## Sample Size for a Mean

Recall that the margin of error for a confidence interval for the population mean is $E = z_{\alpha/2} \cdot \sigma / \sqrt{n}$, where

- $z_{\alpha/2}$ is the critical value from the standard normal distribution corresponding to confidence level $1 - \alpha$,
- $\sigma$ is the population standard deviation (or a reliable estimate of it), and
- $E$ is the desired margin of error (half-width of the confidence interval).

Solving $E = z_{\alpha/2} \cdot \sigma / \sqrt{n}$ for $n$ gives the minimum sample size formula:

$$
n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2
$$

Since $n$ must be a positive integer, always round up the computed value to ensure the margin of error does not exceed $E$.

??? example "Worked Example: Estimating Average Height"

    A researcher wants to estimate the mean height of adult males in a city with a 95% confidence interval whose margin of error is at most $E = 1$ cm. From prior studies, the population standard deviation is estimated as $\sigma = 7$ cm. At the 95% confidence level, $z_{0.025} = 1.96$. The required sample size is

    $$
    n = \left(\frac{1.96 \times 7}{1}\right)^2 = (13.72)^2 = 188.24
    $$

    Rounding up, the researcher needs $n = 189$ participants.

## Sample Size for a Proportion

For a proportion, the margin of error of the confidence interval is $E = z_{\alpha/2} \sqrt{p^*(1 - p^*) / n}$, where $p^*$ is a planning value for the population proportion. Solving for $n$ yields:

$$
n = \frac{z_{\alpha/2}^2 \, p^*(1 - p^*)}{E^2}
$$

Here $p^*$ is a prior estimate or best guess for the true proportion $p$. When no prior estimate is available, setting $p^* = 0.5$ maximizes $p^*(1 - p^*)$ and therefore gives the most conservative (largest) required sample size. As with the mean case, always round up to the next integer.

??? example "Worked Example: Estimating Voter Support"

    A polling firm wants to estimate the proportion of voters who support a ballot measure with a 99% confidence interval and margin of error $E = 0.03$ (3 percentage points). No prior estimate of $p$ is available, so the firm uses $p^* = 0.5$. At the 99% level, $z_{0.005} = 2.576$. The required sample size is

    $$
    n = \frac{(2.576)^2 \times 0.5 \times 0.5}{(0.03)^2} = \frac{6.6358 \times 0.25}{0.0009} = \frac{1.6589}{0.0009} \approx 1843.3
    $$

    Rounding up, the firm needs $n = 1844$ respondents.
