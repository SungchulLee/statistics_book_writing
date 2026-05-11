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

## Exercises

**Exercise 1.**
A pollster wants to estimate a population proportion with a margin of error of 3 percentage points at 95% confidence. Find the required sample size using the conservative estimate $p = 0.5$.

??? success "Solution to Exercise 1"
    The margin of error for a proportion is $E = z_{\alpha/2}\sqrt{p(1-p)/n}$. Solving for $n$:

    $$
    n = \frac{z_{\alpha/2}^2 \cdot p(1-p)}{E^2}
    $$

    With $z_{0.025} = 1.96$, $p = 0.5$ (maximizes $p(1-p) = 0.25$), and $E = 0.03$:

    $$
    n = \frac{1.96^2 \times 0.25}{0.03^2} = \frac{0.9604}{0.0009} = 1067.1
    $$

    Rounding up: $n = 1068$ respondents are needed.

---

**Exercise 2.**
A researcher wants to estimate a population mean with margin of error $E = 2$ units at 99% confidence. A pilot study suggests $\sigma \approx 10$. Find the required sample size.

??? success "Solution to Exercise 2"
    The margin of error for a mean is $E = z_{\alpha/2} \cdot \sigma/\sqrt{n}$. Solving:

    $$
    n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2
    $$

    With $z_{0.005} = 2.576$, $\sigma = 10$, $E = 2$:

    $$
    n = \left(\frac{2.576 \times 10}{2}\right)^2 = (12.88)^2 = 165.9
    $$

    Rounding up: $n = 166$. If the pilot study estimate of $\sigma$ is uncertain, a conservative approach is to use a larger $\sigma$.

---

**Exercise 3.**
Explain why halving the margin of error requires quadrupling the sample size. Derive this relationship from the margin of error formula.

??? success "Solution to Exercise 3"
    The margin of error is $E = z_{\alpha/2}\sigma/\sqrt{n}$, which means $n = z_{\alpha/2}^2\sigma^2/E^2$.

    Since $n \propto 1/E^2$, the relationship between sample size and margin of error is an inverse-square law. If we want $E' = E/2$ (half the margin):

    $$
    n' = \frac{z_{\alpha/2}^2\sigma^2}{(E/2)^2} = \frac{4z_{\alpha/2}^2\sigma^2}{E^2} = 4n
    $$

    This "diminishing returns" property means that precision improvements become increasingly expensive. Going from $E = 4$ to $E = 2$ quadruples $n$; going from $E = 2$ to $E = 1$ quadruples it again. This is a fundamental constraint in survey design and explains why extremely precise estimates require very large samples.

---

**Exercise 4.**
A company has budget for $n = 400$ surveys. What is the best achievable margin of error for a population proportion at 95% confidence? How much does the margin shrink if the budget doubles to $n = 800$?

??? success "Solution to Exercise 4"
    Using the conservative $p = 0.5$:

    For $n = 400$:

    $$
    E = 1.96\sqrt{\frac{0.25}{400}} = 1.96 \times 0.025 = 0.049 = 4.9\%
    $$

    For $n = 800$:

    $$
    E = 1.96\sqrt{\frac{0.25}{800}} = 1.96 \times 0.01768 = 0.0346 = 3.5\%
    $$

    Doubling the sample size reduced the margin from 4.9% to 3.5%, a reduction factor of $\sqrt{2} \approx 1.414$. The margin decreases proportionally to $1/\sqrt{n}$, so doubling $n$ only improves precision by about 29%, not 50%.
