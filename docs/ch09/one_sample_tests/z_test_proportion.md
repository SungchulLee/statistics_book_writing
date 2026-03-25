# Z-Test for a Proportion

## Overview

Many practical questions involve population proportions: Has a manufacturing defect rate changed? Does voter support exceed 50%? Is the response rate for a new drug different from the known rate? The one-sample Z-test for proportions provides a formal framework for answering such questions. It tests whether a population proportion $p$ equals a specific hypothesized value $p_0$, using the normal approximation to the binomial distribution.

## Hypotheses

$$
H_0: p = p_0 \quad \text{vs} \quad H_1: p \neq p_0
$$

For one-sided tests, the alternative is either $H_1: p > p_0$ or $H_1: p < p_0$, depending on the research question.

## Test Statistic

The test statistic measures how many standard errors the sample proportion $\hat{p}$ lies from the hypothesized value $p_0$. Under $H_0$, the standard error of $\hat{p}$ is $\sqrt{p_0(1-p_0)/n}$, so we standardize:

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}
$$

where $\hat{p} = X/n$ is the sample proportion and $X$ is the number of successes in $n$ independent trials. Under $H_0$, $Z$ is approximately standard normal when the sample size conditions below are met.

## Decision Rule

- **Two-sided** ($H_1: p \neq p_0$): Reject $H_0$ if $|Z| > z_{\alpha/2}$
- **Right-sided** ($H_1: p > p_0$): Reject $H_0$ if $Z > z_{\alpha}$
- **Left-sided** ($H_1: p < p_0$): Reject $H_0$ if $Z < -z_{\alpha}$

Equivalently, compute the p-value and reject $H_0$ if $p\text{-value} \leq \alpha$.

## Conditions

The normal approximation to the binomial is adequate when both

$$
np_0 \geq 10 \quad \text{and} \quad n(1-p_0) \geq 10
$$

as a rule of thumb. These conditions ensure that the sampling distribution of $\hat{p}$ is approximately normal under $H_0$, so the Z-statistic has an approximate $N(0,1)$ distribution. Some references use a threshold of 5 instead of 10.

??? example "Worked Example: Testing a Defect Rate"

    A factory historically produces items with a 4% defect rate ($p_0 = 0.04$). After a process change, a random sample of $n = 500$ items contains 28 defectives, giving $\hat{p} = 28/500 = 0.056$. Test whether the defect rate has changed at the $\alpha = 0.05$ level.

    **Check conditions.** $np_0 = 500 \times 0.04 = 20 \geq 10$ and $n(1 - p_0) = 500 \times 0.96 = 480 \geq 10$. Both satisfied.

    **Compute the test statistic.**

    $$
    Z = \frac{0.056 - 0.04}{\sqrt{0.04 \times 0.96 / 500}} = \frac{0.016}{\sqrt{0.0000768}} = \frac{0.016}{0.00877} \approx 1.83
    $$

    **Decision.** For a two-sided test, the critical value is $z_{0.025} = 1.96$. Since $|1.83| < 1.96$, we fail to reject $H_0$. The two-sided p-value is $2P(Z > 1.83) \approx 0.067 > 0.05$, confirming the decision. There is not sufficient evidence at the 5% level to conclude that the defect rate has changed.
