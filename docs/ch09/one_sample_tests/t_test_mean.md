# t-Test for μ (Unknown σ)

## Overview

The one-sample t-test tests hypotheses about the population mean when $\sigma$ is unknown.

## Test Statistic

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1} \text{ under } H_0
$$

## Assumptions

- Random sample from a Normal population (or large $n$ by CLT)
- Independent observations

## Robustness

The t-test is fairly robust to non-normality for moderate to large samples.
