# Z-Test for μ (Known σ)

## Overview

The one-sample Z-test tests hypotheses about the population mean when $\sigma$ is known.

## Hypotheses

$$
H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0 \text{ (or } \mu > \mu_0 \text{ or } \mu < \mu_0\text{)}
$$

## Test Statistic

$$
Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0,1) \text{ under } H_0
$$

## Decision Rule (two-sided)

Reject $H_0$ if $|Z| > z_{\alpha/2}$.
