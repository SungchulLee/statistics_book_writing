# Efficiency of the Sample Mean

## Overview

The sample mean $\bar{X}$ achieves the Cramér–Rao lower bound for estimating the mean of a Normal distribution, making it the most efficient unbiased estimator.

## CRLB for the Normal Mean

For $X \sim N(\mu, \sigma^2)$, the Fisher information for $\mu$ is $I(\mu) = 1/\sigma^2$, giving:

$$
\text{Var}(\hat{\mu}) \geq \frac{1}{nI(\mu)} = \frac{\sigma^2}{n} = \text{Var}(\bar{X})
$$

## Efficiency Under Non-Normality

For non-normal distributions, the sample mean may not be efficient. The **asymptotic relative efficiency (ARE)** compares estimators:

$$
\text{ARE}(\bar{X}, \text{Median}) = \frac{\text{Var}(\text{Median})}{\text{Var}(\bar{X})} = \frac{\pi}{2} \approx 1.57 \quad \text{(for Normal)}
$$
