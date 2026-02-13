# Heavy-Tailed Distributions

## Overview

When the underlying distribution has heavy tails, the sample mean can have poor performance due to extreme observations.

## Characteristics

Heavy-tailed distributions have tails that decay slower than exponential. Examples include Student's $t$ (low df), Cauchy, Pareto, and log-normal distributions.

## Impact on the Sample Mean

- Higher variance of $\bar{X}$ than Normal case
- Slower convergence to normality (CLT)
- Outliers have outsized influence

## Robust Alternatives

For heavy-tailed data, consider:
- Trimmed means
- The median
- Huber's M-estimator
- Winsorized means
