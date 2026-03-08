# Dunnett's Test (vs Control)

## Overview

Dunnett's test compares each of $k-1$ treatment groups to a single control group, while controlling the FWER.

## Test Statistic

For comparing group $i$ to the control:

$$
t_i = \frac{\bar{X}_i - \bar{X}_{\text{control}}}{\sqrt{MSE \cdot (1/n_i + 1/n_{\text{control}})}}
$$

Critical values come from the multivariate $t$-distribution accounting for the correlation between comparisons.
