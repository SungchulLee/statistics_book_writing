# MoM for Common Distributions

## Overview

We derive Method of Moments estimators for several standard distributions.

## Normal Distribution

Matching the first two moments:

$$
\hat{\mu} = \bar{X}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

## Gamma Distribution

For $X \sim \text{Gamma}(\alpha, \beta)$ with $E[X] = \alpha\beta$ and $\text{Var}(X) = \alpha\beta^2$:

$$
\hat{\beta} = \frac{S^2}{\bar{X}}, \quad \hat{\alpha} = \frac{\bar{X}}{\hat{\beta}} = \frac{\bar{X}^2}{S^2}
$$

## Beta Distribution

For $X \sim \text{Beta}(a, b)$:

$$
\hat{a} = \bar{X}\left(\frac{\bar{X}(1-\bar{X})}{S^2} - 1\right), \quad \hat{b} = (1 - \bar{X})\left(\frac{\bar{X}(1-\bar{X})}{S^2} - 1\right)
$$
