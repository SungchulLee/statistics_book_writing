# Two-Sample t-Test (Pooled and Welch)

## Overview

Tests whether two population means differ when variances are unknown.

## Pooled t-Test (Equal Variances)

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}}, \quad S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}
$$

## Welch's t-Test (Unequal Variances)

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

with Satterthwaite degrees of freedom. Welch's test is generally preferred as it does not assume equal variances.
