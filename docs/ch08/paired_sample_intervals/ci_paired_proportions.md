# Paired Interval for Proportions (McNemar)

## Overview

McNemar's test and the associated confidence interval address paired binary data, such as before/after treatment comparisons on the same subjects.

## Setup

For paired binary outcomes, we construct a $2 \times 2$ table of concordant and discordant pairs. The CI focuses on the discordant pairs $b$ and $c$:

$$
\hat{p}_1 - \hat{p}_2 = \frac{b - c}{n}
$$

## McNemar's CI

The confidence interval uses the standard error based on discordant pairs.
