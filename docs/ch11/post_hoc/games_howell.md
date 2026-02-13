# Games–Howell (Unequal Variances)

## Overview

The Games-Howell procedure is a post-hoc test that does not assume equal variances or equal sample sizes.

## Test Statistic

$$
q_{ij} = \frac{\bar{X}_i - \bar{X}_j}{\sqrt{(S_i^2/n_i + S_j^2/n_j)/2}}
$$

with Welch-Satterthwaite degrees of freedom.

## When to Use

Use Games-Howell when Levene's test indicates unequal variances, or when sample sizes are substantially different across groups.
