# Family-Wise Error Rate

## Overview

When performing multiple tests simultaneously, the probability of at least one false rejection increases.

$$
\text{FWER} = P(\text{at least one false rejection}) = 1 - (1-\alpha)^m \approx m\alpha
$$

for $m$ independent tests. Controlling FWER is essential in multiple testing scenarios.
