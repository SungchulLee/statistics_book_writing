# Family-Wise Error Rate


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

When performing multiple tests simultaneously, the probability of at least one false rejection increases.

$$
\text{FWER} = P(\text{at least one false rejection}) = 1 - (1-\alpha)^m \approx m\alpha
$$

for $m$ independent tests. Controlling FWER is essential in multiple testing scenarios.
