# Bonferroni and Holm Corrections

## Overview

## Bonferroni Correction

Reject $H_i$ if $p_i \leq \alpha/m$. Simple but conservative.

## Holm's Step-Down Procedure

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Reject $H_{(i)}$ if $p_{(i)} \leq \alpha/(m-i+1)$ for all prior hypotheses
3. Stop at first non-rejection

Holm's method is uniformly more powerful than Bonferroni while still controlling FWER.
