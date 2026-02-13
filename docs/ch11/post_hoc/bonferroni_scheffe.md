# Bonferroni and Scheffé Methods

## Overview

## Bonferroni Method

Perform each pairwise comparison at $\alpha/C$ where $C = \binom{k}{2}$ is the number of pairs. Simple and applicable to any set of comparisons.

## Scheffé's Method

Controls the FWER for ALL possible contrasts (not just pairwise). Uses the critical value $\sqrt{(k-1)F_{\alpha,k-1,N-k}}$.

## Comparison

- Bonferroni: tighter for a small number of planned comparisons
- Scheffé: better when exploring all possible contrasts
