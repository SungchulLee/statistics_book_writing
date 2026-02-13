# Method of Moments Foundations

## Overview

The **Method of Moments (MoM)** equates population moments to sample moments to estimate parameters.

## Definition

For a distribution with $p$ parameters $\theta_1, \ldots, \theta_p$, the MoM sets:

$$
\mu_k'(\theta_1, \ldots, \theta_p) = m_k' = \frac{1}{n}\sum_{i=1}^n X_i^k, \quad k = 1, \ldots, p
$$

## Properties

- **Consistency**: MoM estimators are consistent under mild conditions
- **Simplicity**: Often yields closed-form solutions
- **Not necessarily efficient**: May have larger variance than MLE
- **May produce inadmissible estimates**: e.g., negative variance estimates
