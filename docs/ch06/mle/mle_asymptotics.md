# Asymptotic Properties of MLE

## Overview

Under regularity conditions, MLE possesses three key asymptotic properties.

## Consistency

$$
\hat{\theta}_{MLE} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty
$$

## Asymptotic Normality

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{I(\theta_0)}\right)
$$

## Asymptotic Efficiency

The MLE achieves the Cramér–Rao lower bound asymptotically, meaning no other consistent estimator has smaller asymptotic variance.

## Regularity Conditions

The above results require: (1) identifiability, (2) common support, (3) $\theta_0$ is an interior point, (4) smoothness of the log-likelihood.
