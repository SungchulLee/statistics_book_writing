# Robust Variance Estimators (MAD, IQR-based)

## Overview

When data contains outliers, classical variance estimators can be severely affected. Robust alternatives include the MAD and IQR-based estimators.

## Median Absolute Deviation (MAD)

$$
\text{MAD} = \text{Median}(|X_i - \text{Median}(X)|)
$$

For Normal data, $\sigma \approx 1.4826 \cdot \text{MAD}$.

## IQR-based Estimator

$$
\hat{\sigma}_{IQR} = \frac{\text{IQR}}{1.349}
$$

## Comparison

| Estimator | Breakdown Point | Efficiency (Normal) |
|---|---|---|
| Sample variance | 0% | 100% |
| MAD | 50% | 37% |
| IQR-based | 25% | 37% |
