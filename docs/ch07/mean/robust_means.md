# Trimmed and Winsorized Means

## Overview

When data contains outliers or comes from heavy-tailed distributions, robust alternatives to the sample mean may perform better.

## Trimmed Mean

The $\alpha$-trimmed mean removes the smallest and largest $\alpha$ fraction of observations:

$$
\bar{X}_{\text{trim}(\alpha)} = \frac{1}{n - 2\lfloor n\alpha \rfloor} \sum_{i=\lfloor n\alpha \rfloor + 1}^{n - \lfloor n\alpha \rfloor} X_{(i)}
$$

## Winsorized Mean

The Winsorized mean replaces extreme values with the nearest non-trimmed value instead of removing them.

## Comparison

| Estimator | Robustness | Efficiency (Normal) |
|---|---|---|
| Sample mean | Low (breakdown = 0) | 100% |
| 10% trimmed mean | Moderate | ~97% |
| Median (50% trim) | High (breakdown = 50%) | ~64% |
