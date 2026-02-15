#!/usr/bin/env python3
# ======================================================================
# 07_robust_01_trimmed_weighted_mad.py
# ======================================================================
# Robust estimators of location and scale:
#   1. Trimmed mean (symmetric trim).
#   2. Weighted mean and weighted median.
#   3. Median Absolute Deviation (MAD).
#   4. Comparison with outlier-contaminated data.
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 1 — Exploratory Data Analysis).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def trimmed_mean(data, proportion=0.1):
    """
    Symmetric trimmed mean — drop the lowest and highest *proportion*
    of observations, then compute the mean of the remainder.
    """
    x = np.sort(data)
    n = len(x)
    k = int(np.floor(n * proportion))
    if k == 0:
        return x.mean()
    return x[k:-k].mean()


def weighted_mean(data, weights):
    """Weighted arithmetic mean."""
    return np.sum(data * weights) / np.sum(weights)


def weighted_median(data, weights):
    """Weighted median (smallest value where cumulative weight >= 0.5)."""
    order = np.argsort(data)
    sorted_data = data[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w) / np.sum(sorted_w)
    idx = np.searchsorted(cum_w, 0.5)
    return sorted_data[idx]


def mad(data):
    """Median Absolute Deviation (MAD)."""
    med = np.median(data)
    return np.median(np.abs(data - med))


def main():
    print("Robust Estimators")
    print("=" * 55)

    # ── Clean data ──
    clean = np.random.normal(loc=50, scale=10, size=100)

    # ── Contaminated data: add 5 extreme outliers ──
    outliers = np.array([200, 250, 300, -100, -150])
    contaminated = np.concatenate([clean, outliers])

    # ── 1. Location estimators ──
    print("\n1. Location Estimators")
    print("-" * 40)
    for label, data in [("Clean", clean), ("Contaminated", contaminated)]:
        m = data.mean()
        med = np.median(data)
        tm10 = trimmed_mean(data, 0.10)
        tm20 = trimmed_mean(data, 0.20)
        print(f"\n   {label} data (n = {len(data)}):")
        print(f"     Mean            = {m:.2f}")
        print(f"     Median          = {med:.2f}")
        print(f"     Trimmed mean 10% = {tm10:.2f}")
        print(f"     Trimmed mean 20% = {tm20:.2f}")

    # ── 2. Weighted mean and median ──
    values = np.array([10, 20, 30, 40, 50])
    weights = np.array([1.0, 1.0, 3.0, 5.0, 2.0])
    wm = weighted_mean(values, weights)
    wmed = weighted_median(values, weights)
    print(f"\n2. Weighted Estimators")
    print(f"   Values  = {values}")
    print(f"   Weights = {weights}")
    print(f"   Weighted mean   = {wm:.2f}")
    print(f"   Weighted median = {wmed:.2f}")

    # ── 3. Scale estimators ──
    print(f"\n3. Scale Estimators")
    print("-" * 40)
    for label, data in [("Clean", clean), ("Contaminated", contaminated)]:
        s = data.std(ddof=1)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        m = mad(data)
        print(f"\n   {label} data:")
        print(f"     Std dev (s) = {s:.2f}")
        print(f"     IQR         = {iqr:.2f}")
        print(f"     MAD         = {m:.2f}")

    # ── Plot: effect of outliers on estimators ──
    # Progressively add outliers and track estimator values
    n_out_range = range(0, 21)
    means, medians, trims = [], [], []
    stds, iqrs, mads_list = [], [], []

    for n_out in n_out_range:
        extra = np.full(n_out, 300.0)
        data = np.concatenate([clean, extra])
        means.append(data.mean())
        medians.append(np.median(data))
        trims.append(trimmed_mean(data, 0.10))
        stds.append(data.std(ddof=1))
        iqrs.append(np.percentile(data, 75) - np.percentile(data, 25))
        mads_list.append(mad(data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(list(n_out_range), means, 'o-', label='Mean', markersize=4)
    ax1.plot(list(n_out_range), medians, 's-', label='Median', markersize=4)
    ax1.plot(list(n_out_range), trims, 'D-', label='Trimmed Mean (10%)', markersize=4)
    ax1.set_xlabel('Number of outliers added (value = 300)')
    ax1.set_ylabel('Estimated location')
    ax1.set_title('Location Estimators vs Outlier Count')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(list(n_out_range), stds, 'o-', label='Std Dev', markersize=4)
    ax2.plot(list(n_out_range), iqrs, 's-', label='IQR', markersize=4)
    ax2.plot(list(n_out_range), mads_list, 'D-', label='MAD', markersize=4)
    ax2.set_xlabel('Number of outliers added (value = 300)')
    ax2.set_ylabel('Estimated scale')
    ax2.set_title('Scale Estimators vs Outlier Count')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
