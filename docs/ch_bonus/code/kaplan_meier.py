#!/usr/bin/env python3
# ======================================================================
# bonus_survival_01_kaplan_meier.py
# ======================================================================
# Kaplan-Meier survival curve estimator and log-rank test for comparing
# two groups.  No external survival-analysis packages required — all
# computations use only NumPy / SciPy.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (10_survival.ipynb).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def kaplan_meier(times, censored):
    """
    Compute the Kaplan-Meier survival function estimate.

    Parameters
    ----------
    times    : 1-d array   Observed times (event or censoring).
    censored : 1-d array   1 = censored (no event), 0 = event observed.

    Returns
    -------
    t_plot : array   Time points for step-plot (includes 0 and max time).
    s_plot : array   Survival probabilities matching t_plot.
    """
    order = np.argsort(times)
    times = times[order]
    censored = censored[order]

    event_times = times[censored == 0]
    unique_events = np.unique(event_times)

    n_total = len(times)
    s = 1.0
    t_list = [0.0]
    s_list = [1.0]

    for t_j in unique_events:
        n_at_risk = np.sum(times >= t_j)
        d_j = np.sum((times == t_j) & (censored == 0))    # events at t_j
        s *= (n_at_risk - d_j) / n_at_risk
        t_list.append(t_j)
        s_list.append(s)

    t_list.append(times.max())
    s_list.append(s_list[-1])

    return np.array(t_list), np.array(s_list)


def logrank_test(times_1, censored_1, times_2, censored_2):
    """
    Two-sample log-rank test.

    Returns
    -------
    chi2   : float   Test statistic (chi-square with 1 df).
    p_value: float   p-value from chi-square(1).
    """
    event_1 = times_1[censored_1 == 0]
    event_2 = times_2[censored_2 == 0]
    all_event_times = np.unique(np.concatenate([event_1, event_2]))

    O1 = 0.0
    E1 = 0.0
    V  = 0.0

    for t_j in all_event_times:
        r1 = np.sum(times_1 >= t_j)
        r2 = np.sum(times_2 >= t_j)
        r  = r1 + r2

        d1 = np.sum(event_1 == t_j)
        d2 = np.sum(event_2 == t_j)
        d  = d1 + d2

        e1 = r1 * d / r if r > 0 else 0
        v  = r1 * r2 * d * (r - d) / (r ** 2 * (r - 1)) if r > 1 else 0

        O1 += d1
        E1 += e1
        V  += v

    chi2 = (O1 - E1) ** 2 / V if V > 0 else 0
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value


def main():
    # ── Simulated survival data for two treatment groups ──
    np.random.seed(0)

    # Group 1: slower event rate
    n1 = 40
    times_1 = np.random.exponential(scale=20, size=n1)
    censored_1 = (np.random.rand(n1) < 0.2).astype(int)

    # Group 2: faster event rate
    n2 = 40
    times_2 = np.random.exponential(scale=12, size=n2)
    censored_2 = (np.random.rand(n2) < 0.2).astype(int)

    # ── Kaplan-Meier curves ──
    t1, s1 = kaplan_meier(times_1, censored_1)
    t2, s2 = kaplan_meier(times_2, censored_2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(t1, s1, where='post', linewidth=2, label='Group 1 (slow)')
    ax.step(t2, s2, where='post', linewidth=2, label='Group 2 (fast)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves')
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # ── Log-rank test ──
    chi2, p = logrank_test(times_1, censored_1, times_2, censored_2)
    print("Log-Rank Test")
    print("=" * 40)
    print(f"  chi-square = {chi2:.4f}")
    print(f"  p-value    = {p:.4f}")
    if p < 0.05:
        print("  -> The two survival curves are significantly different "
              "(alpha = 0.05).")
    else:
        print("  -> No significant difference between survival curves "
              "(alpha = 0.05).")


if __name__ == "__main__":
    main()
