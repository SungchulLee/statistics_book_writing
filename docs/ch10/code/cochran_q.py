#!/usr/bin/env python3
# ======================================================================
# 17_cochran_q_test.py
# ======================================================================
# Cochran's Q test — a generalisation of McNemar's test to k ≥ 2
# related dichotomous outcomes measured on the same subjects.
#
# Source:  Adapted from *Introduction to Statistics with Python*
#          (9_testsCategorical.ipynb).
# ======================================================================

import numpy as np
from scipy import stats

def cochran_q(data):
    """
    Perform Cochran's Q test.

    Parameters
    ----------
    data : array-like, shape (n_subjects, k_conditions)
        Binary (0/1) matrix.  Each row is a subject, each column a
        condition / task.

    Returns
    -------
    Q       : float   Cochran's Q statistic
    p_value : float   p-value from chi-square(k-1) approximation
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape

    # Row and column totals
    T_j = data.sum(axis=0)          # column totals (successes per condition)
    L_i = data.sum(axis=1)          # row totals   (successes per subject)

    grand_T = T_j.sum()

    numerator = (k - 1) * (k * np.sum(T_j ** 2) - grand_T ** 2)
    denominator = k * grand_T - np.sum(L_i ** 2)

    Q = numerator / denominator
    p_value = stats.chi2(k - 1).sf(Q)
    return Q, p_value


def main():
    # ── Example: 12 subjects rated on 3 tasks (pass=1, fail=0) ──
    tasks = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])

    print(f"Data matrix  ({tasks.shape[0]} subjects × {tasks.shape[1]} tasks):")
    print(tasks)
    print()

    Q, p = cochran_q(tasks)

    print(f"Cochran's Q = {Q:.4f}")
    print(f"p-value     = {p:.4f}")
    print()

    if p < 0.05:
        print("→ Reject H₀: the proportion of successes differs "
              "significantly across the tasks (α = 0.05).")
    else:
        print("→ Fail to reject H₀: no significant difference "
              "in success rates across tasks (α = 0.05).")


if __name__ == "__main__":
    main()
