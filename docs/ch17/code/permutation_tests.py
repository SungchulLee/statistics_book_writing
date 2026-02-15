#!/usr/bin/env python3
# ======================================================================
# 25_permutation_01_two_sample_and_anova.py
# ======================================================================
# Permutation tests for comparing groups:
#   1. Two-sample permutation test (difference of means).
#   2. Multi-group permutation test (ANOVA-like).
#   3. Permutation test for proportions (A/B test).
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 3 — Statistical Experiments and Significance Testing).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


def perm_test_two_sample(x, y, n_perm=5000):
    """Two-sample permutation test for difference of means."""
    obs_diff = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n_x = len(x)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_x].mean() - pooled[n_x:].mean()
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, perm_diffs


def perm_test_multi_group(groups, n_perm=5000):
    """Multi-group permutation test (variance of group means)."""
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    obs_var = np.var([g.mean() for g in groups])
    perm_vars = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        idx = 0
        means = []
        for s in sizes:
            means.append(pooled[idx:idx + s].mean())
            idx += s
        perm_vars[i] = np.var(means)
    p_value = np.mean(perm_vars >= obs_var)
    return obs_var, p_value, perm_vars


def perm_test_proportion(n_a, conv_a, n_b, conv_b, n_perm=5000):
    """Permutation test for two proportions (A/B test)."""
    obs_diff = conv_a / n_a - conv_b / n_b
    pooled = np.zeros(n_a + n_b)
    pooled[:conv_a + conv_b] = 1
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, perm_diffs


def main():
    print("Permutation Tests")
    print("=" * 55)

    # ── 1. Two-sample test ──
    page_a = np.random.normal(120, 30, size=36)
    page_b = np.random.normal(135, 30, size=40)
    diff, p, perms = perm_test_two_sample(page_a, page_b)
    print(f"\n1. Two-sample permutation test")
    print(f"   Observed diff = {diff:.2f},  p-value = {p:.4f}")

    # ── 2. Multi-group test (4 groups) ──
    g1 = np.random.normal(160, 25, 30)
    g2 = np.random.normal(170, 25, 30)
    g3 = np.random.normal(155, 25, 30)
    g4 = np.random.normal(180, 25, 30)
    var_obs, p_multi, perm_vars = perm_test_multi_group([g1, g2, g3, g4])
    print(f"\n2. Multi-group permutation test (4 groups)")
    print(f"   Observed variance of means = {var_obs:.2f},  p-value = {p_multi:.4f}")

    # ── 3. Proportion test (A/B) ──
    diff_ab, p_ab, perms_ab = perm_test_proportion(23739, 200, 22588, 182)
    print(f"\n3. A/B proportion test")
    print(f"   Observed diff = {100 * diff_ab:.4f}%,  p-value = {p_ab:.4f}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(perms, bins=40, edgecolor='k', alpha=0.7)
    axes[0].axvline(diff, color='black', linewidth=2)
    axes[0].set_title(f'Two-Sample (p={p:.3f})')
    axes[0].set_xlabel('Difference of means')

    axes[1].hist(perm_vars, bins=40, edgecolor='k', alpha=0.7)
    axes[1].axvline(var_obs, color='black', linewidth=2)
    axes[1].set_title(f'Multi-Group (p={p_multi:.3f})')
    axes[1].set_xlabel('Variance of group means')

    axes[2].hist(100 * perms_ab, bins=40, edgecolor='k', alpha=0.7)
    axes[2].axvline(100 * diff_ab, color='black', linewidth=2)
    axes[2].set_title(f'A/B Proportion (p={p_ab:.3f})')
    axes[2].set_xlabel('Difference in rate (%)')

    for ax in axes:
        ax.set_ylabel('Frequency')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
