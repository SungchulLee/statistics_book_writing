"""
Non-Parametric Tests for Two-Sample and Multi-Group Comparisons

Includes:
- Wilcoxon Rank-Sum Test (two independent samples, no ties)
- Wilcoxon Signed-Rank Test (paired samples)
- Mann-Whitney U Test (two independent samples, handles ties)
- Kruskal-Wallis H Test (k independent samples)
- Mood's Median Test (k independent samples)
"""

import numpy as np
import scipy.stats as stats


def load_student_data():
    """Pre/post treatment scores for 15 students."""
    return np.array([
        [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
        [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
        [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
    ])


def main():
    paired_data = load_student_data()

    # --- Wilcoxon Rank-Sum Test ---
    print("=" * 50)
    print("Wilcoxon Rank-Sum Test (two independent samples)")
    print("=" * 50)
    statistic, p_value = stats.ranksums(
        paired_data[:, 0], paired_data[:, 1],
        alternative="two-sided",
    )
    print(f"{statistic = :.4f}")
    print(f"{p_value   = :.2%}")
    print()

    # --- Wilcoxon Signed-Rank Test ---
    print("=" * 50)
    print("Wilcoxon Signed-Rank Test (paired samples)")
    print("=" * 50)
    statistic, p_value = stats.wilcoxon(
        paired_data[:, 0], paired_data[:, 1],
        alternative="two-sided",
        mode="approx",
        zero_method="pratt"
    )
    print(f"{statistic = }")
    print(f"{p_value   = :.4f}")
    print()

    # --- Mann-Whitney U Test ---
    print("=" * 50)
    print("Mann-Whitney U Test")
    print("=" * 50)
    data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    statistic, p_value = stats.mannwhitneyu(data0, data1)
    print(f"{statistic = }")
    print(f"{p_value   = :.02%}")
    print()

    # --- Kruskal-Wallis H Test ---
    print("=" * 50)
    print("Kruskal-Wallis H Test (k independent samples)")
    print("=" * 50)
    data2 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    statistic, p_value = stats.kruskal(data0, data1, data2)
    print(f"{statistic = :.4f}")
    print(f"{p_value   = :.2%}")
    print()

    # --- Mood's Median Test ---
    print("=" * 50)
    print("Mood's Median Test")
    print("=" * 50)
    result = stats.median_test(data0, data1, data2)
    print(f"Grand median = {result.median}")
    print(f"Contingency table:\n{result.table}")
    print(f"p-value = {result.pvalue:.4f}")


if __name__ == "__main__":
    main()
