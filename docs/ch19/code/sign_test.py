"""
Sign Test for Paired Observations

Tests whether the median difference between paired observations
is zero, using only the signs (directions) of the differences.
Ties (zero differences) are excluded.
"""

import argparse
import numpy as np
import scipy.stats as stats


def load_student_data():
    """Pre/post treatment scores for 15 students."""
    return np.array([
        [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
        [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
        [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
    ])


def sign_test(paired_data, test_type="two-sided"):
    """
    Sign test for paired observations.

    Parameters
    ----------
    paired_data : ndarray of shape (n, 2)
        Column 0 is post-treatment, column 1 is pre-treatment.
    test_type : str
        One of "less", "two-sided", "greater".

    Returns
    -------
    z : float
        The Z test statistic.
    p_value : float
    """
    p_0, q_0 = 0.5, 0.5

    n_plus = np.sum(paired_data[:, 0] > paired_data[:, 1])
    n_minus = np.sum(paired_data[:, 0] < paired_data[:, 1])
    n = n_plus + n_minus  # ties excluded
    p_hat = n_plus / n

    z = (p_hat - p_0) / np.sqrt(p_0 * q_0 / n)

    if test_type == "less":
        p_value = stats.norm.cdf(z)
    elif test_type == "two-sided":
        p_value = 2 * stats.norm.cdf(-abs(z))
    elif test_type == "greater":
        p_value = stats.norm.sf(z)

    return z, p_value


def main():
    parser = argparse.ArgumentParser(description="Sign Test")
    parser.add_argument("--test_type", type=str, default="two-sided",
                        choices=["less", "two-sided", "greater"],
                        help="alternative hypothesis type")
    args = parser.parse_args()

    paired_data = load_student_data()
    z, p_value = sign_test(paired_data, test_type=args.test_type)

    print(f"Test type: {args.test_type}")
    print(f"{z       = :.4f}")
    print(f"{p_value = :.4f}")


if __name__ == "__main__":
    main()
