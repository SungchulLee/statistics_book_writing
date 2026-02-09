"""
Confidence Interval for the Slope of a Simple Linear Regression
===============================================================
Example: Study Hours vs Caffeine Consumption (n=20 students)
"""

from scipy import stats


def main():
    # Regression output
    beta_1_hat = 0.164      # estimated slope
    standard_error = 0.057  # SE of the slope

    # Sample size and degrees of freedom
    n = 20
    df = n - 2

    # 95% confidence interval
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_star = stats.t(df).ppf(1 - alpha / 2)

    margin_of_error = t_star * standard_error

    ci_lower = beta_1_hat - margin_of_error
    ci_upper = beta_1_hat + margin_of_error

    print(f"Slope estimate: {beta_1_hat:.4f}")
    print(f"Standard error: {standard_error:.4f}")
    print(f"t* (df={df}): {t_star:.4f}")
    print(f"Margin of error: {margin_of_error:.4f}")
    print(f"\n{confidence_level:.0%} confidence interval of the slope")
    print(f"{beta_1_hat:.4f} \u00b1 {margin_of_error:.4f}")
    print(f"({ci_lower:.4f}, {ci_upper:.4f})")


if __name__ == "__main__":
    main()
