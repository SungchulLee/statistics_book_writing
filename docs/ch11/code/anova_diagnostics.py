"""
ANOVA Diagnostics - Complete Example

Demonstrates the full ANOVA diagnostic workflow:
1. Fit ANOVA model
2. Check normality (Q-Q plot, Shapiro-Wilk)
3. Check homoscedasticity (Levene's test)
4. Check independence (residual plots)
5. Identify influential points (Cook's distance)
6. Apply alternatives if assumptions are violated

Usage:
    python anova_diagnostics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, levene
from statsmodels.stats.stattools import durbin_watson


def check_normality(model, ax_qq=None, ax_hist=None):
    """Check normality of residuals using Shapiro-Wilk test and Q-Q plot."""
    resid = model.resid

    # Shapiro-Wilk test
    stat, p_value = shapiro(resid)
    print(f"  Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p_value:.4f}")
    print(f"  {'✓ Normality not rejected' if p_value > 0.05 else '✗ Normality rejected'}")

    # Q-Q plot
    if ax_qq is not None:
        sm.qqplot(resid, line='s', ax=ax_qq)
        ax_qq.set_title("Q-Q Plot of Residuals")

    # Histogram
    if ax_hist is not None:
        ax_hist.hist(resid, bins=15, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(resid.min(), resid.max(), 100)
        ax_hist.plot(x, stats.norm(resid.mean(), resid.std()).pdf(x), 'r--', lw=2)
        ax_hist.set_title("Histogram of Residuals")
        ax_hist.set_xlabel("Residuals")

    return stat, p_value


def check_homoscedasticity(data, response_col, group_col):
    """Check homoscedasticity using Levene's test."""
    groups = [
        data[data[group_col] == g][response_col].values
        for g in data[group_col].unique()
    ]

    stat, p_value = levene(*groups)
    print(f"  Levene's Test: F = {stat:.4f}, p-value = {p_value:.4f}")
    print(f"  {'✓ Equal variances not rejected' if p_value > 0.05 else '✗ Equal variances rejected'}")

    # Also run Bartlett's test for comparison
    stat_b, p_value_b = stats.bartlett(*groups)
    print(f"  Bartlett's Test: χ² = {stat_b:.4f}, p-value = {p_value_b:.4f}")

    return stat, p_value


def check_independence(model, ax=None):
    """Check independence using Durbin-Watson test and residual plot."""
    dw = durbin_watson(model.resid)
    print(f"  Durbin-Watson Statistic: {dw:.4f}")
    print(f"  {'✓ No autocorrelation detected' if 1.5 < dw < 2.5 else '⚠ Possible autocorrelation'}")

    if ax is not None:
        ax.scatter(model.fittedvalues, model.resid, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Values")

    return dw


def check_influential_points(model, ax=None):
    """Identify influential points using Cook's distance."""
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(cooks_d)

    n_influential = np.sum(cooks_d > threshold)
    print(f"  Cook's Distance Threshold: {threshold:.4f}")
    print(f"  Influential Points (D > threshold): {n_influential}")

    if ax is not None:
        ax.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.3f}')
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Cook's Distance")
        ax.set_title("Cook's Distance")
        ax.legend()

    return cooks_d


def run_full_diagnostics(data, response_col, group_col, title="ANOVA Diagnostics"):
    """Run complete ANOVA diagnostics pipeline."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    # Fit model
    formula = f'{response_col} ~ {group_col}'
    model = ols(formula, data=data).fit()

    # ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA Table:")
    print(anova_table)

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # 1. Normality
    print("\n1. Normality Check:")
    check_normality(model, ax_qq=axes[0, 0], ax_hist=axes[0, 1])

    # 2. Homoscedasticity
    print("\n2. Homoscedasticity Check:")
    check_homoscedasticity(data, response_col, group_col)

    # 3. Independence
    print("\n3. Independence Check:")
    check_independence(model, ax=axes[1, 0])

    # 4. Influential Points
    print("\n4. Influential Points Check:")
    check_influential_points(model, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    return model


def main():
    # --- Example 1: Iris Dataset ---
    import seaborn as sns
    iris = sns.load_dataset("iris")
    run_full_diagnostics(iris, 'sepal_length', 'species',
                         title="ANOVA: Iris Sepal Length by Species")

    # --- Example 2: Employee Productivity ---
    productivity_data = pd.DataFrame({
        'productivity': [68, 75, 80, 65, 85, 78, 70, 82, 90, 88, 72, 95, 67, 85, 79],
        'environment': ['remote']*5 + ['office']*5 + ['hybrid']*5
    })
    run_full_diagnostics(productivity_data, 'productivity', 'environment',
                         title="ANOVA: Productivity by Environment")

    # --- Example 3: Customer Satisfaction ---
    satisfaction_data = pd.DataFrame({
        'satisfaction': [4.5, 3.8, 4.7, 4.2, 4.9, 4.1, 3.5, 4.3, 4.8, 3.9,
                         4.4, 4.0, 3.7, 4.2, 4.6, 4.8, 3.6, 4.3, 4.1, 4.7],
        'location': ['A']*5 + ['B']*5 + ['C']*5 + ['D']*5
    })
    run_full_diagnostics(satisfaction_data, 'satisfaction', 'location',
                         title="ANOVA: Customer Satisfaction by Location")


if __name__ == "__main__":
    main()
