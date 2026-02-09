"""
Reproducing Linear Regression Output from Scratch
==================================================
Model: Sales ~ TV + Radio + Newspaper (Advertising dataset)
Computes coefficients, standard errors, t-statistics, p-values,
and 95% confidence intervals using the Normal Equation.
"""

import numpy as np
import pandas as pd
from scipy import stats


def load_and_split_data(url, test_ratio=0.3):
    """Load dataset and perform train/test split."""
    data = pd.read_csv(url, usecols=[1, 2, 3, 4])
    n_total = data.shape[0]
    n_train = int(n_total * (1 - test_ratio))
    return data.iloc[:n_train]


def fit_ols(X, y):
    """
    Fit OLS regression via the Normal Equation.

    Parameters
    ----------
    X : ndarray of shape (n, p+1)
        Design matrix with intercept column.
    y : ndarray of shape (n, 1)
        Response vector.

    Returns
    -------
    beta_hat : ndarray of shape (p+1, 1)
        Estimated coefficients.
    s : float
        Residual standard error.
    cov_matrix : ndarray of shape (p+1, p+1)
        (X'X)^{-1} matrix.
    """
    n, k = X.shape
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    residuals = y - y_hat
    s = np.sqrt(np.sum(residuals ** 2) / (n - k))
    cov_matrix = np.linalg.inv(X.T @ X)
    return beta_hat, s, cov_matrix


def regression_table(beta_hat, s, cov_matrix, n, k, var_names):
    """
    Print a formatted regression summary table.

    Parameters
    ----------
    beta_hat : ndarray
        Estimated coefficients.
    s : float
        Residual standard error.
    cov_matrix : ndarray
        (X'X)^{-1} matrix.
    n : int
        Number of observations.
    k : int
        Number of parameters (including intercept).
    var_names : list of str
        Variable names.
    """
    df = n - k
    t_crit = stats.t(df).ppf(0.975)

    print("=" * 100)
    print(f"{'':10}    {'coef':>10} {'std err':>10} "
          f"{'t':>10} {'P>|t|':>10} "
          f"{'[0.025':>10} {'0.975]':>10}")
    print("-" * 100)

    for name, j in zip(var_names, range(k)):
        coef = beta_hat[j, 0]
        v_j = cov_matrix[j, j]
        se = s * np.sqrt(v_j)
        t_stat = coef / se
        p_val = 2 * stats.t(df).sf(np.abs(t_stat))
        ci_lo = coef - t_crit * se
        ci_hi = coef + t_crit * se
        print(f"{name:10}    {coef:10.4f} {se:10.3f} "
              f"{t_stat:10.3f} {p_val:10.3f} "
              f"{ci_lo:10.3f} {ci_hi:10.3f}")

    print("=" * 100)
    print(f"\nResidual standard error: {s:.4f}")
    print(f"Degrees of freedom: {df}")


def main():
    url = ('https://raw.githubusercontent.com/justmarkham/'
           'scikit-learn-videos/master/data/Advertising.csv')

    training_data = load_and_split_data(url)

    # Response and design matrix
    y = np.array(training_data.Sales).reshape(-1, 1)
    n = y.shape[0]
    X = np.concatenate(
        (np.ones((n, 1)), np.array(training_data.iloc[:, :-1])),
        axis=1
    )
    k = X.shape[1]

    # Fit model
    beta_hat, s, cov_matrix = fit_ols(X, y)

    # Display results
    var_names = ["Intercept", "TV", "Radio", "Newspaper"]
    regression_table(beta_hat, s, cov_matrix, n, k, var_names)


if __name__ == "__main__":
    main()
