"""
Confidence and Prediction Intervals for Simple Linear Regression
================================================================
Generates synthetic data, fits OLS, and plots 95% confidence intervals
for E[y|x] and prediction intervals for y|x.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def generate_data(n, sigma, seed=0):
    """
    Generate synthetic linear regression data: y = 1 + 2x + noise.

    Parameters
    ----------
    n : int
        Number of observations.
    sigma : float
        Standard deviation of the noise term.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    x, y : ndarray of shape (n, 1)
        Predictor and response arrays.
    """
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    y = 1 + 2 * x + sigma * np.random.randn(n, 1)
    return x, y


def estimate_regression_line(x, y):
    """
    Estimate the regression line using the correlation formula.

    Returns
    -------
    y_hat : ndarray
        Fitted values.
    beta_hat : float
        Estimated slope.
    y_bar, x_bar : float
        Sample means of y and x.
    """
    x_bar = x.mean()
    y_bar = y.mean()
    s_x = x.std(ddof=1)
    s_y = y.std(ddof=1)
    r = np.corrcoef(np.concatenate([x, y], axis=1), rowvar=False)[1, 0]
    beta_hat = r * s_y / s_x
    y_hat = beta_hat * (x - x_bar) + y_bar
    return y_hat, beta_hat, y_bar, x_bar


def calculate_residual_variance(y, y_hat, n):
    """
    Compute the unbiased residual variance s² and standard deviation s.

    Parameters
    ----------
    y : ndarray
        Observed responses.
    y_hat : ndarray
        Fitted values.
    n : int
        Number of observations.

    Returns
    -------
    s_square : float
        Residual variance estimate.
    s : float
        Residual standard deviation.
    """
    s_square = np.sum((y - y_hat) ** 2) / (n - 2)
    s = np.sqrt(s_square)
    return s_square, s


def confidence_intervals(x, y_hat, beta_hat, x_bar, y_bar, n, s):
    """
    Compute 95% confidence and prediction intervals.

    Returns
    -------
    x0 : ndarray
        Grid of x values.
    lower, upper : ndarray
        Bounds for the mean response CI.
    lower2, upper2 : ndarray
        Bounds for the prediction interval.
    """
    x0 = np.linspace(x.min(), x.max(), 20)
    y0_hat = beta_hat * (x0 - x_bar) + y_bar
    t_val = stats.t(n - 2).ppf(0.975)
    ss_x = np.sum((x - x_bar) ** 2)

    # CI for E[y | x = x0]
    margin = t_val * s * np.sqrt((1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower = y0_hat - margin
    upper = y0_hat + margin

    # PI for y | x = x0
    margin2 = t_val * s * np.sqrt(1 + (1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower2 = y0_hat - margin2
    upper2 = y0_hat + margin2

    return x0, lower, upper, lower2, upper2


def plot_intervals(x, y, y_hat, x0, lower, upper, lower2, upper2):
    """Plot confidence and prediction bands side by side."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    # Confidence interval for E[y]
    ax0.plot(x, y, 'o', alpha=0.5)
    ax0.plot(x, y_hat, '--b', label='Fitted line')
    ax0.plot(x0, upper, '--r', label='95% CI')
    ax0.plot(x0, lower, '--r')
    ax0.set_title(r'95% Confidence Interval for $E[y]$')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.legend()

    # Prediction interval for y
    ax1.plot(x, y, 'o', alpha=0.5)
    ax1.plot(x, y_hat, '--b', label='Fitted line')
    ax1.plot(x0, upper2, '--r', label='95% PI')
    ax1.plot(x0, lower2, '--r')
    ax1.set_title(r'95% Prediction Interval for $y$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()

    plt.tight_layout()
    plt.savefig('ci_pi_bands.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # Parameters
    n = 100
    sigma = 3

    # Generate data (true model: y = 1 + 2x + N(0, sigma²))
    x, y = generate_data(n, sigma)

    # Fit regression
    y_hat, beta_hat, y_bar, x_bar = estimate_regression_line(x, y)

    # Residual variance
    s_square, s = calculate_residual_variance(y, y_hat, n)
    print(f"True sigma^2: {sigma**2}")
    print(f"Estimated s^2: {s_square:.4f}")

    # Compute intervals
    x0, lower, upper, lower2, upper2 = confidence_intervals(
        x, y_hat, beta_hat, x_bar, y_bar, n, s
    )

    # Plot
    plot_intervals(x, y, y_hat, x0, lower, upper, lower2, upper2)


if __name__ == "__main__":
    main()
