#!/usr/bin/env python3
# ======================================================================
# correlation_ellipses.py
# ======================================================================
# Visualize correlation matrices using ellipses, ideal for grayscale or
# print publication. Ellipse width and orientation encode the correlation
# strength and direction.
#
# This visualization is particularly useful for:
# - Print publications (no color required)
# - Accessibility (color-blind friendly)
# - Showing both magnitude and direction simultaneously
#
# Adapted from: https://stackoverflow.com/a/34558488
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


def plot_corr_ellipses(data, figsize=None, **kwargs):
    """
    Create a correlation matrix visualization using ellipses.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Correlation matrix (typically from DataFrame.corr())
    figsize : tuple, optional
        Figure size as (width, height)
    **kwargs : dict
        Additional arguments passed to EllipseCollection (e.g., cmap)

    Returns
    -------
    ec : EllipseCollection
        The ellipse collection object
    ax : matplotlib.axes.Axes
        The axes object

    Notes
    -----
    Ellipse width represents positive correlation (circle when r ≈ 1),
    height represents negative correlation. A perfect positive correlation
    shows a horizontal ellipse; perfect negative shows vertical.
    """
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # Set the relative sizes of the major/minor axes according to the strength
    # of the positive/negative correlation.
    # Width: 1 + 0.01 for perfect positive, near 1.01 for uncorrelated
    # Height: inversely related to absolute correlation strength
    w = np.ones_like(M).ravel() + 0.01  # Start at ~1.01
    h = 1 - np.abs(M).ravel() - 0.01    # Strong corr → small height
    a = 45 * np.sign(M).ravel()         # Rotation: +45° for positive, -45° for negative

    # Create EllipseCollection
    ec = EllipseCollection(
        widths=w, heights=h, angles=a,
        units='x', offsets=xy,
        norm=Normalize(vmin=-1, vmax=1),
        transOffset=ax.transData,
        array=M.ravel(),
        **kwargs
    )
    ax.add_collection(ec)

    # If data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec, ax


def main():
    """
    Example: Correlation ellipses for S&P 500 ETFs.

    This demonstrates how ellipse visualization reveals correlation structure
    without relying on color, making it suitable for print and accessibility.
    """

    # ── Generate sample correlated data ──
    # In practice, you would load actual market data:
    # sp500_sym = pd.read_csv('sp500_sectors.csv')
    # sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)
    # etfs = sp500_px.loc[sp500_px.index > '2012-07-01',
    #                     sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]
    # corr_matrix = etfs.corr()

    # For demonstration, create synthetic correlated data
    np.random.seed(42)
    n = 200
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)

    # Create correlated variables
    x1 = z1
    x2 = 0.8 * z1 + 0.2 * z2          # Strong positive with x1
    x3 = -0.6 * z1 + np.random.randn(n) * 0.7  # Negative with x1
    x4 = 0.3 * z1 + 0.7 * z2          # Moderate positive with x1
    x5 = np.random.randn(n)            # Independent

    data = np.column_stack([x1, x2, x3, x4, x5])
    df = pd.DataFrame(data, columns=['Tech (QQQ)', 'Finance (XLF)',
                                      'Utilities (XLU)', 'Energy (XLE)',
                                      'Commodity (USO)'])

    corr_matrix = df.corr()

    # ── Plot 1: Ellipse visualization (grayscale) ──
    fig, ax = plt.subplots(figsize=(6, 5))
    ec, ax = plot_corr_ellipses(corr_matrix, figsize=(6, 5), cmap='bwr_r')
    cb = fig.colorbar(ec, ax=ax, label='Correlation Coefficient')
    ax.set_title('Correlation Matrix: Ellipse Visualization\n(Suitable for Print)')
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Color-coded for comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ellipse version
    ec, ax1 = plot_corr_ellipses(corr_matrix, cmap='bwr_r')
    ax1.set_title('Ellipse Visualization\n(Grayscale-friendly)')

    # Heatmap version for comparison
    import seaborn as sns
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='bwr_r',
                vmin=-1, vmax=1, ax=ax2, square=True, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Heatmap Visualization\n(Color-based)')

    plt.tight_layout()
    plt.show()

    # ── Print correlation statistics ──
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    print("\nEllipse Interpretation Guide:")
    print("- Width: Positive correlation (wider = stronger positive)")
    print("- Height: Negative correlation (taller = stronger negative)")
    print("- Rotation: +45° (positive), -45° (negative)")
    print("- Circle: Weak/no correlation")


if __name__ == "__main__":
    main()
