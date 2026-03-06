"""
Bivariate Normal Distribution — 3-D Surface and Contour
=========================================================
Adapted from Basic-Statistics-With-Python Chapter 2 notebook.

Plots the bivariate normal PDF as a 3-D surface and as a
contour plot, demonstrating how the shape changes with
different covariance structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def bivariate_normal_pdf(mu, cov, grid_range=10, grid_points=200):
    """Evaluate bivariate normal PDF on a meshgrid."""
    x = np.linspace(-grid_range, grid_range, grid_points)
    y = np.linspace(-grid_range, grid_range, grid_points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=mu, cov=cov)
    Z = rv.pdf(pos)
    return X, Y, Z


def main():
    print("=" * 60)
    print("Bivariate Normal Distribution")
    print("=" * 60)

    configs = [
        {"label": "Independent (rho=0)",
         "mu": [0, 0], "cov": [[4, 0], [0, 4]]},
        {"label": "Positive corr (rho=0.7)",
         "mu": [0, 0], "cov": [[4, 2.8], [2.8, 4]]},
        {"label": "Negative corr (rho=-0.7)",
         "mu": [0, 0], "cov": [[4, -2.8], [-2.8, 4]]},
        {"label": "Unequal variances",
         "mu": [0, 0], "cov": [[7, 0], [0, 15]]},
    ]

    fig = plt.figure(figsize=(18, 12))

    for i, cfg in enumerate(configs):
        mu, cov = cfg["mu"], cfg["cov"]
        X, Y, Z = bivariate_normal_pdf(mu, cov)

        # 3D surface
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85,
                        edgecolor="none")
        ax.set_title(cfg["label"], fontsize=9, pad=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("f(x,y)")

        # contour
        ax2 = fig.add_subplot(2, 4, i + 5)
        ax2.contourf(X, Y, Z, levels=20, cmap="viridis")
        ax2.contour(X, Y, Z, levels=8, colors="white", linewidths=0.5)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(cfg["label"], fontsize=9)
        ax2.set_aspect("equal")

        # print info
        rho = cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])
        print(f"\n  {cfg['label']}:")
        print(f"    mu = {mu}")
        print(f"    Sigma = {cov}")
        print(f"    rho = {rho:.2f}")

    plt.suptitle("Bivariate Normal: 3-D Surface (top) and Contour (bottom)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bivariate_normal.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: bivariate_normal.png")


if __name__ == "__main__":
    main()
