"""
Bivariate Gaussian — Eigendecomposition of the Covariance Matrix
=================================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

The covariance matrix Sigma of a Gaussian random vector can be
decomposed as Sigma = U * D * U^T, where:
  - U = matrix of eigenvectors (principal directions)
  - D = diagonal matrix of eigenvalues (directional variances)

The eigenvectors point along the axes of the probability ellipses,
and sqrt(eigenvalue) gives the standard deviation in each direction.

Demonstrates:
1. 3D surface of the bivariate Gaussian PDF
2. Contour plot with eigenvector arrows
3. Multiple covariance structures compared
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Main
# =============================================================================


def bivariate_normal_pdf(X, Y, inv_Sigma, det_Sigma):
    """Evaluate bivariate Gaussian PDF on a meshgrid."""
    return (np.exp(-(inv_Sigma[0, 0] * X**2
                     + 2 * inv_Sigma[0, 1] * X * Y
                     + inv_Sigma[1, 1] * Y**2) / 2)
            / (2 * np.pi * np.sqrt(det_Sigma)))


def main():
    print("=" * 60)
    print("Bivariate Gaussian — Eigendecomposition of Covariance")
    print("=" * 60)

    configs = [
        {"label": "Sigma = [[0.5, 0.3], [0.3, 0.5]]",
         "Sigma": np.array([[0.5, 0.3], [0.3, 0.5]])},
        {"label": "Sigma = [[1.0, 0.0], [0.0, 0.3]]",
         "Sigma": np.array([[1.0, 0.0], [0.0, 0.3]])},
        {"label": "Sigma = [[0.2, 0.14], [0.14, 0.8]]",
         "Sigma": np.array([[0.2, 0.14], [0.14, 0.8]])},
    ]

    x = np.linspace(-2.5, 2.5, 200)
    y = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(len(configs), 2,
                             figsize=(12, 5 * len(configs)))

    for i, cfg in enumerate(configs):
        Sigma = cfg["Sigma"]
        inv_Sigma = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

        Z = bivariate_normal_pdf(X, Y, inv_Sigma, det_Sigma)

        # Sort eigenvalues descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        print(f"\n  {cfg['label']}:")
        print(f"    Eigenvalues: {eigenvalues}")
        for j in range(2):
            print(f"    u{j+1} = [{eigenvectors[0, j]:.4f}, "
                  f"{eigenvectors[1, j]:.4f}], "
                  f"lambda{j+1} = {eigenvalues[j]:.4f}")

        # Column 1: 3D surface
        axes[i, 0].remove()
        ax3d = fig.add_subplot(len(configs), 2, 2 * i + 1,
                               projection="3d")
        ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85,
                          edgecolor="none")
        ax3d.set_xlabel("x1")
        ax3d.set_ylabel("x2")
        ax3d.set_zlabel("f(x)")
        ax3d.set_title(cfg["label"], fontsize=10, pad=10)

        # Column 2: contour with eigenvectors
        ax = axes[i, 1]
        ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.5)
        ax.contour(X, Y, Z, levels=8, colors="gray", linewidths=0.5)

        # Draw eigenvector arrows scaled by sqrt(eigenvalue)
        colors_ev = ["red", "darkgreen"]
        for j in range(2):
            scale = np.sqrt(eigenvalues[j])
            dx = eigenvectors[0, j] * scale
            dy = eigenvectors[1, j] * scale
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=colors_ev[j],
                                        lw=2.5))
            ax.text(dx * 1.15, dy * 1.15,
                    f"sqrt(l{j+1})*u{j+1}",
                    fontsize=8, color=colors_ev[j],
                    ha="center")

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Contour + Eigenvectors", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig("gaussian_2d_eigendecomposition.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("\nFigure saved: gaussian_2d_eigendecomposition.png")


if __name__ == "__main__":
    main()
