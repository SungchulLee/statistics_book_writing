"""
Bivariate Gaussian — Conditional Distributions
================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

For a bivariate Gaussian (a, b) with correlation rho,
the conditional distribution of b | a = a0 is:
    b | a = a0  ~  N(rho * a0,  1 - rho^2)

Demonstrates:
1. Contour plot of the joint PDF with a vertical slice
2. 3D surface with the conditional slice highlighted
3. The resulting conditional PDF
4. How increasing rho concentrates the conditional around rho*a0
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============================================================================
# Main
# =============================================================================


def bivariate_gaussian_pdf(x, y, rho):
    """Standard bivariate Gaussian PDF with correlation rho."""
    return (np.exp(-(x**2 - 2*rho*x*y + y**2) / (2*(1 - rho**2)))
            / (2 * np.pi * np.sqrt(1 - rho**2)))


def conditional_pdf(x0, y, rho):
    """PDF of b | a = x0 for standard bivariate Gaussian."""
    sigma_cond = np.sqrt(1 - rho**2)
    mu_cond = rho * x0
    return (1 / (np.sqrt(2 * np.pi) * sigma_cond)
            * np.exp(-0.5 * ((y - mu_cond) / sigma_cond)**2))


def main():
    print("=" * 60)
    print("Bivariate Gaussian — Conditional Distributions")
    print("=" * 60)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    rho_vals = [0.0, 0.5, 0.9]
    cond_val = 1.0  # condition on a = 1

    fig, axes = plt.subplots(len(rho_vals), 3,
                             figsize=(15, 4 * len(rho_vals)))

    for i, rho in enumerate(rho_vals):
        Z = bivariate_gaussian_pdf(X, Y, rho)

        # Column 1: contour with vertical slice
        ax = axes[i, 0]
        levels = np.logspace(-7, -1, 8)
        ax.contour(X, Y, Z, levels=levels, colors="gray", linewidths=0.7)
        ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.4)
        ax.axvline(cond_val, color="red", linestyle="--", lw=2,
                   label=f"a = {cond_val}")
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title(f"Joint PDF (rho = {rho})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

        # Column 2: 3D surface with highlighted slice
        axes[i, 1].remove()
        ax3d = fig.add_subplot(len(rho_vals), 3, i * 3 + 2,
                               projection="3d")
        ax3d.plot_surface(X, Y, Z, cmap="Greys", alpha=0.25,
                          edgecolor="none")
        z_slice = bivariate_gaussian_pdf(cond_val, y, rho)
        ax3d.plot(cond_val * np.ones_like(y), y, z_slice,
                  color="red", lw=3)
        ax3d.set_xlabel("a", fontsize=8)
        ax3d.set_ylabel("b", fontsize=8)
        ax3d.set_title(f"Slice at a = {cond_val}", fontsize=10)

        # Column 3: conditional PDF
        ax = axes[i, 2]
        cond_y = conditional_pdf(cond_val, y, rho)
        ax.plot(y, cond_y, "b-", lw=2)
        mu_cond = rho * cond_val
        sigma_cond = np.sqrt(1 - rho**2)
        ax.axvline(mu_cond, color="red", linestyle="--", lw=1.5,
                   label=f"E[b|a=1] = {mu_cond:.2f}")
        ax.fill_between(y, cond_y, alpha=0.15, color="blue")
        ax.set_xlabel("b")
        ax.set_ylabel("f(b | a = 1)")
        ax.set_title(f"Conditional PDF (sd = {sigma_cond:.3f})",
                     fontsize=10)
        ax.legend(fontsize=8)

        # Print summary
        print(f"\n  rho = {rho}:")
        print(f"    E[b | a = {cond_val}] = {mu_cond:.3f}")
        print(f"    Var[b | a = {cond_val}] = {sigma_cond**2:.3f}")

    plt.tight_layout()
    plt.savefig("gaussian_2d_conditionals.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("\nFigure saved: gaussian_2d_conditionals.png")


if __name__ == "__main__":
    main()
