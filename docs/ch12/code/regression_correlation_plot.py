"""
Regression Slope, Error Scale, and Correlation
================================================
Adapted from Basic-Statistics-With-Python plot_material.py (reg_corr_plot).

Shows 8 scatter plots with the true regression line overlaid,
demonstrating how the Pearson correlation varies as a function of:
  1. The slope (beta_2) — steeper slope => higher |r|
  2. The error scale — more noise => lower |r|
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

DATA_SIZE = 100

CONFIGS = [
    # (beta1, beta2, error_scale)
    (2, 0.05, 1),
    (2, -0.6, 1),
    (2, 1.0,  1),
    (2, 3.0,  1),
    (2, 3.0,  3),
    (2, 3.0, 10),
    (2, 3.0, 20),
    (2, 3.0, 50),
]


def generate(beta1, beta2, error_scale, n=DATA_SIZE):
    """Generate (x, y) with y = beta1 + beta2*x + error_scale*u."""
    x = np.random.randint(1, n, n).astype(float)
    y = beta1 + beta2 * x + error_scale * np.random.randn(n)
    r = np.corrcoef(x, y)[0, 1]
    return x, y, r


def main():
    print("=" * 60)
    print("Regression Slope, Error Scale, and Correlation")
    print("=" * 60)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for idx, (b1, b2, es) in enumerate(CONFIGS):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        x, y, r = generate(b1, b2, es)

        ax.scatter(x, y, alpha=0.5, s=15, edgecolors="grey")
        xs = np.sort(x)
        ax.plot(xs, b1 + b2 * xs, color="#FA954D", lw=2, alpha=0.8)
        ax.set_title(f"Y = {b1} + {b2}X + {es}u", fontsize=10)
        ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.9),
                    xycoords="axes fraction", fontsize=11,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        print(f"  Config {idx+1}: beta1={b1}, beta2={b2:>5}, "
              f"error_scale={es:>2}  =>  r = {r:+.4f}")

    fig.suptitle("How Slope and Noise Affect Pearson Correlation",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("regression_correlation_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: regression_correlation_plot.png")


if __name__ == "__main__":
    main()
