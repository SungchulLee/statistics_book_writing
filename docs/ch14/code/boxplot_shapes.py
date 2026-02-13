#!/usr/bin/env python3
# ======================================================================
# 27_boxnorm_04_shapes_via_boxplots_examples.py
# ======================================================================
# Two separate figures showing typical deviations from normality:
#  - Skewed (lognormal)
#  - Heavy-tailed (Student-t)
# Boxplots reveal longer whisker on one side or many outliers.
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

def make_plot(x, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(x, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("Values")
    plt.tight_layout()
    plt.show()

def main():
    rng = np.random.default_rng(3)
    x_skew = rng.lognormal(0.0, 0.7, size=400)
    x_t    = rng.standard_t(df=3, size=400)

    make_plot(x_skew, "Boxplot: skewed distribution (lognormal)")
    make_plot(x_t, "Boxplot: heavy tails (t df=3)")

if __name__ == "__main__":
    main()
