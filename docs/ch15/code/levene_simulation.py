#!/usr/bin/env python3
import numpy as np
from scipy.stats import levene

rng = np.random.default_rng(0)

def simulate_once(n=20, dist="normal"):
    if dist == "normal":
        g1 = rng.normal(0, 1.0, size=n)
        g2 = rng.normal(0, 1.0, size=n)
        g3 = rng.normal(0, 1.0, size=n)
    else:
        g1 = rng.lognormal(0, 1.0, size=n)
        g2 = rng.lognormal(0, 1.0, size=n)
        g3 = rng.lognormal(0, 1.0, size=n)
    W, p = levene(g1, g2, g3, center='median')
    return p

def main():
    alpha = 0.05
    n_sims = 300
    for dist in ["normal", "lognormal"]:
        pvals = [simulate_once(20, dist) for _ in range(n_sims)]
        type1 = np.mean(np.array(pvals) < alpha)
        print(f"Type I (median-centered) under {dist}: {type1:.3f}")

if __name__ == "__main__":
    main()
