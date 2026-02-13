#!/usr/bin/env python3
# ======================================================================
# 14_welch_anova_05_simulation_typeI_power.py
# ======================================================================
# Monte Carlo sketch to compare Type I error (null true) and simple power
# for Welch's ANOVA under heteroscedasticity.
# Keep n_sims modest to stay quick.
# ======================================================================

import numpy as np
import pandas as pd
import pingouin as pg

rng = np.random.default_rng(0)

def simulate_once(null=True):
    # Three groups with unequal variances; sizes unbalanced
    ns = [10, 18, 7]
    sigmas = [1.0, 3.0, 6.0]

    # Under null, same mean. Under alt, shift group C.
    means = [10.0, 10.0, 10.0] if null else [10.0, 10.0, 12.0]

    rows = []
    for i, (n, mu, sd) in enumerate(zip(ns, means, sigmas), start=1):
        x = rng.normal(mu, sd, size=n)
        rows += [{"Group": f"G{i}", "Values": v} for v in x]
    df = pd.DataFrame(rows)

    aov = pg.welch_anova(dv="Values", between="Group", data=df)
    p = float(aov["p-unc"].iloc[0])
    return p

def run(n_sims=500, alpha=0.05):
    # Type I under null, simple power under alt
    pvals_null = [simulate_once(null=True) for _ in range(n_sims)]
    pvals_alt  = [simulate_once(null=False) for _ in range(n_sims)]
    type1 = np.mean(np.array(pvals_null) < alpha)
    power = np.mean(np.array(pvals_alt)  < alpha)
    return type1, power

def main():
    type1, power = run(n_sims=300, alpha=0.05)
    print("=== Welch's ANOVA (Monte Carlo sketch) ===")
    print(f"Estimated Type I error @ α=0.05 : {type1:.3f}")
    print(f"Estimated Power     @ α=0.05 : {power:.3f} (simple mean shift in group C)")

if __name__ == "__main__":
    main()
