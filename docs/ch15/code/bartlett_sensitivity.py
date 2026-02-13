#!/usr/bin/env python3
# ======================================================================
# 20_bartlett_04_sensitivity_non_normality.py
# ======================================================================
# Sketch to show Bartlett's sensitivity to non-normality. We generate
# groups from a skewed distribution (lognormal) and compare tests.
# ======================================================================

import numpy as np
from scipy.stats import bartlett, levene, fligner

rng = np.random.default_rng(0)

def simulate_once(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    if skew:
        # Lognormal: heavy-tailed/skewed
        g = [rng.lognormal(mean=0.0, sigma=s, size=n) for s in sigmas]
    else:
        # Normal
        g = [rng.normal(loc=0.0, scale=s, size=n) for s in sigmas]
    return g

def trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    g1, g2, g3 = simulate_once(n=n, sigmas=sigmas, skew=skew)
    b_stat, b_p = bartlett(g1, g2, g3)
    Wm, pm = levene(g1, g2, g3, center='mean')
    X2, pf = fligner(g1, g2, g3)
    return b_p, pm, pf

def main():
    n_sims = 200
    # Null of equal variances but skewed data
    ps_b, ps_lv, ps_fl = [], [], []
    for _ in range(n_sims):
        p_b, p_l, p_f = trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True)
        ps_b.append(p_b); ps_lv.append(p_l); ps_fl.append(p_f)
    alpha = 0.05
    fp_b  = np.mean(np.array(ps_b)  < alpha)
    fp_lv = np.mean(np.array(ps_lv) < alpha)
    fp_fl = np.mean(np.array(ps_fl) < alpha)
    print("Estimated false-positive rate under skew (equal variances):")
    print(f"  Bartlett  : {fp_b:.3f}")
    print(f"  Levene    : {fp_lv:.3f}")
    print(f"  Fligner   : {fp_fl:.3f}")

if __name__ == "__main__":
    main()
