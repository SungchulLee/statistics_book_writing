#!/usr/bin/env python3
# ======================================================================
# 11_confidence_interval_02_proportion_confidence_interval_simulation.py
# ======================================================================
"""
Single-proportion CI simulation (inline formulas, no helpers).

What this script does
---------------------
- Repeats `n_simulations` Bernoulli experiments with sample size `n` and true p = `p_true`.
- In each experiment, computes a two-sided (1-α) CI for the proportion with one of four methods:
    1) 'wald'   : p̂ ± z* √[ p̂(1−p̂)/n ]
    2) 'wilson' : Wilson score interval (no continuity correction)
    3) 'ac'     : Agresti–Coull ("add z²/2" pseudo counts)
    4) 'cp'     : Clopper–Pearson ("exact" binomial; conservative)
- Plots all intervals; those that **miss** the true p (coverage failure) are colored red.
- The title reports empirical coverage across the simulations.

When to use which method (undergrad-friendly rules of thumb)
------------------------------------------------------------
• WALD  ('wald')
  - Use only when n is **large** and p is **not extreme** (e.g., p in ~[0.1, 0.9]).
  - Quick textbook formula but can **under-cover** badly for small n or p near 0/1.
  - Rule of thumb (very rough): n·p̂ ≥ 10 and n·(1−p̂) ≥ 10. Otherwise avoid.
  - Pedagogical: useful to *illustrate* why better intervals exist.

• WILSON ('wilson')  ▶ **Recommended default**
  - Good coverage even for moderate n; center is adjusted (not simply at p̂).
  - Interval stays within [0,1] and behaves well near boundaries without CC.
  - Great classroom choice: accurate, simple to compute, and widely recommended.

• AGRESTI–COULL ('ac')
  - Add z²/2 pseudo-successes and pseudo-failures; compute Wald on the adjusted counts.
  - Coverage very close to Wilson; easy to explain/compute by hand.
  - Good practical default if you want a one-line formula.

• CLOPPER–PEARSON ('cp')
  - Inverts the exact binomial test → **guaranteed ≥ (1−α) coverage** (conservative).
  - Intervals are often noticeably wider; preferred in small n / regulatory settings
    where conservatism is valued (e.g., clinical, quality control).

Practical teaching summary
--------------------------
- Prefer **Wilson** (or **Agresti–Coull**) for general use and classroom demos.
- Use **Clopper–Pearson** when you need *guaranteed* minimum coverage or n is very small.
- Keep **Wald** for comparison and historical context (expect undercoverage for small n).

Tip: The plotted **red** intervals visually show undercoverage for challenging cases
(e.g., small n or extreme p). Try changing `n` and `p_true` and compare methods.

Sampling design note (10% condition & FPC)
-----------------------------------------
• **When sampling without replacement** from a finite population of size **N**, treating
  observations as independent is a good approximation if **n ≤ 0.10·N** (the "10% condition").
• If the sampling fraction is larger, use the **finite population correction (FPC)** for
  Wald/Agresti–Coull standard errors: multiply the SE (and MOE) by √((N−n)/(N−1)).
• These methods (Wilson/Clopper–Pearson) are derived under a **binomial** model (with replacement).
  Under large sampling fractions the exact model is **hypergeometric**; the 10% condition justifies
  the binomial approximation. For exact finite-population inference, use hypergeometric-based CIs
  (advanced topic).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

# -----------------------------
# Configuration (edit these)
# -----------------------------
rng_seed = None         # set an int (e.g., 0) for reproducibility; None → random each run
n_simulations = 100     # number of repeated experiments (intervals to plot)
n = 20                  # sample size per experiment
p_true = 0.20           # true Bernoulli success probability
alpha = 0.05            # significance level → confidence level = 1 - alpha
method = "wald"       # choose: 'wald' | 'wilson' | 'ac' | 'cp'   (recommended default: 'wilson')

def main():
    # Optional reproducibility
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Simulate number of successes for each experiment: K ~ Binomial(n, p_true)
    k = np.random.binomial(n=n, p=p_true, size=n_simulations)
    phat = k / n  # sample proportion for plotting markers

    lower = np.empty(n_simulations, dtype=float)
    upper = np.empty(n_simulations, dtype=float)

    # Critical z-value for two-sided (1-α) intervals
    z = norm.ppf(1 - alpha/2.0)

    # ---------------------------------------------------------
    # Compute CI for each experiment (method chosen above)
    # ---------------------------------------------------------
    for i, ki in enumerate(k):
        # Use ki (integer count) and p = ki/n (sample proportion)
        p = ki / n

        if method == "wald":
            # Wald: p̂ ± z* * SE, where SE = sqrt(p̂(1−p̂)/n)
            # Fast, familiar, but can under-cover for small n or extreme p (near 0 or 1)
            se = np.sqrt(p * (1 - p) / n)
            lo = p - z * se
            hi = p + z * se

        elif method == "wilson":
            # Wilson score (no continuity correction)
            # Shifts the center and shrinks the half-width in a way that improves coverage.
            # Often preferred as a default in intro stats.
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
            lo = center - half
            hi = center + half

        elif method == "ac":
            # Agresti–Coull: add z²/2 pseudo-successes and pseudo-failures
            # Then apply Wald to the adjusted counts. Coverage ~ Wilson, very simple to compute.
            n_tilde = n + z**2
            p_tilde = (ki + 0.5 * z**2) / n_tilde
            se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
            lo = p_tilde - z * se_tilde
            hi = p_tilde + z * se_tilde

        elif method == "cp":
            # Clopper–Pearson ("exact"): invert the binomial test via Beta quantiles
            # Conservative (actual coverage ≥ nominal), especially for small n.
            if ki == 0:
                lo = 0.0
            else:
                lo = beta.ppf(alpha/2.0, ki, n - ki + 1)
            if ki == n:
                hi = 1.0
            else:
                hi = beta.ppf(1 - alpha/2.0, ki + 1, n - ki)

        else:
            raise ValueError("Unknown method; choose 'wald' | 'wilson' | 'ac' | 'cp'.")

        # Always clip to the parameter space [0,1]
        lower[i] = max(0.0, lo)
        upper[i] = min(1.0, hi)

    # ---------------------------------------------------------
    # Empirical coverage across simulations
    # ---------------------------------------------------------
    covered = (lower <= p_true) & (p_true <= upper)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    # ---------------------------------------------------------
    # Plot: intervals (black) and misses (red)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)   # CI segment
        ax.plot(phat[i], i, marker="o", ms=3, color=color)         # point estimate (p̂)

    # Reference line at the true p
    ax.axvline(p_true, linestyle="--", linewidth=1.5)

    # Title with a compact summary
    ax.set_title(
        f"{n_simulations} {method.upper()} Proportion CIs  |  n={n}, p={p_true:.3f}, "
        f"CL={int((1-alpha)*100)}%  |  Fail={n_fail}  (Coverage ≈ {coverage_pct:.1f}%)",
        fontsize=12
    )

    # Cosmetic clean-up for a decluttered plot
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Proportion value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()