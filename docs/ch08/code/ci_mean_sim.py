#!/usr/bin/env python3
# ================================================================
# 11_confidence_interval_01_mean_confidence_interval_simulation.py
# ================================================================
"""
One file, three toggles: **Z (σ known)**, **Z (σ plug-in s)**, and **t (σ unknown)** CIs for a mean.

Why this file?
--------------
Gives a *full picture* of one-sample mean confidence intervals by simulating and plotting
interval coverage for:
  1) Z-interval with **known** σ          → x̄ ± z* · σ / √n
  2) Z-like interval with **plug-in s**   → x̄ ± z* · s / √n      (common teaching variant)
  3) t-interval with **unknown σ**        → x̄ ± t* · s / √n, df = n−1

You typically run a single method (default: t-interval). For large n, t* ≈ z*.

Notes and teaching tips
-----------------------
• In real work σ is rarely known → the **t-interval** is the default.
• The **plug-in Z** interval (replace σ by s) is popular for large-n intuition but may **under-cover**
  for small n vs the t-interval.
• For **large n**, all three become very similar (t* ≈ z*).

Finite population correction (FPC)
----------------------------------
When sampling without replacement from a finite population of size N, use the FPC factor
  fpc = sqrt((N − n) / (N − 1))
if the sampling fraction n/N is non-negligible. Rule of thumb: if n ≤ 0.10·N, then FPC ≈ 1.

Usage
-----
# Default run (t-interval; good for notebooks/terminals)
python 11_confidence_interval_01_mean_confidence_interval_simulation.py

# Choose a specific method
python 11_confidence_interval_01_mean_confidence_interval_simulation.py --method t
python 11_confidence_interval_01_mean_confidence_interval_simulation.py --method z_plugin
python 11_confidence_interval_01_mean_confidence_interval_simulation.py --method z_known --sigma 1.0

# Increase simulations / change sample size / level
python 11_confidence_interval_01_mean_confidence_interval_simulation.py --method t --n-sim 200 --n 12 --alpha 0.05

# Use a finite population correction (e.g., N=500)
python 11_confidence_interval_01_mean_confidence_interval_simulation.py --method z_plugin --N 500

Notebook note
-------------
This script uses argparse’s parse_known_args() so it **ignores unknown flags** that Jupyter/Colab injects
(e.g., `-f /path/to/kernel.json`), making it safe to run inside notebooks without modification.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def finite_population_correction(n: int, N: int | None) -> float:
    """Return FPC factor sqrt((N−n)/(N−1)) if N is provided; else 1.0."""
    if N is None:
        return 1.0
    if N <= 1 or n >= N:
        raise ValueError("FPC requires N > 1 and n < N.")
    return float(np.sqrt((N - n) / (N - 1)))


def simulate_data(n_sim: int, n: int, mu: float, sigma: float, rng) -> np.ndarray:
    """Return X with shape (n_sim, n): each row ~ N(mu, sigma^2)."""
    return rng.normal(loc=mu, scale=sigma, size=(n_sim, n))


def compute_intervals(xbar: np.ndarray, s: np.ndarray, n: int, alpha: float,
                      method: str, sigma_known: float | None = None, N: int | None = None):
    """
    Compute CI lower/upper depending on method:
      - 'z_known': uses true σ (sigma_known) with z*
      - 'z_plugin': uses sample s with z* (large-n teaching variant)
      - 't': uses sample s with t* and df = n-1 (default in practice)
    Applies FPC to the standard error if N is provided.
    """
    fpc = finite_population_correction(n, N)
    if method == "z_known":
        if sigma_known is None:
            raise ValueError("z_known requires --sigma to be provided as the true known σ.")
        z_star = norm.ppf(1 - alpha/2.0)
        se = sigma_known / np.sqrt(n) * fpc
        moe = z_star * se
    elif method == "z_plugin":
        z_star = norm.ppf(1 - alpha/2.0)
        se = (s / np.sqrt(n)) * fpc
        moe = z_star * se
    elif method == "t":
        df = n - 1
        t_star = t.ppf(1 - alpha/2.0, df=df)
        se = (s / np.sqrt(n)) * fpc
        moe = t_star * se
    else:
        raise ValueError(f"Unknown method: {method}")

    lower = xbar - moe
    upper = xbar + moe
    return lower, upper


def run_once(X: np.ndarray, mu: float, n: int, alpha: float, method: str, sigma_known: float | None, N: int | None):
    """Compute intervals for all rows of X and return dict with results and coverage."""
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)
    lower, upper = compute_intervals(xbar, s, n, alpha, method, sigma_known, N)

    covered = (lower <= mu) & (mu <= upper)
    return {
        "xbar": xbar,
        "lower": lower,
        "upper": upper,
        "covered": covered,
        "coverage_pct": 100.0 * covered.mean(),
        "n_fail": int((~covered).sum()),
    }


def plot_intervals(ax, lower, upper, xbar, covered, mu, title):
    """Plot each interval as a horizontal segment; red indicates miss (does not cover μ)."""
    for i in range(len(xbar)):
        color = "k" if covered[i] else "r"
        ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)
        ax.plot(xbar[i], i, marker="o", ms=3, color=color)

    ax.axvline(mu, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(title, fontsize=12)
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Mean value")


def compare_all(X: np.ndarray, mu: float, n: int, alpha: float, sigma: float, N: int | None):
    """
    (Optional utility) Create a 3-row figure comparing z_known, z_plugin, and t on the same dataset.
    Not called by default; keep for teaching/demo use if you want side-by-side coverage visuals.
    """
    results = {}
    methods = [("z_known", "Z (σ known)"),
               ("z_plugin", "Z (plug-in s)"),
               ("t", "t (σ unknown)")]

    for m, _ in methods:
        results[m] = run_once(X, mu=mu, n=n, alpha=alpha, method=m,
                              sigma_known=sigma if m == "z_known" else None, N=N)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), sharex=True)
    for ax, (m, label) in zip(axes, methods):
        r = results[m]
        title = f"{label} | n={n}, CL={int((1-alpha)*100)}% | Fail={r['n_fail']} (Coverage ≈ {r['coverage_pct']:.1f}%)"
        plot_intervals(ax, r["lower"], r["upper"], r["xbar"], r["covered"], mu, title)

    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Mean CI simulator: Z (known σ), Z (plug-in s), and t."
    )

    # Single method arg, default "t"
    p.add_argument(
        "--method",
        choices=["z_known", "z_plugin", "t"],
        default="t",
        help='Choose a method to run. Default: "t".'
    )
    p.add_argument("--rng-seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--n-sim", type=int, default=100, help="Number of simulated experiments.")
    p.add_argument("--n", type=int, default=10, help="Sample size per experiment.")
    p.add_argument("--mu", type=float, default=0.0, help="True population mean.")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="True population σ (used for data gen; also for z_known).")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level; CI level = 1 − α.")
    p.add_argument("--N", type=int, default=None,
                   help="Finite population size for FPC. If omitted, no FPC is applied.")

    # Parse in a notebook-friendly way: ignore unknown args that Jupyter/Colab inject (e.g., -f path.json).
    args, _unknown = p.parse_known_args()

    rng = np.random.default_rng(args.rng_seed)
    X = simulate_data(args.n_sim, args.n, args.mu, args.sigma, rng)

    # Always run one method (default "t"); no --compare branch.
    res = run_once(
        X, mu=args.mu, n=args.n, alpha=args.alpha,
        method=args.method,
        sigma_known=args.sigma if args.method == "z_known" else None,
        N=args.N
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    title = (f"{args.n_sim} {args.method} CIs | n={args.n}, "
             f"CL={int((1-args.alpha)*100)}% | "
             f"Fail={res['n_fail']} (Coverage ≈ {res['coverage_pct']:.1f}%)")
    plot_intervals(ax, res["lower"], res["upper"], res["xbar"], res["covered"], args.mu, title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()