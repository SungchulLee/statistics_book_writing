
#!/usr/bin/env python3
"""
One-sample proportion 95% CI.

Methods:
  - wald    : p̂ ± z* sqrt(p̂(1-p̂)/n)           (simple, can under-cover)
  - wilson  : Wilson score (recommended)
  - ac      : Agresti–Coull (near Wilson, simple)
  - cp      : Clopper–Pearson exact (conservative)

Inputs:
  - Either counts via --k --n
  - Or CSV with 0/1 in one column via --csv

Examples:
  python ci_proportion_one_sample.py --k 12 --n 50 --method wilson
  python ci_proportion_one_sample.py --csv bernoulli.csv --method cp --cl 0.99
"""
from __future__ import annotations
import argparse, sys, csv, math
import numpy as np
from scipy.stats import norm, beta

def load_data(csv_path: str) -> np.ndarray:
    arr = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    v = float(item)
                    if v not in (0,1):
                        raise ValueError("CSV must contain only 0/1 values for proportion.")
                    arr.append(v)
    if len(arr) == 0:
        raise ValueError("No values found in CSV.")
    return np.array(arr, dtype=float)

def parse_args():
    p = argparse.ArgumentParser(description="One-sample proportion CI.")
    p.add_argument("--csv", type=str, help="CSV file with one column of 0/1 values.")
    p.add_argument("--k", type=int, help="Number of successes.")
    p.add_argument("--n", type=int, help="Sample size.")
    p.add_argument("--method", choices=["wald","wilson","ac","cp"], default="wilson")
    p.add_argument("--cl", type=float, default=0.95)
    return p.parse_args()

def main():
    args = parse_args()
    alpha = 1 - args.cl
    z = norm.ppf(1 - alpha/2)

    if args.csv:
        x = load_data(args.csv)
        n = x.size
        k = int(x.sum())
    else:
        if args.k is None or args.n is None:
            print("Provide --csv OR counts via --k and --n.", file=sys.stderr)
            sys.exit(2)
        k = int(args.k); n = int(args.n)

    phat = k / n

    if args.method == "wald":
        se = math.sqrt(phat*(1-phat)/n)
        lo = phat - z*se
        hi = phat + z*se
    elif args.method == "wilson":
        denom = 1 + z*z/n
        center = (phat + z*z/(2*n))/denom
        half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denom
        lo, hi = center - half, center + half
    elif args.method == "ac":
        n_tilde = n + z*z
        p_tilde = (k + 0.5*z*z)/n_tilde
        se_tilde = math.sqrt(p_tilde*(1-p_tilde)/n_tilde)
        lo = p_tilde - z*se_tilde
        hi = p_tilde + z*se_tilde
    else:  # cp
        if k == 0:
            lo = 0.0
        else:
            lo = beta.ppf(alpha/2, k, n-k+1)
        if k == n:
            hi = 1.0
        else:
            hi = beta.ppf(1 - alpha/2, k+1, n-k)

    lo = max(0.0, lo)
    hi = min(1.0, hi)

    print(f"One-sample proportion CI ({args.method}), CL={args.cl*100:.1f}%")
    print(f"k={k}, n={n}, phat={phat:.6g}")
    print(f"CI: [{lo:.6g}, {hi:.6g}]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
