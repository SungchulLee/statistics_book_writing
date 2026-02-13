
#!/usr/bin/env python3
"""
One-sample mean 95% CI.

Supports:
- z-interval with KNOWN sigma:        xbar ± z* * sigma / sqrt(n)
- t-interval with sample sd (default): xbar ± t* * s / sqrt(n)

Inputs can be either:
  (A) Raw data file via --csv (one column of numbers)
  (B) Summary stats via --n --mean --sd

Usage examples
--------------
# t-interval from CSV
python ci_mean_one_sample.py --csv data.csv

# z-interval with known sigma from summary stats
python ci_mean_one_sample.py --n 25 --mean 3.2 --sd 1.1 --known-sigma 1.0 --method z

# choose confidence level (default 0.95)
python ci_mean_one_sample.py --csv data.csv --cl 0.99
"""
from __future__ import annotations
import argparse, sys, csv, math
import numpy as np
from scipy.stats import norm, t

def load_data(csv_path: str) -> np.ndarray:
    arr = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    arr.append(float(item))
    if len(arr) == 0:
        raise ValueError("No numeric values found in CSV.")
    return np.array(arr, dtype=float)

def parse_args():
    p = argparse.ArgumentParser(description="One-sample mean CI (z or t).")
    p.add_argument("--csv", type=str, help="CSV file with one column of numeric data.")
    p.add_argument("--n", type=int, help="Sample size (if summary stats).")
    p.add_argument("--mean", type=float, help="Sample mean (if summary stats).")
    p.add_argument("--sd", type=float, help="Sample sd (if summary stats).")
    p.add_argument("--method", choices=["t","z"], default="t", help="Method: 't' (default) or 'z' (known sigma).")
    p.add_argument("--known-sigma", type=float, default=None, help="Known population sigma (required if --method z).")
    p.add_argument("--cl", type=float, default=0.95, help="Confidence level, e.g., 0.95.")
    return p.parse_args()

def main():
    args = parse_args()
    alpha = 1 - args.cl

    if args.csv:
        x = load_data(args.csv)
        n = x.size
        xbar = x.mean()
        s = x.std(ddof=1)
    else:
        if args.n is None or args.mean is None or (args.sd is None and args.method=="t"):
            print("Provide --csv OR summary stats --n --mean --sd (sd optional only if using z with known sigma).", file=sys.stderr)
            sys.exit(2)
        n = int(args.n)
        xbar = float(args.mean)
        s = None if args.method=="z" else float(args.sd)

    if args.method == "z":
        if args.known_sigma is None:
            print("Known sigma required for z-interval. Use --known-sigma.", file=sys.stderr)
            sys.exit(2)
        se = args.known_sigma / math.sqrt(n)
        zstar = norm.ppf(1 - alpha/2)
        moe = zstar * se
        lo, hi = xbar - moe, xbar + moe
        method_label = f"z (known σ={args.known_sigma:g})"
        df = None
    else:
        if s is None:
            if args.csv:
                x = load_data(args.csv)
                s = x.std(ddof=1)
            else:
                print("Sample sd required for t-interval.", file=sys.stderr)
                sys.exit(2)
        se = s / math.sqrt(n)
        df = n - 1
        tstar = t.ppf(1 - alpha/2, df=df)
        moe = tstar * se
        lo, hi = xbar - moe, xbar + moe
        method_label = f"t (df={df})"

    print(f"One-sample mean CI ({method_label}), CL={args.cl*100:.1f}%")
    print(f"n={n}, xbar={xbar:.6g}, se={se:.6g}")
    if df is not None:
        print(f"df={df}")
    print(f"CI: [{lo:.6g}, {hi:.6g}]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
