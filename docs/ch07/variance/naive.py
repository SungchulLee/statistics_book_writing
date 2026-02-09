"""
Naive Variance Estimator: Demonstrations
==========================================
  - Bias verification
  - Comparison with Bessel-corrected and MSE-optimal estimators
  - Degrees of freedom intuition
  - Known vs unknown mean
  - Financial: volatility estimation with different divisors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Bias Verification
# ============================================================

def bias_verification(sigma=3.0, n_sim=200_000, seed=42):
    """Verify the bias of the naive variance estimator."""
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("=" * 70)
    print("Naive Variance Estimator: Bias Verification")
    print(f"True sigma^2 = {sigma2}")
    print("=" * 70)
    print(f"\n{'n':>6} {'E[S_tilde2]':>14} {'(n-1)/n*sig2':>14} {'Bias':>10} "
          f"{'Theory':>10}")
    print("-" * 58)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        x_bar = samples.mean(axis=1, keepdims=True)
        s_tilde2 = np.mean((samples - x_bar)**2, axis=1)
        
        e_s2 = s_tilde2.mean()
        theory = (n-1)/n * sigma2
        bias = e_s2 - sigma2
        bias_theory = -sigma2/n
        
        print(f"{n:>6} {e_s2:>14.4f} {theory:>14.4f} {bias:>10.4f} {bias_theory:>10.4f}")
    
    print(f"\nBias = -sigma^2/n: always negative (underestimates).")
    print(f"Bias -> 0 as n -> inf (consistent).")


# ============================================================
# 2. Three Estimators Compared
# ============================================================

def three_estimators_comparison(sigma=3.0, n_sim=100_000, seed=42):
    """Compare divide-by-n, n-1, and n+1 variance estimators."""
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    sigma4 = sigma**4
    
    sample_sizes = [3, 5, 10, 20, 50, 100]
    
    print("\n" + "=" * 80)
    print("Comparison: Divide by n, n-1, n+1")
    print(f"True sigma^2 = {sigma2}")
    print("=" * 80)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        ss = np.sum((samples - samples.mean(axis=1, keepdims=True))**2, axis=1)
        
        s2_n = ss / n         # Naive
        s2_n1 = ss / (n - 1)  # Bessel's
        s2_np1 = ss / (n + 1) # MSE-optimal
        
        results = {}
        for name, est in [('1/n', s2_n), ('1/(n-1)', s2_n1), ('1/(n+1)', s2_np1)]:
            bias = est.mean() - sigma2
            mse = np.mean((est - sigma2)**2)
            results[name] = (bias, mse)
        
        print(f"\nn = {n}:")
        print(f"  {'Divisor':<10} {'Bias':>10} {'MSE':>12} {'MSE(theory)':>14}")
        print("  " + "-" * 50)
        
        # Theoretical MSEs
        mse_n_theory = (2*n - 1) / n**2 * sigma4
        mse_n1_theory = 2 / (n - 1) * sigma4
        mse_np1_theory = 2*(n-1) / (n+1)**2 * sigma4 + 4 / (n+1)**2 * sigma4
        
        for name, theory in [('1/n', mse_n_theory), ('1/(n-1)', mse_n1_theory),
                              ('1/(n+1)', mse_np1_theory)]:
            bias, mse = results[name]
            print(f"  {name:<10} {bias:>10.4f} {mse:>12.4f} {theory:>14.4f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = np.arange(3, 101)
    mse_n = (2*ns - 1) / ns**2 * sigma4
    mse_n1 = 2 / (ns - 1) * sigma4
    mse_np1 = (2*(ns-1) + 4) / (ns+1)**2 * sigma4
    
    ax.plot(ns, mse_n, 'b-', linewidth=2, label='1/n (naive/MLE)')
    ax.plot(ns, mse_n1, 'r-', linewidth=2, label='1/(n-1) (Bessel)')
    ax.plot(ns, mse_np1, 'g-', linewidth=2, label='1/(n+1) (MSE-optimal)')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE of Variance Estimators (Normal Population)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('naive_variance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. Degrees of Freedom Intuition
# ============================================================

def degrees_of_freedom_demo(seed=42):
    """
    Visualize why using X_bar instead of mu loses one degree of freedom.
    Show that sum(Xi - X_bar) = 0 constrains the deviations.
    """
    rng = np.random.default_rng(seed)
    mu = 5.0
    sigma = 2.0
    n = 5
    
    sample = rng.normal(mu, sigma, n)
    x_bar = sample.mean()
    
    print("\n" + "=" * 60)
    print("Degrees of Freedom Intuition")
    print(f"Sample: {np.round(sample, 3)}")
    print(f"X_bar = {x_bar:.3f}, True mu = {mu}")
    print("=" * 60)
    
    dev_xbar = sample - x_bar
    dev_mu = sample - mu
    
    print(f"\n{'i':>3} {'Xi':>8} {'Xi-X_bar':>10} {'Xi-mu':>10}")
    print("-" * 35)
    for i in range(n):
        print(f"{i+1:>3} {sample[i]:>8.3f} {dev_xbar[i]:>10.3f} {dev_mu[i]:>10.3f}")
    print(f"{'Sum':>3} {'':>8} {sum(dev_xbar):>10.3f} {sum(dev_mu):>10.3f}")
    
    print(f"\nSum of (Xi - X_bar) = {sum(dev_xbar):.10f} (exactly 0)")
    print(f"Sum of (Xi - mu)   = {sum(dev_mu):.3f} (not 0)")
    print(f"\nSS from X_bar: {np.sum(dev_xbar**2):.3f}")
    print(f"SS from mu:    {np.sum(dev_mu**2):.3f}")
    print(f"Difference:    {np.sum(dev_mu**2) - np.sum(dev_xbar**2):.3f}")
    print(f"n*(X_bar-mu)^2: {n*(x_bar - mu)**2:.3f}")
    print(f"\nKey identity verified: SS_xbar = SS_mu - n*(X_bar - mu)^2")


# ============================================================
# 4. Known vs Unknown Mean
# ============================================================

def known_vs_unknown_mean(sigma=3.0, n_sim=100_000, seed=42):
    """Compare variance estimation when mu is known vs unknown."""
    rng = np.random.default_rng(seed)
    mu = 5.0
    sigma2 = sigma**2
    
    sample_sizes = [5, 10, 25, 50, 100]
    
    print("\n" + "=" * 65)
    print("Variance Estimation: Known vs Unknown Mean")
    print("=" * 65)
    print(f"\n{'n':>6} {'MSE(known mu)':>14} {'MSE(unknown)':>14} {'Ratio':>8}")
    print("-" * 46)
    
    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        
        # Known mean
        est_known = np.mean((samples - mu)**2, axis=1)
        mse_known = np.mean((est_known - sigma2)**2)
        
        # Unknown mean (naive)
        est_unknown = np.mean((samples - samples.mean(axis=1, keepdims=True))**2, axis=1)
        mse_unknown = np.mean((est_unknown - sigma2)**2)
        
        print(f"{n:>6} {mse_known:>14.4f} {mse_unknown:>14.4f} "
              f"{mse_unknown/mse_known:>8.3f}")
    
    print("\nKnowing the true mean always reduces MSE.")
    print("The ratio approaches 1 as n increases.")


# ============================================================
# 5. Financial: Volatility Estimation
# ============================================================

def volatility_estimation_demo(seed=42):
    """
    Compare variance estimators applied to simulated daily returns.
    Shows practical differences with different divisors.
    """
    rng = np.random.default_rng(seed)
    
    # Simulate daily log-returns
    annual_vol = 0.20
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = 0.08 / 252  # 8% annual return
    
    print("\n" + "=" * 60)
    print("Financial: Daily Return Volatility Estimation")
    print(f"True annual vol: {annual_vol*100:.1f}%")
    print("=" * 60)
    
    windows = [5, 10, 21, 63, 126, 252]  # Trading days
    n_sim = 30_000
    
    print(f"\n{'Window':>8} {'Vol(1/n)':>10} {'Vol(1/n-1)':>12} {'Bias(1/n)':>10} "
          f"{'% Diff':>8}")
    print("-" * 55)
    
    for w in windows:
        vol_n = []
        vol_n1 = []
        
        for _ in range(n_sim):
            returns = rng.normal(daily_mu, daily_vol, w)
            s2_n = np.var(returns, ddof=0)
            s2_n1 = np.var(returns, ddof=1)
            vol_n.append(np.sqrt(s2_n * 252))    # Annualize
            vol_n1.append(np.sqrt(s2_n1 * 252))
        
        vol_n = np.array(vol_n)
        vol_n1 = np.array(vol_n1)
        bias_pct = (vol_n.mean() - annual_vol) / annual_vol * 100
        diff_pct = (vol_n1.mean() - vol_n.mean()) / vol_n.mean() * 100
        
        print(f"{w:>8} {vol_n.mean()*100:>9.2f}% {vol_n1.mean()*100:>11.2f}% "
              f"{bias_pct:>9.2f}% {diff_pct:>7.2f}%")
    
    print("\nFor short windows (5-21 days), the choice of divisor matters.")
    print("For long windows (252 days), the difference is negligible.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("NAIVE VARIANCE ESTIMATOR: DEMONSTRATIONS")
    print("=" * 60)
    
    bias_verification()
    three_estimators_comparison()
    degrees_of_freedom_demo()
    known_vs_unknown_mean()
    volatility_estimation_demo()
