"""
Bessel's Correction: Demonstrations
=====================================
  - Unbiasedness verification
  - Chi-squared distribution of (n-1)S^2/sigma^2
  - Standard deviation bias (Jensen's inequality)
  - Software default comparison
  - Degrees of freedom in regression
  - Financial: tracking error estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma as gamma_func


# ============================================================
# 1. Unbiasedness Verification
# ============================================================

def unbiasedness_verification(sigma=4.0, n_sim=200_000, seed=42):
    """Verify E[S^2] = sigma^2 across sample sizes and distributions."""
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    
    sample_sizes = [3, 5, 10, 20, 50, 100]
    
    print("=" * 65)
    print("Bessel's Correction: Unbiasedness Verification")
    print(f"True sigma^2 = {sigma2}")
    print("=" * 65)
    
    # Normal distribution
    print(f"\nNormal Population:")
    print(f"{'n':>6} {'E[S^2]':>10} {'Bias':>12} {'E[S_tilde^2]':>14}")
    print("-" * 45)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s2 = np.var(samples, axis=1, ddof=1)    # Bessel's
        s2_naive = np.var(samples, axis=1, ddof=0)  # Naive
        print(f"{n:>6} {s2.mean():>10.4f} {s2.mean()-sigma2:>12.6f} {s2_naive.mean():>14.4f}")
    
    # Non-normal distributions
    print(f"\nVerification across distributions (n=20):")
    distributions = {
        'Exponential': lambda: rng.exponential(sigma, 20),
        'Uniform': lambda: rng.uniform(0, 2*sigma*np.sqrt(3), 20),
        'Chi-squared': lambda: rng.chisquare(sigma2, 20),
    }
    
    for name, sampler in distributions.items():
        true_var = {'Exponential': sigma2, 'Uniform': sigma2, 
                    'Chi-squared': 2*sigma2}[name]
        s2_vals = np.array([np.var(sampler(), ddof=1) for _ in range(n_sim)])
        print(f"  {name:<15}: E[S^2] = {s2_vals.mean():.4f}, True = {true_var:.4f}, "
              f"Bias = {s2_vals.mean()-true_var:.4f}")


# ============================================================
# 2. Chi-Squared Distribution
# ============================================================

def chi_squared_distribution(sigma=3.0, n_sim=100_000, seed=42):
    """Verify (n-1)S^2/sigma^2 ~ Chi2(n-1) for normal data."""
    rng = np.random.default_rng(seed)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sample_sizes = [5, 10, 25, 50]
    
    for ax, n in zip(axes.flat, sample_sizes):
        samples = rng.normal(0, sigma, (n_sim, n))
        s2 = np.var(samples, axis=1, ddof=1)
        chi2_vals = (n - 1) * s2 / sigma**2
        
        ax.hist(chi2_vals, bins=80, density=True, alpha=0.6, color='steelblue',
                edgecolor='white', label='Empirical')
        
        x = np.linspace(0, max(chi2_vals.max(), stats.chi2.ppf(0.999, n-1)), 200)
        ax.plot(x, stats.chi2.pdf(x, n-1), 'r-', linewidth=2,
                label=f'Chi2(df={n-1})')
        
        ax.set_title(f'n = {n}: (n-1)S^2/sigma^2', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of (n-1)S^2/sigma^2 (Normal Data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bessels_chi_squared.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 50)
    print("Chi-Squared Distribution Verified")
    print("(n-1)S^2/sigma^2 ~ Chi2(n-1) for Normal data")


# ============================================================
# 3. Standard Deviation Bias (Jensen's Inequality)
# ============================================================

def std_bias_demo(sigma=3.0, n_sim=200_000, seed=42):
    """Show that S is biased for sigma (E[S] < sigma)."""
    rng = np.random.default_rng(seed)
    
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("\n" + "=" * 70)
    print("Standard Deviation Bias: E[S] < sigma (Jensen's Inequality)")
    print(f"True sigma = {sigma}")
    print("=" * 70)
    
    print(f"\n{'n':>6} {'E[S]':>10} {'sigma':>8} {'Bias':>10} {'c4':>8} {'S/c4':>8}")
    print("-" * 55)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s = np.std(samples, axis=1, ddof=1)
        
        # Theoretical correction factor c4
        c4 = np.sqrt(2/(n-1)) * gamma_func(n/2) / gamma_func((n-1)/2)
        unbiased_s = s / c4
        
        print(f"{n:>6} {s.mean():>10.4f} {sigma:>8.4f} {s.mean()-sigma:>10.4f} "
              f"{c4:>8.4f} {unbiased_s.mean():>8.4f}")
    
    print(f"\nc4 = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)")
    print(f"S/c4 is unbiased for sigma.")


# ============================================================
# 4. Software Default Comparison
# ============================================================

def software_defaults_demo():
    """Demonstrate different software defaults for variance."""
    data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    
    print("\n" + "=" * 60)
    print("Software Defaults: var() Behavior")
    print(f"Data: {data}")
    print("=" * 60)
    
    # NumPy
    var_np_default = np.var(data)           # ddof=0 (population)
    var_np_sample = np.var(data, ddof=1)    # ddof=1 (sample)
    
    # Manual calculations
    n = len(data)
    xbar = data.mean()
    ss = np.sum((data - xbar)**2)
    var_n = ss / n
    var_n1 = ss / (n - 1)
    
    print(f"\nnp.var(data)        = {var_np_default:.4f}  (divides by n={n})")
    print(f"np.var(data, ddof=1) = {var_np_sample:.4f}  (divides by n-1={n-1})")
    print(f"\nManual 1/n:   {var_n:.4f}")
    print(f"Manual 1/n-1: {var_n1:.4f}")
    print(f"\nDifference: {var_n1 - var_n:.4f} ({(var_n1/var_n - 1)*100:.1f}% larger)")
    print(f"\nReminder: Always use ddof=1 for sample variance in NumPy!")


# ============================================================
# 5. Independence of X_bar and S^2 (Normal Only)
# ============================================================

def independence_demo(sigma=3.0, n_sim=100_000, seed=42):
    """Show X_bar and S^2 are independent for normal data."""
    rng = np.random.default_rng(seed)
    n = 20
    
    # Normal data (should be independent)
    samples_norm = rng.normal(5, sigma, (n_sim, n))
    xbar_norm = samples_norm.mean(axis=1)
    s2_norm = np.var(samples_norm, axis=1, ddof=1)
    corr_norm = np.corrcoef(xbar_norm, s2_norm)[0, 1]
    
    # Exponential data (NOT independent)
    samples_exp = rng.exponential(sigma, (n_sim, n))
    xbar_exp = samples_exp.mean(axis=1)
    s2_exp = np.var(samples_exp, axis=1, ddof=1)
    corr_exp = np.corrcoef(xbar_exp, s2_exp)[0, 1]
    
    print("\n" + "=" * 60)
    print("Independence of X_bar and S^2")
    print("=" * 60)
    print(f"\nNormal data:     Corr(X_bar, S^2) = {corr_norm:.6f} (should be ~0)")
    print(f"Exponential data: Corr(X_bar, S^2) = {corr_exp:.6f} (not independent)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(xbar_norm[:2000], s2_norm[:2000], s=5, alpha=0.2)
    axes[0].set_title(f'Normal: Corr = {corr_norm:.4f}', fontsize=12)
    axes[0].set_xlabel('X_bar'); axes[0].set_ylabel('S^2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(xbar_exp[:2000], s2_exp[:2000], s=5, alpha=0.2)
    axes[1].set_title(f'Exponential: Corr = {corr_exp:.4f}', fontsize=12)
    axes[1].set_xlabel('X_bar'); axes[1].set_ylabel('S^2')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Independence of X_bar and S^2 (Normal Only)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bessels_independence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. Financial: Tracking Error
# ============================================================

def tracking_error_demo(seed=42):
    """Estimate tracking error with Bessel's correction."""
    rng = np.random.default_rng(seed)
    
    # Simulate portfolio and benchmark returns
    n_months = 36  # 3 years of monthly data
    mu_excess = 0.002  # Monthly alpha
    te_true = 0.01  # Monthly tracking error
    
    n_sim = 50_000
    
    print("\n" + "=" * 60)
    print("Tracking Error Estimation")
    print(f"True monthly TE: {te_true*100:.2f}%")
    print(f"True annualized TE: {te_true*np.sqrt(12)*100:.2f}%")
    print("=" * 60)
    
    te_n = []
    te_n1 = []
    
    for _ in range(n_sim):
        excess_returns = rng.normal(mu_excess, te_true, n_months)
        te_n.append(np.std(excess_returns, ddof=0) * np.sqrt(12))
        te_n1.append(np.std(excess_returns, ddof=1) * np.sqrt(12))
    
    te_n = np.array(te_n)
    te_n1 = np.array(te_n1)
    te_annual_true = te_true * np.sqrt(12)
    
    print(f"\n{'Estimator':<20} {'Mean':>10} {'Bias':>10} {'RMSE':>10}")
    print("-" * 55)
    print(f"{'TE (ddof=0)':<20} {te_n.mean()*100:>9.3f}% "
          f"{(te_n.mean()-te_annual_true)*100:>9.3f}% "
          f"{np.sqrt(np.mean((te_n-te_annual_true)**2))*100:>9.3f}%")
    print(f"{'TE (ddof=1)':<20} {te_n1.mean()*100:>9.3f}% "
          f"{(te_n1.mean()-te_annual_true)*100:>9.3f}% "
          f"{np.sqrt(np.mean((te_n1-te_annual_true)**2))*100:>9.3f}%")
    print(f"{'True':<20} {te_annual_true*100:>9.3f}%")
    
    print(f"\nNote: Even with Bessel's correction, E[S] < sigma (Jensen's inequality).")
    print(f"Both estimators underestimate sigma, but S^2 is unbiased for sigma^2.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("BESSEL'S CORRECTION: DEMONSTRATIONS")
    print("=" * 60)
    
    unbiasedness_verification()
    chi_squared_distribution()
    std_bias_demo()
    software_defaults_demo()
    independence_demo()
    tracking_error_demo()
