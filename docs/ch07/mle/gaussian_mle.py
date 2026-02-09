"""
MLE of μ and σ² for the Gaussian Distribution
===============================================
Demonstrations:
  - Derivation verification (analytical vs numerical MLE)
  - Log-likelihood surface visualization
  - Finite-sample bias of sigma^2 MLE
  - Fisher information and CRLB
  - Confidence intervals (z, t, chi2)
  - Financial: VaR estimation under normality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize


# ============================================================
# 1. MLE Derivation Verification
# ============================================================

def mle_verification(seed=42):
    """
    Verify analytical MLEs match numerical optimization
    of the log-likelihood.
    """
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 2.0
    n = 50
    data = rng.normal(mu_true, sigma_true, n)
    
    # Analytical MLEs
    mu_mle = data.mean()
    sigma2_mle = np.mean((data - mu_mle)**2)
    
    # Numerical MLE
    def neg_loglik(params):
        mu, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)  # Ensure positivity
        return n/2 * np.log(2*np.pi*sigma2) + np.sum((data - mu)**2) / (2*sigma2)
    
    result = optimize.minimize(neg_loglik, x0=[0, 0], method='Nelder-Mead')
    mu_num, sigma2_num = result.x[0], np.exp(result.x[1])
    
    print("=" * 60)
    print("Gaussian MLE Verification")
    print(f"True: mu = {mu_true}, sigma^2 = {sigma_true**2}")
    print("=" * 60)
    print(f"\n{'Method':<15} {'mu_hat':>10} {'sigma2_hat':>12}")
    print("-" * 40)
    print(f"{'Analytical':<15} {mu_mle:>10.6f} {sigma2_mle:>12.6f}")
    print(f"{'Numerical':<15} {mu_num:>10.6f} {sigma2_num:>12.6f}")
    print(f"\nDifference: mu = {abs(mu_mle - mu_num):.2e}, "
          f"sigma^2 = {abs(sigma2_mle - sigma2_num):.2e}")
    
    # Also show Bessel-corrected
    s2 = np.var(data, ddof=1)
    print(f"\nBessel-corrected S^2 = {s2:.6f}")
    print(f"Ratio MLE/Bessel = {sigma2_mle/s2:.6f} = {n-1}/{n} = {(n-1)/n:.6f}")


# ============================================================
# 2. Log-Likelihood Surface
# ============================================================

def loglikelihood_surface(seed=42):
    """Visualize the log-likelihood as a function of (mu, sigma^2)."""
    rng = np.random.default_rng(seed)
    n = 30
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)
    
    mu_mle = data.mean()
    sigma2_mle = np.mean((data - mu_mle)**2)
    
    # Create grid
    mu_range = np.linspace(mu_mle - 2, mu_mle + 2, 200)
    sigma2_range = np.linspace(sigma2_mle * 0.3, sigma2_mle * 3, 200)
    MU, SIG2 = np.meshgrid(mu_range, sigma2_range)
    
    # Compute log-likelihood on grid
    LL = np.zeros_like(MU)
    for i in range(MU.shape[0]):
        for j in range(MU.shape[1]):
            mu, s2 = MU[i, j], SIG2[i, j]
            LL[i, j] = -n/2 * np.log(2*np.pi*s2) - np.sum((data - mu)**2) / (2*s2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Contour plot
    ax = axes[0]
    cs = ax.contour(MU, SIG2, LL, levels=30, cmap='viridis')
    ax.plot(mu_mle, sigma2_mle, 'r*', markersize=15, label='MLE')
    ax.plot(mu_true, sigma_true**2, 'g+', markersize=15, markeredgewidth=3, label='True')
    ax.set_xlabel('mu', fontsize=12)
    ax.set_ylabel('sigma^2', fontsize=12)
    ax.set_title('Log-Likelihood Contours', fontsize=12)
    ax.legend(fontsize=10)
    
    # Profile for mu
    ax = axes[1]
    profile_mu = np.array([-np.sum((data - mu)**2) / (2*sigma2_mle) 
                            for mu in mu_range])
    profile_mu -= profile_mu.max()
    ax.plot(mu_range, profile_mu, 'b-', linewidth=2)
    ax.axvline(mu_mle, color='red', linestyle='--', label=f'MLE = {mu_mle:.3f}')
    ax.axhline(-stats.chi2.ppf(0.95, 1)/2, color='orange', linestyle=':', alpha=0.7,
               label='95% CI')
    ax.set_xlabel('mu', fontsize=12)
    ax.set_ylabel('Relative log-likelihood', fontsize=12)
    ax.set_title('Profile for mu', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Profile for sigma^2
    ax = axes[2]
    profile_sig = np.array([-n/2*np.log(s2) - np.sum((data-mu_mle)**2)/(2*s2)
                             for s2 in sigma2_range])
    profile_sig -= profile_sig.max()
    ax.plot(sigma2_range, profile_sig, 'b-', linewidth=2)
    ax.axvline(sigma2_mle, color='red', linestyle='--', label=f'MLE = {sigma2_mle:.3f}')
    ax.set_xlabel('sigma^2', fontsize=12)
    ax.set_ylabel('Relative log-likelihood', fontsize=12)
    ax.set_title('Profile for sigma^2', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gaussian Log-Likelihood Surface', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gaussian_mle_surface.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. Finite-Sample Bias Analysis
# ============================================================

def finite_sample_bias(n_sim=200_000, seed=42):
    """Verify bias of sigma^2 MLE across sample sizes."""
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 3.0
    sigma2_true = sigma_true**2
    
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("\n" + "=" * 75)
    print("Finite-Sample Properties of Gaussian MLEs")
    print(f"True: mu = {mu_true}, sigma^2 = {sigma2_true}")
    print("=" * 75)
    
    print(f"\n{'n':>6} {'E[mu_hat]':>10} {'E[sig2_mle]':>12} {'E[S^2]':>10} "
          f"{'Bias(mle)':>10} {'(n-1)/n':>8}")
    print("-" * 60)
    
    for n in sample_sizes:
        samples = rng.normal(mu_true, sigma_true, (n_sim, n))
        mu_hat = samples.mean(axis=1)
        sig2_mle = np.var(samples, axis=1, ddof=0)
        s2 = np.var(samples, axis=1, ddof=1)
        
        print(f"{n:>6} {mu_hat.mean():>10.4f} {sig2_mle.mean():>12.4f} "
              f"{s2.mean():>10.4f} {sig2_mle.mean()-sigma2_true:>10.4f} "
              f"{(n-1)/n:>8.4f}")
    
    print(f"\nE[sig2_mle] = (n-1)/n * sigma^2 = {sigma2_true} * (n-1)/n")
    print(f"Bias = -sigma^2/n -> 0 as n -> inf")


# ============================================================
# 4. Fisher Information and CRLB
# ============================================================

def fisher_information_crlb(sigma=3.0, n_sim=100_000, seed=42):
    """Verify MLEs achieve (or approach) CRLB."""
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    
    sample_sizes = [10, 25, 50, 100, 500]
    
    print("\n" + "=" * 70)
    print("Fisher Information and CRLB")
    print("=" * 70)
    
    print(f"\nFor mu (CRLB = sigma^2/n):")
    print(f"{'n':>6} {'Var(mu_hat)':>14} {'CRLB':>10} {'Ratio':>8}")
    print("-" * 42)
    
    for n in sample_sizes:
        mu_hats = np.array([rng.normal(5, sigma, n).mean() for _ in range(n_sim)])
        var_mu = np.var(mu_hats)
        crlb_mu = sigma2 / n
        print(f"{n:>6} {var_mu:>14.6f} {crlb_mu:>10.6f} {var_mu/crlb_mu:>8.4f}")
    
    print(f"\nFor sigma^2 (CRLB = 2*sigma^4/n):")
    print(f"{'n':>6} {'Var(sig2_mle)':>14} {'CRLB':>12} {'Ratio':>8} {'(n-1)/n':>8}")
    print("-" * 50)
    
    for n in sample_sizes:
        sig2_hats = np.array([np.var(rng.normal(5, sigma, n)) for _ in range(n_sim)])
        var_sig2 = np.var(sig2_hats)
        crlb_sig2 = 2 * sigma**4 / n
        theory_var = 2 * (n-1) * sigma**4 / n**2
        print(f"{n:>6} {var_sig2:>14.6f} {crlb_sig2:>12.6f} "
              f"{var_sig2/crlb_sig2:>8.4f} {(n-1)/n:>8.4f}")
    
    print(f"\nmu_hat achieves CRLB exactly (ratio ≈ 1.00)")
    print(f"sig2_mle ratio = (n-1)/n, approaches 1 asymptotically")


# ============================================================
# 5. Confidence Intervals
# ============================================================

def confidence_intervals_demo(seed=42):
    """Construct and verify coverage of confidence intervals."""
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 10.0, 3.0
    n = 25
    n_sim = 50_000
    alpha = 0.05
    
    # Track coverage
    z_cover = 0
    t_cover = 0
    chi2_cover = 0
    
    for _ in range(n_sim):
        data = rng.normal(mu_true, sigma_true, n)
        xbar = data.mean()
        s = data.std(ddof=1)
        s2 = data.var(ddof=1)
        
        # Z-interval for mu (known sigma)
        z_crit = stats.norm.ppf(1 - alpha/2)
        z_lo = xbar - z_crit * sigma_true / np.sqrt(n)
        z_hi = xbar + z_crit * sigma_true / np.sqrt(n)
        if z_lo <= mu_true <= z_hi:
            z_cover += 1
        
        # T-interval for mu (unknown sigma)
        t_crit = stats.t.ppf(1 - alpha/2, n-1)
        t_lo = xbar - t_crit * s / np.sqrt(n)
        t_hi = xbar + t_crit * s / np.sqrt(n)
        if t_lo <= mu_true <= t_hi:
            t_cover += 1
        
        # Chi2-interval for sigma^2
        chi2_lo = (n-1) * s2 / stats.chi2.ppf(1-alpha/2, n-1)
        chi2_hi = (n-1) * s2 / stats.chi2.ppf(alpha/2, n-1)
        if chi2_lo <= sigma_true**2 <= chi2_hi:
            chi2_cover += 1
    
    print("\n" + "=" * 60)
    print(f"Confidence Interval Coverage (n={n}, alpha={alpha})")
    print(f"True: mu={mu_true}, sigma^2={sigma_true**2}")
    print("=" * 60)
    print(f"\n{'Interval':<25} {'Target':>8} {'Actual':>8}")
    print("-" * 45)
    print(f"{'Z-interval (mu, sigma known)':<25} {1-alpha:>8.1%} {z_cover/n_sim:>8.1%}")
    print(f"{'T-interval (mu, sigma unk.)':<25} {1-alpha:>8.1%} {t_cover/n_sim:>8.1%}")
    print(f"{'Chi2-interval (sigma^2)':<25} {1-alpha:>8.1%} {chi2_cover/n_sim:>8.1%}")
    print(f"\nAll intervals achieve approximately {(1-alpha)*100}% coverage.")


# ============================================================
# 6. Financial: VaR Under Normality
# ============================================================

def var_estimation_demo(seed=42):
    """
    Value at Risk estimation using Gaussian MLE.
    Compare parametric (normal) VaR with historical VaR.
    """
    rng = np.random.default_rng(seed)
    
    # Simulate 2 years of daily returns
    mu_daily = 0.08 / 252
    sigma_daily = 0.20 / np.sqrt(252)
    n = 504  # 2 years
    
    # Generate from t-distribution (realistic fat tails)
    df = 5
    returns = mu_daily + sigma_daily * rng.standard_t(df, n) / np.sqrt(df/(df-2))
    
    # MLE estimates (assuming normality)
    mu_hat = returns.mean()
    sigma_hat = np.sqrt(np.mean((returns - mu_hat)**2))
    
    alpha_levels = [0.01, 0.025, 0.05, 0.10]
    
    print("\n" + "=" * 70)
    print("VaR Estimation: Gaussian MLE vs Historical")
    print(f"Data: {n} daily returns (generated from t(df={df}), estimated as Normal)")
    print("=" * 70)
    print(f"\nMLE: mu_hat = {mu_hat*252*100:.2f}% ann., "
          f"sigma_hat = {sigma_hat*np.sqrt(252)*100:.2f}% ann.")
    
    print(f"\n{'alpha':>8} {'Parametric VaR':>16} {'Historical VaR':>16} {'Ratio':>8}")
    print("-" * 52)
    
    for alpha in alpha_levels:
        # Parametric (normal) VaR
        var_param = -(mu_hat + stats.norm.ppf(alpha) * sigma_hat)
        
        # Historical VaR
        var_hist = -np.percentile(returns, alpha * 100)
        
        print(f"{alpha:>8.3f} {var_param*100:>15.3f}% {var_hist*100:>15.3f}% "
              f"{var_hist/var_param:>8.3f}")
    
    print(f"\nRatio > 1: Normal VaR underestimates tail risk (fat tails).")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(returns, bins=60, density=True, alpha=0.5, color='steelblue',
            edgecolor='white', label='Historical')
    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat), 'r-', linewidth=2,
            label=f'Normal MLE')
    
    # Mark VaR
    var_1 = -(mu_hat + stats.norm.ppf(0.01) * sigma_hat)
    var_hist_1 = -np.percentile(returns, 1)
    ax.axvline(-var_1, color='red', linestyle='--', alpha=0.7, label=f'Normal 1% VaR')
    ax.axvline(-var_hist_1, color='blue', linestyle='--', alpha=0.7, label=f'Historical 1% VaR')
    
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('VaR: Gaussian MLE vs Historical', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gaussian_mle_var.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 7. Multivariate Extension
# ============================================================

def multivariate_mle_demo(seed=42):
    """Brief demonstration of multivariate Gaussian MLE."""
    rng = np.random.default_rng(seed)
    
    # 3-asset example
    p = 3
    n = 100
    mu_true = np.array([0.08, 0.05, 0.12]) / 252
    
    # True covariance
    vols = np.array([0.20, 0.15, 0.25]) / np.sqrt(252)
    corr = np.array([[1.0, 0.6, 0.3],
                      [0.6, 1.0, 0.4],
                      [0.3, 0.4, 1.0]])
    Sigma_true = np.diag(vols) @ corr @ np.diag(vols)
    
    # Generate data
    returns = rng.multivariate_normal(mu_true, Sigma_true, n)
    
    # MLEs
    mu_hat = returns.mean(axis=0)
    Sigma_hat_mle = np.cov(returns, rowvar=False, ddof=0)  # Divide by n
    Sigma_hat_unbiased = np.cov(returns, rowvar=False, ddof=1)  # Divide by n-1
    
    print("\n" + "=" * 60)
    print("Multivariate Gaussian MLE")
    print(f"p = {p} assets, n = {n} observations")
    print("=" * 60)
    
    print(f"\nMean vector (annualized %):")
    print(f"  True:  {mu_true * 252 * 100}")
    print(f"  MLE:   {mu_hat * 252 * 100}")
    
    print(f"\nCovariance (annualized vol on diagonal, %):")
    print(f"  True vols: {vols * np.sqrt(252) * 100}")
    print(f"  MLE vols:  {np.sqrt(np.diag(Sigma_hat_mle) * 252) * 100}")
    
    print(f"\nCorrelation matrix:")
    print(f"  True:\n{corr}")
    corr_hat = np.corrcoef(returns.T)
    print(f"  MLE:\n{np.round(corr_hat, 3)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("GAUSSIAN MLE: DEMONSTRATIONS")
    print("=" * 60)
    
    mle_verification()
    loglikelihood_surface()
    finite_sample_bias()
    fisher_information_crlb()
    confidence_intervals_demo()
    var_estimation_demo()
    multivariate_mle_demo()
