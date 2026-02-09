"""
Gaussian MLE
=============
Maximum Likelihood Estimation for Normal distribution:
  - Analytical vs numerical MLE verification
  - Log-likelihood surface visualization
  - Finite-sample bias of σ² MLE
  - Fisher information and CRLB
  - Confidence intervals (z, t, χ²)
  - Multivariate Gaussian MLE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize


def mle_analytical_vs_numerical(seed=42):
    """
    Verify analytical MLEs match numerical optimization.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 2.0
    n = 50
    data = rng.normal(mu_true, sigma_true, n)
    
    mu_mle = data.mean()
    sigma2_mle = np.mean((data - mu_mle)**2)
    
    def neg_ll(params):
        mu, ls2 = params
        s2 = np.exp(ls2)
        return n/2*np.log(2*np.pi*s2) + np.sum((data-mu)**2)/(2*s2)
    
    res = optimize.minimize(neg_ll, [0, 0], method='Nelder-Mead')
    mu_num, s2_num = res.x[0], np.exp(res.x[1])
    
    print("=" * 55)
    print("Gaussian MLE: Analytical vs Numerical")
    print(f"True: μ = {mu_true}, σ² = {sigma_true**2}")
    print("=" * 55)
    print(f"\n{'Method':<14} {'μ̂':>10} {'σ̂²':>12}")
    print("-" * 38)
    print(f"{'Analytical':<14} {mu_mle:>10.6f} {sigma2_mle:>12.6f}")
    print(f"{'Numerical':<14} {mu_num:>10.6f} {s2_num:>12.6f}")
    print(f"\nBessel S² = {np.var(data, ddof=1):.6f}")
    print(f"MLE/Bessel = (n−1)/n = {(n-1)/n:.6f}")


def loglikelihood_surface(seed=42):
    """
    Visualize log-likelihood surface and profile likelihoods.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    n = 30
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, n)
    
    mu_mle = data.mean()
    s2_mle = np.mean((data - mu_mle)**2)
    
    mu_r = np.linspace(mu_mle - 2, mu_mle + 2, 200)
    s2_r = np.linspace(s2_mle * 0.3, s2_mle * 3, 200)
    MU, S2 = np.meshgrid(mu_r, s2_r)
    
    LL = np.zeros_like(MU)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            LL[i, j] = -n/2*np.log(2*np.pi*S2[i,j]) - np.sum((data-MU[i,j])**2)/(2*S2[i,j])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.contour(MU, S2, LL, levels=30, cmap='viridis')
    ax.plot(mu_mle, s2_mle, 'r*', ms=15, label='MLE')
    ax.plot(mu_true, sigma_true**2, 'g+', ms=15, mew=3, label='True')
    ax.set_xlabel('μ'); ax.set_ylabel('σ²')
    ax.set_title('Log-Likelihood Contours'); ax.legend()
    
    ax = axes[1]
    prof_mu = [-np.sum((data-m)**2)/(2*s2_mle) for m in mu_r]
    prof_mu = np.array(prof_mu) - max(prof_mu)
    ax.plot(mu_r, prof_mu, 'b-', lw=2)
    ax.axvline(mu_mle, color='red', ls='--', label=f'MLE={mu_mle:.3f}')
    ax.axhline(-stats.chi2.ppf(0.95,1)/2, color='orange', ls=':', label='95% cutoff')
    ax.set_xlabel('μ'); ax.set_title('Profile for μ'); ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    prof_s = [-n/2*np.log(s)-np.sum((data-mu_mle)**2)/(2*s) for s in s2_r]
    prof_s = np.array(prof_s) - max(prof_s)
    ax.plot(s2_r, prof_s, 'b-', lw=2)
    ax.axvline(s2_mle, color='red', ls='--', label=f'MLE={s2_mle:.3f}')
    ax.set_xlabel('σ²'); ax.set_title('Profile for σ²'); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gaussian Log-Likelihood', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gaussian_mle_surface.png', dpi=150, bbox_inches='tight')
    plt.show()


def finite_sample_bias(n_sim=200_000, seed=42):
    """
    Verify E[σ̂²_MLE] = (n−1)/n · σ².
    
    Parameters
    ----------
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 5.0, 3.0
    sigma2 = sigma_true**2
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("\n" + "=" * 70)
    print(f"Finite-Sample Bias of σ̂²_MLE (True σ² = {sigma2})")
    print("=" * 70)
    print(f"\n{'n':>6} {'E[μ̂]':>8} {'E[σ̂²_MLE]':>12} {'E[S²]':>10} {'Bias(MLE)':>10}")
    print("-" * 50)
    
    for n in sample_sizes:
        samp = rng.normal(mu_true, sigma_true, (n_sim, n))
        mu_hat = samp.mean(axis=1)
        s2_mle = np.var(samp, axis=1, ddof=0)
        s2_ub = np.var(samp, axis=1, ddof=1)
        print(f"{n:>6} {mu_hat.mean():>8.4f} {s2_mle.mean():>12.4f} "
              f"{s2_ub.mean():>10.4f} {s2_mle.mean()-sigma2:>10.4f}")


def fisher_information_crlb(sigma=3.0, n_sim=100_000, seed=42):
    """
    Verify MLEs approach CRLB.
    
    Parameters
    ----------
    sigma : float
        True standard deviation
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sample_sizes = [10, 25, 50, 100, 500]
    
    print("\n" + "=" * 65)
    print("Fisher Information & CRLB")
    print("=" * 65)
    
    print(f"\nFor μ:  CRLB = σ²/n = {sigma**2}/n")
    print(f"{'n':>6} {'Var(μ̂)':>12} {'CRLB':>10} {'Ratio':>8}")
    print("-" * 38)
    for n in sample_sizes:
        mu_h = np.array([rng.normal(5, sigma, n).mean() for _ in range(n_sim)])
        print(f"{n:>6} {mu_h.var():>12.6f} {sigma**2/n:>10.6f} "
              f"{mu_h.var()/(sigma**2/n):>8.4f}")
    
    print(f"\nFor σ²: CRLB = 2σ⁴/n")
    print(f"{'n':>6} {'Var(σ̂²)':>12} {'CRLB':>12} {'Ratio':>8}")
    print("-" * 38)
    for n in sample_sizes:
        s2_h = np.array([np.var(rng.normal(5, sigma, n)) for _ in range(n_sim)])
        print(f"{n:>6} {s2_h.var():>12.6f} {2*sigma**4/n:>12.6f} "
              f"{s2_h.var()/(2*sigma**4/n):>8.4f}")


def confidence_interval_coverage(seed=42):
    """
    Verify coverage of z, t, and χ² confidence intervals.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 10.0, 3.0
    n, alpha, n_sim = 25, 0.05, 50_000
    
    z_ok = t_ok = chi_ok = 0
    for _ in range(n_sim):
        d = rng.normal(mu_true, sigma_true, n)
        xb, s, s2 = d.mean(), d.std(ddof=1), d.var(ddof=1)
        
        z_c = stats.norm.ppf(1 - alpha/2)
        if xb - z_c*sigma_true/np.sqrt(n) <= mu_true <= xb + z_c*sigma_true/np.sqrt(n):
            z_ok += 1
        
        t_c = stats.t.ppf(1 - alpha/2, n-1)
        if xb - t_c*s/np.sqrt(n) <= mu_true <= xb + t_c*s/np.sqrt(n):
            t_ok += 1
        
        lo = (n-1)*s2 / stats.chi2.ppf(1-alpha/2, n-1)
        hi = (n-1)*s2 / stats.chi2.ppf(alpha/2, n-1)
        if lo <= sigma_true**2 <= hi:
            chi_ok += 1
    
    print("\n" + "=" * 55)
    print(f"CI Coverage (n={n}, α={alpha}, {n_sim:,} sims)")
    print("=" * 55)
    print(f"\n{'Interval':<28} {'Target':>8} {'Actual':>8}")
    print("-" * 48)
    print(f"{'z-interval (μ, σ known)':<28} {1-alpha:>8.1%} {z_ok/n_sim:>8.1%}")
    print(f"{'t-interval (μ, σ unknown)':<28} {1-alpha:>8.1%} {t_ok/n_sim:>8.1%}")
    print(f"{'χ²-interval (σ²)':<28} {1-alpha:>8.1%} {chi_ok/n_sim:>8.1%}")


def var_estimation_finance(seed=42):
    """
    Value at Risk under Gaussian MLE vs historical.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu_d = 0.08/252
    sig_d = 0.20/np.sqrt(252)
    n = 504
    df = 5
    
    returns = mu_d + sig_d * rng.standard_t(df, n) / np.sqrt(df/(df-2))
    mu_hat = returns.mean()
    sig_hat = np.sqrt(np.mean((returns - mu_hat)**2))
    
    print("\n" + "=" * 65)
    print("VaR: Gaussian MLE vs Historical")
    print("=" * 65)
    print(f"\n{'α':>8} {'Parametric VaR':>16} {'Historical VaR':>16} {'Ratio':>8}")
    print("-" * 52)
    for alpha in [0.01, 0.025, 0.05, 0.10]:
        v_p = -(mu_hat + stats.norm.ppf(alpha) * sig_hat)
        v_h = -np.percentile(returns, alpha * 100)
        print(f"{alpha:>8.3f} {v_p*100:>15.3f}% {v_h*100:>15.3f}% {v_h/v_p:>8.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=60, density=True, alpha=0.5, color='steelblue', edgecolor='white')
    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu_hat, sig_hat), 'r-', lw=2, label='Normal MLE')
    v1 = -(mu_hat + stats.norm.ppf(0.01)*sig_hat)
    vh1 = -np.percentile(returns, 1)
    ax.axvline(-v1, color='red', ls='--', alpha=0.7, label='Normal 1% VaR')
    ax.axvline(-vh1, color='blue', ls='--', alpha=0.7, label='Historical 1% VaR')
    ax.set_xlabel('Daily Return'); ax.set_title('VaR: Gaussian MLE vs Historical')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gaussian_mle_var.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    mle_analytical_vs_numerical()
    loglikelihood_surface()
    finite_sample_bias()
    fisher_information_crlb()
    confidence_interval_coverage()
    var_estimation_finance()
