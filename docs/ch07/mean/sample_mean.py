"""
Sample Mean as Estimator: Demonstrations and Implementations
=============================================================
This module provides comprehensive demonstrations:
  - Properties verification (unbiasedness, variance, MSE)
  - Sampling distribution and CLT verification
  - Efficiency comparisons with alternative estimators
  - Effect of correlated observations
  - Financial application: expected return estimation precision
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Properties Verification
# ============================================================

def properties_verification(mu=10.0, sigma=3.0, n_simulations=100_000, seed=42):
    """Empirically verify unbiasedness, variance, and MSE of sample mean."""
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 25, 50, 100, 500]
    
    print("=" * 70)
    print("Sample Mean Properties Verification")
    print(f"Population: mu = {mu}, sigma = {sigma}")
    print("=" * 70)
    print(f"\n{'n':>6} {'E[X_bar]':>10} {'Bias':>10} {'Var(X_bar)':>12} "
          f"{'sig2/n':>12} {'MSE':>12}")
    print("-" * 65)
    
    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_simulations, n))
        x_bars = samples.mean(axis=1)
        
        print(f"{n:>6} {x_bars.mean():>10.4f} {x_bars.mean()-mu:>10.6f} "
              f"{x_bars.var():>12.6f} {sigma**2/n:>12.6f} "
              f"{np.mean((x_bars-mu)**2):>12.6f}")


# ============================================================
# 2. Sampling Distribution and CLT
# ============================================================

def sampling_distribution_clt(seed=42):
    """Visualize CLT: sampling distribution of X_bar approaches normal."""
    rng = np.random.default_rng(seed)
    n_sim = 20_000
    
    populations = {
        'Normal(5, 4)': (lambda n: rng.normal(5, 2, n), 5.0, 4.0),
        'Exp(lam=0.5)': (lambda n: rng.exponential(2, n), 2.0, 4.0),
        'Uniform(0,10)': (lambda n: rng.uniform(0, 10, n), 5.0, 100/12),
        'Bernoulli(0.3)': (lambda n: rng.binomial(1, 0.3, n), 0.3, 0.21),
    }
    
    sample_sizes = [2, 5, 30]
    fig, axes = plt.subplots(len(populations), len(sample_sizes), figsize=(15, 12))
    
    for i, (pop_name, (sampler, mu, sigma2)) in enumerate(populations.items()):
        for j, n in enumerate(sample_sizes):
            x_bars = np.array([sampler(n).mean() for _ in range(n_sim)])
            ax = axes[i, j]
            ax.hist(x_bars, bins=60, density=True, alpha=0.6, color='steelblue',
                    edgecolor='white')
            x = np.linspace(x_bars.min(), x_bars.max(), 200)
            se = np.sqrt(sigma2 / n)
            ax.plot(x, stats.norm.pdf(x, mu, se), 'r-', linewidth=2)
            if i == 0: ax.set_title(f'n = {n}', fontsize=12)
            if j == 0: ax.set_ylabel(pop_name, fontsize=10)
            ax.grid(True, alpha=0.2)
    
    plt.suptitle('Sampling Distribution of X_bar (CLT)\nRed: N(mu, sig2/n)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_mean_clt.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. Efficiency Comparison
# ============================================================

def efficiency_comparison(seed=42):
    """Compare mean vs median vs trimmed mean for different distributions."""
    rng = np.random.default_rng(seed)
    n, n_sim = 30, 50_000
    
    print("\n" + "=" * 70)
    print(f"Efficiency Comparison (n = {n})")
    print("=" * 70)
    
    distributions = {
        'Normal(0,1)': lambda: rng.standard_normal(n),
        't(df=3)': lambda: rng.standard_t(3, n),
        't(df=5)': lambda: rng.standard_t(5, n),
        'Contaminated': lambda: np.where(
            rng.uniform(0, 1, n) < 0.1, rng.normal(0, 10, n), rng.standard_normal(n)),
    }
    
    for dist_name, sampler in distributions.items():
        est_vals = {'Mean': [], 'Median': [], 'Trim10%': [], 'Trim20%': []}
        for _ in range(n_sim):
            s = sampler()
            est_vals['Mean'].append(np.mean(s))
            est_vals['Median'].append(np.median(s))
            est_vals['Trim10%'].append(stats.trim_mean(s, 0.1))
            est_vals['Trim20%'].append(stats.trim_mean(s, 0.2))
        
        mse_mean = np.mean(np.array(est_vals['Mean'])**2)
        print(f"\n{dist_name}:")
        print(f"  {'Estimator':<12} {'MSE':>10} {'RE vs Mean':>12}")
        print("  " + "-" * 36)
        for name, vals in est_vals.items():
            mse = np.mean(np.array(vals)**2)
            re = mse_mean / mse
            print(f"  {name:<12} {mse:>10.6f} {re:>12.4f}")


# ============================================================
# 4. Correlated Observations
# ============================================================

def correlated_observations_demo(seed=42):
    """Show how autocorrelation affects Var(X_bar)."""
    rng = np.random.default_rng(seed)
    n, n_sim, sigma = 100, 30_000, 1.0
    rho_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95]
    
    print("\n" + "=" * 60)
    print("Effect of Autocorrelation on Var(X_bar)")
    print(f"AR(1) process, n = {n}")
    print("=" * 60)
    print(f"\n{'rho':>8} {'Var(X_bar)':>14} {'sig2/n':>10} {'Ratio':>8}")
    print("-" * 44)
    
    results = []
    for rho in rho_values:
        x_bars = []
        for _ in range(n_sim):
            x = np.zeros(n)
            innov_sig = sigma * np.sqrt(max(1 - rho**2, 0.01))
            x[0] = rng.normal(0, sigma)
            for t in range(1, n):
                x[t] = rho * x[t-1] + rng.normal(0, innov_sig)
            x_bars.append(x.mean())
        
        var_emp = np.var(x_bars)
        var_iid = sigma**2 / n
        ratio = var_emp / var_iid
        results.append((rho, var_emp, ratio))
        print(f"{rho:>8.2f} {var_emp:>14.6f} {var_iid:>10.6f} {ratio:>8.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    rhos = [r[0] for r in results]
    ratios = [r[2] for r in results]
    ax.bar(range(len(rhos)), ratios, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f'{r:.2f}' for r in rhos])
    ax.axhline(1, color='red', linestyle='--', label='iid baseline')
    ax.set_xlabel('Autocorrelation rho', fontsize=12)
    ax.set_ylabel('Var(X_bar) / (sig2/n)', fontsize=12)
    ax.set_title('Variance Inflation from Autocorrelation', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sample_mean_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. Financial Application: Return Estimation Precision
# ============================================================

def return_estimation_precision(seed=42):
    """
    Demonstrate the imprecision of expected return estimation.
    Show confidence intervals for the expected return estimate.
    """
    rng = np.random.default_rng(seed)
    
    # Typical stock parameters
    mu_annual = 0.08   # 8% expected annual return
    sigma_annual = 0.20  # 20% annual volatility
    
    print("\n" + "=" * 60)
    print("Expected Return Estimation Precision")
    print(f"True annual return: {mu_annual*100:.1f}%, Volatility: {sigma_annual*100:.1f}%")
    print("=" * 60)
    
    years = [5, 10, 20, 30, 50, 100]
    print(f"\n{'Years':>8} {'SE':>10} {'95% CI':>25} {'CI Width':>10}")
    print("-" * 58)
    
    for T in years:
        se = sigma_annual / np.sqrt(T)
        ci_low = mu_annual - 1.96 * se
        ci_high = mu_annual + 1.96 * se
        ci_width = ci_high - ci_low
        print(f"{T:>8} {se*100:>9.2f}% [{ci_low*100:>6.2f}%, {ci_high*100:>6.2f}%] "
              f"{ci_width*100:>9.2f}%")
    
    print("\nEven with 50 years of data, the 95% CI is about ±5.5%.")
    print("Expected returns are fundamentally hard to estimate precisely.")
    
    # Simulation: distribution of Sharpe ratio estimates
    n_sim = 20_000
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    for ax, T, title in [(axes[0], 10, '10 Years'), (axes[1], 50, '50 Years')]:
        # Monthly returns
        n_months = T * 12
        mu_m = mu_annual / 12
        sig_m = sigma_annual / np.sqrt(12)
        
        mean_ests = []
        sr_ests = []
        for _ in range(n_sim):
            returns = rng.normal(mu_m, sig_m, n_months)
            mean_ests.append(returns.mean() * 12)  # Annualize
            sr_ests.append(returns.mean() / returns.std() * np.sqrt(12))
        
        ax.hist(mean_ests, bins=60, density=True, alpha=0.6, color='steelblue',
                edgecolor='white')
        ax.axvline(mu_annual, color='red', linestyle='--', linewidth=2,
                   label=f'True mu = {mu_annual*100:.0f}%')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Estimated Annual Return', fontsize=11)
        ax.set_title(f'{title} of Monthly Data', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Expected Return Estimates', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_mean_return_precision.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. Weighted Mean
# ============================================================

def weighted_mean_demo(seed=42):
    """Demonstrate inverse-variance weighted mean."""
    rng = np.random.default_rng(seed)
    mu = 5.0
    n_sim = 50_000
    
    # Observations with different precisions
    sigmas = np.array([1.0, 2.0, 5.0, 10.0, 0.5])
    n_obs = len(sigmas)
    
    print("\n" + "=" * 60)
    print("Weighted vs Unweighted Mean")
    print(f"True mu = {mu}, Observation SDs: {sigmas}")
    print("=" * 60)
    
    unweighted_ests = []
    weighted_ests = []
    optimal_weights = 1 / sigmas**2
    optimal_weights /= optimal_weights.sum()
    
    for _ in range(n_sim):
        obs = rng.normal(mu, sigmas)
        unweighted_ests.append(obs.mean())
        weighted_ests.append(np.sum(optimal_weights * obs))
    
    unweighted_ests = np.array(unweighted_ests)
    weighted_ests = np.array(weighted_ests)
    
    print(f"\n{'Estimator':<20} {'Mean':>8} {'Var':>10} {'MSE':>10}")
    print("-" * 52)
    print(f"{'Unweighted mean':<20} {unweighted_ests.mean():>8.4f} "
          f"{unweighted_ests.var():>10.6f} {np.mean((unweighted_ests-mu)**2):>10.6f}")
    print(f"{'IV-weighted mean':<20} {weighted_ests.mean():>8.4f} "
          f"{weighted_ests.var():>10.6f} {np.mean((weighted_ests-mu)**2):>10.6f}")
    print(f"\nOptimal weights: {np.round(optimal_weights, 4)}")
    print(f"Variance reduction: {(1 - weighted_ests.var()/unweighted_ests.var())*100:.1f}%")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("SAMPLE MEAN AS ESTIMATOR: DEMONSTRATIONS")
    print("=" * 60)
    
    properties_verification()
    sampling_distribution_clt()
    efficiency_comparison()
    correlated_observations_demo()
    return_estimation_precision()
    weighted_mean_demo()
