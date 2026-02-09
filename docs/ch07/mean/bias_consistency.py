"""
Bias and Consistency of the Sample Mean
========================================
Demonstrations:
  - Unbiasedness verification across distributions
  - Consistency: convergence as n grows
  - Convergence rate (1/sqrt(n))
  - Strong Law of Large Numbers visualization
  - Failure cases: Cauchy distribution
  - Financial: estimation horizon analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. Unbiasedness Across Distributions
# ============================================================

def unbiasedness_verification(n_sim=100_000, seed=42):
    """Verify unbiasedness of sample mean for multiple distributions."""
    rng = np.random.default_rng(seed)
    n = 20
    
    distributions = {
        'Normal(5, 2)': (lambda: rng.normal(5, 2, n), 5.0),
        'Exp(lam=0.5)': (lambda: rng.exponential(2, n), 2.0),
        'Poisson(3.7)': (lambda: rng.poisson(3.7, n), 3.7),
        'Uniform(2,8)': (lambda: rng.uniform(2, 8, n), 5.0),
        'Bernoulli(0.4)': (lambda: rng.binomial(1, 0.4, n), 0.4),
        'Chi2(df=5)': (lambda: rng.chisquare(5, n), 5.0),
    }
    
    print("=" * 60)
    print(f"Unbiasedness Verification (n={n}, sims={n_sim:,})")
    print("=" * 60)
    print(f"\n{'Distribution':<20} {'True mu':>10} {'E[X_bar]':>10} {'Bias':>12}")
    print("-" * 55)
    
    for name, (sampler, true_mu) in distributions.items():
        estimates = np.array([sampler().mean() for _ in range(n_sim)])
        bias = estimates.mean() - true_mu
        print(f"{name:<20} {true_mu:>10.4f} {estimates.mean():>10.4f} {bias:>12.6f}")
    
    print("\nAll biases are essentially zero (within simulation noise).")


# ============================================================
# 2. Consistency: Convergence as n Grows
# ============================================================

def consistency_demo(seed=42):
    """Show sample mean converging to true mean as n increases."""
    rng = np.random.default_rng(seed)
    mu = 5.0
    sigma = 3.0
    
    sample_sizes = np.concatenate([np.arange(1, 100), np.arange(100, 1001, 10),
                                    np.arange(1000, 10001, 100)])
    
    # Several runs to show variability
    n_runs = 20
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distributions = {
        'Normal(5, 9)': (lambda n: rng.normal(mu, sigma, n), mu),
        'Exp(lam=0.2)': (lambda n: rng.exponential(5, n), 5.0),
        'Uniform(0, 10)': (lambda n: rng.uniform(0, 10, n), 5.0),
        'Chi2(df=5)': (lambda n: rng.chisquare(5, n), 5.0),
    }
    
    for ax, (name, (sampler, true_mu)) in zip(axes.flat, distributions.items()):
        for _ in range(n_runs):
            data = sampler(max(sample_sizes))
            running_means = np.cumsum(data[:max(sample_sizes)]) / np.arange(1, max(sample_sizes)+1)
            ax.plot(np.arange(1, max(sample_sizes)+1), running_means, alpha=0.2, linewidth=0.5)
        
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2, label=f'mu = {true_mu}')
        ax.set_xscale('log')
        ax.set_xlabel('Sample Size n')
        ax.set_ylabel('X_bar')
        ax.set_title(f'{name}: Consistency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Consistency of the Sample Mean (SLLN)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bias_consistency_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. Convergence Rate
# ============================================================

def convergence_rate_demo(mu=5.0, sigma=3.0, n_sim=50_000, seed=42):
    """Verify the 1/sqrt(n) convergence rate of standard error."""
    rng = np.random.default_rng(seed)
    
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    empirical_se = []
    theoretical_se = []
    
    for n in sample_sizes:
        estimates = np.array([rng.normal(mu, sigma, n).mean() for _ in range(n_sim)])
        empirical_se.append(estimates.std())
        theoretical_se.append(sigma / np.sqrt(n))
    
    print("\n" + "=" * 50)
    print("Convergence Rate: SE = sigma/sqrt(n)")
    print("=" * 50)
    print(f"\n{'n':>8} {'SE(emp)':>12} {'sig/sqrt(n)':>12} {'Ratio':>8}")
    print("-" * 44)
    for n, se_e, se_t in zip(sample_sizes, empirical_se, theoretical_se):
        print(f"{n:>8} {se_e:>12.6f} {se_t:>12.6f} {se_e/se_t:>8.4f}")
    
    # Log-log plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(sample_sizes, empirical_se, 'bo-', label='Empirical SE')
    ax.loglog(sample_sizes, theoretical_se, 'r--', label='sigma/sqrt(n)', linewidth=2)
    
    # Reference slope
    n_ref = np.array([5, 5000])
    ax.loglog(n_ref, sigma * n_ref**(-0.5), 'g:', alpha=0.5, label='Slope = -1/2')
    
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.set_title('Convergence Rate of Sample Mean', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('bias_consistency_rate.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. Failure Case: Cauchy Distribution
# ============================================================

def cauchy_failure_demo(seed=42):
    """
    Show that the sample mean does NOT converge for Cauchy data
    (infinite mean / undefined expectation).
    """
    rng = np.random.default_rng(seed)
    
    N = 10000
    n_runs = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal (converges)
    ax = axes[0]
    for _ in range(n_runs):
        data = rng.standard_normal(N)
        running_mean = np.cumsum(data) / np.arange(1, N+1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='True mean = 0')
    ax.set_xlabel('n')
    ax.set_ylabel('X_bar')
    ax.set_title('Normal: Sample Mean Converges', fontsize=12)
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cauchy (does NOT converge)
    ax = axes[1]
    for _ in range(n_runs):
        data = rng.standard_cauchy(N)
        running_mean = np.cumsum(data) / np.arange(1, N+1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Median = 0')
    ax.set_xlabel('n')
    ax.set_ylabel('X_bar')
    ax.set_title('Cauchy: Sample Mean Does NOT Converge', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Consistency Failure: Cauchy Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bias_consistency_cauchy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("Cauchy Distribution: Sample Mean Failure")
    print("=" * 60)
    print("The Cauchy distribution has no finite mean (E[|X|] = inf).")
    print("The sample mean does not converge — it keeps fluctuating")
    print("regardless of sample size. The median IS consistent.")


# ============================================================
# 5. MSE Consistency Verification
# ============================================================

def mse_consistency_demo(mu=5.0, sigma=3.0, n_sim=50_000, seed=42):
    """Verify MSE -> 0 as n -> infinity."""
    rng = np.random.default_rng(seed)
    
    sample_sizes = [5, 10, 25, 50, 100, 250, 500, 1000, 5000]
    
    print("\n" + "=" * 50)
    print("MSE Consistency: MSE -> 0 as n -> inf")
    print("=" * 50)
    print(f"\n{'n':>8} {'MSE':>12} {'Bias^2':>12} {'Var':>12}")
    print("-" * 48)
    
    for n in sample_sizes:
        estimates = np.array([rng.normal(mu, sigma, n).mean() for _ in range(n_sim)])
        mse = np.mean((estimates - mu)**2)
        bias_sq = (estimates.mean() - mu)**2
        var = estimates.var()
        print(f"{n:>8} {mse:>12.6f} {bias_sq:>12.8f} {var:>12.6f}")
    
    print("\nMSE = Var -> 0 (Bias^2 always ≈ 0)")


# ============================================================
# 6. Financial: Estimation Horizon Analysis
# ============================================================

def estimation_horizon_analysis(seed=42):
    """
    How many years of data do you need to be confident that
    expected return > 0 (or > risk-free rate)?
    """
    rng = np.random.default_rng(seed)
    
    mu_annual = 0.06  # 6% equity premium
    sigma_annual = 0.20  # 20% volatility
    rf = 0.03  # risk-free rate
    
    print("\n" + "=" * 60)
    print("How Long to Detect Positive Expected Excess Return?")
    print(f"True excess return: {(mu_annual-rf)*100:.1f}%, Volatility: {sigma_annual*100}%")
    print("=" * 60)
    
    years = np.arange(1, 101)
    # Probability that X_bar > rf (i.e., detected positive excess return)
    prob_detect = []
    for T in years:
        se = sigma_annual / np.sqrt(T)
        z = (mu_annual - rf) / se
        prob_detect.append(stats.norm.cdf(z))
    
    # Find years needed for 80%, 90%, 95% power
    for target in [0.80, 0.90, 0.95]:
        idx = np.argmax(np.array(prob_detect) >= target)
        print(f"  {target*100:.0f}% detection probability: ~{years[idx]} years")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, prob_detect, 'b-', linewidth=2)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='80%')
    ax.set_xlabel('Years of Data', fontsize=12)
    ax.set_ylabel('P(X_bar > r_f)', fontsize=12)
    ax.set_title('Probability of Detecting Positive Excess Return', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bias_consistency_horizon.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("BIAS AND CONSISTENCY OF THE SAMPLE MEAN")
    print("=" * 60)
    
    unbiasedness_verification()
    consistency_demo()
    convergence_rate_demo()
    cauchy_failure_demo()
    mse_consistency_demo()
    estimation_horizon_analysis()
