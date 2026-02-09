"""
Consistency and Convergence
============================
Demonstrate convergence properties of the sample mean:
  - Consistency: X̄ →ᵖ μ (WLLN) and X̄ → μ a.s. (SLLN)
  - CLT: sampling distribution approaches Normal
  - Convergence failure for Cauchy distribution
  - Autocorrelation effects on Var(X̄)
  - Financial: estimation horizon for detecting positive expected returns
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def consistency_visualization(seed=42):
    """
    Visualize the Strong Law of Large Numbers: running averages
    converge to the true mean as n grows.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 20
    
    distributions = {
        'Normal(5, 9)': (lambda: rng.normal(5, 3, N), 5.0),
        'Exp(λ=0.5)': (lambda: rng.exponential(2, N), 2.0),
        'Uniform(0, 10)': (lambda: rng.uniform(0, 10, N), 5.0),
        'Chi²(df=5)': (lambda: rng.chisquare(5, N), 5.0),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, (name, (sampler, true_mu)) in zip(axes.flat, distributions.items()):
        for _ in range(n_runs):
            data = sampler()
            running_mean = np.cumsum(data) / np.arange(1, N + 1)
            ax.plot(running_mean, alpha=0.2, linewidth=0.5)
        
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'μ = {true_mu}')
        ax.set_xscale('log')
        ax.set_xlabel('n')
        ax.set_ylabel('X̄ₙ')
        ax.set_title(f'{name}: SLLN', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Strong Law of Large Numbers: X̄ₙ → μ  a.s.',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('consistency_slln.png', dpi=150, bbox_inches='tight')
    plt.show()


def clt_demonstration(seed=42):
    """
    Visualize the Central Limit Theorem: sampling distribution of
    X̄ converges to Normal regardless of population shape.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    n_sim = 20_000
    
    populations = {
        'Normal(5, 4)': (lambda n: rng.normal(5, 2, n), 5.0, 4.0),
        'Exp(λ=0.5)': (lambda n: rng.exponential(2, n), 2.0, 4.0),
        'Uniform(0, 10)': (lambda n: rng.uniform(0, 10, n), 5.0, 100 / 12),
        'Bernoulli(0.3)': (lambda n: rng.binomial(1, 0.3, n), 0.3, 0.21),
    }
    
    sample_sizes = [2, 5, 30]
    fig, axes = plt.subplots(len(populations), len(sample_sizes), figsize=(15, 12))
    
    for i, (pop_name, (sampler, mu, sigma2)) in enumerate(populations.items()):
        for j, n in enumerate(sample_sizes):
            x_bars = np.array([sampler(n).mean() for _ in range(n_sim)])
            ax = axes[i, j]
            ax.hist(x_bars, bins=60, density=True, alpha=0.6,
                    color='steelblue', edgecolor='white')
            x = np.linspace(x_bars.min(), x_bars.max(), 200)
            se = np.sqrt(sigma2 / n)
            ax.plot(x, stats.norm.pdf(x, mu, se), 'r-', linewidth=2)
            if i == 0:
                ax.set_title(f'n = {n}', fontsize=12)
            if j == 0:
                ax.set_ylabel(pop_name, fontsize=10)
            ax.grid(True, alpha=0.2)
    
    plt.suptitle('Central Limit Theorem: X̄ → N(μ, σ²/n)\n'
                 'Red curve: theoretical normal',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('consistency_clt.png', dpi=150, bbox_inches='tight')
    plt.show()


def cauchy_failure(seed=42):
    """
    Show that the sample mean does NOT converge for Cauchy data
    (the mean does not exist).
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal — converges
    ax = axes[0]
    for _ in range(n_runs):
        data = rng.standard_normal(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='True μ = 0')
    ax.set_ylim(-2, 2)
    ax.set_xlabel('n'); ax.set_ylabel('X̄ₙ')
    ax.set_title('Normal: Converges', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Cauchy — does NOT converge
    ax = axes[1]
    for _ in range(n_runs):
        data = rng.standard_cauchy(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Median = 0')
    ax.set_xlabel('n'); ax.set_ylabel('X̄ₙ')
    ax.set_title('Cauchy: Does NOT Converge', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.suptitle('Consistency Failure: Cauchy (E[|X|] = ∞)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('consistency_cauchy_failure.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Cauchy: E[X] undefined → X̄ does not converge.")
    print("The median IS consistent for the Cauchy location parameter.")


def autocorrelation_effect(n=100, n_sim=30_000, seed=42):
    """
    Show how autocorrelation inflates/deflates Var(X̄) relative to σ²/n.
    
    Parameters
    ----------
    n : int
        Time series length
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0
    rho_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95]
    
    print("=" * 55)
    print(f"Autocorrelation Effect on Var(X̄) — AR(1), n = {n}")
    print("=" * 55)
    print(f"\n{'ρ':>6} {'Var(X̄)':>12} {'σ²/n':>10} {'Ratio':>8}")
    print("-" * 40)
    
    ratios = []
    for rho in rho_values:
        x_bars = []
        innov_sig = sigma * np.sqrt(max(1 - rho**2, 0.01))
        for _ in range(n_sim):
            x = np.zeros(n)
            x[0] = rng.normal(0, sigma)
            for t in range(1, n):
                x[t] = rho * x[t - 1] + rng.normal(0, innov_sig)
            x_bars.append(x.mean())
        
        var_emp = np.var(x_bars)
        var_iid = sigma**2 / n
        ratio = var_emp / var_iid
        ratios.append(ratio)
        print(f"{rho:>6.2f} {var_emp:>12.6f} {var_iid:>10.6f} {ratio:>8.2f}")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(rho_values)), ratios, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(rho_values)))
    ax.set_xticklabels([f'{r:.2f}' for r in rho_values])
    ax.axhline(1, color='red', linestyle='--', label='iid baseline')
    ax.set_xlabel('Autocorrelation ρ', fontsize=12)
    ax.set_ylabel('Var(X̄) / (σ²/n)', fontsize=12)
    ax.set_title('Variance Inflation from Autocorrelation', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('consistency_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.show()


def estimation_horizon_analysis(seed=42):
    """
    How many years of data are needed to detect positive expected
    excess return with a given probability?
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    mu_annual = 0.06   # 6% equity premium
    sigma_annual = 0.20 # 20% volatility
    rf = 0.03            # risk-free rate
    
    print("\n" + "=" * 60)
    print("Estimation Horizon: Detecting Positive Excess Return")
    print(f"True excess return: {(mu_annual - rf)*100:.1f}%, σ = {sigma_annual*100}%")
    print("=" * 60)
    
    years = np.arange(1, 101)
    prob_detect = [stats.norm.cdf((mu_annual - rf) / (sigma_annual / np.sqrt(T)))
                   for T in years]
    
    for target in [0.80, 0.90, 0.95]:
        idx = np.argmax(np.array(prob_detect) >= target)
        print(f"  {target*100:.0f}% power: ~{years[idx]} years of data needed")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, prob_detect, 'b-', linewidth=2)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(0.80, color='orange', linestyle='--', alpha=0.5, label='80%')
    ax.set_xlabel('Years of Data', fontsize=12)
    ax.set_ylabel('P(X̄ > rₓ)', fontsize=12)
    ax.set_title('Probability of Detecting Positive Excess Return', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('consistency_horizon.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    consistency_visualization()
    clt_demonstration()
    cauchy_failure()
    autocorrelation_effect()
    estimation_horizon_analysis()
