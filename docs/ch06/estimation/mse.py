"""
Mean Squared Error: Demonstrations and Computations
====================================================
This module provides comprehensive demonstrations of MSE concepts:
  - MSE decomposition verification
  - Comparing estimators by MSE
  - MSE-optimal variance estimator
  - Relative efficiency calculations
  - James-Stein estimator demonstration
  - Financial application: forecast evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. MSE: Direct Computation and Decomposition
# ============================================================

def mse_computation_demo(mu_true=10.0, sigma=4.0, seed=42):
    """
    Compute MSE directly and via decomposition for multiple estimators.
    
    Parameters
    ----------
    mu_true : float
        True population mean
    sigma : float
        Population standard deviation
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    n = 20
    n_sim = 100_000
    
    samples = rng.normal(mu_true, sigma, (n_sim, n))
    x_bar = samples.mean(axis=1)
    
    # Define several estimators of mu
    estimators = {
        'X̄ (sample mean)': x_bar,
        '0.9·X̄ (shrunk)': 0.9 * x_bar,
        'X̄ + 1 (shifted)': x_bar + 1,
        'median': np.median(samples, axis=1),
        'midrange': (samples.max(axis=1) + samples.min(axis=1)) / 2,
    }
    
    print("=" * 75)
    print("MSE Comparison of Estimators for μ")
    print(f"True μ = {mu_true}, σ = {sigma}, n = {n}")
    print("=" * 75)
    print(f"{'Estimator':<22} {'MSE':>10} {'Bias':>10} {'Bias²':>10} "
          f"{'Var':>10} {'Var+Bias²':>10}")
    print("-" * 75)
    
    for name, est in estimators.items():
        mse_direct = np.mean((est - mu_true)**2)
        bias = np.mean(est) - mu_true
        bias_sq = bias**2
        var = np.var(est, ddof=0)
        decomp = var + bias_sq
        
        print(f"{name:<22} {mse_direct:>10.4f} {bias:>10.4f} {bias_sq:>10.4f} "
              f"{var:>10.4f} {decomp:>10.4f}")
    
    print("-" * 75)
    print("Note: MSE ≈ Var + Bias² confirms the decomposition for all estimators.")


# ============================================================
# 2. Comparing Variance Estimators by MSE
# ============================================================

def variance_estimator_mse(sigma_true=3.0, n_simulations=100_000, seed=42):
    """
    Compare MSE of different variance estimators:
      - S² (divide by n-1, unbiased)
      - S̃² (divide by n, biased)
      - Ŝ² (divide by n+1, MSE-optimal for normal)
    
    Parameters
    ----------
    sigma_true : float
        True population standard deviation
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigma2_true = sigma_true**2
    
    sample_sizes = [5, 10, 20, 50, 100, 200]
    
    print("\n" + "=" * 80)
    print("MSE Comparison: Variance Estimators (Normal Population)")
    print(f"True σ² = {sigma2_true}")
    print("=" * 80)
    
    results = {n: {} for n in sample_sizes}
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma_true, (n_simulations, n))
        ss = np.sum((samples - samples.mean(axis=1, keepdims=True))**2, axis=1)
        
        s2_n1 = ss / (n - 1)      # Unbiased (Bessel's)
        s2_n = ss / n              # Naive (biased)
        s2_n1_opt = ss / (n + 1)   # MSE-optimal
        
        for name, est in [('S²(n-1)', s2_n1), ('S̃²(n)', s2_n), ('Ŝ²(n+1)', s2_n1_opt)]:
            mse = np.mean((est - sigma2_true)**2)
            bias = np.mean(est) - sigma2_true
            results[n][name] = {'mse': mse, 'bias': bias}
    
    # Print table
    print(f"\n{'n':>6}", end='')
    for name in ['S²(n-1)', 'S̃²(n)', 'Ŝ²(n+1)']:
        print(f"  {'MSE':>10} {'Bias':>8}", end='')
    print()
    print("-" * 70)
    
    for n in sample_sizes:
        print(f"{n:>6}", end='')
        for name in ['S²(n-1)', 'S̃²(n)', 'Ŝ²(n+1)']:
            r = results[n][name]
            print(f"  {r['mse']:>10.4f} {r['bias']:>8.4f}", end='')
        print()
    
    print("\nNote: S̃²(n) and Ŝ²(n+1) are biased but have lower MSE than S²(n-1).")
    print("      Ŝ²(n+1) achieves the minimum MSE among all c·ΣΣ(Xi-X̄)² estimators.")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE comparison
    ax = axes[0]
    for name, marker, color in [('S²(n-1)', 'o', 'red'), ('S̃²(n)', 's', 'blue'),
                                  ('Ŝ²(n+1)', '^', 'green')]:
        mses = [results[n][name]['mse'] for n in sample_sizes]
        ax.plot(sample_sizes, mses, f'{color[0]}-{marker}', label=name, linewidth=1.5)
    
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE of Variance Estimators', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Theoretical MSE as function of divisor c
    ax = axes[1]
    n = 10
    sigma4 = sigma2_true**2
    c_vals = np.linspace(5, 20, 200)
    mse_c = ((n-1)/c_vals - 1)**2 * sigma4 + 2*(n-1)/c_vals**2 * sigma4
    
    ax.plot(c_vals, mse_c, 'b-', linewidth=2)
    for c, label, color in [(n-1, 'n-1 (unbiased)', 'red'), (n, 'n (naive)', 'blue'),
                             (n+1, 'n+1 (optimal)', 'green')]:
        mse_val = ((n-1)/c - 1)**2 * sigma4 + 2*(n-1)/c**2 * sigma4
        ax.scatter([c], [mse_val], color=color, s=100, zorder=5, label=f'c={c}: {label}')
    
    ax.set_xlabel('Divisor c', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(f'MSE vs Divisor c (n={n})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mse_variance_estimators.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 3. Relative Efficiency
# ============================================================

def relative_efficiency_demo(sigma=5.0, n_simulations=100_000, seed=42):
    """
    Compute relative efficiency of mean vs median for normal and
    heavy-tailed distributions.
    
    Parameters
    ----------
    sigma : float
        Scale parameter
    n_simulations : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    sample_sizes = [10, 20, 50, 100, 200]
    
    print("\n" + "=" * 70)
    print("Relative Efficiency: Mean vs Median")
    print("=" * 70)
    
    distributions = {
        'Normal(0, σ²)': lambda n: rng.normal(0, sigma, n),
        't(df=3)': lambda n: rng.standard_t(3, n) * sigma / np.sqrt(3),
        'Cauchy': lambda n: rng.standard_cauchy(n),
    }
    
    for dist_name, sampler in distributions.items():
        print(f"\nDistribution: {dist_name}")
        print(f"{'n':>8} {'MSE(mean)':>12} {'MSE(median)':>12} {'RE(med/mean)':>14}")
        print("-" * 50)
        
        for n in sample_sizes:
            mean_estimates = []
            median_estimates = []
            
            for _ in range(n_simulations):
                sample = sampler(n)
                mean_estimates.append(np.mean(sample))
                median_estimates.append(np.median(sample))
            
            mean_estimates = np.array(mean_estimates)
            median_estimates = np.array(median_estimates)
            
            mse_mean = np.mean(mean_estimates**2)    # True location is 0
            mse_median = np.mean(median_estimates**2)
            
            # Clip extreme MSE values for Cauchy
            if mse_mean > 1e6:
                re_str = "mean ≫ median"
            else:
                re = mse_mean / mse_median
                re_str = f"{re:.4f}"
            
            print(f"{n:>8} {mse_mean:>12.4f} {mse_median:>12.4f} {re_str:>14}")
    
    print("\nNote: For Normal, RE ≈ π/2 ≈ 1.57 (mean is more efficient).")
    print("      For heavy-tailed distributions, median can be more efficient.")


# ============================================================
# 4. James-Stein Estimator
# ============================================================

def james_stein_demo(p=10, mu_norm=1.0, n_simulations=50_000, seed=42):
    """
    Demonstrate that the James-Stein estimator dominates the sample mean
    for multivariate normal means when p >= 3.
    
    Parameters
    ----------
    p : int
        Dimension (number of means to estimate)
    mu_norm : float
        Norm of the true mean vector
    n_simulations : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    # True mean: spread mu_norm across dimensions
    mu = np.full(p, mu_norm / np.sqrt(p))
    
    print("\n" + "=" * 60)
    print(f"James-Stein Estimator (p = {p})")
    print("=" * 60)
    
    dimensions = [3, 5, 10, 20, 50]
    
    results_table = []
    
    for p in dimensions:
        mu = np.full(p, mu_norm / np.sqrt(p))
        
        mse_xbar_total = 0
        mse_js_total = 0
        
        for _ in range(n_simulations):
            X = rng.normal(mu, 1.0)  # Single observation per component
            
            # Sample mean estimator (= X itself with one observation)
            theta_xbar = X.copy()
            
            # James-Stein estimator
            shrinkage = max(0, 1 - (p - 2) / np.sum(X**2))
            theta_js = shrinkage * X
            
            mse_xbar_total += np.sum((theta_xbar - mu)**2)
            mse_js_total += np.sum((theta_js - mu)**2)
        
        mse_xbar = mse_xbar_total / n_simulations
        mse_js = mse_js_total / n_simulations
        improvement = (1 - mse_js / mse_xbar) * 100
        
        results_table.append((p, mse_xbar, mse_js, improvement))
    
    print(f"\n{'p':>5} {'MSE(X̄)':>10} {'MSE(JS)':>10} {'Improvement':>12}")
    print("-" * 40)
    for p, mse_x, mse_j, imp in results_table:
        print(f"{p:>5} {mse_x:>10.4f} {mse_j:>10.4f} {imp:>11.1f}%")
    
    print("\nThe James-Stein estimator always has lower total MSE when p ≥ 3.")
    print("This shows the sample mean is inadmissible in high dimensions!")


# ============================================================
# 5. Financial Application: Forecast Evaluation
# ============================================================

def forecast_evaluation_demo(seed=42):
    """
    Use MSE (and RMSE) to evaluate financial forecasting models.
    Compare naive, EWMA, and GARCH-like volatility forecasts.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    T = 500
    
    # Simulate GARCH(1,1)-like returns
    omega, alpha, beta = 0.01, 0.08, 0.90
    sigma2 = np.zeros(T)
    returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    
    # Forecasting models (one-step-ahead)
    window = 50
    forecast_start = window + 1
    
    actual_vol = sigma2[forecast_start:]
    n_forecast = len(actual_vol)
    
    # Model 1: Historical variance (rolling window)
    hist_forecast = np.array([np.var(returns[t-window:t]) 
                               for t in range(forecast_start, T)])
    
    # Model 2: EWMA (lambda = 0.94)
    lam = 0.94
    ewma_var = np.zeros(T)
    ewma_var[0] = np.var(returns[:window])
    for t in range(1, T):
        ewma_var[t] = lam * ewma_var[t-1] + (1 - lam) * returns[t-1]**2
    ewma_forecast = ewma_var[forecast_start:]
    
    # Model 3: Simple GARCH-like (using true parameters, cheating)
    garch_forecast = np.array([omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
                                for t in range(forecast_start, T)])
    
    # Evaluate
    models = {
        'Historical': hist_forecast,
        'EWMA (λ=0.94)': ewma_forecast,
        'GARCH(1,1)': garch_forecast,
    }
    
    print("\n" + "=" * 60)
    print("Volatility Forecast Evaluation")
    print("=" * 60)
    print(f"{'Model':<20} {'MSE':>12} {'RMSE':>12} {'Bias':>12} {'Rel Eff':>10}")
    print("-" * 68)
    
    mse_values = {}
    for name, forecast in models.items():
        mse = np.mean((forecast - actual_vol[:len(forecast)])**2)
        rmse = np.sqrt(mse)
        bias = np.mean(forecast - actual_vol[:len(forecast)])
        mse_values[name] = mse
        print(f"{name:<20} {mse:>12.6f} {rmse:>12.6f} {bias:>12.6f} {'--':>10}")
    
    # Relative efficiency (relative to Historical)
    base_mse = mse_values['Historical']
    print()
    for name, mse in mse_values.items():
        re = base_mse / mse
        print(f"  RE({name} vs Historical) = {re:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t_plot = range(len(actual_vol))
    ax = axes[0]
    ax.plot(t_plot, actual_vol, 'k-', alpha=0.5, linewidth=0.8, label='True σ²')
    for name, forecast, color in [('Historical', hist_forecast, 'blue'),
                                    ('EWMA', ewma_forecast, 'red'),
                                    ('GARCH', garch_forecast, 'green')]:
        ax.plot(t_plot[:len(forecast)], forecast, color=color, alpha=0.7, 
                linewidth=0.8, label=name)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Volatility Forecasts vs True Variance', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Squared errors
    ax = axes[1]
    for name, forecast, color in [('Historical', hist_forecast, 'blue'),
                                    ('EWMA', ewma_forecast, 'red'),
                                    ('GARCH', garch_forecast, 'green')]:
        sq_err = (forecast - actual_vol[:len(forecast)])**2
        ax.plot(t_plot[:len(sq_err)], sq_err, color=color, alpha=0.5, linewidth=0.8, label=name)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Squared Error', fontsize=12)
    ax.set_title('Squared Forecast Errors', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mse_forecast_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6. MSE as a Function of Sample Size
# ============================================================

def mse_vs_sample_size(mu_true=5.0, sigma=3.0, n_simulations=50_000, seed=42):
    """
    Show how MSE decreases with sample size for different estimators.
    
    Parameters
    ----------
    mu_true : float
        True mean
    sigma : float
        True standard deviation
    n_simulations : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    sample_sizes = np.arange(5, 201, 5)
    
    mse_mean = []
    mse_median = []
    mse_trim = []  # 10% trimmed mean
    theoretical_mse_mean = sigma**2 / sample_sizes
    
    for n in sample_sizes:
        samples = rng.normal(mu_true, sigma, (n_simulations, n))
        
        means = samples.mean(axis=1)
        medians = np.median(samples, axis=1)
        trimmed = stats.trim_mean(samples, 0.1, axis=1)
        
        mse_mean.append(np.mean((means - mu_true)**2))
        mse_median.append(np.mean((medians - mu_true)**2))
        mse_trim.append(np.mean((trimmed - mu_true)**2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sample_sizes, mse_mean, 'b-', linewidth=2, label='Mean (empirical)')
    ax.plot(sample_sizes, theoretical_mse_mean, 'b--', linewidth=1, 
            label=r'Mean (theoretical: $\sigma^2/n$)')
    ax.plot(sample_sizes, mse_median, 'r-', linewidth=2, label='Median')
    ax.plot(sample_sizes, mse_trim, 'g-', linewidth=2, label='Trimmed Mean (10%)')
    
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE vs Sample Size for Normal Population', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('mse_vs_sample_size.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("MSE vs Sample Size (Normal Population)")
    print("=" * 60)
    print("For Normal data, Mean is the most efficient estimator.")
    print(f"Asymptotic RE(Median/Mean) ≈ π/2 ≈ {np.pi/2:.4f}")
    print(f"Empirical RE at n=200: {mse_mean[-1]/mse_median[-1]:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("MEAN SQUARED ERROR DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. MSE computation and decomposition
    mse_computation_demo()
    
    # 2. Variance estimator comparison
    variance_estimator_mse()
    
    # 3. Relative efficiency
    relative_efficiency_demo()
    
    # 4. James-Stein estimator
    james_stein_demo()
    
    # 5. Financial forecast evaluation
    forecast_evaluation_demo()
    
    # 6. MSE vs sample size
    mse_vs_sample_size()
