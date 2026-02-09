"""
Bias–Variance Tradeoff: Demonstrations and Visualizations
=========================================================
This module provides comprehensive demonstrations of the bias–variance
tradeoff in statistical estimation, including:
  - Empirical verification of the MSE decomposition
  - Shrinkage estimator analysis
  - Dartboard visualization (geometric interpretation)
  - U-shaped MSE curve for model complexity
  - Financial application: Ledoit-Wolf shrinkage
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 1. MSE Decomposition: Empirical Verification
# ============================================================

def mse_decomposition_demo(mu_true=5.0, sigma=3.0, n=10, n_simulations=100_000,
                            seed=42):
    """
    Empirically verify MSE = Variance + Bias^2 for the sample mean.
    
    Parameters
    ----------
    mu_true : float
        True population mean
    sigma : float
        Population standard deviation
    n : int
        Sample size
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    
    Returns
    -------
    dict with empirical MSE, variance, bias, and theoretical values
    """
    rng = np.random.default_rng(seed)
    
    # Generate many samples and compute sample means
    samples = rng.normal(mu_true, sigma, size=(n_simulations, n))
    estimates = samples.mean(axis=1)
    
    # Empirical quantities
    empirical_mse = np.mean((estimates - mu_true) ** 2)
    empirical_bias = np.mean(estimates) - mu_true
    empirical_variance = np.var(estimates, ddof=0)
    
    # Theoretical quantities (sample mean is unbiased)
    theoretical_mse = sigma**2 / n
    theoretical_bias = 0.0
    theoretical_variance = sigma**2 / n
    
    results = {
        'empirical_mse': empirical_mse,
        'empirical_bias': empirical_bias,
        'empirical_bias_sq': empirical_bias**2,
        'empirical_variance': empirical_variance,
        'empirical_decomposition': empirical_variance + empirical_bias**2,
        'theoretical_mse': theoretical_mse,
        'theoretical_bias': theoretical_bias,
        'theoretical_variance': theoretical_variance,
    }
    
    print("=" * 60)
    print("MSE Decomposition: Empirical Verification")
    print("=" * 60)
    print(f"True μ = {mu_true}, σ = {sigma}, n = {n}")
    print(f"Simulations: {n_simulations:,}")
    print("-" * 60)
    print(f"{'Quantity':<25} {'Empirical':>12} {'Theoretical':>12}")
    print("-" * 60)
    print(f"{'Bias':<25} {empirical_bias:>12.6f} {theoretical_bias:>12.6f}")
    print(f"{'Bias²':<25} {empirical_bias**2:>12.6f} {theoretical_bias**2:>12.6f}")
    print(f"{'Variance':<25} {empirical_variance:>12.6f} {theoretical_variance:>12.6f}")
    print(f"{'MSE (direct)':<25} {empirical_mse:>12.6f} {theoretical_mse:>12.6f}")
    print(f"{'Var + Bias²':<25} {empirical_variance + empirical_bias**2:>12.6f} "
          f"{theoretical_variance + theoretical_bias**2:>12.6f}")
    print("-" * 60)
    print(f"Decomposition check: MSE ≈ Var + Bias²? "
          f"{np.isclose(empirical_mse, empirical_variance + empirical_bias**2, rtol=0.01)}")
    
    return results


# ============================================================
# 2. Shrinkage Estimator: Bias–Variance Tradeoff
# ============================================================

def shrinkage_estimator_analysis(mu_true=2.0, sigma=5.0, n=10,
                                  n_simulations=50_000, seed=42):
    """
    Compare sample mean vs shrinkage estimator λ*X̄.
    Show how introducing bias can reduce MSE.
    
    Parameters
    ----------
    mu_true : float
        True population mean
    sigma : float
        Population standard deviation
    n : int
        Sample size
    n_simulations : int
        Number of simulations
    seed : int
        Random seed
    
    Returns
    -------
    dict with lambda values, MSE curves, and optimal lambda
    """
    rng = np.random.default_rng(seed)
    
    # Generate samples
    samples = rng.normal(mu_true, sigma, size=(n_simulations, n))
    x_bar = samples.mean(axis=1)
    
    # Range of shrinkage factors
    lambdas = np.linspace(0.01, 1.5, 200)
    
    # Compute MSE components for each lambda
    mse_empirical = np.array([np.mean((lam * x_bar - mu_true)**2) for lam in lambdas])
    bias_sq = np.array([(lam - 1)**2 * mu_true**2 for lam in lambdas])
    variance = np.array([lam**2 * sigma**2 / n for lam in lambdas])
    mse_theoretical = variance + bias_sq
    
    # Optimal lambda (theoretical)
    lambda_opt = (n * mu_true**2) / (n * mu_true**2 + sigma**2)
    mse_opt = lambda_opt**2 * sigma**2 / n + (1 - lambda_opt)**2 * mu_true**2
    
    # MSE of unbiased estimator (lambda=1)
    mse_unbiased = sigma**2 / n
    
    print("\n" + "=" * 60)
    print("Shrinkage Estimator Analysis: λ·X̄")
    print("=" * 60)
    print(f"True μ = {mu_true}, σ = {sigma}, n = {n}")
    print(f"Optimal λ* = {lambda_opt:.4f}")
    print(f"MSE(X̄) [λ=1]     = {mse_unbiased:.4f}")
    print(f"MSE(λ*·X̄)        = {mse_opt:.4f}")
    print(f"MSE reduction     = {(1 - mse_opt/mse_unbiased)*100:.1f}%")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: MSE decomposition
    ax = axes[0]
    ax.plot(lambdas, mse_theoretical, 'k-', linewidth=2, label='MSE = Var + Bias²')
    ax.plot(lambdas, variance, 'b--', linewidth=1.5, label='Variance')
    ax.plot(lambdas, bias_sq, 'r--', linewidth=1.5, label='Bias²')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label=f'λ=1 (unbiased)')
    ax.axvline(x=lambda_opt, color='green', linestyle=':', alpha=0.7, 
               label=f'λ*={lambda_opt:.3f} (optimal)')
    ax.scatter([lambda_opt], [mse_opt], color='green', zorder=5, s=80)
    ax.scatter([1.0], [mse_unbiased], color='gray', zorder=5, s=80)
    ax.set_xlabel('Shrinkage Factor λ', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias–Variance Tradeoff: Shrinkage Estimator', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # Right: Empirical vs theoretical MSE
    ax = axes[1]
    ax.plot(lambdas, mse_empirical, 'b-', alpha=0.7, linewidth=1.5, label='Empirical MSE')
    ax.plot(lambdas, mse_theoretical, 'r--', linewidth=1.5, label='Theoretical MSE')
    ax.axvline(x=lambda_opt, color='green', linestyle=':', alpha=0.7)
    ax.set_xlabel('Shrinkage Factor λ', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Empirical vs Theoretical MSE', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_variance_shrinkage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'lambdas': lambdas,
        'mse_theoretical': mse_theoretical,
        'mse_empirical': mse_empirical,
        'lambda_opt': lambda_opt,
        'mse_opt': mse_opt,
        'mse_unbiased': mse_unbiased,
    }


# ============================================================
# 3. Dartboard Visualization (Geometric Interpretation)
# ============================================================

def dartboard_visualization(n_darts=200, seed=42):
    """
    Visualize bias and variance using a dartboard analogy.
    Four scenarios: all combinations of low/high bias and variance.
    
    Parameters
    ----------
    n_darts : int
        Number of darts per scenario
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    scenarios = [
        ('Low Bias, Low Variance', 0.0, 0.0, 0.3),
        ('Low Bias, High Variance', 0.0, 0.0, 1.2),
        ('High Bias, Low Variance', 1.5, 1.0, 0.3),
        ('High Bias, High Variance', 1.5, 1.0, 1.2),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, (title, bias_x, bias_y, std) in zip(axes.flat, scenarios):
        # Generate dart positions
        x = rng.normal(bias_x, std, n_darts)
        y = rng.normal(bias_y, std, n_darts)
        
        # Draw target circles
        for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
            circle = plt.Circle((0, 0), r, fill=False, color='lightgray', 
                              linestyle='--', linewidth=0.8)
            ax.add_patch(circle)
        
        # Plot darts
        ax.scatter(x, y, c='steelblue', alpha=0.4, s=15, edgecolors='none')
        
        # Mark center (true value) and mean (expected value)
        ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='True value')
        ax.plot(bias_x, bias_y, 'ko', markersize=8, label=f'E[θ̂]')
        
        # Compute MSE
        mse = np.mean(x**2 + y**2)
        bias_sq = bias_x**2 + bias_y**2
        var_val = np.var(x) + np.var(y)
        
        ax.set_title(f'{title}\nMSE={mse:.2f} = Var({var_val:.2f}) + Bias²({bias_sq:.2f})', 
                     fontsize=11)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Dartboard Analogy: Bias vs Variance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bias_variance_dartboard.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. U-Shaped MSE Curve: Model Complexity
# ============================================================

def u_shaped_mse_curve(n_train=30, degree_max=15, n_simulations=500, seed=42):
    """
    Demonstrate the U-shaped MSE curve using polynomial regression.
    As polynomial degree increases: bias ↓, variance ↑, MSE is U-shaped.
    
    Parameters
    ----------
    n_train : int
        Training sample size
    degree_max : int
        Maximum polynomial degree to evaluate
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    # True function: cubic polynomial
    def f_true(x):
        return 0.5 * x**3 - 2 * x**2 + 3 * x + 1
    
    sigma_noise = 3.0
    degrees = range(1, degree_max + 1)
    
    # Test point for evaluation
    x_test = 1.5
    y_true = f_true(x_test)
    
    # Storage for predictions at test point
    predictions = {d: [] for d in degrees}
    
    for _ in range(n_simulations):
        # Generate training data
        x_train = rng.uniform(-2, 4, n_train)
        y_train = f_true(x_train) + rng.normal(0, sigma_noise, n_train)
        
        for d in degrees:
            # Fit polynomial
            try:
                coeffs = np.polyfit(x_train, y_train, d)
                y_pred = np.polyval(coeffs, x_test)
                predictions[d].append(y_pred)
            except (np.linalg.LinAlgError, np.RankWarning):
                predictions[d].append(np.nan)
    
    # Compute bias², variance, MSE for each degree
    bias_sq_vals = []
    var_vals = []
    mse_vals = []
    
    for d in degrees:
        preds = np.array(predictions[d])
        preds = preds[np.isfinite(preds)]
        
        bias = np.mean(preds) - y_true
        var = np.var(preds)
        mse = np.mean((preds - y_true) ** 2)
        
        bias_sq_vals.append(bias**2)
        var_vals.append(var)
        mse_vals.append(mse)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    degrees_list = list(degrees)
    ax.plot(degrees_list, mse_vals, 'k-o', linewidth=2, markersize=6, label='MSE')
    ax.plot(degrees_list, bias_sq_vals, 'r--s', linewidth=1.5, markersize=5, label='Bias²')
    ax.plot(degrees_list, var_vals, 'b--^', linewidth=1.5, markersize=5, label='Variance')
    
    # Mark optimal degree
    opt_idx = np.argmin(mse_vals)
    ax.axvline(x=degrees_list[opt_idx], color='green', linestyle=':', alpha=0.7,
               label=f'Optimal degree = {degrees_list[opt_idx]}')
    
    # Annotate regions
    ax.annotate('Underfitting\n(High Bias)', xy=(2, max(mse_vals)*0.7), fontsize=11,
                ha='center', color='red', fontstyle='italic')
    ax.annotate('Overfitting\n(High Variance)', xy=(degree_max - 1, max(mse_vals)*0.7), 
                fontsize=11, ha='center', color='blue', fontstyle='italic')
    
    ax.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
    ax.set_ylabel('Error at Test Point', fontsize=12)
    ax.set_title('U-Shaped MSE Curve: Bias–Variance Tradeoff', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(degrees_list)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('bias_variance_u_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("U-Shaped MSE Curve Results")
    print("=" * 60)
    print(f"True function: f(x) = 0.5x³ - 2x² + 3x + 1 (degree 3)")
    print(f"Optimal polynomial degree: {degrees_list[opt_idx]}")
    print(f"Minimum MSE: {mse_vals[opt_idx]:.4f}")
    print(f"\n{'Degree':>8} {'Bias²':>10} {'Variance':>10} {'MSE':>10}")
    print("-" * 40)
    for d, b, v, m in zip(degrees_list, bias_sq_vals, var_vals, mse_vals):
        print(f"{d:>8} {b:>10.4f} {v:>10.4f} {m:>10.4f}")


# ============================================================
# 5. Financial Application: Covariance Shrinkage
# ============================================================

def covariance_shrinkage_demo(n_assets=10, n_obs=30, n_simulations=1000, seed=42):
    """
    Demonstrate bias–variance tradeoff in covariance matrix estimation.
    Compare sample covariance (unbiased) vs shrinkage estimator.
    
    Parameters
    ----------
    n_assets : int
        Number of assets
    n_obs : int
        Number of return observations
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    # True covariance matrix (random positive definite)
    A = rng.standard_normal((n_assets, n_assets))
    Sigma_true = A @ A.T / n_assets + np.eye(n_assets) * 0.5
    
    # Shrinkage target: scaled identity
    mu_target = np.trace(Sigma_true) / n_assets
    Target = mu_target * np.eye(n_assets)
    
    alphas = np.linspace(0, 1, 50)
    frobenius_errors = {alpha: [] for alpha in alphas}
    
    for _ in range(n_simulations):
        # Generate returns from true distribution
        returns = rng.multivariate_normal(np.zeros(n_assets), Sigma_true, size=n_obs)
        S = np.cov(returns, rowvar=False)  # Sample covariance
        
        for alpha in alphas:
            # Shrinkage estimator: (1-α)S + αT
            Sigma_shrunk = (1 - alpha) * S + alpha * Target
            error = np.linalg.norm(Sigma_shrunk - Sigma_true, 'fro')**2
            frobenius_errors[alpha].append(error)
    
    # Average errors
    mean_errors = [np.mean(frobenius_errors[a]) for a in alphas]
    
    # Find optimal alpha
    opt_idx = np.argmin(mean_errors)
    alpha_opt = alphas[opt_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas, mean_errors, 'b-', linewidth=2)
    ax.axvline(x=alpha_opt, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal α = {alpha_opt:.3f}')
    ax.scatter([0], [mean_errors[0]], color='green', s=100, zorder=5, 
               label=f'Sample Cov (α=0): {mean_errors[0]:.2f}')
    ax.scatter([alpha_opt], [mean_errors[opt_idx]], color='red', s=100, zorder=5,
               label=f'Shrinkage (α={alpha_opt:.2f}): {mean_errors[opt_idx]:.2f}')
    
    ax.set_xlabel('Shrinkage Intensity α', fontsize=12)
    ax.set_ylabel('Mean Squared Frobenius Error', fontsize=12)
    ax.set_title(f'Covariance Shrinkage: {n_assets} Assets, {n_obs} Observations', 
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_variance_cov_shrinkage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("Covariance Shrinkage Results")
    print("=" * 60)
    print(f"Assets: {n_assets}, Observations: {n_obs}")
    print(f"Sample Covariance MSE:    {mean_errors[0]:.4f}")
    print(f"Optimal Shrinkage MSE:    {mean_errors[opt_idx]:.4f} (α = {alpha_opt:.3f})")
    print(f"MSE Reduction:            {(1 - mean_errors[opt_idx]/mean_errors[0])*100:.1f}%")


# ============================================================
# 6. Effect of Sample Size on Bias and Variance
# ============================================================

def sample_size_effect(mu_true=5.0, sigma=3.0, n_simulations=50_000, seed=42):
    """
    Show how bias and variance change with sample size for
    biased and unbiased estimators.
    
    Parameters
    ----------
    mu_true : float
        True mean
    sigma : float
        True standard deviation
    n_simulations : int
        Number of simulations per sample size
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    results = {'n': [], 'xbar_mse': [], 'xbar_var': [], 'xbar_bias_sq': [],
               'biased_mse': [], 'biased_var': [], 'biased_bias_sq': []}
    
    # Biased estimator: use n/(n+1) * sample mean (shrinkage toward 0)
    for n in sample_sizes:
        samples = rng.normal(mu_true, sigma, size=(n_simulations, n))
        x_bar = samples.mean(axis=1)
        shrunk = (n / (n + 1)) * x_bar  # Biased estimator
        
        # Sample mean statistics
        results['n'].append(n)
        results['xbar_mse'].append(np.mean((x_bar - mu_true)**2))
        results['xbar_var'].append(np.var(x_bar))
        results['xbar_bias_sq'].append((np.mean(x_bar) - mu_true)**2)
        
        # Shrinkage estimator statistics
        results['biased_mse'].append(np.mean((shrunk - mu_true)**2))
        results['biased_var'].append(np.var(shrunk))
        results['biased_bias_sq'].append((np.mean(shrunk) - mu_true)**2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, prefix, title in zip(axes, ['xbar', 'biased'],
                                  ['Sample Mean (Unbiased)', 
                                   'Shrinkage Estimator n/(n+1)·X̄ (Biased)']):
        ax.plot(results['n'], results[f'{prefix}_mse'], 'k-o', label='MSE', linewidth=2)
        ax.plot(results['n'], results[f'{prefix}_var'], 'b--s', label='Variance', linewidth=1.5)
        ax.plot(results['n'], results[f'{prefix}_bias_sq'], 'r--^', label='Bias²', linewidth=1.5)
        ax.set_xlabel('Sample Size n', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Sample Size on Bias and Variance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bias_variance_sample_size.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("BIAS–VARIANCE TRADEOFF DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. MSE decomposition
    mse_decomposition_demo()
    
    # 2. Shrinkage estimator
    shrinkage_estimator_analysis()
    
    # 3. Dartboard visualization
    dartboard_visualization()
    
    # 4. U-shaped MSE curve
    u_shaped_mse_curve()
    
    # 5. Covariance shrinkage (finance)
    covariance_shrinkage_demo()
    
    # 6. Sample size effects
    sample_size_effect()
