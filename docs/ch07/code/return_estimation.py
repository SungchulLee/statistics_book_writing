"""
Return Estimation
==================
Financial applications of mean and variance estimation:
  - Expected return estimation precision
  - Sharpe ratio estimation uncertainty
  - Realized volatility with different windows
  - Estimation risk in portfolio optimization
  - Annualization conventions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def expected_return_precision(seed=42):
    """
    Demonstrate the fundamental imprecision of expected return estimation.
    Show confidence intervals and standard errors by estimation horizon.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu_annual = 0.08
    sigma_annual = 0.20
    
    print("=" * 60)
    print("Expected Return Estimation Precision")
    print(f"True μ = {mu_annual*100:.1f}% annual, σ = {sigma_annual*100:.1f}% annual")
    print("=" * 60)
    
    years = [5, 10, 20, 30, 50, 100]
    print(f"\n{'Years':>8} {'SE':>8} {'95% CI':>28} {'Width':>8}")
    print("-" * 56)
    for T in years:
        se = sigma_annual / np.sqrt(T)
        lo = mu_annual - 1.96 * se
        hi = mu_annual + 1.96 * se
        print(f"{T:>8} {se*100:>7.2f}% [{lo*100:>7.2f}%, {hi*100:>7.2f}%] "
              f"{(hi-lo)*100:>7.2f}%")
    
    # Simulation
    n_sim = 20_000
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, T, title in [(axes[0], 10, '10 Years'), (axes[1], 50, '50 Years')]:
        n_m = T * 12
        mu_m = mu_annual / 12
        sig_m = sigma_annual / np.sqrt(12)
        ests = np.array([rng.normal(mu_m, sig_m, n_m).mean() * 12 for _ in range(n_sim)])
        ax.hist(ests, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='white')
        ax.axvline(mu_annual, color='red', ls='--', lw=2, label=f'True μ = {mu_annual*100:.0f}%')
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Estimated Annual Return')
        ax.set_title(f'{title} of Monthly Data')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.suptitle('Distribution of Expected Return Estimates', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('return_estimation_precision.png', dpi=150, bbox_inches='tight')
    plt.show()


def sharpe_ratio_uncertainty(seed=42):
    """
    Show the sampling distribution of estimated Sharpe ratios.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    true_sr = 0.5  # Annual Sharpe ratio
    mu_annual = 0.08
    sigma_annual = mu_annual / true_sr
    n_sim = 30_000
    
    print("\n" + "=" * 60)
    print("Sharpe Ratio Estimation Uncertainty")
    print(f"True SR = {true_sr}")
    print("=" * 60)
    
    horizons = [3, 5, 10, 20, 50]
    print(f"\n{'Years':>8} {'E[SR̂]':>8} {'SD(SR̂)':>8} {'P(SR̂<0)':>10}")
    print("-" * 38)
    
    for T in horizons:
        n_m = T * 12
        mu_m, sig_m = mu_annual / 12, sigma_annual / np.sqrt(12)
        sr_ests = []
        for _ in range(n_sim):
            r = rng.normal(mu_m, sig_m, n_m)
            sr_ests.append(r.mean() / r.std() * np.sqrt(12))
        sr_ests = np.array(sr_ests)
        print(f"{T:>8} {sr_ests.mean():>8.3f} {sr_ests.std():>8.3f} "
              f"{(sr_ests < 0).mean():>10.1%}")
    
    # Plot for 10 years
    fig, ax = plt.subplots(figsize=(9, 5))
    for T, color in [(5, 'steelblue'), (10, 'coral'), (30, 'seagreen')]:
        n_m = T * 12
        sr_ests = [rng.normal(mu_annual/12, sigma_annual/np.sqrt(12), n_m).mean() /
                   rng.normal(mu_annual/12, sigma_annual/np.sqrt(12), n_m).std() *
                   np.sqrt(12) for _ in range(n_sim)]
        ax.hist(sr_ests, bins=60, density=True, alpha=0.4, color=color, label=f'{T}yr')
    ax.axvline(true_sr, color='red', ls='--', lw=2, label=f'True SR = {true_sr}')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Estimated Sharpe Ratio'); ax.set_ylabel('Density')
    ax.set_title('Sharpe Ratio Estimation Uncertainty')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('return_sharpe_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.show()


def realized_volatility_windows(seed=42):
    """
    Compare realized volatility estimates across different window sizes.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    # Simulate GARCH-like returns
    T = 756  # 3 years daily
    omega, alpha, beta = 0.00001, 0.08, 0.90
    sigma2 = np.zeros(T)
    returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    
    windows = [5, 10, 21, 63, 126, 252]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    true_vol = np.sqrt(sigma2 * 252) * 100
    ax.plot(true_vol, 'k-', alpha=0.3, lw=0.8, label='True σ (GARCH)')
    
    for w, color in zip([21, 63, 252], ['blue', 'red', 'green']):
        rv = np.array([np.std(returns[max(0,t-w):t], ddof=1) * np.sqrt(252) * 100
                       for t in range(w, T)])
        ax.plot(range(w, T), rv, color=color, alpha=0.7, lw=0.8, label=f'{w}d window')
    
    ax.set_xlabel('Trading Day'); ax.set_ylabel('Annualized Vol (%)')
    ax.set_title('Realized Volatility: Window Size Comparison')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('return_realized_vol.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("Realized Volatility: Bias–Variance by Window")
    print("=" * 60)
    print(f"\n{'Window':>8} {'Mean Vol%':>10} {'SD Vol%':>10}")
    print("-" * 32)
    for w in windows:
        rv = [np.std(returns[max(0,t-w):t], ddof=1)*np.sqrt(252)*100 for t in range(w, T)]
        print(f"{w:>8} {np.mean(rv):>10.2f} {np.std(rv):>10.2f}")
    print("\nShort windows: low bias, high variance (noisy)")
    print("Long windows: high bias (lagging), low variance (smooth)")


def annualization_conventions():
    """Demonstrate standard annualization of mean and volatility."""
    print("\n" + "=" * 60)
    print("Annualization Conventions")
    print("=" * 60)
    
    # Example daily statistics
    mu_d = 0.0003  # Daily mean return
    sigma_d = 0.012  # Daily volatility
    
    print(f"\nDaily: μ = {mu_d*100:.4f}%, σ = {sigma_d*100:.4f}%")
    print(f"\nAnnualized (252 trading days):")
    print(f"  μ_annual = μ_daily × 252 = {mu_d*252*100:.2f}%")
    print(f"  σ_annual = σ_daily × √252 = {sigma_d*np.sqrt(252)*100:.2f}%")
    print(f"  SR_annual = (μ/σ)_daily × √252 = {mu_d/sigma_d*np.sqrt(252):.3f}")
    
    print(f"\nFrom monthly (12 months):")
    mu_m = mu_d * 21  # Approx monthly
    sigma_m = sigma_d * np.sqrt(21)
    print(f"  Monthly: μ = {mu_m*100:.3f}%, σ = {sigma_m*100:.3f}%")
    print(f"  μ_annual = μ_monthly × 12 = {mu_m*12*100:.2f}%")
    print(f"  σ_annual = σ_monthly × √12 = {sigma_m*np.sqrt(12)*100:.2f}%")


if __name__ == "__main__":
    expected_return_precision()
    sharpe_ratio_uncertainty()
    realized_volatility_windows()
    annualization_conventions()
