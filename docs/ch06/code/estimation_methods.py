"""
Chapter 6: General Statistical Estimation — Code Examples
==========================================================
Demonstrates bias-variance tradeoff, MSE computation,
MLE vs Method of Moments comparison, and Fisher information.
"""

import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# 1. Bias-Variance Tradeoff: Variance Estimators
# =============================================================================

def compare_variance_estimators(mu=5, sigma2=4, n=10, n_sim=50_000):
    """Compare three variance estimators: MLE (n), Bessel (n-1), MSE-optimal (n+1)."""
    sigma = np.sqrt(sigma2)
    estimates = {r"$\hat{\sigma}^2_{n}$ (MLE)": [],
                 r"$S^2_{n-1}$ (Bessel)": [],
                 r"$\hat{\sigma}^2_{n+1}$ (MSE-opt)": []}

    for _ in range(n_sim):
        x = np.random.normal(mu, sigma, n)
        ss = np.sum((x - x.mean())**2)
        estimates[r"$\hat{\sigma}^2_{n}$ (MLE)"].append(ss / n)
        estimates[r"$S^2_{n-1}$ (Bessel)"].append(ss / (n - 1))
        estimates[r"$\hat{\sigma}^2_{n+1}$ (MSE-opt)"].append(ss / (n + 1))

    print(f"True σ² = {sigma2}, n = {n}, simulations = {n_sim}")
    print(f"{'Estimator':<30} {'Mean':>8} {'Bias':>8} {'Var':>8} {'MSE':>8}")
    print("-" * 70)
    for name, vals in estimates.items():
        vals = np.array(vals)
        bias = vals.mean() - sigma2
        var = vals.var()
        mse = np.mean((vals - sigma2)**2)
        print(f"{name:<30} {vals.mean():>8.4f} {bias:>8.4f} {var:>8.4f} {mse:>8.4f}")

    # Theoretical values
    print("\nTheoretical MSE:")
    print(f"  MLE (n):      {(2*n - 1) * sigma2**2 / n**2:.4f}")
    print(f"  Bessel (n-1): {2 * sigma2**2 / (n - 1):.4f}")
    print(f"  Optimal (n+1): {2 * sigma2**2 / (n + 1):.4f}")

compare_variance_estimators()


# =============================================================================
# 2. Shrinkage Estimator: Bias-Variance Tradeoff
# =============================================================================

def shrinkage_mse(mu_true=3, sigma2=4, n=20):
    """Show how shrinkage toward 0 trades bias for variance."""
    lambdas = np.linspace(0.01, 1.5, 200)
    bias_sq = (lambdas - 1)**2 * mu_true**2
    variance = lambdas**2 * sigma2 / n
    mse = bias_sq + variance

    lambda_opt = mu_true**2 / (mu_true**2 + sigma2 / n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, bias_sq, "--", label=r"Bias$^2$", linewidth=1.5)
    ax.plot(lambdas, variance, "--", label="Variance", linewidth=1.5)
    ax.plot(lambdas, mse, "-", label="MSE", linewidth=2)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label=r"$\lambda=1$ (unbiased)")
    ax.axvline(lambda_opt, color="red", linestyle=":", alpha=0.8,
               label=rf"$\lambda^*={lambda_opt:.3f}$ (MSE-optimal)")
    ax.set_xlabel(r"Shrinkage factor $\lambda$")
    ax.set_ylabel("Value")
    ax.set_title(rf"Bias-Variance Tradeoff: $\hat{{\mu}}_\lambda = \lambda \bar{{X}}$"
                 f"\n" + rf"$\mu={mu_true}, \sigma^2={sigma2}, n={n}$")
    ax.legend()
    ax.set_xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig("shrinkage_bv_tradeoff.png", dpi=150)
    plt.show()

shrinkage_mse()


# =============================================================================
# 3. MLE: Normal Distribution (numerical + closed-form)
# =============================================================================

def mle_normal_demo(n=100):
    """Demonstrate MLE for Normal distribution."""
    mu_true, sigma_true = 5.0, 2.0
    data = np.random.normal(mu_true, sigma_true, n)

    # Closed-form MLE
    mu_hat = data.mean()
    sigma2_hat = np.mean((data - mu_hat)**2)

    # Log-likelihood function
    def neg_log_lik(params, x):
        mu, log_sigma2 = params  # log transform for unconstrained optimization
        sigma2 = np.exp(log_sigma2)
        n = len(x)
        return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((x - mu)**2) / (2 * sigma2)

    # Numerical MLE
    result = optimize.minimize(neg_log_lik, x0=[0, 0], args=(data,), method="Nelder-Mead")
    mu_num, sigma2_num = result.x[0], np.exp(result.x[1])

    print("MLE for Normal Distribution")
    print(f"  True:        μ = {mu_true:.4f}, σ² = {sigma_true**2:.4f}")
    print(f"  Closed-form: μ = {mu_hat:.4f}, σ² = {sigma2_hat:.4f}")
    print(f"  Numerical:   μ = {mu_num:.4f}, σ² = {sigma2_num:.4f}")

    # Plot log-likelihood surface
    mus = np.linspace(mu_hat - 1, mu_hat + 1, 100)
    ll = np.array([-neg_log_lik([m, np.log(sigma2_hat)], data) for m in mus])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mus, ll, linewidth=2)
    ax.axvline(mu_hat, color="red", linestyle="--", label=rf"$\hat{{\mu}}_{{MLE}} = {mu_hat:.3f}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\ell(\mu)$")
    ax.set_title("Log-Likelihood as a Function of μ")
    ax.legend()
    plt.tight_layout()
    plt.savefig("loglik_normal.png", dpi=150)
    plt.show()

mle_normal_demo()


# =============================================================================
# 4. MLE vs Method of Moments: Gamma Distribution
# =============================================================================

def mle_vs_mom_gamma(alpha_true=3, beta_true=2, n=200, n_sim=5000):
    """Compare MLE and MoM for Gamma distribution."""
    mle_alpha, mle_beta = [], []
    mom_alpha, mom_beta = [], []

    for _ in range(n_sim):
        data = np.random.gamma(alpha_true, beta_true, n)

        # Method of Moments
        m1 = data.mean()
        m2 = np.mean(data**2)
        v = m2 - m1**2
        a_mom = m1**2 / v
        b_mom = v / m1
        mom_alpha.append(a_mom)
        mom_beta.append(b_mom)

        # MLE (scipy fit uses shape, loc, scale parameterization)
        a_mle, _, b_mle = stats.gamma.fit(data, floc=0)
        mle_alpha.append(a_mle)
        mle_beta.append(b_mle)

    print(f"Gamma(α={alpha_true}, β={beta_true}), n={n}, {n_sim} simulations")
    print(f"\n{'Parameter':<12} {'Method':<8} {'Mean':>8} {'Bias':>8} {'Var':>10} {'MSE':>10}")
    print("-" * 60)
    for name, true, mle_vals, mom_vals in [
        ("α", alpha_true, mle_alpha, mom_alpha),
        ("β", beta_true, mle_beta, mom_beta),
    ]:
        for method, vals in [("MLE", mle_vals), ("MoM", mom_vals)]:
            vals = np.array(vals)
            bias = vals.mean() - true
            var = vals.var()
            mse = np.mean((vals - true)**2)
            print(f"{name:<12} {method:<8} {vals.mean():>8.4f} {bias:>8.4f} {var:>10.6f} {mse:>10.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, true, mle_vals, mom_vals in [
        (axes[0], "α", alpha_true, mle_alpha, mom_alpha),
        (axes[1], "β", beta_true, mle_beta, mom_beta),
    ]:
        ax.hist(mle_vals, bins=50, alpha=0.6, density=True, label="MLE")
        ax.hist(mom_vals, bins=50, alpha=0.6, density=True, label="MoM")
        ax.axvline(true, color="black", linestyle="--", linewidth=2, label=f"True {name}")
        ax.set_title(f"Sampling Distribution of {name}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("mle_vs_mom_gamma.png", dpi=150)
    plt.show()

mle_vs_mom_gamma()


# =============================================================================
# 5. Cramér-Rao Lower Bound Verification
# =============================================================================

def cramer_rao_demo(n=50, n_sim=20_000):
    """Verify that sample mean achieves the CRLB for Normal mean."""
    mu_true, sigma = 5.0, 2.0
    crlb = sigma**2 / n

    means = [np.random.normal(mu_true, sigma, n).mean() for _ in range(n_sim)]
    means = np.array(means)

    print("Cramér-Rao Lower Bound Verification (Normal Mean)")
    print(f"  CRLB = σ²/n = {crlb:.6f}")
    print(f"  Var(X̄)     = {means.var():.6f}")
    print(f"  Ratio       = {means.var() / crlb:.4f}  (should be ≈ 1.0)")

    # Compare with median
    medians = [np.median(np.random.normal(mu_true, sigma, n)) for _ in range(n_sim)]
    medians = np.array(medians)
    print(f"\n  Var(median) = {medians.var():.6f}")
    print(f"  RE(mean, median) = {medians.var() / means.var():.4f}  (theoretical: π/2 ≈ 1.571)")

cramer_rao_demo()


# =============================================================================
# 6. Fisher Information: Numerical Computation
# =============================================================================

def fisher_information_numerical(dist_name="norm", true_params={"loc": 5, "scale": 2},
                                  param_name="loc", n_samples=100_000, delta=1e-5):
    """Compute Fisher information numerically via score variance."""
    dist = getattr(stats, dist_name)
    data = dist.rvs(size=n_samples, **true_params)

    theta = true_params[param_name]

    # Score: d/dθ log f(x; θ)
    params_plus = {**true_params, param_name: theta + delta}
    params_minus = {**true_params, param_name: theta - delta}

    logf_plus = dist.logpdf(data, **params_plus)
    logf_minus = dist.logpdf(data, **params_minus)
    score = (logf_plus - logf_minus) / (2 * delta)

    I_numerical = np.var(score)

    # Theoretical for Normal mean: I(μ) = 1/σ²
    I_theoretical = 1 / true_params["scale"]**2

    print("Fisher Information (Numerical vs Theoretical)")
    print(f"  Numerical:   I(μ) = {I_numerical:.6f}")
    print(f"  Theoretical: I(μ) = {I_theoretical:.6f}")

fisher_information_numerical()
