"""
Linear Regression — Weed Prices vs State Demographics
======================================================
Adapted from intro2stats "Linear Regression" notebook.

Predicts high-quality weed price from state demographic
features (population, income, racial composition).

Demonstrates:
1. Exploratory scatter / heatmap / pair plots
2. Single-variable OLS  (sklearn + statsmodels)
3. Multi-variable OLS   with RMSE evaluation
4. Train / test split and prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── Synthetic state-level dataset ───────────────────────────
# Inspired by the original Weed_Price.csv + Demographics_State.csv
# 42 states for training, 8 states for testing.

def build_dataset():
    """
    Create a synthetic DataFrame with columns:
      state, HighQ, total_population, per_capita_income,
      percent_white, percent_black, percent_hispanic
    """
    states = [
        "alabama", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii",
        "idaho", "illinois", "indiana", "kansas", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "montana", "nebraska", "new hampshire",
        "new mexico", "new york", "north carolina", "north dakota",
        "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
        "south carolina", "tennessee", "texas", "utah", "vermont",
        "virginia", "washington", "west virginia", "wisconsin",
        # test states
        "iowa", "kentucky", "missouri", "nevada",
        "wyoming", "south dakota", "new jersey", "colorado_extra",
    ]
    n = len(states)
    rng = np.random.default_rng(42)

    pop     = rng.integers(500_000, 30_000_000, n).astype(float)
    income  = rng.integers(20_000, 55_000, n).astype(float)
    pwhite  = rng.uniform(40, 95, n)
    pblack  = rng.uniform(2, 40, n)
    phisp   = rng.uniform(2, 45, n)

    # price ~ intercept + noise driven mainly by income and pwhite
    price = (200 + 0.4 * (income - 30_000) / 1_000
             + 0.3 * pwhite
             + rng.normal(0, 15, n))

    df = pd.DataFrame({
        "state": states,
        "HighQ": np.round(price, 2),
        "total_population": pop,
        "per_capita_income": income,
        "percent_white": np.round(pwhite, 1),
        "percent_black": np.round(pblack, 1),
        "percent_hispanic": np.round(phisp, 1),
    })
    return df


# ── Train / test split ─────────────────────────────────────
TEST_STATES = {"iowa", "kentucky", "missouri", "nevada",
               "wyoming", "south dakota", "new jersey", "colorado_extra"}


def split_data(df):
    mask = df["state"].isin(TEST_STATES)
    return df[~mask].copy(), df[mask].copy()


# ── Main ────────────────────────────────────────────────────
def main():
    df = build_dataset()
    train, test = split_data(df)

    print("=" * 60)
    print("Linear Regression — Weed Price vs Demographics")
    print("=" * 60)
    print(f"  Training samples: {len(train)},  Test samples: {len(test)}")

    features = ["total_population", "per_capita_income",
                "percent_white", "percent_black", "percent_hispanic"]

    # ── 1. Correlation matrix ──────────────────────────────
    print("\n--- Correlation with HighQ ---")
    for col in features:
        r = train["HighQ"].corr(train[col])
        print(f"  {col:>25s}:  r = {r:+.3f}")

    # ── 2. Single-variable model (sklearn) ─────────────────
    print("\n--- Model 1: HighQ ~ total_population (sklearn) ---")
    feat1 = ["total_population"]
    model1 = LinearRegression().fit(train[feat1], train["HighQ"])
    pred1  = model1.predict(test[feat1])
    rmse1  = np.sqrt(np.mean((test["HighQ"] - pred1) ** 2))
    print(f"  intercept = {model1.intercept_:.4f}")
    print(f"  coef      = {model1.coef_[0]:.8f}")
    print(f"  Test RMSE = {rmse1:.2f}")

    # ── 3. Multi-variable model (sklearn) ──────────────────
    feat2 = ["total_population", "per_capita_income"]
    print(f"\n--- Model 2: HighQ ~ {' + '.join(feat2)} ---")
    model2 = LinearRegression().fit(train[feat2], train["HighQ"])
    pred2  = model2.predict(test[feat2])
    rmse2  = np.sqrt(np.mean((test["HighQ"] - pred2) ** 2))
    print(f"  intercept = {model2.intercept_:.4f}")
    print(f"  coefs     = {dict(zip(feat2, model2.coef_))}")
    print(f"  Test RMSE = {rmse2:.2f}")

    # ── 4. statsmodels OLS (full summary) ──────────────────
    print(f"\n--- Model 3: statsmodels OLS (all features) ---")
    formula = "HighQ ~ total_population + per_capita_income + percent_white"
    sm_model = smf.ols(formula=formula, data=train).fit()
    print(sm_model.summary())

    pred3 = sm_model.predict(test)
    rmse3 = np.sqrt(np.mean((test["HighQ"] - pred3) ** 2))
    print(f"\n  Test RMSE = {rmse3:.2f}")
    print(f"  R-squared = {sm_model.rsquared:.4f}")
    print(f"  Adj R-sq  = {sm_model.rsquared_adj:.4f}")

    # ── 5. Prediction table ────────────────────────────────
    print(f"\n--- Predictions (Model 3) ---")
    result = pd.DataFrame({
        "state":     test["state"].values,
        "actual":    test["HighQ"].values,
        "predicted": np.round(pred3.values, 2),
    })
    result["error"] = np.round(result["actual"] - result["predicted"], 2)
    print(result.to_string(index=False))

    # ── Visualisation ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # scatter: HighQ vs per_capita_income
    ax = axes[0]
    ax.scatter(train["per_capita_income"], train["HighQ"],
               alpha=0.6, label="Train")
    ax.scatter(test["per_capita_income"], test["HighQ"],
               marker="x", s=80, color="red", label="Test")
    z = np.polyfit(train["per_capita_income"], train["HighQ"], 1)
    xs = np.linspace(train["per_capita_income"].min(),
                     train["per_capita_income"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "r--")
    ax.set_xlabel("Per Capita Income ($)")
    ax.set_ylabel("HighQ Price ($)")
    ax.set_title("HighQ vs Income")
    ax.legend(fontsize=8)

    # correlation heatmap
    ax = axes[1]
    corr = train[["HighQ"] + features].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    labels = [c[:8] for c in corr.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=6)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # actual vs predicted
    ax = axes[2]
    ax.scatter(result["actual"], result["predicted"], s=60, edgecolors="grey")
    lims = [min(result["actual"].min(), result["predicted"].min()) - 2,
            max(result["actual"].max(), result["predicted"].max()) + 2]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Actual vs Predicted  (RMSE = {rmse3:.1f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig("regression_weed_demographics.png", dpi=150)
    plt.show()
    print("\nFigure saved: regression_weed_demographics.png")


if __name__ == "__main__":
    main()
