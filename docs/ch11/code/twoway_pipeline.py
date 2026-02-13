#!/usr/bin/env python3
# ======================================================================
# 13_anova2_09_end_to_end_pipeline.py
# ======================================================================
# End-to-end pipeline on ToothGrowth:
#   - Two-way ANOVA (Type II)
#   - Tukey HSD for main effects
#   - Interaction Tukey
#   - Quick interaction plot
# ======================================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot

def main():
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2, 3])

    # Two-way ANOVA (Type II is often recommended when design is balanced/near-balanced)
    model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
    aov2 = anova_lm(model, typ=2)
    print("=== Two-Way ANOVA (Type II) ===")
    print(aov2, "\n")

    print("=== Tukey HSD: main effect of dose ===")
    print(pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05), "\n")

    print("=== Tukey HSD: main effect of supp ===")
    print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05), "\n")

    df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)
    print("=== Tukey HSD: interaction groups (supp × dose) ===")
    print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05), "\n")

    # Interaction plot
    fig, ax = plt.subplots(figsize=(8, 4))
    interaction_plot(df['dose'], df['supp'], df['len'], ax=ax, markers=['o','s'], linestyles=['--','-.'])
    ax.set_title("Interaction: dose × supp")
    ax.set_xlabel("dose"); ax.set_ylabel("len")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
