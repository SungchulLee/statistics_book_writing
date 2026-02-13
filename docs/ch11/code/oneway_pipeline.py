#!/usr/bin/env python3
# ======================================================================
# 13_anova_06_end_to_end_pipeline.py
# ======================================================================
# End-to-end:
#   1) One-way ANOVA (statsmodels)
#   2) Tukey HSD (statsmodels)
#   3) Pairwise Welch t-tests + Bonferroni correction
#   4) Quick boxplot to visualize groups
# ======================================================================

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

def main():
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])  # group, weight

    # 1) ANOVA (statsmodels)
    model = ols('weight ~ C(group)', data=df).fit()
    aov = anova_lm(model)
    print("=== One-Way ANOVA (statsmodels) ===")
    print(aov, "\n")

    # 2) Tukey HSD (statsmodels)
    tukey = pairwise_tukeyhsd(endog=df['weight'], groups=df['group'], alpha=0.05)
    print("=== Tukey's HSD (statsmodels) ===")
    print(tukey, "\n")

    # 3) Pairwise Welch t-tests + Bonferroni
    groups = df['group'].unique()
    p_raw = []
    labels = []
    for g1, g2 in combinations(groups, 2):
        x = df.loc[df['group'] == g1, 'weight'].values
        y = df.loc[df['group'] == g2, 'weight'].values
        stat, p = ttest_ind(x, y, equal_var=False)
        p_raw.append(p)
        labels.append(f"{g1} vs {g2}")
    _, p_bonf, _, _ = multipletests(p_raw, alpha=0.05, method='bonferroni')

    print("=== Pairwise Welch t-tests (Bonferroni-corrected) ===")
    for lbl, p, pb in zip(labels, p_raw, p_bonf):
        print(f"{lbl:<12}  p = {p:8.4f}   p_bonf = {pb:8.4f}")
    print()

    # 4) Quick boxplot
    order = ['ctrl', 'trt1', 'trt2']
    data = [df.loc[df['group'] == g, 'weight'].values for g in order]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=order)
    plt.title('PlantGrowth weights by group')
    plt.xlabel('Group')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
