#!/usr/bin/env python3
# ======================================================================
# 13_anova_04_anova_scipy_with_plots.py
# ======================================================================
# One-way ANOVA with SciPy's f_oneway + visualization:
# 1) Boxplots of group distributions.
# 2) F-distribution PDF with shaded critical region (beyond observed F).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def load_data():
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])
    g = df.groupby('group')
    ctrl = g.get_group('ctrl').weight.values
    trt1 = g.get_group('trt1').weight.values
    trt2 = g.get_group('trt2').weight.values
    # degrees of freedom
    n_total = len(df)
    k = df['group'].nunique()
    df1 = k - 1
    df2 = n_total - k
    return df, (ctrl, trt1, trt2), df1, df2

def main():
    df, (ctrl, trt1, trt2), df1, df2 = load_data()
    F, p = stats.f_oneway(ctrl, trt1, trt2)
    print(f"ANOVA (SciPy): F = {F:.4f}, p = {p:.4f}  (df1={df1}, df2={df2})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot
    ax1.boxplot([ctrl, trt1, trt2], labels=['ctrl', 'trt1', 'trt2'])
    ax1.set_xlabel('Group'); ax1.set_ylabel('Weight')
    ax1.set_title('Plant weights by group')

    # F PDF with shaded tail beyond observed F
    x = np.linspace(0, 8, 400)
    pdf = stats.f(df1, df2).pdf(x)
    ax2.plot(x, pdf, label=f'F(df1={df1}, df2={df2}) PDF')
    mask = x >= F
    ax2.fill_between(x[mask], pdf[mask], alpha=0.3, label='Observed tail (p-value region)')

    ax2.set_title('F-distribution & observed tail')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
