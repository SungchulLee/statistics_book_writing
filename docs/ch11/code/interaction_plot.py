#!/usr/bin/env python3
# ======================================================================
# 13_anova2_05_interaction_plot.py
# ======================================================================
# Visualize the interaction (dose × supp) on len using statsmodels' interaction_plot.
# ======================================================================

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot

def main():
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 4))
    interaction_plot(df['dose'], df['supp'], df['len'],
                     ax=ax, markers=['o', 's'], linestyles=['--', '-.'])
    ax.set_title("Interaction: dose × supp on tooth length")
    ax.set_xlabel("dose")
    ax.set_ylabel("len")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
