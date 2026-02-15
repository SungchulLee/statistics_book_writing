# Heatmaps for Correlation Matrices

## Overview

A **heatmap** is a two-dimensional visualization that uses color intensity to represent numerical values in a matrix. For correlation matrices, heatmaps reveal patterns of association across many variables simultaneously, making them indispensable for exploratory analysis of multivariate datasets. Colors (typically blue for negative correlation, white for zero, red for positive) encode the strength and direction of relationships at a glance.

---

## Basic Heatmap of a Correlation Matrix

### Financial Example: S&P 500 Exchange-Traded Funds (ETFs)

Exchange-traded funds (ETFs) track broad market segments. By examining correlations among sector ETFs, investors can assess diversification—whether holdings move independently or in lockstep.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load S&P 500 data with ETF symbols
sp500_sym = pd.read_csv('sp500_sectors.csv')
sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)

# Filter for ETFs only (major exchanges), from July 2012 onward
etfs = sp500_px.loc[sp500_px.index > '2012-07-01',
                    sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]

# Compute correlation matrix
corr_matrix = etfs.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix,
            vmin=-1, vmax=1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            ax=ax,
            square=True,
            cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Heatmap: S&P 500 ETFs (2012–2015)')
plt.tight_layout()
plt.show()
```

### Interpreting the Heatmap

The heatmap reveals:

- **Positive correlations (red):** Sector ETFs that rise and fall together (e.g., technology and communication services often move in sync during growth phases)
- **Negative correlations (blue):** Sectors that tend to diverge (e.g., defensive sectors like utilities versus cyclical sectors like consumer discretionary)
- **Near-zero correlations (white):** Independent movements, beneficial for portfolio diversification
- **Diagonal (all 1.0, dark red):** Each ETF is perfectly correlated with itself

### Portfolio Insight

High positive correlations limit diversification benefits. If you hold two ETFs with a correlation of 0.8, they move nearly in lockstep, providing less risk reduction than two uncorrelated ETFs. A well-diversified portfolio targets low or negative correlations.

---

## Heatmap with Annotation

Adding numerical values to heatmap cells aids interpretation:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and filter data (same as above)
sp500_sym = pd.read_csv('sp500_sectors.csv')
sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)
etfs = sp500_px.loc[sp500_px.index > '2012-07-01',
                    sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]

corr_matrix = etfs.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix,
            vmin=-1, vmax=1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            annot=True,  # Show correlation values
            fmt='.2f',   # Format to 2 decimal places
            ax=ax,
            square=True,
            cbar_kws={'label': 'Correlation'},
            cbar=False)  # Optional: remove colorbar if space is tight
ax.set_title('Annotated Correlation Heatmap: S&P 500 ETFs')
plt.tight_layout()
plt.show()
```

With annotations, specific correlation pairs are easy to identify:
- SPY (S&P 500 total market) correlates highly with QQQ (tech-heavy Nasdaq) and DIA (large-cap), as expected
- GLD (gold) often shows lower or negative correlation with equity ETFs, making it a hedge

---

## Handling Large Correlation Matrices

When many variables create dense, hard-to-read heatmaps, consider:

### 1. Clustering (Reordering)

Use hierarchical clustering to group similar variables:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Compute and cluster
corr_matrix = etfs.corr()
linkage_matrix = linkage(1 - corr_matrix, method='ward')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix,
            cmap=sns.diverging_palette(20, 220, as_cmap=True),
            vmin=-1, vmax=1,
            ax=ax,
            square=True)
ax.set_title('Clustered Correlation Heatmap')
plt.tight_layout()
plt.show()
```

Clustering reorders rows and columns so that strongly correlated variables cluster together, revealing block structure in the correlation matrix.

### 2. Subsetting

Select only a subset of variables of interest:

```python
# Focus on sector funds only (exclude single-asset ETFs)
sector_etfs = ['XLI', 'QQQ', 'XLE', 'XLY', 'XLU', 'XLB', 'XLV', 'XLP', 'XLF', 'XLK']
subset_corr = corr_matrix.loc[sector_etfs, sector_etfs]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(subset_corr, annot=True, fmt='.2f', ax=ax, square=True)
ax.set_title('Sector ETF Correlations')
plt.tight_layout()
plt.show()
```

---

## Grayscale Heatmap (For Print)

When publishing in monochrome or with printing constraints, use a grayscale colormap and add visual cues:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = etfs.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix,
            cmap='gray',  # Grayscale colormap
            vmin=-1, vmax=1,
            ax=ax,
            square=True,
            cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Heatmap (Grayscale)')
plt.tight_layout()
plt.show()
```

---

## Real-World Application: Risk Management

Portfolio managers use correlation heatmaps to assess:

1. **Systemic risk:** Do all holdings rise and fall together? High correlations indicate vulnerability to market-wide shocks.
2. **Hedging effectiveness:** Do some holdings move opposite to others? Negative correlations provide natural hedges.
3. **Sector concentration:** Are positions redundant (high correlation) or diversified?

A portfolio with correlations near 0.8 offers poor diversification. A portfolio with a mix of positive, near-zero, and negative correlations provides genuine risk mitigation.

---

## Limitations and Considerations

- **Correlation ≠ Causation:** A high correlation between two variables doesn't imply one causes the other.
- **Time-varying correlations:** Correlations change over market regimes. Heatmaps represent a static snapshot.
- **Non-linear relationships:** Heatmaps display Pearson correlation (linear). Non-linear dependencies may be missed.
- **Outliers:** Extreme events can distort correlation estimates; consider robust alternatives for contaminated data.

---

## Summary

Heatmaps transform a correlation matrix into a visual format where patterns emerge instantly. For investors, they reveal diversification potential; for data scientists, they expose multicollinearity. Combined with clustering or subsetting techniques, heatmaps remain one of the most practical tools for multivariate exploratory analysis.

