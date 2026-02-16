"""
3D RSS Surface Visualization

This script creates powerful 3D visualizations of the Residual Sum of Squares (RSS)
surface as a function of regression coefficients β₀ (intercept) and β₁ (slope).

These visualizations provide intuition about:
- How RSS changes across different coefficient values
- The optimal coefficients that minimize RSS (the global minimum)
- The convex nature of the RSS function
- How the contour plot relates to the 3D surface

The script demonstrates:
1. Fitting a simple linear regression model
2. Creating a 3D surface mesh of RSS values
3. Visualizing the surface as a 3D plot
4. Visualizing contours (2D projections of the surface)
5. Marking the optimal coefficients found by the model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Generate or Load Data
# ============================================================
# For demonstration, we create a synthetic advertising dataset
# In practice, you would load the real ISLR Advertising dataset

np.random.seed(42)
n_samples = 100
TV = np.random.uniform(0, 300, n_samples)
# Sales linearly related to TV with noise
Sales = 7 + 0.05 * TV + np.random.normal(0, 2, n_samples)

# Create a DataFrame similar to ISLR's Advertising dataset
advertising = pd.DataFrame({
    'TV': TV,
    'Sales': Sales
})

print("Data Summary:")
print(advertising.describe())
print()

# ============================================================
# 2. Fit Simple Linear Regression
# ============================================================
# Extract features and target
X = advertising['TV'].values.reshape(-1, 1)
y = advertising['Sales'].values

# Scale X (optional but helps with visualization range)
scaler = StandardScaler(with_mean=True, with_std=False)
X_scaled = scaler.fit_transform(X)

# Fit model
model = LinearRegression()
model.fit(X_scaled, y)

beta_0 = model.intercept_  # β₀
beta_1 = model.coef_[0]    # β₁

print(f"Fitted Model:")
print(f"  β₀ (intercept) = {beta_0:.4f}")
print(f"  β₁ (slope)     = {beta_1:.4f}")
print()

# ============================================================
# 3. Create Parameter Grid
# ============================================================
# Define ranges for β₀ and β₁ around the optimal values
B0_range = np.linspace(beta_0 - 2, beta_0 + 2, 50)
B1_range = np.linspace(beta_1 - 0.05, beta_1 + 0.05, 50)

# Create meshgrid
B0_mesh, B1_mesh = np.meshgrid(B0_range, B1_range, indexing='xy')

# ============================================================
# 4. Compute RSS for Each Coefficient Pair
# ============================================================
RSS = np.zeros_like(B0_mesh)

for i in range(B0_mesh.shape[0]):
    for j in range(B0_mesh.shape[1]):
        b0 = B0_mesh[i, j]
        b1 = B1_mesh[i, j]
        # Predicted values: y_hat = b0 + b1 * X
        y_pred = b0 + b1 * X_scaled
        # Residual sum of squares
        RSS[i, j] = np.sum((y - y_pred) ** 2)

# Scale RSS for better visualization
RSS_scaled = RSS / 1000.0

print(f"RSS Statistics:")
print(f"  Minimum RSS: {RSS.min():.2f}")
print(f"  Maximum RSS: {RSS.max():.2f}")
print(f"  RSS at fitted model: {np.sum((y - model.predict(X_scaled)) ** 2):.2f}")
print()

# ============================================================
# 5. Create Visualizations
# ============================================================
fig = plt.figure(figsize=(16, 6))

# --- Subplot 1: Contour Plot (2D projection) ---
ax1 = fig.add_subplot(121)

contour = ax1.contour(B0_mesh, B1_mesh, RSS_scaled, levels=20, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)

# Mark the optimal coefficients
ax1.plot(beta_0, beta_1, 'r*', markersize=20, label='Optimal (β₀, β₁)')

ax1.set_xlabel('β₀ (Intercept)', fontsize=12)
ax1.set_ylabel('β₁ (Slope)', fontsize=12)
ax1.set_title('RSS Contour Plot (2D Projection)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: 3D Surface Plot ---
ax2 = fig.add_subplot(122, projection='3d')

# Plot surface
surface = ax2.plot_surface(B0_mesh, B1_mesh, RSS_scaled,
                           cmap='viridis', alpha=0.8,
                           edgecolor='none', antialiased=True)

# Mark the optimal point on the surface
rss_min_idx = np.unravel_index(np.argmin(RSS), RSS.shape)
z_min = RSS_scaled[rss_min_idx]
ax2.scatter([beta_0], [beta_1], [z_min], color='red', s=100, marker='*',
            label='Optimal (β₀, β₁)', zorder=10)

ax2.set_xlabel('β₀ (Intercept)', fontsize=11)
ax2.set_ylabel('β₁ (Slope)', fontsize=11)
ax2.set_zlabel('RSS / 1000', fontsize=11)
ax2.set_title('RSS 3D Surface', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)

# Add colorbar
cbar = plt.colorbar(surface, ax=ax2, pad=0.1, shrink=0.8)
cbar.set_label('RSS / 1000', fontsize=10)

plt.tight_layout()
plt.savefig('rss_surface_visualization.png', dpi=150, bbox_inches='tight')
print("Saved: rss_surface_visualization.png")
plt.show()

# ============================================================
# 6. Interpretation
# ============================================================
print("\nInterpretation:")
print("=" * 70)
print("""
The contour plot (left) and 3D surface (right) show how RSS changes
as we vary the regression coefficients β₀ and β₁.

Key observations:
1. CONVEXITY: The RSS surface is convex (bowl-shaped), with a single
   global minimum at the optimal coefficients.

2. GRADIENT: The density of contour lines indicates the slope of the
   surface. Tightly packed contours mean the RSS changes rapidly.

3. OPTIMAL POINT: The red star marks the coefficients (β₀, β₁) that
   minimize RSS—these are the values found by ordinary least squares (OLS).

4. SURFACE SHAPE: The contours are roughly elliptical, not circular,
   because the RSS function is more sensitive to changes in some
   directions than others (related to correlation between features).

5. INTERPRETATION FOR PREDICTION: Any point away from the red star
   yields worse predictions (higher RSS) than the optimal model.
""")
print("=" * 70)
