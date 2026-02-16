"""
3D Multiple Regression Plane Visualization

This script creates a 3D visualization of a multiple linear regression fit
with two continuous predictors. It shows:

1. The regression plane in 3D space defined by the two predictors
2. The actual data points scattered in 3D
3. How the plane fits through the data cloud

This is particularly useful for understanding:
- How multiple regression fits a hyperplane through multivariate data
- The relationship between the predictors and the response
- The magnitude of residuals (distances from points to the plane)
- Visualization limitations when moving to higher dimensions (3+ predictors)

The example uses Sales as the response and Radio and TV as predictors,
demonstrating a model without interaction terms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Generate or Load Data
# ============================================================
# For demonstration, we create a synthetic advertising dataset
# In practice, you would load the real ISLR Advertising dataset

np.random.seed(42)
n_samples = 150
TV = np.random.uniform(0, 300, n_samples)
Radio = np.random.uniform(0, 50, n_samples)
# Sales linearly related to TV and Radio with noise
Sales = 5 + 0.04 * TV + 0.15 * Radio + np.random.normal(0, 1.5, n_samples)

# Create a DataFrame similar to ISLR's Advertising dataset
advertising = pd.DataFrame({
    'TV': TV,
    'Radio': Radio,
    'Sales': Sales
})

print("Data Summary:")
print(advertising.describe())
print()

# ============================================================
# 2. Fit Multiple Linear Regression
# ============================================================
X = advertising[['Radio', 'TV']].values  # Note: Radio first, then TV
y = advertising['Sales'].values

model = LinearRegression()
model.fit(X, y)

beta_0 = model.intercept_
beta_1 = model.coef_[0]  # Coefficient for Radio
beta_2 = model.coef_[1]  # Coefficient for TV

print(f"Fitted Model:")
print(f"  Sales = {beta_0:.4f} + {beta_1:.4f}*Radio + {beta_2:.4f}*TV")
print()

# Model performance
y_pred = model.predict(X)
rss = np.sum((y - y_pred) ** 2)
tss = np.sum((y - y.mean()) ** 2)
r_squared = 1 - (rss / tss)

print(f"Model Performance:")
print(f"  RSS = {rss:.2f}")
print(f"  R² = {r_squared:.4f}")
print()

# ============================================================
# 3. Create the Regression Plane Mesh
# ============================================================
# Define ranges for Radio and TV
Radio_range = np.arange(0, 50, 5)
TV_range = np.arange(0, 300, 30)

# Create meshgrid
Radio_mesh, TV_mesh = np.meshgrid(Radio_range, TV_range, indexing='xy')

# Compute predicted Sales values on the plane
Sales_mesh = np.zeros_like(Radio_mesh, dtype=float)
for i in range(Radio_mesh.shape[0]):
    for j in range(Radio_mesh.shape[1]):
        sales_pred = beta_0 + beta_1 * Radio_mesh[i, j] + beta_2 * TV_mesh[i, j]
        Sales_mesh[i, j] = sales_pred

print(f"Regression Plane Range:")
print(f"  Sales predictions: {Sales_mesh.min():.2f} to {Sales_mesh.max():.2f}")
print()

# ============================================================
# 4. Create 3D Visualization
# ============================================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Plot the regression plane ---
surface = ax.plot_surface(Radio_mesh, TV_mesh, Sales_mesh,
                          alpha=0.3, cmap='coolwarm',
                          edgecolor='none', antialiased=True)

# --- Plot the actual data points ---
scatter = ax.scatter(advertising['Radio'], advertising['TV'], advertising['Sales'],
                     c='blue', marker='o', s=50, alpha=0.6,
                     edgecolors='darkblue', linewidth=0.5,
                     label='Actual Data Points')

# ============================================================
# 5. Customize Plot
# ============================================================
ax.set_xlabel('Radio (Advertising Spend)', fontsize=12, fontweight='bold')
ax.set_ylabel('TV (Advertising Spend)', fontsize=12, fontweight='bold')
ax.set_zlabel('Sales', fontsize=12, fontweight='bold')
ax.set_title('Multiple Linear Regression: Sales ~ Radio + TV\n(Regression Plane with Data Points)',
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar for surface
cbar = plt.colorbar(surface, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Predicted Sales', fontsize=11)

# Adjust viewing angle for better visualization
ax.view_init(elev=20, azim=45)

# Add legend
ax.legend(['Data Points'], loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('regression_plane_3d.png', dpi=150, bbox_inches='tight')
print("Saved: regression_plane_3d.png")
plt.show()

# ============================================================
# 6. Alternative View: Different Angle
# ============================================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Plot the regression plane ---
surface = ax.plot_surface(Radio_mesh, TV_mesh, Sales_mesh,
                          alpha=0.3, cmap='coolwarm',
                          edgecolor='none', antialiased=True)

# --- Plot the actual data points ---
ax.scatter(advertising['Radio'], advertising['TV'], advertising['Sales'],
           c='blue', marker='o', s=50, alpha=0.6,
           edgecolors='darkblue', linewidth=0.5,
           label='Actual Data Points')

# --- Optional: Draw vertical lines from points to plane (residuals) ---
y_pred_for_residuals = model.predict(X)
for i in range(0, len(X), 5):  # Plot every 5th residual to avoid clutter
    ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]],
            [y[i], y_pred_for_residuals[i]],
            'r-', alpha=0.3, linewidth=0.8)

ax.set_xlabel('Radio (Advertising Spend)', fontsize=12, fontweight='bold')
ax.set_ylabel('TV (Advertising Spend)', fontsize=12, fontweight='bold')
ax.set_zlabel('Sales', fontsize=12, fontweight='bold')
ax.set_title('Multiple Linear Regression with Residuals\n(Distances from Points to Plane)',
             fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(surface, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Predicted Sales', fontsize=11)

# Different viewing angle
ax.view_init(elev=15, azim=120)

ax.legend(['Data Points'], loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('regression_plane_3d_residuals.png', dpi=150, bbox_inches='tight')
print("Saved: regression_plane_3d_residuals.png")
plt.show()

# ============================================================
# 7. Residual Analysis Summary
# ============================================================
print("\nResidual Analysis:")
print("=" * 70)
residuals = y - y_pred
print(f"Mean Residual:     {residuals.mean():.6f} (should be ≈ 0)")
print(f"Std Dev Residuals: {residuals.std():.4f}")
print(f"Min Residual:      {residuals.min():.4f}")
print(f"Max Residual:      {residuals.max():.4f}")
print(f"90th percentile:   {np.percentile(np.abs(residuals), 90):.4f}")
print("=" * 70)

# ============================================================
# 8. Interpretation
# ============================================================
print("\nInterpretation:")
print("=" * 70)
print(f"""
The 3D visualization shows:

1. REGRESSION PLANE (colored surface):
   A flat plane in 3D space that represents the linear relationship:
   Sales = {beta_0:.2f} + {beta_1:.4f}*Radio + {beta_2:.4f}*TV

2. DATA POINTS (blue dots):
   The actual observations in 3D. These points are scattered around the
   plane—some above it (positive residuals) and some below (negative).

3. PLANE ORIENTATION:
   - The slope with respect to Radio is {beta_1:.4f}
   - The slope with respect to TV is {beta_2:.4f}
   - These indicate how Sales change for unit increases in each predictor

4. FIT QUALITY:
   - R² = {r_squared:.4f} means {r_squared*100:.1f}% of variation is explained
   - Residuals (red lines in second plot) show prediction errors

5. LIMITATIONS:
   - With 3+ predictors, we move to higher dimensions that cannot be
     visualized directly. The plane becomes a hyperplane.
   - Interpretation extends naturally: each predictor has a coefficient
     that represents its marginal effect on the response.

6. ASSUMPTIONS CHECK:
   - Points should scatter randomly around the plane (no patterns)
   - No funnel-shaped pattern (constant variance assumption)
   - No heavy clustering far from the plane (outliers)
""")
print("=" * 70)
