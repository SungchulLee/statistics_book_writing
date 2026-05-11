# Correlation Ellipse Plot

## Overview

Ellipse plots provide an alternative to color-coded heatmaps for visualizing correlation matrices. Each cell in the matrix is represented by an ellipse whose shape, orientation, and size encode the sign and magnitude of the correlation. This technique is particularly useful for grayscale publications and is accessible to color-blind readers.

---

## Ellipse Encoding

The correspondence between ellipse geometry and correlation is as follows:

| Correlation | Ellipse Shape | Rotation |
|:---:|:---|:---:|
| $r = +1$ | Thin line (degenerate ellipse) | $+45°$ |
| $0 < r < 1$ | Narrow ellipse | $+45°$ |
| $r = 0$ | Circle | $0°$ |
| $-1 < r < 0$ | Narrow ellipse | $-45°$ |
| $r = -1$ | Thin line (degenerate ellipse) | $-45°$ |

The key idea is that the **eccentricity** of the ellipse encodes $|r|$ while the **orientation** encodes the sign of $r$. A perfect circle indicates zero correlation; as $|r| \to 1$, the ellipse collapses toward a line.

---

## The Ellipse Construction

For a correlation value $r_{ij}$, the ellipse parameters are:

- **Width**: $w = 1 + \delta$ (approximately constant)
- **Height**: $h = 1 - |r_{ij}| - \delta$ (shrinks as $|r_{ij}|$ increases)
- **Angle**: $\theta = 45° \cdot \text{sign}(r_{ij})$

where $\delta$ is a small constant for numerical stability. The ratio of width to height determines the eccentricity, and the rotation angle distinguishes positive from negative correlations.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


def plot_corr_ellipses(data, figsize=None, **kwargs):
    M = np.array(data)
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    w = np.ones_like(M).ravel() + 0.01
    h = 1 - np.abs(M).ravel() - 0.01
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(
        widths=w, heights=h, angles=a,
        units='x', offsets=xy,
        norm=Normalize(vmin=-1, vmax=1),
        transOffset=ax.transData,
        array=M.ravel(),
        **kwargs
    )
    ax.add_collection(ec)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec, ax
```

---

## Example with Synthetic Data

We create five correlated variables and visualize their correlation structure using the ellipse plot:

```python
np.random.seed(42)
n = 200
z1 = np.random.randn(n)
z2 = np.random.randn(n)

x1 = z1
x2 = 0.8 * z1 + 0.2 * z2
x3 = -0.6 * z1 + np.random.randn(n) * 0.7
x4 = 0.3 * z1 + 0.7 * z2
x5 = np.random.randn(n)

df = pd.DataFrame(
    np.column_stack([x1, x2, x3, x4, x5]),
    columns=['Tech', 'Finance', 'Utilities', 'Energy', 'Commodity']
)

corr_matrix = df.corr()

ec, ax = plot_corr_ellipses(corr_matrix, figsize=(6, 5), cmap='bwr_r')
plt.colorbar(ec, ax=ax, label='Correlation Coefficient')
ax.set_title('Correlation Matrix: Ellipse Visualization')
plt.tight_layout()
plt.show()
```

---

## Reading the Ellipse Plot

Inspecting the output, we observe:

- **Tech--Finance**: a narrow ellipse tilted at $+45°$, indicating strong positive correlation ($r \approx 0.8$).
- **Tech--Utilities**: a narrow ellipse tilted at $-45°$, indicating moderate negative correlation ($r \approx -0.6$).
- **Commodity** row/column: nearly circular ellipses, indicating near-zero correlation with all other variables.
- **Diagonal**: degenerate ellipses (lines at $+45°$) corresponding to $r = 1$.

---

## Interpretation

Ellipse plots offer several advantages over standard heatmaps:

1. **Grayscale compatibility.** Even without color, the ellipse shape and orientation convey full information about the correlation.
2. **Accessibility.** Color-blind readers can interpret the plot without any loss of information.
3. **Dual encoding.** Both magnitude ($|r|$ via eccentricity) and sign ($\text{sign}(r)$ via rotation) are encoded simultaneously.

The main disadvantage is that exact numerical values are harder to read compared to annotated heatmaps. For publications, it is common to present both an ellipse plot and a numerical table of correlations.

---

## Exercises

**Exercise 1.**
Create a $4 \times 4$ correlation matrix by hand with the following properties: $r_{12} = 0.9$, $r_{13} = -0.7$, $r_{14} = 0$, $r_{23} = -0.5$, $r_{24} = 0.3$, $r_{34} = -0.2$. Plot it using the ellipse function and verify visually that the ellipse shapes match your expectations.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    import pandas as pd

    corr = np.array([
        [1.0,  0.9, -0.7,  0.0],
        [0.9,  1.0, -0.5,  0.3],
        [-0.7, -0.5,  1.0, -0.2],
        [0.0,  0.3, -0.2,  1.0]
    ])
    df_corr = pd.DataFrame(corr, columns=['A', 'B', 'C', 'D'],
                           index=['A', 'B', 'C', 'D'])

    ec, ax = plot_corr_ellipses(df_corr, figsize=(5, 5), cmap='bwr_r')
    ax.set_title('Hand-crafted Correlation Matrix')
    import matplotlib.pyplot as plt
    plt.colorbar(ec, ax=ax)
    plt.tight_layout()
    plt.show()
    ```

    The ellipse for $(A, B)$ should be very narrow at $+45°$ (strong positive). The ellipse for $(A, C)$ should be moderately narrow at $-45°$ (strong negative). The ellipse for $(A, D)$ should be nearly circular (zero correlation). $\square$

---

**Exercise 2.**
Explain why the matrix $\begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix}$ is a valid correlation matrix but $\begin{pmatrix} 1 & 1.2 \\ 1.2 & 1 \end{pmatrix}$ is not. State the necessary and sufficient conditions for a matrix to be a valid correlation matrix.

??? success "Solution to Exercise 2"

    A matrix $\mathbf{R}$ is a valid correlation matrix if and only if:

    1. It is symmetric: $R_{ij} = R_{ji}$.
    2. All diagonal entries equal 1: $R_{ii} = 1$.
    3. All off-diagonal entries satisfy $|R_{ij}| \le 1$.
    4. It is positive semi-definite: $\mathbf{v}^\top \mathbf{R}\, \mathbf{v} \ge 0$ for all $\mathbf{v}$.

    For the first matrix, the eigenvalues are $1 + 0.9 = 1.9$ and $1 - 0.9 = 0.1$, both non-negative, so it is valid.

    For the second matrix, the entry $1.2$ violates condition 3 since $|1.2| > 1$. Additionally, the determinant is $1 \cdot 1 - 1.2 \cdot 1.2 = -0.44 < 0$, so the matrix has a negative eigenvalue and fails condition 4. Correlation coefficients are bounded by $[-1, 1]$ by the Cauchy--Schwarz inequality, so $r = 1.2$ is impossible. $\square$

---

**Exercise 3.**
Modify the `plot_corr_ellipses` function to display the numerical correlation value inside each ellipse. Test on a $5 \times 5$ matrix.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import Normalize

    def plot_corr_ellipses_annotated(data, figsize=None, **kwargs):
        M = np.array(data)
        fig, ax = plt.subplots(1, 1, figsize=figsize,
                               subplot_kw={'aspect': 'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)
        ax.invert_yaxis()

        xy = np.indices(M.shape)[::-1].reshape(2, -1).T
        w = np.ones_like(M).ravel() + 0.01
        h = 1 - np.abs(M).ravel() - 0.01
        a = 45 * np.sign(M).ravel()

        ec = EllipseCollection(
            widths=w, heights=h, angles=a,
            units='x', offsets=xy,
            norm=Normalize(vmin=-1, vmax=1),
            transOffset=ax.transData,
            array=M.ravel(), **kwargs
        )
        ax.add_collection(ec)

        # Add text annotations
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.2f}',
                        ha='center', va='center', fontsize=8)

        if isinstance(data, pd.DataFrame):
            ax.set_xticks(np.arange(M.shape[1]))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticks(np.arange(M.shape[0]))
            ax.set_yticklabels(data.index)

        return ec, ax

    # Test
    np.random.seed(42)
    data = np.random.randn(200, 5)
    df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 6)])
    ec, ax = plot_corr_ellipses_annotated(df.corr(), figsize=(6, 6),
                                          cmap='bwr_r')
    plt.colorbar(ec, ax=ax)
    plt.tight_layout()
    plt.show()
    ```

    The annotation loop iterates over all $(i, j)$ entries and places the numerical value at the center of each ellipse using `ax.text`. This combines the advantages of ellipse visualization (shape encoding) with exact numerical readability. $\square$

---

**Exercise 4.**
For a $2 \times 2$ correlation matrix $\mathbf{R} = \begin{pmatrix} 1 & r \\ r & 1 \end{pmatrix}$, derive the eigenvalues and eigenvectors in terms of $r$. Show that the principal axes of the concentration ellipse of a bivariate normal distribution align with the eigenvectors of $\mathbf{R}$.

??? success "Solution to Exercise 4"

    The characteristic equation is:

    $$
    \det(\mathbf{R} - \lambda \mathbf{I}) = (1 - \lambda)^2 - r^2 = 0
    $$

    Solving: $\lambda_1 = 1 + r$ and $\lambda_2 = 1 - r$.

    For $\lambda_1 = 1 + r$: $(\mathbf{R} - \lambda_1 \mathbf{I})\mathbf{v} = 0$ gives $-r v_1 + r v_2 = 0$, so $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^\top$.

    For $\lambda_2 = 1 - r$: similarly, $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^\top$.

    The concentration ellipse of a bivariate normal with correlation $r$ is the set $\{(x, y) : \mathbf{z}^\top \mathbf{R}^{-1} \mathbf{z} = c\}$ for some constant $c$, where $\mathbf{z} = (x, y)^\top$. The principal axes of this ellipse are the eigenvectors of $\mathbf{R}^{-1}$ (equivalently, of $\mathbf{R}$, since they share eigenvectors). The axes point along $\frac{1}{\sqrt{2}}(1, 1)^\top$ and $\frac{1}{\sqrt{2}}(1, -1)^\top$, which are the $+45°$ and $-45°$ directions. This is why the ellipse plot uses $\pm 45°$ rotations. $\square$

---

**Exercise 5.**
Given $k$ variables, the ellipse plot contains $k^2$ ellipses. How many of these are redundant (i.e., can be inferred from other ellipses)? Propose a modified version that shows only the non-redundant ellipses.

??? success "Solution to Exercise 5"

    Since the correlation matrix is symmetric ($R_{ij} = R_{ji}$), the upper and lower triangles are mirror images. The diagonal always shows $r = 1$. Therefore:

    - Total ellipses: $k^2$
    - Diagonal (trivial, $r = 1$): $k$
    - Unique off-diagonal: $\frac{k(k-1)}{2}$
    - Redundant: $k + \frac{k(k-1)}{2} = \frac{k(k+1)}{2}$

    A non-redundant version shows only the $\frac{k(k-1)}{2}$ lower-triangle ellipses:

    ```python
    def plot_lower_triangle_ellipses(data, figsize=None, **kwargs):
        M = np.array(data)
        k = M.shape[0]
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={'aspect': 'equal'})
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(-0.5, k - 0.5)
        ax.invert_yaxis()

        # Only lower triangle
        for i in range(k):
            for j in range(i):
                r = M[i, j]
                from matplotlib.patches import Ellipse
                e = Ellipse(xy=(j, i),
                            width=1.01,
                            height=1 - abs(r) - 0.01,
                            angle=45 * np.sign(r))
                e.set_facecolor(plt.cm.bwr_r((r + 1) / 2))
                ax.add_patch(e)

        if isinstance(data, pd.DataFrame):
            ax.set_xticks(range(k))
            ax.set_xticklabels(data.columns, rotation=90)
            ax.set_yticks(range(k))
            ax.set_yticklabels(data.index)
        plt.tight_layout()
        return ax
    ```

    This reduces visual clutter and focuses on the $\binom{k}{2}$ unique correlation values. $\square$
