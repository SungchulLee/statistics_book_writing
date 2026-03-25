# Exercises

Work through the following exercises to test your understanding of the mathematical and computational foundations covered in this chapter. Problems are organized by section and progress from conceptual to computational.

---

## Exercise 1: Negation of Quantified Statements

Write the formal negation of each statement.

**(a)** $\forall\, x \in \mathbb{R},\; x^2 \geq 0$

**(b)** $\exists\, \varepsilon > 0 \text{ such that } \forall\, n \in \mathbb{N},\; a_n > \varepsilon$

**(c)** $\forall\, \varepsilon > 0,\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon$

### Solution

**Part (a):**

$$
\exists\, x \in \mathbb{R} \text{ such that } x^2 < 0
$$

**Part (b):**

$$
\forall\, \varepsilon > 0,\; \exists\, n \in \mathbb{N} \text{ such that } a_n \leq \varepsilon
$$

**Part (c):** This is the negation of "$a_n$ converges to $L$":

$$
\exists\, \varepsilon > 0 \text{ such that } \forall\, N \in \mathbb{N},\; \exists\, n > N \text{ with } |a_n - L| \geq \varepsilon
$$

---

## Exercise 2: Set Operations and De Morgan's Laws

Let $A = \{1, 2, 3, 4, 5\}$, $B = \{3, 4, 5, 6, 7\}$, and $\Omega = \{1, 2, 3, 4, 5, 6, 7, 8\}$.

**(a)** Compute $A \cap B$, $A \cup B$, $A \setminus B$, and $A^c$ (complement with respect to $\Omega$).

**(b)** Verify De Morgan's law $(A \cup B)^c = A^c \cap B^c$ for these sets.

### Solution

**Part (a):**

- $A \cap B = \{3, 4, 5\}$
- $A \cup B = \{1, 2, 3, 4, 5, 6, 7\}$
- $A \setminus B = \{1, 2\}$
- $A^c = \{6, 7, 8\}$

**Part (b):**

- $(A \cup B)^c = \{1,2,3,4,5,6,7\}^c = \{8\}$
- $A^c \cap B^c = \{6,7,8\} \cap \{1,2,8\} = \{8\}$

Both sides equal $\{8\}$, confirming De Morgan's law.

---

## Exercise 3: Convergence Using the Epsilon-N Definition

Prove directly from the $\varepsilon$-$N$ definition that

$$
\lim_{n \to \infty} \frac{3n + 1}{n + 2} = 3
$$

### Solution

We need to show: for every $\varepsilon > 0$, there exists $N$ such that $n > N$ implies $|a_n - 3| < \varepsilon$.

Compute:

$$
\left| \frac{3n+1}{n+2} - 3 \right| = \left| \frac{3n+1 - 3(n+2)}{n+2} \right| = \left| \frac{-5}{n+2} \right| = \frac{5}{n+2}
$$

We want $\frac{5}{n+2} < \varepsilon$, which gives $n > \frac{5}{\varepsilon} - 2$.

Choose $N = \left\lceil \frac{5}{\varepsilon} - 2 \right\rceil$. Then for all $n > N$:

$$
\left| \frac{3n+1}{n+2} - 3 \right| = \frac{5}{n+2} < \varepsilon
$$

This completes the proof. $\square$

---

## Exercise 4: Geometric Series and Convergence

**(a)** Show that the geometric series $\sum_{k=0}^{\infty} r^k$ converges if and only if $|r| < 1$, and find its sum.

**(b)** Use part (a) to evaluate $\sum_{k=1}^{\infty} \frac{3}{4^k}$.

### Solution

**Part (a):** The partial sums are:

$$
S_n = \sum_{k=0}^{n} r^k = \frac{1 - r^{n+1}}{1 - r} \quad (r \neq 1)
$$

When $|r| < 1$, $r^{n+1} \to 0$ as $n \to \infty$, so $S_n \to \frac{1}{1 - r}$. When $|r| \geq 1$, the terms do not approach zero, so the series diverges.

**Part (b):**

$$
\sum_{k=1}^{\infty} \frac{3}{4^k} = 3 \sum_{k=1}^{\infty} \left(\frac{1}{4}\right)^k = 3 \left( \frac{1/4}{1 - 1/4} \right) = 3 \cdot \frac{1}{3} = 1
$$

---

## Exercise 5: Matrix Multiplication and the Design Matrix

Consider the design matrix for simple linear regression with intercept:

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}
$$

**(a)** Compute $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$ where $\mathbf{y} = (5, 9, 13)^T$.

**(b)** Solve the normal equations $\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}$ to find the least squares estimates $\hat{\beta}_0$ and $\hat{\beta}_1$.

### Solution

**Part (a):**

$$
\mathbf{X}^T \mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 4 & 6 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix} = \begin{pmatrix} 3 & 12 \\ 12 & 56 \end{pmatrix}
$$

$$
\mathbf{X}^T \mathbf{y} = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 4 & 6 \end{pmatrix} \begin{pmatrix} 5 \\ 9 \\ 13 \end{pmatrix} = \begin{pmatrix} 27 \\ 124 \end{pmatrix}
$$

**Part (b):** Solving $\begin{pmatrix} 3 & 12 \\ 12 & 56 \end{pmatrix} \hat{\boldsymbol{\beta}} = \begin{pmatrix} 27 \\ 124 \end{pmatrix}$:

The determinant is $3 \cdot 56 - 12 \cdot 12 = 168 - 144 = 24$.

$$
(\mathbf{X}^T \mathbf{X})^{-1} = \frac{1}{24} \begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix}
$$

$$
\hat{\boldsymbol{\beta}} = \frac{1}{24} \begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix} \begin{pmatrix} 27 \\ 124 \end{pmatrix} = \frac{1}{24} \begin{pmatrix} 1512 - 1488 \\ -324 + 372 \end{pmatrix} = \frac{1}{24} \begin{pmatrix} 24 \\ 48 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

The fitted line is $\hat{y} = 1 + 2x$, which passes through all three points exactly (the data are perfectly collinear).

---

## Exercise 6: Idempotent and Projection Matrices

A matrix $\mathbf{P}$ is **idempotent** if $\mathbf{P}^2 = \mathbf{P}$.

**(a)** Prove that if $\mathbf{P}$ is idempotent, then $\mathbf{I} - \mathbf{P}$ is also idempotent.

**(b)** Prove that the eigenvalues of an idempotent matrix are either 0 or 1.

**(c)** Show that $\text{tr}(\mathbf{P}) = \text{rank}(\mathbf{P})$ for an idempotent matrix.

### Solution

**Part (a):**

$$
(\mathbf{I} - \mathbf{P})^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P} = \mathbf{I} - \mathbf{P}
$$

$\square$

**Part (b):** Let $\lambda$ be an eigenvalue with eigenvector $\mathbf{v} \neq \mathbf{0}$. Then $\mathbf{P}\mathbf{v} = \lambda \mathbf{v}$. Applying $\mathbf{P}$ again:

$$
\mathbf{P}^2 \mathbf{v} = \mathbf{P}(\lambda \mathbf{v}) = \lambda^2 \mathbf{v}
$$

Since $\mathbf{P}^2 = \mathbf{P}$, we also have $\mathbf{P}^2 \mathbf{v} = \lambda \mathbf{v}$. Therefore $\lambda^2 = \lambda$, giving $\lambda(\lambda - 1) = 0$, so $\lambda = 0$ or $\lambda = 1$. $\square$

**Part (c):** The trace equals the sum of eigenvalues. By part (b), every eigenvalue is 0 or 1. The number of eigenvalues equal to 1 equals the rank (since the rank equals the dimension of the column space, which corresponds to the eigenvalue-1 eigenspace for an idempotent matrix). Therefore $\text{tr}(\mathbf{P}) = \text{rank}(\mathbf{P})$. $\square$

---

## Exercise 7: Symmetric Matrices and the Spectral Theorem

Let $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$.

**(a)** Find the eigenvalues and eigenvectors of $\mathbf{A}$.

**(b)** Write the spectral decomposition $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$.

**(c)** Verify that $\mathbf{A}$ is positive definite.

### Solution

**Part (a):** The characteristic equation is:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0
$$

The eigenvalues are $\lambda_1 = 1$ and $\lambda_2 = 3$.

For $\lambda_1 = 1$: $(\mathbf{A} - \mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$.

For $\lambda_2 = 3$: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$.

**Part (b):**

$$
\mathbf{A} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
$$

**Part (c):** A symmetric matrix is positive definite if and only if all eigenvalues are strictly positive. Since $\lambda_1 = 1 > 0$ and $\lambda_2 = 3 > 0$, the matrix $\mathbf{A}$ is positive definite.

Equivalently, for any nonzero $\mathbf{x} = (x_1, x_2)^T$:

$$
\mathbf{x}^T \mathbf{A} \mathbf{x} = 2x_1^2 + 2x_1 x_2 + 2x_2^2 = x_1^2 + (x_1 + x_2)^2 + x_2^2 > 0
$$

---

## Exercise 8: The Hat Matrix

In linear regression with design matrix $\mathbf{X}$, the hat matrix is defined as:

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

**(a)** Prove that $\mathbf{H}$ is idempotent and symmetric.

**(b)** Prove that $\mathbf{I} - \mathbf{H}$ is also idempotent and symmetric.

**(c)** Show that $\mathbf{H}\mathbf{X} = \mathbf{X}$.

### Solution

**Part (a):**

Idempotent:

$$
\mathbf{H}^2 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
$$

Symmetric:

$$
\mathbf{H}^T = (\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)^T = \mathbf{X}((\mathbf{X}^T\mathbf{X})^{-1})^T \mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
$$

where we used the fact that $(\mathbf{X}^T\mathbf{X})^{-1}$ is symmetric because $\mathbf{X}^T\mathbf{X}$ is symmetric. $\square$

**Part (b):** Idempotency follows from Exercise 6(a). Symmetry: $(\mathbf{I} - \mathbf{H})^T = \mathbf{I}^T - \mathbf{H}^T = \mathbf{I} - \mathbf{H}$. $\square$

**Part (c):**

$$
\mathbf{H}\mathbf{X} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X} = \mathbf{X}\mathbf{I} = \mathbf{X}
$$

$\square$

---

## Exercise 9: Quadratic Forms and the Chi-Squared Distribution

Let $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ be a standard multivariate normal vector and let $\mathbf{P}$ be an $n \times n$ symmetric idempotent matrix with $\text{rank}(\mathbf{P}) = r$.

**(a)** State (without proof) the distribution of $\mathbf{Z}^T \mathbf{P} \mathbf{Z}$.

**(b)** In simple linear regression with $n = 20$ observations, the residual maker matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ has rank $n - 2 = 18$. What is the distribution of $\frac{\mathbf{e}^T \mathbf{e}}{\sigma^2}$ where $\mathbf{e} = \mathbf{M}\mathbf{y}$ is the residual vector?

### Solution

**Part (a):** By the theorem on quadratic forms of normal vectors:

$$
\mathbf{Z}^T \mathbf{P} \mathbf{Z} \sim \chi^2(r)
$$

The degrees of freedom equal the rank of $\mathbf{P}$, which equals $\text{tr}(\mathbf{P})$ by Exercise 6(c).

**Part (b):** Under the regression model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$, the residual vector is $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\boldsymbol{\varepsilon}$ (since $\mathbf{M}\mathbf{X} = \mathbf{0}$). Then:

$$
\frac{\mathbf{e}^T\mathbf{e}}{\sigma^2} = \frac{\boldsymbol{\varepsilon}^T \mathbf{M} \boldsymbol{\varepsilon}}{\sigma^2} \sim \chi^2(n - 2) = \chi^2(18)
$$

---

## Exercise 10: NumPy Verification

Use NumPy to verify your answers from Exercise 5.

### Solution

```python
import numpy as np

# Design matrix and response
X = np.array([[1, 2],
              [1, 4],
              [1, 6]])
y = np.array([5, 9, 13])

# Part (a): X^T X and X^T y
XtX = X.T @ X
Xty = X.T @ y
print("X^T X =")
print(XtX)
print("\nX^T y =", Xty)

# Part (b): Solve normal equations
beta_hat = np.linalg.solve(XtX, Xty)
print("\nbeta_hat =", beta_hat)

# Verify: fitted values
y_hat = X @ beta_hat
print("Fitted values:", y_hat)
print("Residuals:", y - y_hat)
```

Expected output:

```
X^T X =
[[ 3 12]
 [12 56]]

X^T y = [ 27 124]

beta_hat = [1. 2.]
Fitted values: [ 5.  9. 13.]
Residuals: [0. 0. 0.]
```
