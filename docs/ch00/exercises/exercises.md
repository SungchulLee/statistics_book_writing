# Exercises

These exercises cover the mathematical and computational foundations from Chapter 0. Problems progress from logic and set theory through sequences and limits to linear algebra, matrix theory, and their connections to statistical inference.

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

**Part (c):** This is the negation of the statement "$a_n$ converges to $L$":

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

Left side: $(A \cup B)^c = \{1,2,3,4,5,6,7\}^c = \{8\}$

Right side: $A^c \cap B^c = \{6,7,8\} \cap \{1,2,8\} = \{8\}$

Both sides equal $\{8\}$, confirming De Morgan's law.

---

## Exercise 3: Convergence via the Epsilon-N Definition

Prove directly from the $\varepsilon$-$N$ definition that

$$
\lim_{n \to \infty} \frac{3n + 1}{n + 2} = 3
$$

### Solution

We must show that for every $\varepsilon > 0$, there exists $N$ such that $n > N$ implies $|a_n - 3| < \varepsilon$. Compute the difference:

$$
\left| \frac{3n+1}{n+2} - 3 \right| = \left| \frac{3n+1 - 3(n+2)}{n+2} \right| = \frac{5}{n+2}
$$

We need $\frac{5}{n+2} < \varepsilon$, which gives $n > \frac{5}{\varepsilon} - 2$. Choose $N = \left\lceil \frac{5}{\varepsilon} - 2 \right\rceil$. Then for all $n > N$:

$$
\left| \frac{3n+1}{n+2} - 3 \right| = \frac{5}{n+2} < \varepsilon
$$

$\square$

---

## Exercise 4: Geometric Series

**(a)** Find the sum of the geometric series $\sum_{k=0}^{\infty} r^k$ when $|r| < 1$.

**(b)** Evaluate $\sum_{k=1}^{\infty} \frac{3}{4^k}$.

### Solution

**Part (a):** The partial sums satisfy

$$
S_n = \sum_{k=0}^{n} r^k = \frac{1 - r^{n+1}}{1 - r}
$$

When $|r| < 1$, we have $r^{n+1} \to 0$, so $S_n \to \frac{1}{1 - r}$.

**Part (b):**

$$
\sum_{k=1}^{\infty} \frac{3}{4^k} = 3 \sum_{k=1}^{\infty} \left(\frac{1}{4}\right)^k = 3 \cdot \frac{1/4}{1 - 1/4} = 3 \cdot \frac{1}{3} = 1
$$

---

## Exercise 5: Matrix Multiplication and the Normal Equations

Consider the design matrix for simple linear regression with intercept:

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 5 \\ 9 \\ 13 \end{pmatrix}
$$

**(a)** Compute $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$.

**(b)** Solve the normal equations $\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}$ for $\hat{\boldsymbol{\beta}} = (\hat{\beta}_0, \hat{\beta}_1)^T$.

### Solution

**Part (a):**

$$
\mathbf{X}^T \mathbf{X} = \begin{pmatrix} 3 & 12 \\ 12 & 56 \end{pmatrix}, \qquad \mathbf{X}^T \mathbf{y} = \begin{pmatrix} 27 \\ 124 \end{pmatrix}
$$

**Part (b):** The determinant is $3 \cdot 56 - 12 \cdot 12 = 24$, so

$$
(\mathbf{X}^T \mathbf{X})^{-1} = \frac{1}{24} \begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix}
$$

$$
\hat{\boldsymbol{\beta}} = \frac{1}{24} \begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix} \begin{pmatrix} 27 \\ 124 \end{pmatrix} = \frac{1}{24} \begin{pmatrix} 24 \\ 48 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

The fitted line is $\hat{y} = 1 + 2x$.

---

## Exercise 6: Idempotent Matrices

A matrix $\mathbf{P}$ is **idempotent** if $\mathbf{P}^2 = \mathbf{P}$.

**(a)** Prove that if $\mathbf{P}$ is idempotent, then $\mathbf{I} - \mathbf{P}$ is also idempotent.

**(b)** Prove that the eigenvalues of an idempotent matrix are either 0 or 1.

**(c)** Show that $\operatorname{tr}(\mathbf{P}) = \operatorname{rank}(\mathbf{P})$ for any idempotent matrix $\mathbf{P}$.

### Solution

**Part (a):**

$$
(\mathbf{I} - \mathbf{P})^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P} = \mathbf{I} - \mathbf{P}
$$

$\square$

**Part (b):** Let $\lambda$ be an eigenvalue with eigenvector $\mathbf{v} \neq \mathbf{0}$. Then $\mathbf{P}\mathbf{v} = \lambda \mathbf{v}$, so

$$
\mathbf{P}^2 \mathbf{v} = \lambda^2 \mathbf{v}
$$

Since $\mathbf{P}^2 = \mathbf{P}$, we have $\lambda^2 \mathbf{v} = \lambda \mathbf{v}$, giving $\lambda(\lambda - 1) = 0$. Therefore $\lambda = 0$ or $\lambda = 1$. $\square$

**Part (c):** The trace equals the sum of the eigenvalues. By part (b), every eigenvalue is 0 or 1, and the number of eigenvalues equal to 1 equals the dimension of the column space, which is the rank. Therefore $\operatorname{tr}(\mathbf{P}) = \operatorname{rank}(\mathbf{P})$. $\square$

---

## Exercise 7: Spectral Decomposition

Let $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$.

**(a)** Find the eigenvalues and normalized eigenvectors of $\mathbf{A}$.

**(b)** Write the spectral decomposition $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$.

**(c)** Verify that $\mathbf{A}$ is positive definite.

### Solution

**Part (a):** The characteristic equation is

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0
$$

The eigenvalues are $\lambda_1 = 1$ and $\lambda_2 = 3$. The corresponding normalized eigenvectors are

$$
\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}, \qquad \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

**Part (b):**

$$
\mathbf{A} = \underbrace{\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}}_{\mathbf{Q}} \underbrace{\begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix}}_{\boldsymbol{\Lambda}} \underbrace{\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}}_{\mathbf{Q}^T}
$$

**Part (c):** A real symmetric matrix is positive definite if and only if all its eigenvalues are strictly positive. Since $\lambda_1 = 1 > 0$ and $\lambda_2 = 3 > 0$, the matrix $\mathbf{A}$ is positive definite.

---

## Exercise 8: The Hat Matrix

In linear regression, the hat matrix is $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$.

**(a)** Prove that $\mathbf{H}$ is symmetric and idempotent.

**(b)** Prove that $\mathbf{I} - \mathbf{H}$ is also symmetric and idempotent.

**(c)** Show that $\mathbf{H}\mathbf{X} = \mathbf{X}$.

### Solution

**Part (a):** Symmetry: since $(\mathbf{X}^T\mathbf{X})^{-1}$ is symmetric (as the inverse of a symmetric matrix),

$$
\mathbf{H}^T = \mathbf{X}\bigl((\mathbf{X}^T\mathbf{X})^{-1}\bigr)^T \mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
$$

Idempotency:

$$
\mathbf{H}^2 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
$$

$\square$

**Part (b):** Symmetry follows from $(\mathbf{I} - \mathbf{H})^T = \mathbf{I} - \mathbf{H}^T = \mathbf{I} - \mathbf{H}$. Idempotency follows from Exercise 6(a). $\square$

**Part (c):**

$$
\mathbf{H}\mathbf{X} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X} = \mathbf{X}\mathbf{I} = \mathbf{X}
$$

$\square$

---

## Exercise 9: Quadratic Forms and Chi-Squared

Let $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and let $\mathbf{P}$ be an $n \times n$ symmetric idempotent matrix with $\operatorname{rank}(\mathbf{P}) = r$.

**(a)** State the distribution of $\mathbf{Z}^T \mathbf{P} \mathbf{Z}$.

**(b)** In simple linear regression with $n = 20$ observations, the residual maker matrix is $\mathbf{M} = \mathbf{I} - \mathbf{H}$ with $\operatorname{rank}(\mathbf{M}) = n - 2 = 18$. Under the standard normal-error model, state the distribution of $\mathbf{e}^T\mathbf{e}/\sigma^2$ where $\mathbf{e} = \mathbf{M}\mathbf{y}$.

### Solution

**Part (a):** By the theorem on quadratic forms of standard normal vectors,

$$
\mathbf{Z}^T \mathbf{P} \mathbf{Z} \sim \chi^2(r)
$$

**Part (b):** Under $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$, the residual vector satisfies $\mathbf{e} = \mathbf{M}\boldsymbol{\varepsilon}$ (since $\mathbf{M}\mathbf{X} = \mathbf{0}$). Therefore

$$
\frac{\mathbf{e}^T\mathbf{e}}{\sigma^2} = \frac{\boldsymbol{\varepsilon}^T \mathbf{M} \boldsymbol{\varepsilon}}{\sigma^2} \sim \chi^2(18)
$$

---

## Exercise 10: NumPy Verification

Use NumPy to verify your answers to Exercise 5.

### Solution

```python
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 6]])
y = np.array([5, 9, 13])

XtX = X.T @ X
Xty = X.T @ y
print("X^T X =\n", XtX)
print("X^T y =", Xty)

beta_hat = np.linalg.solve(XtX, Xty)
print("beta_hat =", beta_hat)
print("Fitted values:", X @ beta_hat)
print("Residuals:", y - X @ beta_hat)
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
