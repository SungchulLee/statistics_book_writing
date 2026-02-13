# Maximum Likelihood Estimation

## Gradient of the Cross-Entropy Loss

The cross-entropy loss is

$$
\ell = -\sum_{i=1}^{n}\bigl[y^{(i)}\log\sigma^{(i)} + (1-y^{(i)})\log(1-\sigma^{(i)})\bigr]
$$

Using the sigmoid derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$, the
gradient with respect to $\boldsymbol{\theta}$ simplifies to a clean
matrix expression.

### Element-wise Derivation

$$
\frac{\partial\ell}{\partial\boldsymbol{\theta}}
= -\sum_{i=1}^{n}\left[
  \frac{y^{(i)}}{\sigma^{(i)}}\,\sigma^{(i)}(1-\sigma^{(i)})\,A[i,:]^T
  - \frac{1-y^{(i)}}{1-\sigma^{(i)}}\,\sigma^{(i)}(1-\sigma^{(i)})\,A[i,:]^T
\right]
$$

The sigmoid terms cancel, leaving

$$
\nabla\ell = \sum_{i=1}^{n}\bigl(\sigma^{(i)}-y^{(i)}\bigr)\,A[i,:]^T
$$

### Matrix Form

Writing the residuals as a column vector
$\boldsymbol{\sigma}-\mathbf{y}$ and stacking the design-matrix rows:

$$
\nabla\ell = A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

This is the same form as the gradient for linear regression with squared
loss — except the residual $\hat{\mathbf{y}}-\mathbf{y}$ is replaced by
$\boldsymbol{\sigma}-\mathbf{y}$.

## Hessian

Differentiating the gradient a second time:

$$
\nabla^2\ell
= \sum_{i=1}^{n}\sigma^{(i)}(1-\sigma^{(i)})\;A[i,:]^T\,A[i,:]
= A^T B\, A
$$

where $B$ is the $n\times n$ diagonal matrix

$$
B = \operatorname{diag}\!\bigl(\sigma^{(1)}(1-\sigma^{(1)}),\;\ldots,\;\sigma^{(n)}(1-\sigma^{(n)})\bigr)
$$

Since every diagonal entry of $B$ satisfies
$0<\sigma(1-\sigma)\le\tfrac14$, the Hessian is **positive
semi-definite** for any $\boldsymbol{\theta}$, confirming that the
cross-entropy loss is convex.

## Gradient Descent

The first-order update is

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\,\nabla\ell
= \boldsymbol{\theta} - \alpha\,A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

where $\alpha$ is the learning rate.

## Newton's Method and IRLS

The Newton (second-order) update uses the Hessian:

$$
\boldsymbol{\theta}
= \boldsymbol{\theta}_0 - (A^TBA)^{-1}\,A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

This can be rewritten as a **weighted least-squares** normal equation.
Define the *working response*

$$
\mathbf{z} = A\boldsymbol{\theta}_0 - B^{-1}(\boldsymbol{\sigma}-\mathbf{y})
$$

Then the update becomes

$$
\boldsymbol{\theta} = (A^TBA)^{-1}\,A^TB\,\mathbf{z}
$$

This is exactly the solution to the weighted least-squares problem
$\min_{\boldsymbol{\theta}}\|B^{1/2}(\mathbf{z}-A\boldsymbol{\theta})\|^2$.

### Iteratively Reweighted Least Squares (IRLS)

Because the weight matrix $B$ depends on the current parameter vector
$\boldsymbol{\theta}$ (through $\boldsymbol{\sigma}$), we must apply the
normal equations iteratively, recomputing $B$ and $\mathbf{z}$ at each
step. This procedure is known as **iteratively reweighted least squares
(IRLS)** (Rubin, 1983).

!!! info "IRLS Algorithm"
    1. Initialize $\boldsymbol{\theta}_0$.
    2. Compute $\boldsymbol{\sigma} = \sigma(A\boldsymbol{\theta}_0)$.
    3. Form $B = \operatorname{diag}(\sigma^{(i)}(1-\sigma^{(i)}))$.
    4. Compute working response $\mathbf{z} = A\boldsymbol{\theta}_0 - B^{-1}(\boldsymbol{\sigma}-\mathbf{y})$.
    5. Solve $\boldsymbol{\theta} = (A^TBA)^{-1}A^TB\mathbf{z}$.
    6. Set $\boldsymbol{\theta}_0 \leftarrow \boldsymbol{\theta}$ and repeat until convergence.

IRLS typically converges in a small number of iterations and is the
algorithm behind many classical logistic regression solvers.

## Implementation: Logistic Regression with Gradient Descent

The following NumPy implementation trains logistic regression from
scratch using first-order gradient descent.

### Data Loading

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(seed=1):
    url = ('https://raw.githubusercontent.com/codebasics/py/'
           'master/ML/7_logistic_reg/insurance_data.csv')
    df = pd.read_csv(url)
    x = df[['age']].values.reshape((-1, 1))
    y = df.bought_insurance.values.reshape((-1,))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.5, random_state=seed)
    return x_train, x_test, y_train, y_test
```

### Model Class

```python
import numpy as np

class LogisticRegression:
    def __init__(self, x, y, lr=2e-4, epochs=100_000, theta=None):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.theta = (theta if theta is not None
                      else np.random.normal(size=(x.shape[1] + 1, 1)))

    @staticmethod
    def design_matrix(x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, x):
        A = self.design_matrix(x)
        z = A @ self.theta
        return self.sigmoid(z).reshape((-1,))

    def predict(self, x):
        p = self.predict_proba(x)
        return (p > 0.5).astype(float)

    def loss(self):
        p = self.predict_proba(self.x)
        eps = 1e-6
        return -np.mean(
            self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps))

    def gradient(self):
        A = self.design_matrix(self.x)
        p = self.predict_proba(self.x).reshape((-1, 1))
        y = self.y.reshape((-1, 1))
        return A.T @ (p - y)

    def train(self):
        for _ in range(self.epochs):
            self.theta -= self.lr * self.gradient()
```

### Training

```python
x_train, x_test, y_train, y_test = load_data()

model = LogisticRegression(x_train, y_train)
model.train()

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)
```

## Implementation: Logistic Regression with Scikit-Learn

For comparison, the same task using `sklearn`:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
```

Scikit-learn's `LogisticRegression` uses L-BFGS (a quasi-Newton method)
by default, which approximates the Hessian without forming or inverting
it explicitly. For small datasets the `solver='newton-cg'` option gives
exact Newton steps, equivalent to IRLS.
