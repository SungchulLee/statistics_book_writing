# Softmax Regression Implementation

## Overview

This page provides a hands-on, from-scratch implementation of softmax regression in NumPy. We build each component of the pipeline -- the softmax function, the cross-entropy loss, the gradient computation, and the training loop -- step by step. The goal is to connect the mathematical formulas from earlier sections to working code and to verify correctness against scikit-learn.

---

## The Softmax Function

The softmax function maps a vector of real-valued logits $\mathbf{z} = (z_1, \ldots, z_C)^\top$ to a probability distribution over $C$ classes:

$$
\hat{p}_k = \operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}, \qquad k = 1, \ldots, C
$$

A naive implementation would compute `np.exp(z)` directly, but this overflows for large logits. The **log-sum-exp trick** subtracts $\max_k z_k$ before exponentiation, exploiting the shift invariance of softmax.

```python
import numpy as np

def softmax(z):
    """Numerically stable softmax.

    Parameters
    ----------
    z : ndarray, shape (n, C)
        Logit matrix — one row per observation.

    Returns
    -------
    ndarray, shape (n, C)
        Probability matrix with rows summing to 1.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

We can verify the implementation on the logit vector $\mathbf{z} = (2, 1, -1)^\top$:

```python
z = np.array([[2.0, 1.0, -1.0]])
print(softmax(z))
# [[0.7054  0.2595  0.0351]]
```

---

## Cross-Entropy Loss

Given $n$ observations with one-hot encoded labels $\mathbf{Y} \in \{0,1\}^{n \times C}$ and predicted probabilities $\hat{\mathbf{Y}} \in (0,1)^{n \times C}$, the cross-entropy loss is

$$
J = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}
$$

Because each row of $\mathbf{Y}$ is one-hot, only the term corresponding to the true class survives. A small constant $\varepsilon$ prevents $\log(0)$.

```python
def cross_entropy_loss(Y, Y_hat, eps=1e-12):
    """Average cross-entropy loss.

    Parameters
    ----------
    Y : ndarray, shape (n, C)
        One-hot label matrix.
    Y_hat : ndarray, shape (n, C)
        Predicted probability matrix.

    Returns
    -------
    float
        Scalar loss value.
    """
    n = Y.shape[0]
    return -np.sum(Y * np.log(Y_hat + eps)) / n
```

---

## Gradient of the Loss with Respect to the Logits

One of the most elegant results in softmax regression is that the gradient of the cross-entropy loss with respect to the logit matrix $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}^\top$ simplifies to

$$
\frac{\partial J}{\partial \mathbf{Z}} = \frac{1}{n}(\hat{\mathbf{Y}} - \mathbf{Y})
$$

This means the gradient at each observation is the difference between the predicted probabilities and the one-hot labels. Applying the chain rule to the weight matrix $\mathbf{W} \in \mathbb{R}^{d \times C}$ and bias $\mathbf{b} \in \mathbb{R}^C$:

$$
\frac{\partial J}{\partial \mathbf{W}} = \frac{1}{n} \mathbf{X}^\top (\hat{\mathbf{Y}} - \mathbf{Y}), \qquad
\frac{\partial J}{\partial \mathbf{b}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{\mathbf{y}}_i - \mathbf{y}_i)
$$

```python
def compute_gradients(X, Y, Y_hat):
    """Compute gradients of cross-entropy w.r.t. W and b.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    Y : ndarray, shape (n, C)
    Y_hat : ndarray, shape (n, C)

    Returns
    -------
    dW : ndarray, shape (d, C)
    db : ndarray, shape (C,)
    """
    n = X.shape[0]
    error = Y_hat - Y                   # (n, C)
    dW = X.T @ error / n                # (d, C)
    db = np.mean(error, axis=0)         # (C,)
    return dW, db
```

---

## One-Hot Encoding

The training labels $y_i \in \{0, 1, \ldots, C-1\}$ must be converted to one-hot vectors. For label $y_i = k$ the one-hot vector has a 1 in position $k$ and zeros elsewhere.

```python
def one_hot(y, C):
    """Convert integer labels to a one-hot matrix.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer class labels in {0, ..., C-1}.
    C : int
        Number of classes.

    Returns
    -------
    ndarray, shape (n, C)
    """
    n = y.shape[0]
    Y = np.zeros((n, C))
    Y[np.arange(n), y] = 1.0
    return Y
```

---

## Putting It All Together -- Training Loop

We now combine the components into a complete gradient descent training loop.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load and prepare data ---
iris = load_iris()
X, y = iris.data, iris.target
C = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y_train = one_hot(y_train, C)
Y_test = one_hot(y_test, C)

# --- Initialize parameters ---
d = X_train.shape[1]
np.random.seed(0)
W = np.random.randn(d, C) * 0.01
b = np.zeros(C)

# --- Training ---
lr = 0.5
epochs = 200
loss_history = []

for epoch in range(epochs):
    Z = X_train @ W + b                # logits: (n, C)
    Y_hat = softmax(Z)                  # probabilities: (n, C)
    loss = cross_entropy_loss(Y_train, Y_hat)
    loss_history.append(loss)
    dW, db = compute_gradients(X_train, Y_train, Y_hat)
    W -= lr * dW
    b -= lr * db

print(f"Final training loss: {loss_history[-1]:.4f}")
```

---

## Evaluation

After training, we compute predictions on the test set and report accuracy.

```python
Z_test = X_test @ W + b
Y_hat_test = softmax(Z_test)
y_pred = np.argmax(Y_hat_test, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

A well-tuned softmax regression on the Iris dataset typically achieves around 96--98% accuracy.

---

## Verification Against scikit-learn

Comparing our from-scratch implementation with scikit-learn's `LogisticRegression` (which uses softmax for multiclass problems via `multi_class='multinomial'`) provides a useful sanity check.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                         max_iter=1000)
clf.fit(X_train, y_train)
print(f"scikit-learn accuracy: {clf.score(X_test, y_test):.4f}")
```

Both implementations should yield similar test accuracy, confirming the correctness of our gradient derivation and training loop.

---

## Interpretation

The softmax regression implementation reveals several important points:

1. **Numerical stability matters.** Without the log-sum-exp trick, large logits cause overflow and produce `nan` values. This is not a theoretical nicety but a practical necessity.
2. **The gradient has a clean form.** The fact that $\partial J / \partial \mathbf{Z} = (\hat{\mathbf{Y}} - \mathbf{Y}) / n$ makes implementation straightforward and efficient. There is no need to differentiate through the softmax separately.
3. **Feature scaling is critical.** Standardizing the features ensures that all dimensions contribute equally to the logits and that the learning rate works uniformly across features.
4. **Softmax regression is a linear classifier.** Despite the nonlinear softmax transformation, the decision boundaries are hyperplanes in feature space. The model cannot capture nonlinear relationships without explicit feature engineering.

---

## Exercises

**Exercise 1.**
Starting from the cross-entropy loss $J = -\frac{1}{n}\sum_i \sum_c y_{ic}\log\hat{y}_{ic}$ and the softmax definition $\hat{y}_{ic} = e^{z_{ic}} / \sum_{j} e^{z_{ij}}$, derive the gradient $\partial J / \partial z_{ik} = (\hat{y}_{ik} - y_{ik})/n$ for a single observation $i$.

??? success "Solution to Exercise 1"
    Fix observation $i$ and drop the $1/n$ factor for clarity. The loss for this observation is

    $$
    \ell_i = -\sum_c y_{ic} \log \hat{y}_{ic}
    $$

    The softmax Jacobian gives $\partial \hat{y}_{ic} / \partial z_{ik} = \hat{y}_{ic}(\delta_{ck} - \hat{y}_{ik})$ where $\delta_{ck}$ is the Kronecker delta. Applying the chain rule:

    $$
    \frac{\partial \ell_i}{\partial z_{ik}} = -\sum_c y_{ic} \frac{1}{\hat{y}_{ic}} \cdot \hat{y}_{ic}(\delta_{ck} - \hat{y}_{ik})
    = -\sum_c y_{ic}(\delta_{ck} - \hat{y}_{ik})
    $$

    Expanding:

    $$
    = -y_{ik} + \hat{y}_{ik}\sum_c y_{ic} = -y_{ik} + \hat{y}_{ik} \cdot 1 = \hat{y}_{ik} - y_{ik}
    $$

    where we used $\sum_c y_{ic} = 1$ (one-hot). Including the $1/n$ factor, $\partial J / \partial z_{ik} = (\hat{y}_{ik} - y_{ik})/n$. $\square$

---

**Exercise 2.**
Implement an $L_2$-regularized version of the training loop. Add the penalty term $\frac{\lambda}{2}\|\mathbf{W}\|_F^2$ to the loss and update the gradient accordingly. Train on the Iris dataset with $\lambda = 0.1$ and compare the test accuracy to the unregularized version.

??? success "Solution to Exercise 2"
    The regularized loss is

    $$
    J_{\text{reg}} = J + \frac{\lambda}{2}\|\mathbf{W}\|_F^2
    $$

    The gradient with respect to $\mathbf{W}$ picks up an additional term:

    $$
    \frac{\partial J_{\text{reg}}}{\partial \mathbf{W}} = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{Y}} - \mathbf{Y}) + \lambda \mathbf{W}
    $$

    The bias gradient is unchanged (we do not regularize the bias).

    ```python
    lam = 0.1
    W_reg = np.random.randn(d, C) * 0.01
    b_reg = np.zeros(C)

    for epoch in range(200):
        Z = X_train @ W_reg + b_reg
        Y_hat = softmax(Z)
        loss = cross_entropy_loss(Y_train, Y_hat) + 0.5 * lam * np.sum(W_reg ** 2)
        dW, db = compute_gradients(X_train, Y_train, Y_hat)
        dW += lam * W_reg       # regularization gradient
        W_reg -= lr * dW
        b_reg -= lr * db

    y_pred_reg = np.argmax(softmax(X_test @ W_reg + b_reg), axis=1)
    print(f"Regularized accuracy: {np.mean(y_pred_reg == y_test):.4f}")
    ```

    On Iris the regularized and unregularized versions typically achieve comparable accuracy (~96--98%) because the dataset is small and well-separated. The benefit of regularization becomes more pronounced with high-dimensional or noisy features.

---

**Exercise 3.**
The softmax function is shift-invariant: $\operatorname{softmax}(\mathbf{z} + c\mathbf{1}) = \operatorname{softmax}(\mathbf{z})$. Write a NumPy experiment that computes the softmax of $\mathbf{z} = (1000, 1001, 999)^\top$ both with and without the max-subtraction trick, and show that the naive version produces `nan` while the stable version does not.

??? success "Solution to Exercise 3"
    ```python
    z = np.array([[1000.0, 1001.0, 999.0]])

    # Naive softmax (no shift)
    exp_z_naive = np.exp(z)
    softmax_naive = exp_z_naive / np.sum(exp_z_naive, axis=1, keepdims=True)
    print("Naive:", softmax_naive)
    # Output: [[nan nan nan]]  (because np.exp(1001) = inf)

    # Stable softmax (max subtraction)
    print("Stable:", softmax(z))
    # Output: [[0.2447  0.6652  0.0900]]
    ```

    The naive version overflows because $e^{1001}$ exceeds the maximum representable float64 value ($\approx 1.8 \times 10^{308}$). The stable version subtracts $\max(\mathbf{z}) = 1001$, computing $e^{-1}, e^{0}, e^{-2}$ instead, which are all safe. The mathematical result is identical by shift invariance.

---

**Exercise 4.**
Prove that the decision boundary between class $j$ and class $k$ in softmax regression is a hyperplane. That is, the set $\{\mathbf{x} : \hat{p}_j(\mathbf{x}) = \hat{p}_k(\mathbf{x})\}$ is a $(d-1)$-dimensional affine subspace of $\mathbb{R}^d$.

??? success "Solution to Exercise 4"
    The predicted probabilities are $\hat{p}_c(\mathbf{x}) = \operatorname{softmax}(\mathbf{W}\mathbf{x} + \mathbf{b})_c$. Setting $\hat{p}_j = \hat{p}_k$:

    $$
    \frac{e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}{\sum_m e^{\mathbf{w}_m^\top \mathbf{x} + b_m}}
    = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_m e^{\mathbf{w}_m^\top \mathbf{x} + b_m}}
    $$

    The denominators cancel, giving $e^{\mathbf{w}_j^\top \mathbf{x} + b_j} = e^{\mathbf{w}_k^\top \mathbf{x} + b_k}$. Taking logarithms:

    $$
    \mathbf{w}_j^\top \mathbf{x} + b_j = \mathbf{w}_k^\top \mathbf{x} + b_k
    $$

    Rearranging:

    $$
    (\mathbf{w}_j - \mathbf{w}_k)^\top \mathbf{x} + (b_j - b_k) = 0
    $$

    This is the equation of a hyperplane with normal vector $\mathbf{w}_j - \mathbf{w}_k$ and offset $b_j - b_k$. As long as $\mathbf{w}_j \neq \mathbf{w}_k$, this defines a $(d-1)$-dimensional affine subspace. $\square$

---

**Exercise 5.**
Implement a learning rate schedule that multiplies the learning rate by a decay factor $\gamma = 0.95$ every 50 epochs. Train the softmax model on the Iris dataset for 500 epochs with an initial learning rate of $\eta_0 = 1.0$. Plot the training loss curve and compare it with the constant-learning-rate version.

??? success "Solution to Exercise 5"
    ```python
    import matplotlib.pyplot as plt

    W_sched = np.random.randn(d, C) * 0.01
    b_sched = np.zeros(C)
    W_const = W_sched.copy()
    b_const = b_sched.copy()

    eta0 = 1.0
    gamma = 0.95
    total_epochs = 500
    loss_sched, loss_const = [], []

    for epoch in range(total_epochs):
        # Scheduled learning rate
        lr_t = eta0 * (gamma ** (epoch // 50))

        # --- Scheduled ---
        Z = X_train @ W_sched + b_sched
        Yh = softmax(Z)
        loss_sched.append(cross_entropy_loss(Y_train, Yh))
        dW, db = compute_gradients(X_train, Y_train, Yh)
        W_sched -= lr_t * dW
        b_sched -= lr_t * db

        # --- Constant ---
        Z = X_train @ W_const + b_const
        Yh = softmax(Z)
        loss_const.append(cross_entropy_loss(Y_train, Yh))
        dW, db = compute_gradients(X_train, Y_train, Yh)
        W_const -= eta0 * dW
        b_const -= eta0 * db

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_sched, label="Scheduled LR")
    ax.plot(loss_const, label="Constant LR", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss: Scheduled vs Constant Learning Rate")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    The scheduled version typically shows smoother convergence in later epochs. With a constant high learning rate, the loss may oscillate around the minimum rather than settling. The decay schedule reduces the step size over time, allowing the optimizer to take finer steps near the optimum. On a well-behaved dataset like Iris, both reach similar final accuracy, but the scheduled version produces a more stable loss curve.
