# Cross-Entropy Loss


## Definition

For $n$ observations with $C$ classes, the **categorical cross-entropy
loss** is

$$
J = -\sum_{i=0}^{n-1}\sum_{c=0}^{C-1} y_{ic}\,\log\hat{y}_{ic}
$$

where $\mathbf{Y}$ is the $n\times C$ one-hot label matrix and
$\hat{\mathbf{Y}}$ is the $n\times C$ matrix of predicted probabilities
from the softmax.

## Derivation of the Gradient dJ/dZ^o
This gradient is the starting point of backpropagation through the
softmax layer and has a beautifully simple form.

### Step 1 — Rewrite the Loss

Since $\hat{y}_{ic} = e^{z_{ic}^o}\big/\sum_{c'}e^{z_{ic'}^o}$ :

$$
J = -\sum_i\sum_c y_{ic}\,z_{ic}^o

    + \sum_i\sum_c y_{ic}\,\log\sum_{c'}e^{z_{ic'}^o}
$$

Using $\sum_c y_{ic}=1$ (one-hot):

$$
J = -\sum_i\sum_c y_{ic}\,z_{ic}^o

    + \sum_i\log\sum_{c'}e^{z_{ic'}^o}
$$

### Step 2 — Differentiate

$$
\frac{\partial J}{\partial z_{ic}^o}
= -y_{ic} + \frac{e^{z_{ic}^o}}{\sum_{c'}e^{z_{ic'}^o}}
= \hat{y}_{ic} - y_{ic}
$$

### Matrix Form

$$
\frac{\partial J}{\partial \mathbf{Z}^o}
= \hat{\mathbf{Y}} - \mathbf{Y}
$$

This is the same "prediction minus target" residual that appears in
binary logistic regression — the softmax + cross-entropy combination
produces a clean gradient regardless of the number of classes.

## Relationship to KL Divergence

The cross-entropy decomposes as

$$
H(\mathbf{y}_i,\hat{\mathbf{y}}_i)
= H(\mathbf{y}_i) + D_{\mathrm{KL}}(\mathbf{y}_i\|\hat{\mathbf{y}}_i)
$$

For one-hot labels $H(\mathbf{y}_i)=0$, so minimizing cross-entropy is
equivalent to minimizing the KL divergence between the true and
predicted distributions.

## Numerical Stability

In practice the loss is computed from logits $\mathbf{z}$ directly using
the **log-sum-exp** trick:

$$
\log\hat{y}_{ic}
= z_{ic} - \log\sum_{c'}\exp(z_{ic'})
= z_{ic} - \Bigl(m_i + \log\sum_{c'}\exp(z_{ic'}-m_i)\Bigr)
$$

where $m_i=\max_c z_{ic}$.  This avoids both overflow and loss of
precision in the logarithm.  PyTorch's `nn.CrossEntropyLoss` and
TensorFlow's `tf.nn.softmax_cross_entropy_with_logits` implement this
automatically.

## Exercises

**Exercise 1.**
Cross-Entropy Loss

Consider a single training example with true class label $y = 2$ (out of $C = 4$ classes) and predicted probability vector $\hat{\mathbf{p}} = (0.1, 0.6, 0.2, 0.1)^\top$.

**(a)** Write the one-hot encoding $\mathbf{y}$ for this example.

**(b)** Compute the cross-entropy loss:

$$
L = -\sum_{k=1}^C y_k \log \hat{p}_k
$$

**(c)** Suppose another model predicts $\hat{\mathbf{p}}' = (0.05, 0.85, 0.05, 0.05)^\top$. Compute its cross-entropy loss and explain which model is better.

**(d)** What is the minimum possible cross-entropy loss for a correctly classified example? When is it achieved?

??? success "Solution to Exercise 1"

    **(a)** The one-hot encoding for class 2 (using 1-based indexing) is:

    $$
    \mathbf{y} = (0, 1, 0, 0)^\top
    $$

    **(b)** Since only $y_2 = 1$, the sum reduces to a single term:

    $$
    L = -\log \hat{p}_2 = -\log(0.6) \approx 0.511
    $$

    **(c)** For the second model:

    $$
    L' = -\log(0.85) \approx 0.163
    $$

    Since $L' < L$, the second model is better — it assigns higher probability to the correct class. Cross-entropy penalizes low confidence in the true class. The closer $\hat{p}_y$ is to 1, the lower the loss.

    **(d)** The minimum cross-entropy loss is 0, achieved when $\hat{p}_y = 1$ (i.e., the model assigns probability 1 to the correct class). Since $-\log(1) = 0$, a perfectly confident and correct prediction incurs zero loss.

---

**Exercise 2.**
Cross-Entropy Gradient

Consider softmax regression with $C$ classes. The predicted probability for class $k$ is $\hat{p}_k = \text{softmax}(\mathbf{z})_k$ where $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

**(a)** Show that the derivative of the softmax function satisfies:

$$
\frac{\partial \hat{p}_k}{\partial z_j} = \hat{p}_k(\delta_{kj} - \hat{p}_j)
$$

where $\delta_{kj}$ is the Kronecker delta.

**(b)** Using the result from (a), derive the gradient of the cross-entropy loss with respect to the logits:

$$
\frac{\partial L}{\partial z_j} = \hat{p}_j - y_j
$$

**(c)** Interpret this gradient. What happens when the model is very confident and correct? What about when it is confident and wrong?

??? success "Solution to Exercise 2"

    **(a)** The softmax function is $\hat{p}_k = e^{z_k} / S$ where $S = \sum_m e^{z_m}$.

    **Case $k = j$:** Using the quotient rule:

    $$
    \frac{\partial \hat{p}_k}{\partial z_k} = \frac{e^{z_k} \cdot S - e^{z_k} \cdot e^{z_k}}{S^2} = \frac{e^{z_k}}{S} - \left(\frac{e^{z_k}}{S}\right)^2 = \hat{p}_k - \hat{p}_k^2 = \hat{p}_k(1 - \hat{p}_k)
    $$

    **Case $k \ne j$:**

    $$
    \frac{\partial \hat{p}_k}{\partial z_j} = \frac{0 - e^{z_k} \cdot e^{z_j}}{S^2} = -\hat{p}_k \hat{p}_j
    $$

    Both cases are unified by $\frac{\partial \hat{p}_k}{\partial z_j} = \hat{p}_k(\delta_{kj} - \hat{p}_j)$.

    **(b)** The cross-entropy loss is $L = -\sum_k y_k \log \hat{p}_k$. By the chain rule:

    $$
    \frac{\partial L}{\partial z_j} = -\sum_k y_k \frac{1}{\hat{p}_k} \frac{\partial \hat{p}_k}{\partial z_j} = -\sum_k y_k \frac{1}{\hat{p}_k} \hat{p}_k(\delta_{kj} - \hat{p}_j)
    $$

    $$
    = -\sum_k y_k (\delta_{kj} - \hat{p}_j) = -y_j + \hat{p}_j \sum_k y_k
    $$

    Since $\mathbf{y}$ is one-hot, $\sum_k y_k = 1$, giving:

    $$
    \frac{\partial L}{\partial z_j} = \hat{p}_j - y_j
    $$

    In vector form: $\nabla_{\mathbf{z}} L = \hat{\mathbf{p}} - \mathbf{y}$.

    **(c)** The gradient $\hat{p}_j - y_j$ has a clean interpretation:

    - **Correct and confident** ($y_j = 1$, $\hat{p}_j \approx 1$): gradient $\approx 0$. The model is already correct, so little update is needed.
    - **Correct but uncertain** ($y_j = 1$, $\hat{p}_j \approx 0.3$): gradient $\approx -0.7$. The negative gradient pushes $z_j$ upward to increase $\hat{p}_j$.
    - **Wrong and confident** ($y_j = 0$, $\hat{p}_j \approx 0.9$): gradient $\approx 0.9$. The large positive gradient pushes $z_j$ downward to decrease $\hat{p}_j$.

    This "residual" form $(\hat{\mathbf{p}} - \mathbf{y})$ parallels the gradient of squared error in linear regression $(\hat{\mathbf{y}} - \mathbf{y})$, making gradient descent updates intuitive.
