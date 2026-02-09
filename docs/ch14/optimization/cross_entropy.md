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

## Derivation of the Gradient $\partial J/\partial \mathbf{Z}^o$

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
