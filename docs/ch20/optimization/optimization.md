# Gradient-Based Optimization

## Full Gradient Derivation (Two-Layer Network)

We derive the gradients for the two-layer model (hidden layer with
logistic activation, output layer with softmax) applied to MNIST.

### Output-Layer Gradients

Starting from $\partial J/\partial \mathbf{Z}^o = \hat{\mathbf{Y}}-\mathbf{Y}$,
the chain rule gives:

$$
\underset{100 \times 10}{\frac{\partial J}{\partial \mathbf{W}^o}}
= \underset{100 \times n}{\mathbf{H}^T}\;
  \bigl(\underset{n \times 10}{\hat{\mathbf{Y}}-\mathbf{Y}}\bigr)
$$

$$
\underset{1 \times 10}{\frac{\partial J}{\partial \mathbf{b}^o}}
= \underset{1 \times n}{\mathbf{1}}\;
  \bigl(\underset{n \times 10}{\hat{\mathbf{Y}}-\mathbf{Y}}\bigr)
$$

??? note "Element-wise proof for $\partial J/\partial \mathbf{W}^o$"
    Since $z_{ic}^o = \sum_\alpha h_{i\alpha}\,w_{\alpha c}^o + b_{1c}^o$:

    $$
    \frac{\partial J}{\partial w_{\alpha c}^o}
    = \sum_i \frac{\partial J}{\partial z_{ic}^o}\,\frac{\partial z_{ic}^o}{\partial w_{\alpha c}^o}
    = \sum_i (\hat{y}_{ic}-y_{ic})\,h_{i\alpha}
    = \bigl[\mathbf{H}^T(\hat{\mathbf{Y}}-\mathbf{Y})\bigr]_{\alpha c}
    $$

??? note "Element-wise proof for $\partial J/\partial \mathbf{b}^o$"
    $$
    \frac{\partial J}{\partial b_{1c}^o}
    = \sum_i (\hat{y}_{ic}-y_{ic})
    = \bigl[\mathbf{1}(\hat{\mathbf{Y}}-\mathbf{Y})\bigr]_{1c}
    $$

### Backpropagation to the Hidden Layer

$$
\underset{n \times 100}{\frac{\partial J}{\partial \mathbf{H}}}
= \bigl(\hat{\mathbf{Y}}-\mathbf{Y}\bigr)\;\mathbf{W}^{oT}
$$

??? note "Element-wise proof"
    $$
    \frac{\partial J}{\partial h_{i\alpha}}
    = \sum_c (\hat{y}_{ic}-y_{ic})\,w_{\alpha c}^o
    = \sum_c (\hat{y}_{ic}-y_{ic})\,w_{c\alpha}^{oT}
    = \bigl[(\hat{\mathbf{Y}}-\mathbf{Y})\,\mathbf{W}^{oT}\bigr]_{i\alpha}
    $$

Passing through the logistic activation
$\mathbf{H}=\operatorname{logistic}(\mathbf{Z}^h)$:

$$
\underset{n \times 100}{\frac{\partial J}{\partial \mathbf{Z}^h}}
= \mathbf{H}\odot(1-\mathbf{H})\odot
  \bigl[(\hat{\mathbf{Y}}-\mathbf{Y})\,\mathbf{W}^{oT}\bigr]
$$

where $\odot$ denotes element-wise (Hadamard) multiplication.

### Hidden-Layer Gradients

$$
\underset{784 \times 100}{\frac{\partial J}{\partial \mathbf{W}^h}}
= \mathbf{X}^T\;
  \bigl[\mathbf{H}\odot(1-\mathbf{H})\odot
        (\hat{\mathbf{Y}}-\mathbf{Y})\,\mathbf{W}^{oT}\bigr]
$$

$$
\underset{1 \times 100}{\frac{\partial J}{\partial \mathbf{b}^h}}
= \mathbf{1}\;
  \bigl[\mathbf{H}\odot(1-\mathbf{H})\odot
        (\hat{\mathbf{Y}}-\mathbf{Y})\,\mathbf{W}^{oT}\bigr]
$$

## Implementation: NumPy from Scratch

### Model Functions

```python
import numpy as np

logistic = lambda z: 1 / (1 + np.exp(-z))
softmax  = lambda z: np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def initialize_weights():
    w_h = np.random.randn(784, 100)
    b_h = np.random.randn(1, 100)
    w_o = np.random.randn(100, 10)
    b_o = np.random.randn(1, 10)
    return w_h, b_h, w_o, b_o

def feed_forward(x, y, y_cls, w_h, b_h, w_o, b_o):
    z_h = x @ w_h + b_h
    h = logistic(z_h)
    z_o = h @ w_o + b_o
    y_hat = softmax(z_o)
    y_hat_cls = np.argmax(y_hat, axis=1)
    loss = -(y * np.log(y_hat)).sum()
    accuracy = (y_cls == y_hat_cls).sum() / y_cls.size
    return h, y_hat, y_hat_cls, loss, accuracy

def back_propagation(x, y, h, y_hat, w_o):
    loss_grad = y_hat - y                           # n × 10
    w_o_grad  = h.T @ loss_grad                     # 100 × 10
    b_o_grad  = np.sum(loss_grad, axis=0, keepdims=True)  # 1 × 10
    h_grad    = h * (1 - h) * (loss_grad @ w_o.T)   # n × 100
    w_h_grad  = x.T @ h_grad                        # 784 × 100
    b_h_grad  = np.sum(h_grad, axis=0, keepdims=True)     # 1 × 100
    return w_h_grad, b_h_grad, w_o_grad, b_o_grad
```

### Training Loop with Mini-Batches

```python
def run_train_loop(x_train, y_train, y_train_cls,
                   w_h, b_h, w_o, b_o,
                   lr=1e-2, epochs=50, batch_size=100):
    loss_trace, accuracy_trace = [], []
    for epoch in range(epochs):
        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        x_epoch = x_train[idx]
        y_epoch = y_train[idx]
        y_cls_epoch = y_train_cls[idx]

        loss_temp, acc_temp = [], []
        for k in range(x_train.shape[0] // batch_size):
            sl = slice(k * batch_size, (k + 1) * batch_size)
            x = x_epoch[sl]
            y = y_epoch[sl]
            y_cls = y_cls_epoch[sl]

            h, y_hat, _, loss, acc = feed_forward(
                x, y, y_cls, w_h, b_h, w_o, b_o)
            grads = back_propagation(x, y, h, y_hat, w_o)
            for para, grad in zip([w_h, b_h, w_o, b_o], grads):
                para -= lr * grad
            loss_temp.append(loss)
            acc_temp.append(acc)

        loss_trace.append(np.mean(loss_temp))
        accuracy_trace.append(np.mean(acc_temp))
        print(f'{epoch+1}/{epochs}  loss {loss_trace[-1]:.1f}  '
              f'acc {accuracy_trace[-1]:.4f}')

    return w_h, b_h, w_o, b_o, loss_trace, accuracy_trace
```

### Data Loading

```python
import tensorflow as tf

def load_data():
    (x_train, y_train_cls), (x_test, y_test_cls) = (
        tf.keras.datasets.mnist.load_data())
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_train = np.eye(10)[y_train_cls].astype(np.float32)
    y_test  = np.eye(10)[y_test_cls].astype(np.float32)
    return (x_train, y_train, y_train_cls.astype(np.int32),
            x_test,  y_test,  y_test_cls.astype(np.int32))
```

## Gradient Descent Illustrated

A simple visualization of gradient descent on $L(x)=x^2$:

```python
import matplotlib.pyplot as plt

lr, x, steps = 0.1, 5.0, 20
xs, losses = [x], [x * x]

for _ in range(steps):
    x = x - lr * 2 * x
    xs.append(x)
    losses.append(x * x)

plt.figure(figsize=(6, 4))
t = np.linspace(-5, 5, 100)
plt.plot(t, t**2, '--k', alpha=0.4)
plt.plot(xs, losses, marker='o')
plt.xlabel('x')
plt.ylabel('Loss')
plt.title('Gradient Descent on L(x) = x²')
plt.grid(True)
plt.show()
```
