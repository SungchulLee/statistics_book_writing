# MNIST Classification

## Overview

This page walks through a complete MNIST handwritten digit classification pipeline using PyTorch. We implement three models of increasing complexity -- a single linear layer (softmax regression), a two-layer feedforward network, and a convolutional neural network (CNN) -- and compare their performance. The goal is to see how model architecture affects accuracy on a real-world 10-class image classification task.

---

## The MNIST Dataset

MNIST consists of 60,000 training and 10,000 test grayscale images of handwritten digits (0--9), each $28 \times 28$ pixels. The task is multiclass classification with $C = 10$ classes. Each pixel value lies in $[0, 1]$ after normalization.

```python
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")
print(f"Image shape:      {train_dataset[0][0].shape}")
```

---

## Visualizing Sample Images

Before modeling, it is essential to inspect the data.

```python
images, labels = next(iter(test_loader))
img_grid = torchvision.utils.make_grid(images[:32], nrow=8, padding=2)

plt.figure(figsize=(8, 4))
plt.imshow(img_grid.permute(1, 2, 0), cmap='gray')
plt.axis('off')
plt.title("Sample MNIST Images")
plt.show()
```

---

## Model 1 -- Softmax Regression (Single Linear Layer)

The simplest approach flattens each $28 \times 28$ image into a 784-dimensional vector and applies a single linear transformation:

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \qquad \hat{\mathbf{p}} = \operatorname{softmax}(\mathbf{z})
$$

where $\mathbf{W} \in \mathbb{R}^{10 \times 784}$ and $\mathbf{b} \in \mathbb{R}^{10}$.

```python
import torch.nn as nn
import torch.optim as optim

class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
```

PyTorch's `nn.CrossEntropyLoss` combines log-softmax and negative log-likelihood in a single, numerically stable operation.

---

## Training Loop

The training loop is shared across all three models.

```python
def train_model(model, train_loader, epochs=5, lr=0.1):
    """Train a model with SGD and cross-entropy loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history
```

Training the softmax regression model:

```python
model_linear = SoftmaxRegression()
loss_linear = train_model(model_linear, train_loader, epochs=5, lr=0.1)
```

---

## Evaluation

```python
def evaluate(model, test_loader):
    """Compute test accuracy and per-class accuracy."""
    model.eval()
    correct = total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for lbl, pred in zip(labels, predicted):
                class_total[lbl] += 1
                if lbl == pred:
                    class_correct[lbl] += 1

    acc = 100 * correct / total
    print(f"Overall accuracy: {acc:.2f}%")
    for c in range(10):
        print(f"  Digit {c}: {100 * class_correct[c] / class_total[c]:.1f}%")
    return acc

acc_linear = evaluate(model_linear, test_loader)
```

**Typical result: approximately 92% test accuracy.**

---

## Model 2 -- Two-Layer Feedforward Network

Adding a hidden layer with ReLU activation allows the model to learn nonlinear feature combinations:

$$
\mathbf{h} = \operatorname{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \qquad \mathbf{z} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
$$

```python
class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))

model_twolayer = TwoLayerNet(hidden_size=256)
loss_twolayer = train_model(model_twolayer, train_loader, epochs=5, lr=0.1)
acc_twolayer = evaluate(model_twolayer, test_loader)
```

**Typical result: approximately 97% test accuracy.** The hidden layer learns stroke patterns and curves that are more discriminative than raw pixel values.

---

## Model 3 -- Convolutional Neural Network

A CNN exploits the spatial structure of images through local receptive fields and weight sharing:

```python
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> 16 x 28 x 28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 32 x 14 x 14
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)    # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)    # 14 -> 7
        return self.fc(x.view(x.size(0), -1))

model_cnn = SimpleCNN()
loss_cnn = train_model(model_cnn, train_loader, epochs=5, lr=0.01)
acc_cnn = evaluate(model_cnn, test_loader)
```

**Typical result: approximately 98--99% test accuracy.** Convolutional layers detect local patterns (edges, corners, loops) regardless of position, making them highly effective for image data.

---

## Model Comparison

| Model | Parameters | Test Accuracy |
|---|---|---|
| Softmax regression (linear) | $10 \times 784 + 10 = 7{,}850$ | ~92% |
| Two-layer network (256 hidden) | $784 \times 256 + 256 + 256 \times 10 + 10 = 203{,}530$ | ~97% |
| Simple CNN (16, 32 filters) | ~26,000 | ~98--99% |

The parameter count of the CNN is much smaller than the two-layer network, yet it achieves higher accuracy. This efficiency comes from weight sharing: a $3 \times 3$ convolutional filter has only 9 weights but is applied at every spatial location.

---

## Training Loss Curves

Plotting the loss curves for all three models reveals their convergence behavior.

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_linear, marker='o', label='Softmax Regression')
ax.plot(loss_twolayer, marker='s', label='Two-Layer Net')
ax.plot(loss_cnn, marker='^', label='Simple CNN')
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Cross-Entropy Loss")
ax.set_title("Training Loss Comparison")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Confusion Matrix Visualization

The confusion matrix for the CNN reveals which digit pairs the model still confuses.

```python
import numpy as np

def get_predictions(model, loader):
    """Collect all true labels and predictions."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            _, predicted = torch.max(model(images), 1)
            all_true.extend(labels.numpy())
            all_pred.extend(predicted.numpy())
    return np.array(all_true), np.array(all_pred)

y_true, y_pred = get_predictions(model_cnn, test_loader)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("CNN Confusion Matrix on MNIST")
plt.tight_layout()
plt.show()
```

Common confusions include 4 vs 9 (both have a vertical stroke on the right) and 3 vs 5 (similar upper curves).

---

## Interpretation

1. **Linear models have a ceiling.** Softmax regression achieves ~92% on MNIST, which is respectable but far from state-of-the-art. The limitation is that pixel intensities are not linearly separable by digit class -- a handwritten "1" shifted by a few pixels looks very different in raw pixel space.
2. **Hidden layers learn features.** The two-layer network overcomes the linear limitation by learning an intermediate representation $\mathbf{h}$ where digits are more separable. The 256-dimensional hidden layer acts as a learned feature extractor.
3. **CNNs exploit spatial structure.** Convolutional layers are translation-equivariant: a filter that detects an edge at one location can detect the same edge anywhere in the image. This inductive bias dramatically reduces the number of parameters needed and improves generalization.
4. **Cross-entropy loss naturally pairs with softmax.** PyTorch's `nn.CrossEntropyLoss` implements the log-sum-exp trick internally, avoiding the numerical instability of computing softmax and then taking its logarithm.

---

## Exercises

**Exercise 1.**
The softmax regression model has $\mathbf{W} \in \mathbb{R}^{10 \times 784}$. Each row $\mathbf{w}_k$ can be reshaped into a $28 \times 28$ image. Visualize the 10 weight vectors as images and interpret what they represent.

??? success "Solution to Exercise 1"
    ```python
    W = model_linear.fc.weight.data.numpy()   # shape (10, 784)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for k, ax in enumerate(axes.flat):
        ax.imshow(W[k].reshape(28, 28), cmap='seismic', vmin=-0.5, vmax=0.5)
        ax.set_title(f"Digit {k}")
        ax.axis('off')
    plt.suptitle("Learned Weight Templates")
    plt.tight_layout()
    plt.show()
    ```

    Each weight image $\mathbf{w}_k$ acts as a **template** for digit $k$. The logit $z_k = \mathbf{w}_k^\top \mathbf{x} + b_k$ computes the inner product between the template and the input image. Positive (red) regions indicate pixels whose presence supports class $k$; negative (blue) regions indicate pixels whose presence argues against class $k$. For example, the template for "0" typically shows a positive ring shape and a negative center, matching the visual structure of the digit zero.

---

**Exercise 2.**
Compute the number of floating-point multiply-accumulate operations (MACs) for a single forward pass through each of the three models. Use this to explain why the CNN is more computationally expensive per sample than the linear model despite having fewer parameters.

??? success "Solution to Exercise 2"
    **Softmax regression:** A single matrix-vector product $\mathbf{W}\mathbf{x}$ with $\mathbf{W} \in \mathbb{R}^{10 \times 784}$ requires $10 \times 784 = 7{,}840$ MACs.

    **Two-layer network:**

    - First layer: $784 \times 256 = 200{,}704$ MACs
    - Second layer: $256 \times 10 = 2{,}560$ MACs
    - Total: $\approx 203{,}264$ MACs

    **Simple CNN:**

    - Conv1: $16$ filters of size $3 \times 3 \times 1$ applied to $28 \times 28$ spatial locations: $16 \times 9 \times 28 \times 28 = 112{,}896$ MACs
    - Conv2: $32$ filters of size $3 \times 3 \times 16$ applied to $14 \times 14$ locations: $32 \times 144 \times 14 \times 14 = 903{,}168$ MACs
    - FC layer: $32 \times 7 \times 7 \times 10 = 15{,}680$ MACs
    - Total: $\approx 1{,}031{,}744$ MACs

    The CNN has fewer parameters (~26K vs ~203K) because each convolutional filter is shared across all spatial locations. However, it has more MACs because the same small filter is applied to every position in the feature map. The CNN trades parameter efficiency for computational cost, gaining translation equivariance in the process.

---

**Exercise 3.**
Modify the two-layer network to use dropout with probability $p = 0.5$ after the ReLU activation. Train for 10 epochs and compare the train/test accuracy gap with and without dropout. Explain why dropout acts as a regularizer.

??? success "Solution to Exercise 3"
    ```python
    class TwoLayerDropout(nn.Module):
        def __init__(self, hidden_size=256, p=0.5):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, hidden_size)
            self.dropout = nn.Dropout(p)
            self.fc2 = nn.Linear(hidden_size, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    model_drop = TwoLayerDropout()
    loss_drop = train_model(model_drop, train_loader, epochs=10, lr=0.1)
    acc_drop = evaluate(model_drop, test_loader)
    ```

    Without dropout, the two-layer network may achieve ~99% training accuracy but ~97% test accuracy (a 2-point gap). With dropout, the training accuracy is lower (~97%) but the test accuracy is similar or slightly better, reducing the gap.

    Dropout works as a regularizer because during training it randomly sets each hidden unit to zero with probability $p$. This means the network cannot rely on any single neuron and must spread its learned representation across many units. Effectively, dropout trains an exponentially large ensemble of sub-networks (all $2^{256}$ possible masks) and averages their predictions at test time (by scaling weights by $1-p$). This reduces co-adaptation of features and improves generalization.

---

**Exercise 4.**
Prove that the number of parameters in a convolutional layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, and kernel size $k \times k$ is $C_{\text{out}}(C_{\text{in}} k^2 + 1)$. Verify this for the two convolutional layers in our CNN.

??? success "Solution to Exercise 4"
    Each of the $C_{\text{out}}$ filters has a $k \times k$ spatial kernel for each of the $C_{\text{in}}$ input channels, plus one bias term. Thus the total parameter count is:

    $$
    C_{\text{out}} \times (C_{\text{in}} \times k^2 + 1)
    $$

    **Conv1:** $C_{\text{in}} = 1$, $C_{\text{out}} = 16$, $k = 3$:

    $$
    16 \times (1 \times 9 + 1) = 16 \times 10 = 160
    $$

    **Conv2:** $C_{\text{in}} = 16$, $C_{\text{out}} = 32$, $k = 3$:

    $$
    32 \times (16 \times 9 + 1) = 32 \times 145 = 4{,}640
    $$

    **FC layer:** $32 \times 7 \times 7 = 1{,}568$ inputs, 10 outputs: $1{,}568 \times 10 + 10 = 15{,}690$.

    **Total:** $160 + 4{,}640 + 15{,}690 = 20{,}490$ parameters.

    We can verify in PyTorch:

    ```python
    total = sum(p.numel() for p in model_cnn.parameters())
    print(f"Total CNN parameters: {total}")
    ```

    $\square$

---

**Exercise 5.**
The softmax regression model on MNIST learns linear decision boundaries in 784-dimensional pixel space. Give a concrete example of two images that belong to different classes but have a small Euclidean distance in pixel space, and explain why this is problematic for a linear classifier. Then explain how a CNN overcomes this limitation.

??? success "Solution to Exercise 5"
    Consider a digit "1" drawn as a thin vertical stroke centered in the image, and the same "1" shifted 3 pixels to the right. In pixel space, every nonzero pixel in the original has moved, so the Euclidean distance between the two images is substantial:

    $$
    \|\mathbf{x}_{\text{centered}} - \mathbf{x}_{\text{shifted}}\|_2 = \sqrt{\sum_{i} (x_i^{\text{cen}} - x_i^{\text{shift}})^2} > 0
    $$

    even though both are clearly "1". Conversely, a "7" written with a particular stroke style might happen to activate similar pixels as the shifted "1", giving it a small Euclidean distance despite being a different class.

    For a linear classifier, the logit $z_k = \mathbf{w}_k^\top \mathbf{x} + b_k$ depends on the absolute pixel positions. A 3-pixel shift changes every $z_k$, potentially changing the predicted class. The linear model must learn separate templates for each position variant, which is impossible with limited data.

    A CNN overcomes this through **translation equivariance**. Convolutional layers apply the same learned filter at every spatial position. If a filter detects a vertical edge at position $(i, j)$, it also detects that same edge at $(i, j+3)$. The subsequent max-pooling layers introduce approximate **translation invariance**, further reducing sensitivity to small shifts. The CNN learns to recognize the stroke pattern "vertical line" regardless of its position, which is precisely the invariance needed for digit recognition.
