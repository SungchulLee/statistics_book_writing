# MNIST Case Study

## Overview

This section applies the theory from Chapters 13–14 to the MNIST
handwritten digit dataset (60 000 training images, 10 000 test images,
$28\times 28$ pixels, 10 classes).  We compare three model
architectures of increasing complexity: a single linear layer, a
two-layer network, and a simple CNN — all implemented in PyTorch.

---

## 1  Visualizing the Data

```python
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=True)

images, labels = next(iter(test_loader))
img_grid = torchvision.utils.make_grid(images, nrow=8, padding=2)

plt.figure(figsize=(8, 8))
plt.imshow(img_grid.permute(1, 2, 0), cmap='gray')
plt.axis('off')
plt.show()
```

---

## 2  Single Linear Layer (Softmax Regression)

The simplest model: flatten the image and apply one linear
transformation followed by softmax (via `CrossEntropyLoss`).

```python
import torch.nn as nn
import torch.optim as optim

class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

model = SimpleMNIST()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 6):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Typical test accuracy: ~92%.**

---

## 3  Before vs After Training Comparison

Visualizing predictions on the same batch of images before and after
training illustrates how the model transitions from random guessing to
meaningful classification.

```python
def show_images(images, true_labels, pred_labels, title):
    plt.figure(figsize=(10, 10))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(images[i][0], cmap='binary')
        plt.axis("off")
        plt.title(f"T:{true_labels[i]} P:{pred_labels[i]}", fontsize=6)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Before training
with torch.no_grad():
    _, preds = torch.max(model_untrained(fixed_images), 1)
show_images(fixed_images, fixed_labels, preds, "Before Training")

# After training
with torch.no_grad():
    _, preds = torch.max(model_trained(fixed_images), 1)
show_images(fixed_images, fixed_labels, preds, "After Training")
```

---

## 4  Simple CNN

Adding two convolutional layers dramatically improves accuracy.

```python
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # → 16×28×28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # → 32×14×14
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28→14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14→7
        return self.fc(x.view(x.size(0), -1))

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 6):
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Typical test accuracy: ~98–99%.**

---

## 5  PyTorch Softmax Regression (Full Pipeline)

A complete, production-style PyTorch pipeline with device handling,
model save/load, and per-class accuracy reporting.

### Model

```python
class Net(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.layer(torch.flatten(x, 1))
```

### Training

```python
def train(model, loader, criterion, optimizer, epochs=2, device='cpu'):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch+1}, {i+1:5d}] '
                      f'loss: {running_loss/2000:.3f}')
                running_loss = 0.0
```

### Evaluation

```python
def compute_accuracy(model, loader, classes, device='cpu'):
    model.eval()
    correct = total = 0
    class_correct = {c: 0 for c in classes}
    class_total   = {c: 0 for c in classes}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for lbl, pred in zip(labels, predicted):
                if lbl == pred:
                    class_correct[classes[lbl]] += 1
                class_total[classes[lbl]] += 1

    print(f'Overall accuracy: {100 * correct / total:.1f}%')
    for c in classes:
        print(f'  {c}: {100 * class_correct[c] / class_total[c]:.1f}%')
```

### Save and Load

```python
torch.save(model.state_dict(), './model/model.pth')

model = Net()
model.load_state_dict(torch.load('./model/model.pth'))
```

---

## Model Comparison

| Model | Parameters | Test Accuracy |
|---|---|---|
| Linear (softmax regression) | ~7.9 K | ~92% |
| Two-layer network (100 hidden) | ~89 K | ~97% |
| Simple CNN (16→32 filters) | ~26 K | ~98–99% |

The jump from a linear model to a single hidden layer is substantial
because the hidden layer can learn non-linear feature combinations.
The CNN goes further by exploiting the spatial structure of images
through weight sharing and local connectivity.
