# AdamW + DataLoader + LeNet-5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement AdamW optimizer, DataLoader, and LeNet-5 model to achieve >95% accuracy on MNIST within 1 minute training.

**Architecture:** Build three independent modules (optim.py, data.py, lenet5.py) that integrate with existing micrograd engine. LeNet-5 uses Conv2d/AvgPool2d layers following the classic architecture.

**Tech Stack:** micrograd (Tensor, autograd), numpy, scikit-learn (MNIST)

---

## File Structure

```
micrograd/
├── optim.py       # NEW: AdamW optimizer
├── data.py        # NEW: DataLoader + MNIST loading
├── lenet5.py      # NEW: Conv2d, AvgPool2d, LeNet5
└── train_mnist.py # NEW: Training script
```

---

## Task 1: AdamW Optimizer

**Files:**
- Create: `micrograd/optim.py`
- Test: `test/test_optim.py`

- [ ] **Step 1: Write failing test for AdamW**

```python
# test/test_optim.py
import numpy as np
from micrograd.engine import Tensor
from micrograd.optim import AdamW

def test_adamw_initialization():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0], requires_grad=True)
    optim = AdamW([p1, p2], lr=0.001, weight_decay=0.01)
    assert optim.lr == 0.001
    assert optim.weight_decay == 0.01
    assert len(optim.params) == 2

def test_adamw_step():
    p = Tensor([1.0], requires_grad=True)
    p.grad = Tensor([0.5])
    optim = AdamW([p], lr=0.01, weight_decay=0.0)
    old_val = p.data.copy()
    optim.step()
    # SGD-like update: p = p - lr * grad
    assert np.allclose(p.data, old_val - 0.01 * 0.5)

def test_adamw_zero_grad():
    p = Tensor([1.0], requires_grad=True)
    p.grad = Tensor([0.5])
    optim = AdamW([p])
    optim.zero_grad()
    assert p.grad is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_optim.py -v`
Expected: ERROR - AdamW not defined

- [ ] **Step 3: Write minimal AdamW implementation**

```python
# micrograd/optim.py
import random
import numpy as np
from micrograd.engine import Tensor

class AdamW:
    def __init__(self, parameters, lr=0.001, weight_decay=0.01,
                 betas=(0.9, 0.999), eps=1e-8):
        self.params = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # Initialize first and second moment estimates
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data

            # Decoupled weight decay (AdamW)
            p.data = p.data - self.lr * self.weight_decay * p.data

            # Momentum update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Velocity update
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_optim.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/optim.py test/test_optim.py
git commit -m "feat: add AdamW optimizer"
```

---

## Task 2: DataLoader

**Files:**
- Create: `micrograd/data.py`
- Test: `test/test_data.py`

- [ ] **Step 1: Write failing tests for DataLoader**

```python
# test/test_data.py
import numpy as np
from micrograd.data import Dataset, DataLoader

def test_dataset():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    ds = Dataset(X, y)
    assert len(ds) == 3
    x, label = ds[0]
    assert np.array_equal(x, [1, 2])
    assert label == 0

def test_dataloader_iteration():
    X = np.arange(10)
    y = np.arange(10)
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=3)
    batches = list(loader)
    assert len(batches) == 4  # 10/3 rounded up
    assert np.array_equal(batches[0][0], [0, 1, 2])

def test_dataloader_shuffle():
    X = np.arange(10)
    y = np.arange(10)
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=3, shuffle=True)
    # Just verify it runs without error
    batches = list(loader)
    assert len(batches) == 4

def test_dataloader_no_shuffle():
    X = np.arange(10)
    y = np.arange(10)
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batches = list(loader)
    # First batch should be [0,1,2]
    assert np.array_equal(batches[0][0], [0, 1, 2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_data.py -v`
Expected: ERROR - Dataset not defined

- [ ] **Step 3: Write Dataset and DataLoader implementation**

```python
# micrograd/data.py
import random
import numpy as np

class Dataset:
    """Simple in-memory dataset."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    """Batch iterator with optional shuffling."""
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[i:i + self.batch_size]
            X_batch = [self.dataset.X[j] for j in batch_idx]
            y_batch = [self.dataset.y[j] for j in batch_idx]
            yield X_batch, y_batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/data.py test/test_data.py
git commit -m "feat: add Dataset and DataLoader"
```

---

## Task 3: MNIST Loading

**Files:**
- Modify: `micrograd/data.py` (add load_mnist function)
- Test: `test/test_data.py` (add MNIST test)

- [ ] **Step 1: Write failing test for MNIST loading**

```python
# Add to test/test_data.py
def test_load_mnist():
    from micrograd.data import load_mnist
    (X_train, y_train), (X_test, y_test) = load_mnist()
    assert len(X_train) == 60000
    assert len(X_test) == 10000
    assert X_train.shape[1] == 28
    assert X_train.shape[2] == 28
    assert set(y_train) == set(range(10))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_data.py::test_load_mnist -v`
Expected: ERROR - load_mnist not defined

- [ ] **Step 3: Implement load_mnist**

```python
# Add to micrograd/data.py
def load_mnist():
    """Load MNIST dataset using sklearn."""
    from sklearn.datasets import fetch_openml

    # Fetch MNIST from OpenML
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.reshape(-1, 28, 28).astype(np.float64)
    y = mnist.target.astype(np.int32)

    # Split into train (first 60000) and test (last 10000)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return (X_train, y_train), (X_test, y_test)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_data.py::test_load_mnist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/data.py test/test_data.py
git commit -m "feat: add MNIST loading with sklearn"
```

---

## Task 4: Conv2d and AvgPool2d Layers

**Files:**
- Create: `micrograd/lenet5.py`
- Test: `test/test_lenet5.py`

- [ ] **Step 1: Write failing tests for Conv2d and AvgPool2d**

```python
# test/test_lenet5.py
import numpy as np
from micrograd.engine import Tensor
from micrograd.lenet5 import Conv2d, AvgPool2d

def test_conv2d_forward():
    # Input: (batch=1, channels=1, H=4, W=4)
    x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
    conv = Conv2d(1, 1, kernel_size=2)
    out = conv(x)
    # Output: (1, 1, 3, 3) assuming no padding, stride=1
    assert out.data.shape == (1, 1, 3, 3)

def test_conv2d_parameters():
    conv = Conv2d(3, 16, kernel_size=5)
    params = conv.parameters()
    assert len(params) == 2  # w and b
    assert params[0].data.shape == (16, 3, 5, 5)
    assert params[1].data.shape == (16,)

def test_avgpool2d_forward():
    # Input: (batch=1, channels=1, H=4, W=4)
    x = Tensor(np.ones((1, 1, 4, 4)), requires_grad=True)
    pool = AvgPool2d(kernel_size=2, stride=2)
    out = pool(x)
    # Output: (1, 1, 2, 2)
    assert out.data.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_lenet5.py -v`
Expected: ERROR - Conv2d not defined

- [ ] **Step 3: Write Conv2d implementation**

```python
# micrograd/lenet5.py
import random
import numpy as np
from micrograd.engine import Tensor
from micrograd.nn import Module

def conv2d_forward(input, weight, bias, stride=1, padding=0):
    """2D convolution forward pass."""
    # input: (batch, in_channels, H, W)
    # weight: (out_channels, in_channels, kernel_h, kernel_w)
    batch, in_c, in_h, in_w = input.shape
    out_c, _, kernel_h, kernel_w = weight.shape

    out_h = (in_h - kernel_h + 2 * padding) // stride + 1
    out_w = (in_w - kernel_w + 2 * padding) // stride + 1

    output = np.zeros((batch, out_c, out_h, out_w))

    for b in range(batch):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    patch = input[b, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                    output[b, oc, oh, ow] = np.sum(patch * weight[oc]) + bias[oc]

    return output


class Conv2d(Module):
    """2D Convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0

        # Initialize weights: Gaussian with std=0.1
        self.w = Tensor(
            random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1,
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_channels), requires_grad=True)

    def __call__(self, x):
        return Tensor(conv2d_forward(x.data, self.w.data, self.b.data,
                                      self.stride, self.padding),
                      requires_grad=x.requires_grad)

    def parameters(self):
        return [self.w, self.b]


class AvgPool2d(Module):
    """Average pooling layer."""
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        batch, channels, h, w = x.data.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        output = np.zeros((batch, channels, out_h, out_w))

        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        patch = x.data[b, c, h_start:h_start+self.kernel_size,
                                                    w_start:w_start+self.kernel_size]
                        output[b, c, oh, ow] = np.mean(patch)

        return Tensor(output, requires_grad=x.requires_grad)

    def parameters(self):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_lenet5.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/lenet5.py test/test_lenet5.py
git commit -m "feat: add Conv2d and AvgPool2d layers"
```

---

## Task 5: LeNet-5 Model

**Files:**
- Modify: `micrograd/lenet5.py` (add LeNet5 class)
- Test: `test/test_lenet5.py` (add LeNet5 test)

- [ ] **Step 1: Write failing test for LeNet5**

```python
# Add to test/test_lenet5.py
def test_lenet5_forward():
    from micrograd.lenet5 import LeNet5
    model = LeNet5()
    # Input: batch of 2, 28x28 images
    x = Tensor(np.random.randn(2, 1, 28, 28), requires_grad=True)
    out = model(x)
    # Output: (batch, 10)
    assert out.data.shape == (2, 10)

def test_lenet5_parameters():
    from micrograd.lenet5 import LeNet5
    model = LeNet5()
    params = model.parameters()
    # Should have conv weights + fc weights + biases
    assert len(params) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_lenet5.py::test_lenet5_forward -v`
Expected: ERROR - LeNet5 not defined

- [ ] **Step 3: Write LeNet5 implementation**

```python
# Add to micrograd/lenet5.py

class LeNet5(Module):
    """LeNet-5 for MNIST (28x28 input)."""
    def __init__(self):
        # Conv layers
        self.conv1 = Conv2d(1, 6, kernel_size=5)
        self.pool1 = AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)

        # FC layers (using existing Layer class)
        from micrograd.nn import Layer
        # After conv+pool: 16 channels * 4*4 = 256 features
        self.fc1 = Layer(256, 120)
        self.fc2 = Layer(120, 84)
        self.fc3 = Layer(84, 10, nonlin=False)  # No activation on output

    def __call__(self, x):
        # x: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = x.relu()
        x = self.pool1(x)

        x = self.conv2(x)
        x = x.relu()
        x = self.pool2(x)

        # Flatten
        batch = x.data.shape[0]
        x = x.reshape(batch, -1)

        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        return x

    def parameters(self):
        return (self.conv1.parameters() + self.pool1.parameters() +
                self.conv2.parameters() + self.pool2.parameters() +
                self.fc1.parameters() + self.fc2.parameters() +
                self.fc3.parameters())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_lenet5.py::test_lenet5_forward test/test_lenet5.py::test_lenet5_parameters -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/lenet5.py test/test_lenet5.py
git commit -m "feat: add LeNet5 model"
```

---

## Task 6: Training Script with Cross-Entropy Loss

**Files:**
- Create: `train_mnist.py`

- [ ] **Step 1: Create training script**

```python
#!/usr/bin/env python
"""Train LeNet-5 on MNIST using micrograd."""

import numpy as np
from micrograd.engine import Tensor
from micrograd.optim import AdamW
from micrograd.data import DataLoader, Dataset, load_mnist
from micrograd.lenet5 import LeNet5


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss.

    logits: list of Tensors, each (10,)
    targets: Tensor of shape (batch,)
    """
    batch_size = len(logits)

    # Compute softmax probabilities
    exp_logits = [l.exp() for l in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    # Negative log-likelihood
    losses = []
    for i, probs_i in enumerate(probs):
        target = targets.data[i]
        loss_i = -probs_i[target].log()
        losses.append(loss_i)

    return sum(losses) / batch_size


def accuracy(logits, targets):
    """Compute accuracy."""
    preds = [l.data.argmax() for l in logits]
    correct = sum(p == t for p, t in zip(preds, targets.data))
    return correct / len(targets.data)


def main():
    print("Loading MNIST...")
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create datasets and dataloaders
    train_ds = Dataset(X_train, y_train)
    test_ds = Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

    print("Creating model...")
    model = LeNet5()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    epochs = 5
    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_acc = 0
        batches = 0

        for X_batch, y_batch in train_loader:
            # Convert to Tensors
            X_batch = [Tensor(x) for x in X_batch]  # Each is 28x28
            y_batch = Tensor(y_batch)

            # Forward pass
            logits = model(X_batch)

            # Loss
            loss = cross_entropy_loss(logits, y_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

            total_loss += loss.data
            total_acc += accuracy(logits, y_batch)
            batches += 1

        avg_loss = total_loss / batches
        avg_acc = total_acc / batches
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc*100:.1f}%")

    # Final test accuracy
    model.eval()
    total_acc = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch = [Tensor(x) for x in X_batch]
        y_batch = Tensor(y_batch)
        logits = model(X_batch)
        preds = [l.data.argmax() for l in logits]
        correct = sum(p == t for p, t in zip(preds, y_batch.data))
        total_acc += correct
        total += len(y_batch.data)

    print(f"Test accuracy: {total_acc/total*100:.1f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training script**

Run: `python train_mnist.py`
Expected: Training output with loss decreasing and accuracy increasing

- [ ] **Step 3: Verify >95% test accuracy**

Check final test accuracy output

- [ ] **Step 4: Commit**

```bash
git add train_mnist.py
git commit -m "feat: add MNIST training script with LeNet-5"
```

---

## Task 7: Integration Test

**Files:**
- Create: `test/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# test/test_integration.py
import numpy as np
from micrograd.engine import Tensor
from micrograd.optim import AdamW
from micrograd.data import DataLoader, Dataset
from micrograd.lenet5 import LeNet5


def test_lenet5_training_one_epoch():
    """Test that LeNet-5 can train for one epoch without error."""
    # Create small synthetic dataset
    X = np.random.randn(100, 1, 28, 28) / 255.0
    y = np.random.randint(0, 10, 100)
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=10)

    model = LeNet5()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    for X_batch, y_batch in loader:
        X_batch = [Tensor(x) for x in X_batch]
        y_batch = Tensor(y_batch)

        # Forward
        logits = model(X_batch)

        # Simple loss (just to verify backward works)
        loss = sum(logits).sum() / len(logits)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Verify parameters were updated
    # (gradients should have been computed)
    for p in model.parameters():
        assert p.grad is not None
```

- [ ] **Step 2: Run integration test**

Run: `pytest test/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/test_integration.py
git commit -m "test: add integration test for LeNet-5 training"
```

---

## Self-Review Checklist

- [ ] Spec coverage: AdamW, DataLoader, MNIST loading, Conv2d, AvgPool2d, LeNet5, training script all have tasks
- [ ] No placeholders: All code is complete
- [ ] Type consistency: Class names (AdamW, Dataset, DataLoader, Conv2d, AvgPool2d, LeNet5) consistent across tasks
- [ ] TDD approach: Each task starts with failing test
- [ ] YAGNI: Only essential features implemented
