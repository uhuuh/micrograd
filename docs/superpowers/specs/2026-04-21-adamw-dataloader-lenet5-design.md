# Design: AdamW + DataLoader + LeNet-5 Pipeline

## Overview

Implement three components to create a complete training pipeline for MNIST handwritten digit recognition:
1. **AdamW optimizer** - memory-efficient weight decay
2. **DataLoader** - batch iteration with shuffling
3. **LeNet-5** - convolutional neural network for MNIST

Target: >95% accuracy on MNIST test set, training under 1 minute.

## File Structure

```
micrograd/
├── __init__.py
├── engine.py      # Tensor class (existing)
├── nn.py          # Module, Neuron, Layer, MLP (existing)
├── optim.py       # NEW: AdamW optimizer
├── data.py        # NEW: DataLoader + MNIST dataset
├── lenet5.py      # NEW: LeNet-5 model
└── train_mnist.py # NEW: Training script
```

## 1. AdamW Optimizer (`optim.py`)

### API

```python
class AdamW:
    def __init__(self, parameters, lr=0.001, weight_decay=0.01,
                 betas=(0.9, 0.999), eps=1e-8):
        """
        AdamW optimizer with decoupled weight decay.

        Args:
            parameters: iterable of Tensors to optimize
            lr: learning rate (default: 0.001)
            weight_decay: L2 regularization strength (default: 0.01)
            betas: (beta1, beta2) exponential decay rates (default: (0.9, 0.999))
            eps: epsilon for numerical stability (default: 1e-8)
        """

    def step(self):
        """Perform single optimization step."""

    def zero_grad(self):
        """Clear gradients of all parameters."""
```

### Algorithm (AdamW = Adam + decoupled weight decay)

For each parameter `p` with gradient `g`:
```
t += 1  # timestep
m = beta1 * m + (1 - beta1) * g           # momentum
v = beta2 * v + (1 - beta2) * g^2         # velocity
m_hat = m / (1 - beta1^t)                 # bias correction
v_hat = v / (1 - beta2^t)
p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
```

## 2. DataLoader (`data.py`)

### Dataset Class

```python
class Dataset:
    """Simple in-memory dataset."""
    def __init__(self, X, y):
        self.X = X  # numpy array of shape (N, H, W) or (N,)
        self.y = y  # numpy array of labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### DataLoader Class

```python
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

### MNIST Loading

```python
def load_mnist():
    """Load MNIST dataset from sklearn or numpy files."""
    # Option: Use sklearn's fetch_openml or bundled .npy files
    # Returns (X_train, y_train), (X_test, y_test)
```

## 3. LeNet-5 (`lenet5.py`)

### Architecture

Original LeNet-5 for 32x32 input, adapted for 28x28 MNIST:

```
Conv1: 1 channel → 6 channels, 5x5 kernel, stride=1
  → ReLU
  → AvgPool: 2x2, stride=2

Conv2: 6 channels → 16 channels, 5x5 kernel, stride=1
  → ReLU
  → AvgPool: 2x2, stride=2

FC1: 16*4*4 = 256 → 120 neurons
  → ReLU

FC2: 120 → 84 neurons
  → ReLU

FC3: 84 → 10 neurons  (output logits for 10 digits)
```

### Module Integration

```python
class Conv2d(Module):
    """2D Convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Initialize weights: (out_channels, in_channels, H, W)
        self.w = Tensor(random.randn(out_channels, in_channels,
                                     kernel_size, kernel_size) * 0.1)
        self.b = Tensor(zeros(out_channels))
        self.stride = 1
        self.padding = 0

    def __call__(self, x):
        # x: (batch, in_channels, H, W)
        # Implement 2D convolution forward
        return ...

    def parameters(self):
        return [self.w, self.b]

class AvgPool2d(Module):
    """Average pooling layer."""
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        # x: (batch, channels, H, W)
        return ...

    def parameters(self):
        return []

class LeNet5(Module):
    """LeNet-5 for MNIST (28x28 input)."""
    def __init__(self):
        self.conv1 = Conv2d(1, 6, kernel_size=5)
        self.pool1 = AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)
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
        x = x.reshape(x.shape[0], -1)  # Flatten
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

### Conv2d Backward

Convolution backward requires:
- `grad_w`: sum over output gradients * input
- `grad_b`: sum over output gradients
- `grad_input`: convolution of input gradients with flipped weights

## 4. Training Script (`train_mnist.py`)

### Training Loop

```python
def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss."""
    # logits: (batch, 10), targets: (batch,)
    # Simple implementation: -log(softmax(logits)[target])
    exp_logits = [l.exp() for l in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    losses = [-p[target].log() for p, target in zip(probs, targets)]
    return sum(losses) / len(losses)

# Main
train_loader = MNISTLoader(batch_size=32, shuffle=True)
model = LeNet5()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(5):
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

    print(f"Epoch {epoch}: loss={loss.data}")
```

## 5. Dependencies

New dependencies required:
- `scikit-learn` (for MNIST loading via `sklearn.datasets`)
- Already available: `numpy`, `matplotlib`

## 6. Testing

- Unit tests for AdamW (parameter update correctness)
- Unit tests for DataLoader (batch iteration, shuffling)
- Integration test: train LeNet-5 for 1 epoch, verify loss decreases
- Accuracy test: >95% on test set after 5 epochs

## 7. Implementation Order

1. `optim.py` - AdamW (simplest, no dependencies)
2. `data.py` - DataLoader + MNIST loading (depends on numpy)
3. `lenet5.py` - Conv2d, AvgPool2d, LeNet5 (depends on Tensor operations)
4. `train_mnist.py` - Integration script

## 8. Weight Initialization

- Conv2d: Gaussian with std=0.1
- FC layers: Use existing Neuron initialization (uniform [-1, 1])
- Biases: Zero initialization
