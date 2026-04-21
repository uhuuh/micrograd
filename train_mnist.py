#!/usr/bin/env python
"""Train LeNet-5 on MNIST using micrograd."""

import numpy as np
from micrograd.engine import Tensor
from micrograd.optim import AdamW
from micrograd.data import DataLoader, Dataset, load_mnist
from micrograd.lenet5 import LeNet5


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss.

    logits: Tensor of shape (batch, 10)
    targets: Tensor of shape (batch,)
    """
    # logits: (batch, 10), each row is logit for that sample
    batch_size = logits.data.shape[0]

    # Compute softmax per row using numpy
    exp_data = np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
    probs_data = exp_data / exp_data.sum(axis=1, keepdims=True)

    losses = []
    for i in range(batch_size):
        target = int(targets.data[i])
        loss_i = -np.log(probs_data[i, target] + 1e-8)
        losses.append(loss_i)

    return Tensor(sum(losses) / batch_size, copy=False)


def accuracy(logits, targets):
    """Compute accuracy."""
    preds = logits.data.argmax(axis=1)
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
