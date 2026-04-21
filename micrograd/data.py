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
