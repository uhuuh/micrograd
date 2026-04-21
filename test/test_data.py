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

def test_load_mnist():
    from micrograd.data import load_mnist
    (X_train, y_train), (X_test, y_test) = load_mnist()
    assert len(X_train) == 60000
    assert len(X_test) == 10000
    assert X_train.shape[1] == 28
    assert X_train.shape[2] == 28
    assert set(y_train) == set(range(10))
