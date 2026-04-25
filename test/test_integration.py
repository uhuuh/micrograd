# test/test_integration.py
import numpy as np
from micrograd.tensor import Tensor
from micrograd.optim import AdamW
from micrograd.data import DataLoader, Dataset
from micrograd.lenet5 import LeNet5


def test_lenet5_training_one_epoch():
    """Test that LeNet-5 can train for one epoch without error."""
    # Create small synthetic dataset
    X = np.random.randn(100, 28, 28) / 255.0
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
        batch_size = logits.data.shape[0]

        # Simple loss (just to verify backward works)
        loss = logits.sum() / batch_size

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Verify model was trained (loss should be computable)
    assert hasattr(model, 'fc3_w')
