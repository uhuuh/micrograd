import numpy as np
from micrograd.engine import Tensor

def test_tensor_creation():
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(t.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert t.requires_grad == False
    assert t.grad is None

def test_tensor_requires_grad():
    t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    assert t.requires_grad == True

def test_add_forward():
    a = Tensor(np.array([1.0, 2.0]))
    b = Tensor(np.array([3.0, 4.0]))
    c = a + b
    assert np.array_equal(c.data, np.array([4.0, 6.0]))

def test_add_backward():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    c = a + b
    c.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0]))
    assert np.array_equal(b.grad.data, np.array([1.0, 1.0]))

def test_mul_backward():
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = a * b
    c.backward()
    assert np.array_equal(a.grad.data, np.array([4.0, 5.0]))
    assert np.array_equal(b.grad.data, np.array([2.0, 3.0]))

def test_relu_backward():
    a = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]), requires_grad=True)
    b = a.relu()
    b.backward()
    assert np.array_equal(a.grad.data, np.array([0.0, 1.0, 0.0, 1.0]))

def test_matmul_forward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    c = a @ b
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.array_equal(c.data, expected)

def test_matmul_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    c = a @ b
    c.backward()
    # dC = [[1,1],[1,1]] (gradient of sum)
    # dA = dC @ b.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    # dB = a.T @ dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    expected_grad_a = np.array([[11.0, 15.0], [11.0, 15.0]])
    expected_grad_b = np.array([[4.0, 4.0], [6.0, 6.0]])
    assert np.array_equal(a.grad.data, expected_grad_a)
    assert np.array_equal(b.grad.data, expected_grad_b)

def test_reflected_operations():
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = 5.0 - a  # rsub
    assert np.array_equal(b.data, np.array([4.0, 3.0, 2.0]))
    b.sum().backward()
    assert np.array_equal(a.grad.data, np.array([-1.0, -1.0, -1.0]))

    c = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    d = 10.0 / c  # rtruediv
    assert np.allclose(d.data, np.array([5.0, 2.5]))
    d.sum().backward()
    assert np.allclose(c.grad.data, np.array([-2.5, -0.625]))


def test_tensor_ops_match_pytorch():
    """Test that our tensor operations produce same gradients as PyTorch."""
    import torch

    # Test element-wise ops
    a_np = np.random.randn(3, 4).astype(np.float64)
    b_np = np.random.randn(3, 4).astype(np.float64)

    a_t = Tensor(a_np, requires_grad=True)
    b_t = Tensor(b_np, requires_grad=True)

    a_th = torch.tensor(a_np, requires_grad=True)
    b_th = torch.tensor(b_np, requires_grad=True)

    # Test add
    c_t = a_t + b_t
    c_th = a_th + b_th
    assert np.allclose(c_t.data, c_th.detach().numpy())

    c_t.sum().backward()
    c_th.sum().backward()
    assert np.allclose(a_t.grad.data, a_th.grad.numpy())
    assert np.allclose(b_t.grad.data, b_th.grad.numpy())

    # Test matmul
    x_np = np.random.randn(2, 3).astype(np.float64)
    y_np = np.random.randn(3, 4).astype(np.float64)

    x_t = Tensor(x_np, requires_grad=True)
    y_t = Tensor(y_np, requires_grad=True)

    x_th = torch.tensor(x_np, requires_grad=True)
    y_th = torch.tensor(y_np, requires_grad=True)

    z_t = x_t @ y_t
    z_th = x_th @ y_th
    assert np.allclose(z_t.data, z_th.detach().numpy())

    z_t.sum().backward()
    z_th.sum().backward()
    assert np.allclose(x_t.grad.data, x_th.grad.numpy())
    assert np.allclose(y_t.grad.data, y_th.grad.numpy())


def test_relu_match_pytorch():
    """Test that ReLU produces same gradients as PyTorch."""
    import torch

    x_np = np.random.randn(4, 5).astype(np.float64)
    x_t = Tensor(x_np, requires_grad=True)
    x_th = torch.tensor(x_np, requires_grad=True)

    y_t = x_t.relu()
    y_th = torch.relu(x_th)

    assert np.allclose(y_t.data, y_th.detach().numpy())

    y_t.sum().backward()
    y_th.sum().backward()
    assert np.allclose(x_t.grad.data, x_th.grad.numpy())


# Tests for nn module (based on demo.ipynb)
import random
from micrograd.nn import Neuron, Layer, MLP

def test_neuron():
    """Test Neuron creation and forward pass."""
    neuron = Neuron(2)
    assert len(neuron.w) == 2
    assert neuron.b is not None
    assert neuron.nonlin == True

    # Forward pass
    x = [Tensor(1.0), Tensor(2.0)]
    out = neuron(x)
    assert out is not None

def test_neuron_no_activation():
    """Test Neuron without activation (linear)."""
    neuron = Neuron(2, nonlin=False)
    assert neuron.nonlin == False

    x = [Tensor(1.0), Tensor(2.0)]
    out = neuron(x)
    assert out is not None

def test_layer():
    """Test Layer creation and forward pass."""
    layer = Layer(2, 3)
    assert len(layer.neurons) == 3

    x = [Tensor(1.0), Tensor(2.0)]
    out = layer(x)
    assert len(out) == 3

def test_layer_single_neuron():
    """Test Layer with single neuron returns tensor not list."""
    layer = Layer(2, 1)
    x = [Tensor(1.0), Tensor(2.0)]
    out = layer(x)
    assert not isinstance(out, list)

def test_mlp():
    """Test MLP creation."""
    mlp = MLP(2, [4, 1])
    assert len(mlp.layers) == 2
    assert mlp.parameters() is not None

def test_mlp_forward():
    """Test MLP forward pass."""
    mlp = MLP(2, [4, 1])
    x = [Tensor(1.0), Tensor(2.0)]
    out = mlp(x)
    assert out is not None

def test_mlp_parameters():
    """Test MLP parameters method."""
    mlp = MLP(2, [16, 1])
    params = mlp.parameters()
    assert len(params) > 0

def test_mlp_zero_grad():
    """Test MLP zero_grad method."""
    mlp = MLP(2, [4, 1])
    x = [Tensor(1.0), Tensor(2.0)]
    out = mlp(x)
    out.backward()

    # After backward, all params should have grad
    for p in mlp.parameters():
        assert p.grad is not None

    # zero_grad should set all grads to zero
    mlp.zero_grad()
    for p in mlp.parameters():
        assert np.allclose(p.grad.data, np.zeros_like(p.grad.data))

def test_mlp_training_loop():
    """Test MLP training loop (like in demo.ipynb)."""
    # Create simple dataset
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = np.array([1, 1, -1, -1])  # binary classification targets

    mlp = MLP(2, [8, 1])

    # Training loop
    for epoch in range(50):
        # Forward
        losses = []
        for xi, yi in zip(X, y):
            x_tensor = [Tensor(x) for x in xi]
            score = mlp(x_tensor)
            # SVM loss: (1 + -yi*score).relu()
            loss = (Tensor(1.0) + Tensor(-yi) * score).relu()
            losses.append(loss)

        total_loss = sum(losses) * (1.0 / len(losses))

        # Backward
        mlp.zero_grad()
        total_loss.backward()

        # Update
        for p in mlp.parameters():
            p.data -= 0.1 * p.grad.data

    # Final forward pass - predictions should be reasonable
    predictions = []
    for xi in X:
        x_tensor = [Tensor(x) for x in xi]
        score = mlp(x_tensor)
        predictions.append(score.data > 0)

    # At least 3 out of 4 should be correct
    correct = sum([(p == (y[i] > 0)) for i, p in enumerate(predictions)])
    assert correct >= 3

def test_mlp_backward():
    """Test MLP backward pass."""
    mlp = MLP(2, [4, 1])

    x = [Tensor(1.0), Tensor(2.0)]
    score = mlp(x)

    score.backward()

    # All parameters should have gradients
    for p in mlp.parameters():
        assert p.grad is not None

def test_neuron_parameters():
    """Test Neuron parameters method."""
    neuron = Neuron(3)
    params = neuron.parameters()
    assert len(params) == 4  # 3 weights + 1 bias

def test_layer_parameters():
    """Test Layer parameters method."""
    layer = Layer(2, 3)
    params = layer.parameters()
    assert len(params) == 9  # 3 neurons * (2 weights + 1 bias)

def test_cycle_detection():
    """Test that backward detects cycles in computation graph."""
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    c = a + b

    # Manually create a cycle: c depends on itself
    c._ctx.saved_tensors = (c,)
    c.out_degree = 1

    try:
        c.backward()
        assert False, "Should have raised RuntimeError for cycle"
    except RuntimeError as e:
        assert "Cycle detected" in str(e)