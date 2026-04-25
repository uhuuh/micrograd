import pytest
from micrograd import Tensor

def test_leaf_tensor_accumulates_gradient():
    """Leaf tensor (requires_grad=True, _ctx=None) should accumulate gradient."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()

    assert x.grad is not None
    assert x.grad.numpy().tolist() == [2.0, 2.0, 2.0]

def test_non_leaf_tensor_no_gradient():
    """Non-leaf tensor (operation output) should not store gradient."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()

    assert y.grad is None
    assert z.grad is None

def test_leaf_without_requires_grad_no_gradient():
    """Calling backward on tensor without requires_grad should raise RuntimeError."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
    y = x * 2
    z = y.sum()
    
    with pytest.raises(RuntimeError, match="requires_grad"):
        z.backward()
    
    assert x.grad is None

def test_shared_input_gradient_accumulation():
    """When multiple outputs use same leaf input, gradients should accumulate."""
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = x * 4
    w = y + z
    w.backward()

    assert x.grad is not None
    assert x.grad.numpy().tolist() == [7.0]  # 3 + 4