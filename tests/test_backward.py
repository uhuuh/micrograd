import pytest
import numpy as np
from micrograd.tensor import Tensor

def test_tensor_backward_requires_grad():
    t = Tensor([1.0], requires_grad=False)
    
    with pytest.raises(RuntimeError, match="requires_grad=False"):
        t.backward()

def test_tensor_backward_leaf_error():
    t = Tensor([1.0], requires_grad=True)
    
    with pytest.raises(RuntimeError, match="leaf tensor"):
        t.backward()

def test_tensor_backward_on_output():
    from micrograd.function import Function
    from micrograd.ops import Add
    
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    out = Add.apply(a, b)
    
    out.backward()
    
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_array_almost_equal(a.grad.numpy(), [1.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [1.0])