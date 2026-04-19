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