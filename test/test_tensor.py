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

def test_sum_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = a.sum()
    b.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0, 1.0]))

def test_broadcast_add_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0]), requires_grad=True)  # broadcasts to [4.0, 4.0, 4.0]
    c = a + b
    assert np.array_equal(c.data, np.array([5.0, 6.0, 7.0]))
    c.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(b.grad.data, np.array([3.0]))

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