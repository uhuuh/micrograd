# Tensor Autograd System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar `Value` class with a full `Tensor` class supporting n-dimensional arrays with PyTorch-style autograd using numpy backend.

**Architecture:** Tensor class wraps numpy arrays. All operations are implemented as static `Function` classes in ops.py following PyTorch's autograd.Function pattern. Each op's backward knows how to compute gradients matching the forward operation's shape handling.

**Tech Stack:** Python, numpy

---

## File Structure

```
micrograd/
├── engine.py      # Modify: Replace Value with Tensor class (with ops reference)
├── ops.py         # Create: All autograd Function classes
└── __init__.py    # Modify: Update exports

test/
└── test_tensor.py # Create: Tensor tests
```

---

## Task 1: Create ops.py with base Function and basic operations

**Files:**
- Create: `micrograd/ops.py`

- [ ] **Step 1: Write test for Tensor basic creation and requires_grad**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_tensor_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'micrograd.tensor'"

- [ ] **Step 3: Write minimal Tensor stub in engine.py**

```python
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_tensor_creation -v`
Expected: PASS

- [ ] **Step 5: Write test for Add forward**

```python
def test_add_forward():
    a = Tensor(np.array([1.0, 2.0]))
    b = Tensor(np.array([3.0, 4.0]))
    c = a + b
    assert np.array_equal(c.data, np.array([4.0, 6.0]))
```

- [ ] **Step 6: Run test to verify it fails (no __add__)**

Run: `python -m pytest test/test_tensor.py::test_add_forward -v`
Expected: FAIL with "TypeError: unsupported operand type(s) for +"

- [ ] **Step 7: Write minimal __add__ without autograd**

```python
def __add__(self, other):
    if isinstance(other, Tensor):
        return Tensor(self.data + other.data)
    return Tensor(self.data + other)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_add_forward -v`
Expected: PASS

- [ ] **Step 9: Write test for Add backward**

```python
def test_add_backward():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    c = a + b
    c.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0]))
    assert np.array_equal(b.grad.data, np.array([1.0, 1.0]))
```

- [ ] **Step 10: Run test to verify it fails (no backward, no ctx)**

Run: `python -m pytest test/test_tensor.py::test_add_backward -v`
Expected: FAIL

- [ ] **Step 11: Write ops.py with Function base class and Add op**

```python
import numpy as np

class Function:
    """Base class for autograd operations. Subclass with forward/backward static methods."""

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        """Apply the operation: creates ctx, runs forward, attaches grad_fn to output."""
        ctx = cls()
        output = cls.forward(ctx, *inputs)
        output._ctx = ctx
        ctx._grad_fn = cls
        return output


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.data * np.ones_like(a.data)
        grad_b = grad_output.data * np.ones_like(b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Context:
    """Storage for op execution context, used to save tensors for backward."""
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
```

- [ ] **Step 12: Update Tensor class to use ops**

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other)
```

- [ ] **Step 13: Implement backward() on Tensor**

```python
def backward(self):
    """Compute gradient of this tensor with respect to leaf tensors."""
    if self._ctx is None:
        # This tensor is a leaf (no ops applied), just set grad to 1
        self.grad = Tensor(np.ones_like(self.data))
        return

    # Build topological order of the computation graph
    topo = []
    visited = set()
    def build_topo(v):
        if id(v) not in visited:
            visited.add(id(v))
            if v._ctx is not None:
                for child in v._ctx.saved_tensors:
                    build_topo(child)
            topo.append(v)
    build_topo(self)

    # Initialize gradient at output = 1
    self.grad = Tensor(np.ones_like(self.data))
    grad_table = {id(self): self.grad}

    # Process in reverse topological order
    for v in reversed(topo):
        if v._ctx is not None:
            grad_fn = v._ctx._grad_fn
            inputs = v._ctx.saved_tensors
            grad_output = grad_table.get(id(v), Tensor(np.zeros_like(v.data)))

            # Call the op's backward
            grads = grad_fn.backward(v._ctx, grad_output)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # Accumulate gradients for each input
            for inp, g in zip(inputs, grads):
                if inp.requires_grad and g is not None:
                    if id(inp) in grad_table:
                        grad_table[id(inp)].data += g.data
                    else:
                        grad_table[id(inp)] = g
```

- [ ] **Step 14: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_add_backward -v`
Expected: PASS

- [ ] **Step 16: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_add_backward -v`
Expected: PASS

- [ ] **Step 17: Write test for Mul backward**

```python
def test_mul_backward():
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = a * b
    c.backward()
    assert np.array_equal(a.grad.data, np.array([4.0, 5.0]))
    assert np.array_equal(b.grad.data, np.array([2.0, 3.0]))
```

- [ ] **Step 18: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_mul_backward -v`
Expected: FAIL (no __mul__)

- [ ] **Step 19: Add __mul__ to Tensor using Mul.apply**

```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    return Mul.apply(self, other)
```

- [ ] **Step 20: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_mul_backward -v`
Expected: PASS

- [ ] **Step 21: Add all remaining element-wise ops to ops.py**

```python
class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, Tensor(-grad_output.data)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.data / b.data
        grad_b = -a.data * grad_output.data / (b.data ** 2)
        return Tensor(grad_a), Tensor(grad_b)

class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return Tensor(-a.data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return Tensor(-grad_output.data)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return Tensor(grad_output.data * (a.data > 0).astype(float))

class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a, exponent)
        return Tensor(a.data ** exponent, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, exponent = ctx.saved_tensors
        return Tensor(exponent * (a.data ** (exponent - 1)) * grad_output.data), None
```

- [ ] **Step 22: Write test for ReLU backward**

```python
def test_relu_backward():
    a = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]), requires_grad=True)
    b = a.relu()
    b.backward()
    assert np.array_equal(a.grad.data, np.array([0.0, 1.0, 0.0, 1.0]))
```

- [ ] **Step 23: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_relu_backward -v`
Expected: PASS

- [ ] **Step 24: Commit**

```bash
git add micrograd/ops.py test/test_tensor.py
git commit -m "feat: add ops.py with Function base class and element-wise ops"
```

---

## Task 2: Add MatMul operation

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write test for MatMul forward**

```python
def test_matmul_forward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    c = a @ b
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.array_equal(c.data, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_matmul_forward -v`
Expected: FAIL

- [ ] **Step 3: Write MatMul class in ops.py**

```python
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.data @ b.data.T
        grad_b = a.data.T @ grad_output.data
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 4: Add __matmul__ to Tensor**

```python
def __matmul__(self, other):
    return MatMul.apply(self, other)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_matmul_forward -v`
Expected: PASS

- [ ] **Step 6: Write test for MatMul backward**

```python
def test_matmul_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    c = a @ b
    c.backward()
    expected_grad_a = np.array([[5.0, 7.0], [6.0, 8.0]])  # b.T
    expected_grad_b = np.array([[1.0, 3.0], [2.0, 4.0]])  # a.T
    assert np.array_equal(a.grad.data, expected_grad_a)
    assert np.array_equal(b.grad.data, expected_grad_b)
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_matmul_backward -v`
Expected: FAIL

- [ ] **Step 8: Fix MatMul backward (need to sum along correct axes)**

```python
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # For 2D: grad_a = grad @ b.T, grad_b = a.T @ grad
        # But need to handle if a or b had extra dimensions
        grad_a = np.sum(grad_output.data * b.data.T, axis=1) if a.data.ndim == 1 else grad_output.data @ b.data.T
        grad_b = np.sum(a.data.T * grad_output.data, axis=1) if b.data.ndim == 1 else a.data.T @ grad_output.data
        return Tensor(grad_a), Tensor(grad_b)
```

Actually, simpler:
```python
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # grad_a = grad @ b.T
        # grad_b = a.T @ grad
        grad_a = grad_output.data @ b.data.swapaxes(-1, -2)
        grad_b = a.data.swapaxes(-1, -2) @ grad_output.data
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_matmul_backward -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add micrograd/ops.py
git commit -m "feat: add MatMul operation"
```

---

## Task 3: Add shape operations (Sum, Reshape, Transpose)

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write test for Sum forward and backward**

```python
def test_sum_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = a.sum()
    b.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0, 1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_sum_backward -v`
Expected: FAIL

- [ ] **Step 3: Write Sum class in ops.py**

```python
class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a, dim, keepdim)
        out = np.sum(a.data, axis=dim, keepdims=keepdim)
        return Tensor(out, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, dim, keepdim = ctx.saved_tensors
        shape = list(a.data.shape)
        if dim is None:
            return Tensor(np.ones_like(a.data) * grad_output.data), None, None
        shape[dim] = 1
        grad_a = np.ones(shape) * grad_output.data
        if not keepdim:
            grad_a = np.squeeze(grad_a, axis=dim)
        return Tensor(grad_a), None, None
```

- [ ] **Step 4: Add sum method to Tensor**

```python
def sum(self, dim=None, keepdim=False):
    return Sum.apply(self, dim, keepdim)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_sum_backward -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add micrograd/ops.py
git commit -m "feat: add Sum operation"
```

---

## Task 4: Add broadcasted element-wise operations

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write test for broadcasted Add backward**

```python
def test_broadcast_add_backward():
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0]), requires_grad=True)  # broadcasts to [4.0, 4.0, 4.0]
    c = a + b
    assert np.array_equal(c.data, np.array([5.0, 6.0, 7.0]))
    c.backward()
    assert np.array_equal(a.grad.data, np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(b.grad.data, np.array([3.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_broadcast_add_backward -v`
Expected: FAIL

- [ ] **Step 3: Add BroadcastTo helper function and update Add**

```python
def _broadcast_to(a, b):
    """Broadcast a to match b's shape, return (a_expanded, grad_fn)"""
    if a.data.shape == b.data.shape:
        return a, lambda x: x
    # Compute output shape
    shape = np.broadcast_shapes(a.data.shape, b.data.shape)
    # Expand a
    a_expanded = np.broadcast_to(a.data, shape)
    # For gradient: sum over dims that were broadcast
    broadcast_axes = [i for i, (ua, ub) in enumerate(zip(a.data.shape, (1,)*len(a.data.shape) + b.data.shape[:len(a.data.shape)]))) if ua == 1 and ub > 1]
    # simpler: sum over dims where a.shape[i] == 1
    return Tensor(a_expanded, requires_grad=a.requires_grad), broadcast_axes

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # Broadcast both to common shape
        broad_a = np.broadcast_to(a.data, np.broadcast_shapes(a.data.shape, b.data.shape))
        broad_b = np.broadcast_to(b.data, np.broadcast_shapes(a.data.shape, b.data.shape))
        return Tensor(broad_a + broad_b, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        out_shape = np.broadcast_shapes(a.data.shape, b.data.shape)
        grad_a = np.sum(grad_output.data, axis=[i for i in range(len(out_shape)) if a.data.shape[i] == 1 and out_shape[i] > 1])
        grad_b = np.sum(grad_output.data, axis=[i for i in range(len(out_shape)) if b.data.shape[i] == 1 and out_shape[i] > 1])
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_broadcast_add_backward -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add micrograd/ops.py
git commit -m "feat: add broadcasting support to element-wise ops"
```

---

## Task 5: Add rsub, rtruediv and other reflected operations

**Files:**
- Modify: `micrograd/engine.py`

- [ ] **Step 1: Write test for reflected operations**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_tensor.py::test_reflected_operations -v`
Expected: FAIL

- [ ] **Step 3: Add __rsub__ and __rtruediv__ to Tensor**

```python
def __rsub__(self, other):
    return Tensor(other) - self

def __rtruediv__(self, other):
    return Tensor(other) / self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_tensor.py::test_reflected_operations -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/engine.py
git commit -m "feat: add reflected arithmetic operations"
```

---

## Task 6: Full PyTorch gradient validation tests

**Files:**
- Modify: `test/test_tensor.py`

- [ ] **Step 1: Write comprehensive PyTorch validation test**

```python
import torch
import numpy as np
from micrograd.tensor import Tensor

def test_tensor_ops_match_pytorch():
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
    x_np = np.random.randn(4, 5).astype(np.float64)
    x_t = Tensor(x_np, requires_grad=True)
    x_th = torch.tensor(x_np, requires_grad=True)

    y_t = x_t.relu()
    y_th = torch.relu(x_th)

    assert np.allclose(y_t.data, y_th.detach().numpy())

    y_t.sum().backward()
    y_th.sum().backward()
    assert np.allclose(x_t.grad.data, x_th.grad.numpy())
```

- [ ] **Step 2: Run test**

Run: `python -m pytest test/test_tensor.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add test/test_tensor.py
git commit -m "test: add comprehensive PyTorch gradient validation tests"
```

---

## Task 7: Update nn.py to work with Tensor

**Files:**
- Modify: `micrograd/nn.py`

- [ ] **Step 1: Update nn.py to use Tensor instead of Value**

Replace all `Value` with `Tensor`, adjust weight initialization to use proper tensor shapes.

- [ ] **Step 2: Test MLP training still works**

Run demo.ipynb or write a simple training loop test.

- [ ] **Step 3: Commit**

```bash
git add micrograd/nn.py
git commit -m "feat: update nn.py to use Tensor class"
```

---

## Verification Checklist

- [ ] All ops (Add, Sub, Mul, Div, Neg, ReLU, Pow, MatMul, Sum) implemented
- [ ] Broadcasting works correctly for element-wise ops
- [ ] Gradients match PyTorch for all operations
- [ ] nn.py works with new Tensor class
- [ ] All existing tests pass
- [ ] No placeholder/TODO comments in code
