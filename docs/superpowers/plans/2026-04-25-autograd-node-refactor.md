# Autograd Node Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor autograd to use unified Node class for computation graph and backward propagation.

**Architecture:** Single Node class stores graph structure (prev/next/leaf/saved_tensors) and forward/backward logic. Tensor only stores data/grad, grad_fn points to Node. BFS backward with automatic graph cleanup.

**Tech Stack:** Python, NumPy, pytest

---

## File Structure

| File | Responsibility |
|------|----------------|
| `micrograd/node.py` | Unified Node class (new) |
| `micrograd/no_grad.py` | no_grad decorator/context manager (new) |
| `micrograd/tensor.py` | Tensor class, remove _ctx/_use_count, add grad_fn |
| `micrograd/ops.py` | All ops extend Node instead of Function |
| `micrograd/__init__.py` | Export Node, no_grad |
| `micrograd/function.py` | Delete (merged into Node) |
| `tests/test_node.py` | Tests for new Node class (new) |
| `tests/test_backward.py` | Tests for backward with cleanup (new) |

---

### Task 1: Create Node class base

**Files:**
- Create: `micrograd/node.py`
- Create: `tests/test_node.py`

- [ ] **Step 1: Write failing test for Node initialization**

```python
# tests/test_node.py
import pytest
from micrograd.node import Node

def test_node_initialization():
    node = Node()
    assert node.saved_tensors == ()
    assert node.saved_data == []
    assert node.prev == set()
    assert node.next == set()
    assert node.leaf == set()

def test_node_save_for_backward():
    from micrograd.tensor import Tensor
    node = Node()
    t1 = Tensor([1.0])
    t2 = Tensor([2.0])
    node.save_for_backward(t1, t2)
    assert node.saved_tensors == (t1, t2)

def test_node_save_data_for_backward():
    node = Node()
    node.save_data_for_backward(2, 3)
    assert node.saved_data == [2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'micrograd.node'"

- [ ] **Step 3: Create Node class with base structure**

```python
# micrograd/node.py
from collections import deque

class Node:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = []
        self.prev: set[Node] = set()
        self.next: set[Node] = set()
        self.leaf: set['Tensor'] = set()
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    
    def save_data_for_backward(self, *data):
        self.saved_data = list(data)
    
    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify base tests pass**

Run: `python -m pytest tests/test_node.py::test_node_initialization tests/test_node.py::test_node_save_for_backward tests/test_node.py::test_node_save_data_for_backward -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/node.py tests/test_node.py
git commit -m "feat: create Node class with base structure"
```

---

### Task 2: Implement Node.apply and backward

**Files:**
- Modify: `micrograd/node.py`
- Modify: `tests/test_node.py`

- [ ] **Step 1: Write failing test for Node.apply**

```python
# tests/test_node.py (add to existing file)
import numpy as np
from micrograd.tensor import Tensor

class MockAdd(Node):
    @staticmethod
    def forward(ctx, a, b):
        return a.data + b.data
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

def test_node_apply_creates_output():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    
    out = MockAdd.apply(a, b)
    
    assert isinstance(out, Tensor)
    assert np.allclose(out.data.numpy(), [3.0])
    assert out.requires_grad == True
    assert out.grad_fn is not None
    assert isinstance(out.grad_fn, MockAdd)

def test_node_apply_builds_graph():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    
    out = MockAdd.apply(a, b)
    
    # Check graph connections
    assert a in out.grad_fn.leaf
    assert b in out.grad_fn.leaf
    assert out.grad_fn.prev == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_node_apply_creates_output -v`
Expected: FAIL with "NotImplementedError" or similar

- [ ] **Step 3: Implement Node.apply**

```python
# micrograd/node.py (replace apply method)
from . import no_grad

@classmethod
def apply(cls, *inputs):
    ctx = cls()
    
    needs_grad = any(t.requires_grad for t in inputs if isinstance(t, 'Tensor')) and not no_grad.enabled
    
    # Import Tensor locally to avoid circular dependency
    from .tensor import Tensor
    
    output_storage = cls.forward(ctx, *inputs)
    output = Tensor(output_storage, requires_grad=needs_grad)
    
    if needs_grad:
        output.grad_fn = ctx
        
        for t in inputs:
            if not isinstance(t, Tensor) or not t.requires_grad:
                continue
            if t.grad_fn is None:
                ctx.leaf.add(t)
            else:
                ctx.prev.add(t.grad_fn)
                t.grad_fn.next.add(ctx)
    
    return output
```

- [ ] **Step 4: Create stub no_grad module**

```python
# micrograd/no_grad.py
class no_grad:
    enabled = False
    
    def __init__(self):
        self._prev = None
    
    def __enter__(self):
        self._prev = no_grad.enabled
        no_grad.enabled = True
        return self
    
    def __exit__(self, *args):
        no_grad.enabled = self._prev
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
```

- [ ] **Step 5: Run test to verify apply tests pass**

Run: `python -m pytest tests/test_node.py::test_node_apply_creates_output tests/test_node.py::test_node_apply_builds_graph -v`
Expected: PASS

- [ ] **Step 6: Write failing test for Node.backward**

```python
# tests/test_node.py (add)
def test_node_backward_accumulates_grad():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    
    out = MockAdd.apply(a, b)
    out.backward()
    
    assert a.grad is not None
    assert np.allclose(a.grad.data.numpy(), [1.0])
    assert b.grad is not None
    assert np.allclose(b.grad.data.numpy(), [1.0])

def test_node_backward_clears_graph():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    
    out = MockAdd.apply(a, b)
    out.backward()
    
    # Graph should be cleared
    assert a.grad_fn is None
    assert b.grad_fn is None
    assert out.grad_fn.prev == set()
    assert out.grad_fn.leaf == set()
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_node_backward_accumulates_grad -v`
Expected: FAIL with "NotImplementedError"

- [ ] **Step 8: Implement Node.backward**

```python
# micrograd/node.py (replace backward method)
def backward(self, grad_output):
    from .tensor import Tensor
    
    queue = deque([(self, grad_output)])
    
    while queue:
        node, grad = queue.popleft()
        
        grads = type(node).backward(node, grad)
        
        for i, t in enumerate(node.saved_tensors):
            if not t.requires_grad:
                continue
            g = grads[i] if i < len(grads) else None
            if g is None:
                continue
            
            if t.grad_fn is None:
                if t.grad is None:
                    t.grad = g
                else:
                    t.grad = Tensor(t.grad.data.numpy() + g.data.numpy(), copy=False)
            else:
                t.grad_fn.next.discard(node)
                if len(t.grad_fn.next) == 0:
                    queue.append((t.grad_fn, g))
        
        node.prev.clear()
        node.leaf.clear()
        node.saved_tensors = ()
```

- [ ] **Step 9: Run test to verify backward tests pass**

Run: `python -m pytest tests/test_node.py::test_node_backward_accumulates_grad tests/test_node.py::test_node_backward_clears_graph -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add micrograd/node.py micrograd/no_grad.py tests/test_node.py
git commit -m "feat: implement Node.apply and backward with graph cleanup"
```

---

### Task 3: Update Tensor class

**Files:**
- Modify: `micrograd/tensor.py`

- [ ] **Step 1: Write failing test for new Tensor.backward**

```python
# tests/test_backward.py (new file)
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
    from micrograd.node import Node
    
    class SimpleAdd(Node):
        @staticmethod
        def forward(ctx, a, b):
            return a.data + b.data
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output
    
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    out = SimpleAdd.apply(a, b)
    
    out.backward()
    
    assert a.grad is not None
    assert b.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backward.py -v`
Expected: FAIL (current Tensor.backward doesn't check grad_fn)

- [ ] **Step 3: Update Tensor class**

```python
# micrograd/tensor.py (modify __init__)
def __init__(self, data, requires_grad=False, device="cpu", copy=True):
    if isinstance(data, Storage):
        self._storage = data
    elif device == "cpu":
        if copy:
            self._storage = CPUStorage(np.array(data, dtype=np.float64))
        else:
            self._storage = CPUStorage(np.asarray(data, dtype=np.float64))
    else:
        raise ValueError(f"Unknown device: {device}")
    
    self.requires_grad = requires_grad
    self.grad = None
    self.grad_fn = None  # NEW: points to Node or None
    # DELETE: self._ctx = None
    # DELETE: self._use_count = 0

# micrograd/tensor.py (replace backward method)
def backward(self):
    if not self.requires_grad:
        raise RuntimeError("backward() called on tensor with requires_grad=False")
    
    if self.grad_fn is None:
        raise RuntimeError("backward() called on leaf tensor. "
                          "Call backward on output of computation.")
    
    grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)
    self.grad_fn.backward(grad)
    self.grad_fn = None

# micrograd/tensor.py (DELETE _compute_use_counts method)
```

- [ ] **Step 4: Run test to verify passes**

Run: `python -m pytest tests/test_backward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/tensor.py tests/test_backward.py
git commit -m "refactor: update Tensor class, remove _ctx/_use_count, add grad_fn"
```

---

### Task 4: Update Add/Sub/Div/Neg ops (backward doesn't need inputs)

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write failing test for Add with new Node**

```python
# tests/test_node.py (add)
from micrograd.ops import Add

def test_add_forward_no_save():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    
    out = Add.apply(a, b)
    
    assert np.allclose(out.data.numpy(), [4.0, 6.0])
    assert out.grad_fn.saved_tensors == ()  # Add doesn't need inputs

def test_add_backward_no_saved_tensors():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    out = Add.apply(a, b)
    out.backward()
    
    assert np.allclose(a.grad.data.numpy(), [1.0, 1.0])
    assert np.allclose(b.grad.data.numpy(), [1.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_add_forward_no_save -v`
Expected: FAIL (Add currently uses Function from function.py)

- [ ] **Step 3: Update Add op to extend Node**

```python
# micrograd/ops.py (replace Add class)
from .node import Node

class Add(Node):
    @staticmethod
    def forward(ctx, a, b):
        forward_fn, _ = registry.dispatch("add", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
```

- [ ] **Step 4: Update Sub op**

```python
# micrograd/ops.py (replace Sub class)
class Sub(Node):
    @staticmethod
    def forward(ctx, a, b):
        forward_fn, _ = registry.dispatch("sub", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, Tensor(-grad_output.data.numpy(), copy=False)
```

- [ ] **Step 5: Update Div op**

```python
# micrograd/ops.py (replace Div class)
class Div(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("div", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("div", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 6: Update Neg op**

```python
# micrograd/ops.py (replace Neg class)
class Neg(Node):
    @staticmethod
    def forward(ctx, a):
        forward_fn, _ = registry.dispatch("neg", a.device)
        return forward_fn(a.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return Tensor(-grad_output.data.numpy(), copy=False)
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_node.py::test_add_forward_no_save tests/test_node.py::test_add_backward_no_saved_tensors -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add micrograd/ops.py tests/test_node.py
git commit -m "refactor: update Add/Sub/Div/Neg ops to extend Node"
```

---

### Task 5: Update Mul/MatMul ops (backward needs inputs)

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write failing test for Mul with saved tensors**

```python
# tests/test_node.py (add)
from micrograd.ops import Mul

def test_mul_forward_saves_tensors():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    
    out = Mul.apply(a, b)
    
    assert np.allclose(out.data.numpy(), [8.0, 15.0])
    assert a in out.grad_fn.saved_tensors
    assert b in out.grad_fn.saved_tensors

def test_mul_backward_uses_saved():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    
    out = Mul.apply(a, b)
    out.backward()
    
    assert np.allclose(a.grad.data.numpy(), [4.0, 5.0])  # grad_a = b * grad
    assert np.allclose(b.grad.data.numpy(), [2.0, 3.0])  # grad_b = a * grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_mul_forward_saves_tensors -v`
Expected: FAIL (Mul currently uses Function)

- [ ] **Step 3: Update Mul op**

```python
# micrograd/ops.py (replace Mul class)
class Mul(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("mul", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("mul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 4: Update MatMul op**

```python
# micrograd/ops.py (replace MatMul class)
class MatMul(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("matmul", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("matmul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_node.py::test_mul_forward_saves_tensors tests/test_node.py::test_mul_backward_uses_saved -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add micrograd/ops.py tests/test_node.py
git commit -m "refactor: update Mul/MatMul ops to extend Node with saved tensors"
```

---

### Task 6: Update ReLU/Pow ops

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Write failing test for ReLU**

```python
# tests/test_node.py (add)
from micrograd.ops import ReLU

def test_relu_forward_saves_input():
    a = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    
    out = ReLU.apply(a)
    
    assert np.allclose(out.data.numpy(), [0.0, 2.0, 0.0, 4.0])
    assert a in out.grad_fn.saved_tensors

def test_relu_backward_mask():
    a = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    
    out = ReLU.apply(a)
    out.backward()
    
    assert np.allclose(a.grad.data.numpy(), [0.0, 1.0, 0.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_relu_forward_saves_input -v`
Expected: FAIL

- [ ] **Step 3: Update ReLU op**

```python
# micrograd/ops.py (replace ReLU class)
class ReLU(Node):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("relu", a.device)
        return forward_fn(a.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        _, backward_fn = registry.dispatch("relu", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data)
        return Tensor(grad_a)
```

- [ ] **Step 4: Write failing test for Pow**

```python
# tests/test_node.py (add)
from micrograd.ops import Pow

def test_pow_forward_saves_tensor_only():
    a = Tensor([2.0, 3.0], requires_grad=True)
    
    out = Pow.apply(a, 2)
    
    assert np.allclose(out.data.numpy(), [4.0, 9.0])
    assert a in out.grad_fn.saved_tensors
    assert out.grad_fn.saved_data == [2]  # exponent in saved_data

def test_pow_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    
    out = Pow.apply(a, 2)
    out.backward()
    
    # grad = 2 * a^(2-1) = 2 * a
    assert np.allclose(a.grad.data.numpy(), [4.0, 6.0])
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_node.py::test_pow_forward_saves_tensor_only -v`
Expected: FAIL

- [ ] **Step 6: Update Pow op**

```python
# micrograd/ops.py (replace Pow class)
class Pow(Node):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(exponent)
        forward_fn, _ = registry.dispatch("pow", a.device)
        return forward_fn(a.data, exponent)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        exponent = ctx.saved_data[0]
        _, backward_fn = registry.dispatch("pow", grad_output.device)
        grad_a, _ = backward_fn(grad_output.data, a.data, exponent)
        return Tensor(grad_a)
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_node.py::test_relu_forward_saves_input tests/test_node.py::test_pow_forward_saves_tensor_only -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add micrograd/ops.py tests/test_node.py
git commit -m "refactor: update ReLU/Pow ops to extend Node"
```

---

### Task 7: Update Sum/Slice ops

**Files:**
- Modify: `micrograd/ops.py`

- [ ] **Step 1: Update Sum op**

```python
# micrograd/ops.py (replace Sum class)
class Sum(Node):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(dim, keepdim)
        forward_fn, _ = registry.dispatch("sum", a.device)
        return forward_fn(a.data, dim, keepdim)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        dim, keepdim = ctx.saved_data
        _, backward_fn = registry.dispatch("sum", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, dim, keepdim)
        return Tensor(grad_a)
```

- [ ] **Step 2: Update Slice op**

```python
# micrograd/ops.py (replace Slice class)
class Slice(Node):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(key)
        forward_fn, _ = registry.dispatch("slice", a.device)
        return forward_fn(a.data, key)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        key = ctx.saved_data[0]
        _, backward_fn = registry.dispatch("slice", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, key)
        return Tensor(grad_a)
```

- [ ] **Step 3: Remove old Function import**

```python
# micrograd/ops.py (modify imports)
from .node import Node
from .dispatch import registry
# DELETE: from .function import Function
```

- [ ] **Step 4: Run existing tensor tests**

Run: `python -m pytest tests/test_leaf_gradient.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add micrograd/ops.py
git commit -m "refactor: update Sum/Slice ops to extend Node"
```

---

### Task 8: Update tensor.py operator methods

**Files:**
- Modify: `micrograd/tensor.py`

- [ ] **Step 1: Update operator methods to use new ops**

```python
# micrograd/tensor.py (update imports at end of file)
from .ops import Add, Mul, Sub, Div, Neg, ReLU, Pow, MatMul, Sum, Slice
```

No changes needed to operator methods (__add__, __mul__, etc.) since they already call Add.apply, Mul.apply, etc.

- [ ] **Step 2: Verify existing tests pass**

Run: `python -m pytest tests/ -v`
Expected: Most PASS (some may need import updates)

- [ ] **Step 3: Commit**

```bash
git add micrograd/tensor.py
git commit -m "refactor: tensor operators use Node-based ops"
```

---

### Task 9: Update __init__.py exports

**Files:**
- Modify: `micrograd/__init__.py`

- [ ] **Step 1: Update exports**

```python
# micrograd/__init__.py
from .tensor import Tensor
from .node import Node
from .no_grad import no_grad
from .dispatch import registry
from .storage import Storage, CPUStorage
from . import ops
from . import kernels

__all__ = ["Tensor", "Node", "no_grad", "registry", "Storage", "CPUStorage", "ops", "kernels"]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from micrograd import Tensor, Node, no_grad; print('imports OK')"`
Expected: "imports OK"

- [ ] **Step 3: Commit**

```bash
git add micrograd/__init__.py
git commit -m "refactor: export Node and no_grad from __init__.py"
```

---

### Task 10: Delete function.py

**Files:**
- Delete: `micrograd/function.py`

- [ ] **Step 1: Delete function.py**

```bash
rm micrograd/function.py
```

- [ ] **Step 2: Verify nothing references function.py**

Run: `grep -r "from .function import" micrograd/`
Expected: No results

- [ ] **Step 3: Commit**

```bash
git rm micrograd/function.py
git commit -m "refactor: delete function.py (merged into Node)"
```

---

### Task 11: Update test imports

**Files:**
- Modify: `tests/test_leaf_gradient.py`
- Modify: `test/test_tensor.py`
- Modify: `test/test_optim.py`
- Modify: `test/test_lenet5.py`
- Modify: `test/test_integration.py`

- [ ] **Step 1: Fix test imports (replace micrograd.engine with micrograd.tensor)**

```python
# tests/test_leaf_gradient.py (already correct)
from micrograd import Tensor

# test/test_tensor.py
from micrograd.tensor import Tensor  # change from micrograd.engine

# test/test_optim.py
from micrograd.tensor import Tensor  # change from micrograd.engine

# test/test_lenet5.py
from micrograd.tensor import Tensor  # change from micrograd.engine

# test/test_integration.py
from micrograd.tensor import Tensor  # change from micrograd.engine
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ test/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/ test/
git commit -m "fix: update test imports to use micrograd.tensor"
```

---

### Task 12: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run train_mnist.py import check**

Run: `python -c "from train_mnist import *"`
Expected: No import errors

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "refactor: complete autograd Node refactor"
```