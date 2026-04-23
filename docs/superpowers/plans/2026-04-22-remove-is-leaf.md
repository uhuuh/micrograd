# 移除 is_leaf 属性实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除 `is_leaf` 属性，backward 时自动判断 leaf tensor 并仅对 leaf tensor 累加梯度

**Architecture:** 在 backward 的梯度传播循环中判断 `requires_grad and _ctx is None`，满足条件才累加梯度

**Tech Stack:** Python, NumPy

---

### Task 1: 移除 Tensor.is_leaf 属性

**Files:**
- Modify: `micrograd/engine.py:184-185`

- [ ] **Step 1: 删除 Tensor.__init__ 中的 is_leaf 属性**

修改 `Tensor.__init__`，删除 `self.is_leaf = True` 行：

```python
def __init__(self, data, requires_grad=False, copy=True):
    if copy:
        self.data = np.array(data, dtype=np.float64)
    else:
        self.data = np.asarray(data, dtype=np.float64)
    self.requires_grad = requires_grad
    self.grad = None
    self._ctx = None
    self.out_degree = 0
```

- [ ] **Step 2: 运行现有测试验证**

Run: `pytest tests/ -v`
Expected: PASS（如果现有测试通过）或 FAIL（需要后续修复）

- [ ] **Step 3: 暂不提交（等待 Task 2 完成）**

---

### Task 2: 移除 Function.apply 中设置 is_leaf

**Files:**
- Modify: `micrograd/engine.py:33`

- [ ] **Step 1: 删除 Function.apply 中设置 is_leaf=False**

修改 `Function.apply`，删除 `output.is_leaf = False` 行：

```python
@classmethod
def apply(cls, *inputs):
    """Apply the operation: creates ctx, runs forward, attaches grad_fn to output."""
    ctx = cls()
    output = cls.forward(ctx, *inputs)
    output._ctx = ctx
    ctx._grad_fn = cls
    for t in inputs:
        if isinstance(t, Tensor):
            t.out_degree += 1
    return output
```

- [ ] **Step 2: 运行现有测试验证**

Run: `pytest tests/ -v`
Expected: FAIL（backward 逻辑需要修复）

---

### Task 3: 修改 backward 梯度累加逻辑

**Files:**
- Modify: `micrograd/engine.py` backward 方法中的梯度累加部分

- [ ] **Step 1: 修改 backward 中的梯度累加条件**

在 backward 方法的 BFS 循环中，修改梯度累加逻辑，只对 leaf tensor 累加：

```python
for t, g in zip(v._ctx.saved_tensors, grads):
    if g is not None and isinstance(t, Tensor):
        # 只对 leaf tensor 累加梯度（requires_grad=True 且 _ctx=None）
        if t.requires_grad and t._ctx is None:
            if t.grad is None:
                t.grad = g
            else:
                t.grad = Tensor(t.grad.data + g.data, copy=False)

        # 继续传播（无论是否 leaf）
        num_outputs[id(t)] -= 1
        if num_outputs[id(t)] == 0:
            queue.append(t)
```

- [ ] **Step 2: 运行现有测试验证**

Run: `pytest tests/ -v`
Expected: PASS

---

### Task 4: 添加 leaf tensor 梯度测试

**Files:**
- Create: `tests/test_leaf_gradient.py`

- [ ] **Step 1: 编写测试验证 leaf tensor 梯度行为**

```python
import pytest
from micrograd.engine import Tensor

def test_leaf_tensor_accumulates_gradient():
    """Leaf tensor (requires_grad=True, _ctx=None) should accumulate gradient."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()

    assert x.grad is not None
    assert x.grad.data.tolist() == [2.0, 2.0, 2.0]

def test_non_leaf_tensor_no_gradient():
    """Non-leaf tensor (operation output) should not store gradient."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()

    assert y.grad is None
    assert z.grad is None

def test_leaf_without_requires_grad_no_gradient():
    """Leaf tensor without requires_grad should not accumulate gradient."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
    y = x * 2
    z = y.sum()
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
    assert x.grad.data.tolist() == [7.0]  # 3 + 4
```

- [ ] **Step 2: 运行测试验证**

Run: `pytest tests/test_leaf_gradient.py -v`
Expected: PASS

- [ ] **Step 3: 提交所有改动**

```bash
git add micrograd/engine.py tests/test_leaf_gradient.py
git commit -m "refactor: remove is_leaf, auto-detect leaf tensors in backward"
```