# backward 重构设计

## 目标

1. 移除 `out_degree`，改用 `use_count`（预处理时静态计算）
2. 移除 `path` 和 `visited`（环检测功能）
3. 重构 backward，精简逻辑

## 变更

### 1. Tensor 类

```python
class Tensor:
    def __init__(self, ...):
        # 移除 out_degree
        # 添加 use_count
        self.use_count = 0
```

### 2. Function.apply

移除 `t.out_degree += 1`。

### 3. backward 预处理

一次 BFS/DFS，从 root 出发，统计每个 tensor 的 `use_count`：

```python
def _compute_use_counts(self):
    """预处理：从 root 开始，统计每个 tensor 被多少子节点依赖。"""
    queue = deque([self])
    visited = set()
    while queue:
        t = queue.popleft()
        if id(t) in visited:
            continue
        visited.add(id(t))
        if t._ctx is not None:
            for child in t._ctx.saved_tensors:
                if isinstance(child, Tensor):
                    child.use_count += 1
                    if id(child) not in visited:
                        queue.append(child)
```

### 4. backward 主流程

```python
def backward(self):
    if self._ctx is None:
        self.grad = Tensor(np.ones_like(self.data), copy=False)
        return

    # 预处理
    self._compute_use_counts()

    queue = deque([self])
    self.grad = Tensor(np.ones_like(self.data), copy=False)

    while queue:
        v = queue.popleft()

        if v._ctx is None:
            continue  # leaf tensor，不传播

        grads = v._ctx._grad_fn.backward(v._ctx, v.grad)

        for t, g in zip(v._ctx.saved_tensors, grads):
            if g is None or not isinstance(t, Tensor):
                continue

            # 累加梯度
            if t.grad is None:
                t.grad = g
            else:
                t.grad = Tensor(t.grad.data + g.data, copy=False)

            # 传播给子节点后 use_count 减一
            t.use_count -= 1
            if t.use_count == 0:
                queue.append(t)

        # 处理完 non-leaf 删除 grad（leaf 保留）
        v.grad = None
```

## 规则

- `_ctx is None`（leaf）：保留 grad
- `_ctx is not None`（non-leaf）：处理完后删除 grad

## 测试

确保现有测试通过：
- `test_leaf_gradient.py` 验证 leaf tensor 梯度正确
- `test_tensor.py` 验证梯度计算正确
- `test_cycle_detection` 需要更新（移除环检测）