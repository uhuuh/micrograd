# backward 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 backward 方法，移除 out_degree，改用预处理 use_count；移除 path/visited；简化逻辑

**Architecture:** 移除图追踪相关的动态状态（out_degree），改用 backward 前静态预计算 use_count；移除环检测；统一用 _ctx 判断是否 leaf

**Tech Stack:** Python, numpy, pytest

---

## 任务总览

1. 修改 Tensor 类：移除 out_degree，添加 use_count
2. 修改 Function.apply：移除 out_degree 累加
3. 实现 _compute_use_counts 预处理方法
4. 重构 backward 主流程
5. 更新测试（移除环检测测试）
6. 运行完整测试验证

---

## Task 1: 修改 Tensor 类

**Files:**
- Modify: `micrograd/engine.py:174-183`

- [ ] **Step 1: 移除 out_degree，添加 use_count**

读取当前 engine.py 第 174-183 行（Tensor.__init__），找到：
```python
self.out_degree = 0
```

改为：
```python
self.use_count = 0
```

Run: `grep -n "out_degree" micrograd/engine.py`
Expected: 仅在使用处（待移除），而非定义处

---

## Task 2: 修改 Function.apply

**Files:**
- Modify: `micrograd/engine.py:28-37`

- [ ] **Step 1: 移除 out_degree 累加**

读取当前 engine.py 第 28-37 行（Function.apply），找到：
```python
for t in inputs:
    if isinstance(t, Tensor):
        t.out_degree += 1
```

删除这 3 行。

Run: `grep -n "out_degree" micrograd/engine.py`
Expected: 无输出

---

## Task 3: 实现 _compute_use_counts 预处理方法

**Files:**
- Modify: `micrograd/engine.py:233-292`（backward 方法区域）

- [ ] **Step 1: 在 backward 方法前添加 _compute_use_counts 方法**

在 backward 方法之前插入：

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

Run: `grep -n "_compute_use_counts" micrograd/engine.py`
Expected: 找到方法定义

---

## Task 4: 重构 backward 主流程

**Files:**
- Modify: `micrograd/engine.py:234-292`

- [ ] **Step 1: 读取当前 backward 实现**

当前 backward 实现约 60 行（第 234-292 行），需要完全重写。

- [ ] **Step 2: 替换 backward 方法为新实现**

将整个 backward 方法替换为：

```python
def backward(self):
    """Compute gradient using BFS with static use_count."""
    if self._ctx is None:
        self.grad = Tensor(np.ones_like(self.data), copy=False)
        return

    # 预处理：计算 use_count
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

            # 累加梯度到子节点
            if t.grad is None:
                t.grad = g
            else:
                t.grad = Tensor(t.grad.data + g.data, copy=False)

            # 传播给子节点后 use_count 减一，归零时入队
            t.use_count -= 1
            if t.use_count == 0:
                queue.append(t)

        # 处理完 non-leaf 删除 grad（leaf 保留）
        v.grad = None
```

Run: `python -c "from micrograd.engine import Tensor; print('OK')"`
Expected: 无错误

---

## Task 5: 更新测试

**Files:**
- Modify: `test/test_tensor.py:279-293`

- [ ] **Step 1: 读取并移除环检测测试**

读取 test_tensor.py 第 279-293 行（test_cycle_detection），删除该测试函数。

Run: `grep -n "test_cycle_detection" test/test_tensor.py`
Expected: 无输出

---

## Task 6: 运行完整测试验证

**Files:**
- Test: `test/test_tensor.py`
- Test: `test/test_integration.py`
- Test: `tests/test_leaf_gradient.py`

- [ ] **Step 1: 运行所有测试**

Run: `python -m pytest test/test_tensor.py tests/test_leaf_gradient.py test/test_integration.py -v`
Expected: 所有测试 PASS（除已删除的 test_cycle_detection）

---

## 验证清单

- [ ] `out_degree` 完全移除
- [ ] `path` 和 `visited` 完全移除
- [ ] `use_count` 在 backward 前预计算
- [ ] non-leaf tensor 处理完后 grad 设为 None
- [ ] leaf tensor grad 保留
- [ ] 所有现有测试通过