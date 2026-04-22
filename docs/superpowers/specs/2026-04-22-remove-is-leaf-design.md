# 移除 is_leaf 属性，自动判断 leaf tensor

## 背景

当前 `Tensor` 类有 `is_leaf` 属性，在 `Function.apply` 中显式设置为 `False`。这是冗余设计，因为 leaf tensor 可以通过已有属性自动判断。

## 目标

- 移除 `is_leaf` 属性
- backward 时自动判断 leaf tensor 并累加梯度
- 非 leaf tensor 不存储梯度

## Leaf Tensor 定义

`requires_grad == True` 且 `_ctx == None` 的 tensor。

- `requires_grad=True`: 用户显式要求追踪梯度
- `_ctx == None`: 不是任何算子的输出（直接创建）

## 实现方案

### 1. 移除内容

- `Tensor.__init__`: 删除 `self.is_leaf = True`
- `Function.apply`: 删除 `output.is_leaf = False`

### 2. backward 修改

在梯度传播循环中，判断是否为 leaf tensor：

```python
for t, g in zip(v._ctx.saved_tensors, grads):
    if g is not None and isinstance(t, Tensor):
        # 只对 leaf tensor 累加梯度
        if t.requires_grad and t._ctx is None:
            if t.grad is None:
                t.grad = g
            else:
                t.grad = Tensor(t.grad.data + g.data, copy=False)

        # 非 leaf tensor: 梯度只用于传播，不存储

        # 继续传播（无论是否 leaf）
        num_outputs[id(t)] -= 1
        if num_outputs[id(t)] == 0:
            queue.append(t)
```

## 数据流示例

```
x = Tensor([1,2], requires_grad=True)  # leaf: requires_grad=True, _ctx=None
y = x * 2                               # non-leaf: requires_grad=True, _ctx=Mul
z = y.sum()                             # non-leaf: requires_grad=True, _ctx=Sum

z.backward()
# x.grad 存储（leaf）
# y.grad, z.grad 不存储（non-leaf）
```

## 影响范围

- `micrograd/engine.py`: Tensor 类和 backward 方法
- 无其他文件依赖 `is_leaf` 属性

## 验证

- 运行现有测试确保 backward 行为正确
- leaf tensor 正确累加梯度
- 非 leaf tensor 的 grad 为 None