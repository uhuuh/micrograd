# Storage/Dispatch 重构实现计划 (Phase 1: CPU)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 micrograd，添加 Storage 抽象层和 Dispatch 算子注册机制，仅实现 CPU 后端，保持现有测试通过。

**Architecture:** Tensor.data 从 numpy array 改为 Storage 抽象，算子通过 OpRegistry dispatch 到注册的 kernel，kernels/cpu.py 实现 CPU numpy kernel。

**Tech Stack:** Python, NumPy

---

## 文件结构

**Create:**
- `micrograd/storage.py` - Storage 基类, CPUStorage
- `micrograd/dispatch.py` - OpRegistry 算子注册表
- `micrograd/kernels/__init__.py` - kernels 模块入口
- `micrograd/kernels/cpu.py` - CPU kernel 实现
- `micrograd/function.py` - Function 基类 (从 engine.py 提取)
- `micrograd/ops.py` - 算子类 (从 engine.py 提取)
- `micrograd/tensor.py` - Tensor 类 + backward (从 engine.py 重命名)

**Modify:**
- `micrograd/__init__.py` - 更新导出
- `micrograd/nn.py` - 导入路径调整

**Delete:**
- `micrograd/engine.py` - 拆分后删除

---

### Task 1: 创建 Storage 抽象层

**Files:**
- Create: `micrograd/storage.py`

- [ ] **Step 1: 创建 storage.py 文件**

```python
# micrograd/storage.py
import numpy as np
from abc import ABC, abstractmethod


class Storage(ABC):
    """抽象存储层，统一 CPU/GPU 数据访问接口"""
    
    @property
    @abstractmethod
    def device(self) -> str:
        """设备类型: "cpu" 或 "cuda""""
        pass
    
    @property
    @abstractmethod
    def shape(self) -> tuple:
        """数据形状"""
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """数据类型"""
        pass
    
    @abstractmethod
    def numpy(self) -> np.ndarray:
        """获取 numpy 数组"""
        pass
    
    @abstractmethod
    def copy_to(self, other: "Storage"):
        """数据传输到其他 Storage"""
        pass
    
    @property
    def numel(self) -> int:
        """元素数量"""
        return int(np.prod(self.shape))
    
    @abstractmethod
    def ptr(self) -> int:
        """内存指针"""
        pass


class CPUStorage(Storage):
    """CPU 存储：numpy ndarray"""
    
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64)
    
    @property
    def device(self) -> str:
        return "cpu"
    
    @property
    def shape(self) -> tuple:
        return self._data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype
    
    def numpy(self) -> np.ndarray:
        return self._data
    
    def ptr(self) -> int:
        return self._data.ctypes.data
    
    def copy_to(self, other: Storage):
        if other.device == "cpu":
            other._data[:] = self._data
        else:
            raise NotImplementedError("CUDA not yet implemented")
```

- [ ] **Step 2: 运行测试验证 storage.py 可导入**

Run: `python -c "from micrograd.storage import Storage, CPUStorage; s = CPUStorage([1,2,3]); print(s.numpy())"`
Expected: 输出 `[1. 2. 3.]`

- [ ] **Step 3: 暂不提交**

---

### Task 2: 创建 Dispatch 注册表

**Files:**
- Create: `micrograd/dispatch.py`

- [ ] **Step 1: 创建 dispatch.py 文件**

```python
# micrograd/dispatch.py
from typing import Callable, Dict
import functools


class OpRegistry:
    """
    算子注册表，根据 device 自动 dispatch 到对应 kernel。
    
    支持两种注册方式：
    1. 装饰器: @registry.register_op("add", "cpu")
    2. 直接调用: registry.register("add", "cpu", forward_fn, backward_fn)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Dict[str, dict]] = {}
        return cls._instance
    
    def register_op(self, op_name: str, device: str, is_backward: bool = False):
        """
        装饰器：注册 forward 或 backward kernel。
        
        Usage:
            @registry.register_op("add", "cpu")
            def add_forward(a, b):
                ...
            
            @registry.register_op("add", "cpu", is_backward=True)
            def add_backward(grad_out, a, b):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            if op_name not in self._registry:
                self._registry[op_name] = {}
            if device not in self._registry[op_name]:
                self._registry[op_name][device] = {"forward": None, "backward": None}
            
            key = "backward" if is_backward else "forward"
            self._registry[op_name][device][key] = fn
            
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator
    
    def register(self, op_name: str, device: str, 
                 forward: Callable, backward: Callable):
        """直接注册 forward/backward kernel"""
        if op_name not in self._registry:
            self._registry[op_name] = {}
        self._registry[op_name][device] = {"forward": forward, "backward": backward}
    
    def dispatch(self, op_name: str, device: str) -> tuple[Callable, Callable]:
        """根据 op_name 和 device 选择 kernel"""
        if op_name not in self._registry:
            raise KeyError(f"Op '{op_name}' not registered")
        if device not in self._registry[op_name]:
            raise KeyError(f"Op '{op_name}' has no kernel for device '{device}'")
        entry = self._registry[op_name][device]
        return entry["forward"], entry["backward"]
    
    def get_devices(self, op_name: str) -> list[str]:
        """获取某算子支持的所有设备"""
        return list(self._registry.get(op_name, {}).keys())
    
    def is_registered(self, op_name: str, device: str) -> bool:
        """检查算子是否在某设备上注册"""
        return op_name in self._registry and device in self._registry[op_name]


# 全局单例
registry = OpRegistry()
```

- [ ] **Step 2: 运行测试验证 dispatch.py 可导入**

Run: `python -c "from micrograd.dispatch import registry; print(registry)"`
Expected: 输出 `<micrograd.dispatch.OpRegistry object at ...>`

- [ ] **Step 3: 暂不提交**

---

### Task 3: 创建 kernels 模块和 CPU kernel

**Files:**
- Create: `micrograd/kernels/__init__.py`
- Create: `micrograd/kernels/cpu.py`

- [ ] **Step 1: 创建 kernels 目录和 __init__.py**

```python
# micrograd/kernels/__init__.py
from .cpu import (
    add_forward, add_backward,
    mul_forward, mul_backward,
    sub_forward, sub_backward,
    div_forward, div_backward,
    neg_forward, neg_backward,
    relu_forward, relu_backward,
    pow_forward, pow_backward,
    matmul_forward, matmul_backward,
    sum_forward, sum_backward,
    slice_forward, slice_backward,
)

__all__ = [
    "add_forward", "add_backward",
    "mul_forward", "mul_backward",
    "sub_forward", "sub_backward",
    "div_forward", "div_backward",
    "neg_forward", "neg_backward",
    "relu_forward", "relu_backward",
    "pow_forward", "pow_backward",
    "matmul_forward", "matmul_backward",
    "sum_forward", "sum_backward",
    "slice_forward", "slice_backward",
]
```

- [ ] **Step 2: 创建 cpu.py 实现所有 CPU kernel**

```python
# micrograd/kernels/cpu.py
import numpy as np
from ..dispatch import registry
from ..storage import CPUStorage


# === Add ===
@registry.register_op("add", "cpu")
def add_forward(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.numpy() + b.numpy())

@registry.register_op("add", "cpu", is_backward=True)
def add_backward(grad_output: CPUStorage, a: CPUStorage, b: CPUStorage) -> tuple[CPUStorage, CPUStorage]:
    return grad_output, grad_output


# === Mul ===
@registry.register_op("mul", "cpu")
def mul_forward(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.numpy() * b.numpy())

@registry.register_op("mul", "cpu", is_backward=True)
def mul_backward(grad_output: CPUStorage, a: CPUStorage, b: CPUStorage) -> tuple[CPUStorage, CPUStorage]:
    return CPUStorage(b.numpy() * grad_output.numpy()), CPUStorage(a.numpy() * grad_output.numpy())


# === Sub ===
@registry.register_op("sub", "cpu")
def sub_forward(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.numpy() - b.numpy())

@registry.register_op("sub", "cpu", is_backward=True)
def sub_backward(grad_output: CPUStorage, a: CPUStorage, b: CPUStorage) -> tuple[CPUStorage, CPUStorage]:
    return grad_output, CPUStorage(-grad_output.numpy())


# === Div ===
@registry.register_op("div", "cpu")
def div_forward(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.numpy() / b.numpy())

@registry.register_op("div", "cpu", is_backward=True)
def div_backward(grad_output: CPUStorage, a: CPUStorage, b: CPUStorage) -> tuple[CPUStorage, CPUStorage]:
    a_np = a.numpy()
    b_np = b.numpy()
    grad_np = grad_output.numpy()
    return CPUStorage(grad_np / b_np), CPUStorage(-a_np * grad_np / (b_np ** 2))


# === Neg ===
@registry.register_op("neg", "cpu")
def neg_forward(a: CPUStorage) -> CPUStorage:
    return CPUStorage(-a.numpy())

@registry.register_op("neg", "cpu", is_backward=True)
def neg_backward(grad_output: CPUStorage, a: CPUStorage) -> CPUStorage:
    return CPUStorage(-grad_output.numpy())


# === ReLU ===
@registry.register_op("relu", "cpu")
def relu_forward(a: CPUStorage) -> CPUStorage:
    return CPUStorage(np.maximum(0, a.numpy()))

@registry.register_op("relu", "cpu", is_backward=True)
def relu_backward(grad_output: CPUStorage, a: CPUStorage) -> CPUStorage:
    return CPUStorage(grad_output.numpy() * (a.numpy() > 0).astype(float))


# === Pow ===
@registry.register_op("pow", "cpu")
def pow_forward(a: CPUStorage, exponent: float) -> CPUStorage:
    return CPUStorage(a.numpy() ** exponent)

@registry.register_op("pow", "cpu", is_backward=True)
def pow_backward(grad_output: CPUStorage, a: CPUStorage, exponent: float) -> tuple[CPUStorage, None]:
    a_np = a.numpy()
    grad_np = grad_output.numpy()
    return CPUStorage(exponent * (a_np ** (exponent - 1)) * grad_np), None


# === MatMul ===
@registry.register_op("matmul", "cpu")
def matmul_forward(a: CPUStorage, b: CPUStorage) -> CPUStorage:
    return CPUStorage(a.numpy() @ b.numpy())

@registry.register_op("matmul", "cpu", is_backward=True)
def matmul_backward(grad_output: CPUStorage, a: CPUStorage, b: CPUStorage) -> tuple[CPUStorage, CPUStorage]:
    return (CPUStorage(grad_output.numpy() @ b.numpy().swapaxes(-1, -2)),
            CPUStorage(a.numpy().swapaxes(-1, -2) @ grad_output.numpy()))


# === Sum ===
@registry.register_op("sum", "cpu")
def sum_forward(a: CPUStorage, dim=None, keepdim=False) -> CPUStorage:
    out = np.sum(a.numpy(), axis=dim, keepdims=keepdim)
    return CPUStorage(out)

@registry.register_op("sum", "cpu", is_backward=True)
def sum_backward(grad_output: CPUStorage, a: CPUStorage, dim, keepdim) -> CPUStorage:
    if dim is None:
        return CPUStorage(np.ones_like(a.numpy()) * grad_output.numpy())
    else:
        shape = list(a.numpy().shape)
        shape[dim] = 1
        grad_a = np.ones(shape) * grad_output.numpy()
        if not keepdim:
            grad_a = np.squeeze(grad_a, axis=dim)
        return CPUStorage(grad_a)


# === Slice ===
@registry.register_op("slice", "cpu")
def slice_forward(a: CPUStorage, key) -> CPUStorage:
    return CPUStorage(a.numpy()[key])

@registry.register_op("slice", "cpu", is_backward=True)
def slice_backward(grad_output: CPUStorage, a: CPUStorage, key) -> CPUStorage:
    out = np.zeros_like(a.numpy())
    out[key] = grad_output.numpy()
    return CPUStorage(out)
```

- [ ] **Step 3: 运行测试验证 kernel 注册成功**

Run: `python -c "from micrograd.kernels import cpu; from micrograd.dispatch import registry; print(registry.get_devices('add'))"`
Expected: 输出 `['cpu']`

- [ ] **Step 4: 暂不提交**

---

### Task 4: 提取 Function 基类

**Files:**
- Create: `micrograd/function.py`

- [ ] **Step 1: 创建 function.py**

```python
# micrograd/function.py


class Function:
    """Base class for autograd operations. Subclass with forward/backward static methods."""

    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = []

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def save_data_for_backward(self, *data):
        self.saved_data = list(data)

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output) -> tuple["Tensor | None", ...]:
        """Backward pass. Must return a tuple of gradients (Tensor or None) for each input."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        """Apply the operation: creates ctx, runs forward, attaches grad_fn to output."""
        ctx = cls()
        output = cls.forward(ctx, *inputs)
        output._ctx = ctx
        ctx._grad_fn = cls
        return output
```

- [ ] **Step 2: 验证可导入**

Run: `python -c "from micrograd.function import Function; print(Function)"`
Expected: 输出 `<class 'micrograd.function.Function'>`

- [ ] **Step 3: 暂不提交**

---

### Task 5: 创建 ops.py 算子类

**Files:**
- Create: `micrograd/ops.py`

- [ ] **Step 1: 创建 ops.py 文件**

```python
# micrograd/ops.py
from .function import Function
from .dispatch import registry


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("add", a.device)
        storage = forward_fn(a.data, b.data)
        return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("add", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("mul", a.device)
        storage = forward_fn(a.data, b.data)
        return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("mul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("sub", a.device)
        storage = forward_fn(a.data, b.data)
        return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("sub", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("div", a.device)
        storage = forward_fn(a.data, b.data)
        return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("div", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("neg", a.device)
        storage = forward_fn(a.data)
        return Tensor(storage, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        _, backward_fn = registry.dispatch("neg", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data)
        return (Tensor(grad_a),)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("relu", a.device)
        storage = forward_fn(a.data)
        return Tensor(storage, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        _, backward_fn = registry.dispatch("relu", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data)
        return (Tensor(grad_a),)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(exponent)
        forward_fn, _ = registry.dispatch("pow", a.device)
        storage = forward_fn(a.data, exponent)
        return Tensor(storage, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        exponent, = ctx.saved_data
        _, backward_fn = registry.dispatch("pow", grad_output.device)
        grad_a, _ = backward_fn(grad_output.data, a.data, exponent)
        return Tensor(grad_a), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("matmul", a.device)
        storage = forward_fn(a.data, b.data)
        return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("matmul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(dim, keepdim)
        forward_fn, _ = registry.dispatch("sum", a.device)
        storage = forward_fn(a.data, dim, keepdim)
        return Tensor(storage, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.saved_data
        _, backward_fn = registry.dispatch("sum", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, dim, keepdim)
        return (Tensor(grad_a),)


class Slice(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(key)
        forward_fn, _ = registry.dispatch("slice", a.device)
        storage = forward_fn(a.data, key)
        return Tensor(storage, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        key, = ctx.saved_data
        _, backward_fn = registry.dispatch("slice", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, key)
        return (Tensor(grad_a),)


# Tensor 前向引用，在 ops.py 末尾导入避免循环依赖
from .tensor import Tensor
```

- [ ] **Step 2: 暂不提交 (等待 Tensor 创建)**

---

### Task 6: 创建 Tensor 类 (从 engine.py 重构)

**Files:**
- Create: `micrograd/tensor.py`

- [ ] **Step 1: 创建 tensor.py 文件**

```python
# micrograd/tensor.py
import numpy as np
from collections import deque
from .storage import Storage, CPUStorage
from .ops import Add, Mul, Sub, Div, Neg, ReLU, Pow, MatMul, Sum, Slice


class Tensor:
    def __init__(self, data, requires_grad=False, device="cpu"):
        if isinstance(data, Storage):
            self._storage = data
        elif device == "cpu":
            self._storage = CPUStorage(np.array(data, dtype=np.float64))
        else:
            raise ValueError(f"Unknown device: {device}")
        
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self._use_count = 0
    
    @property
    def data(self) -> Storage:
        return self._storage
    
    @property
    def device(self) -> str:
        return self._storage.device
    
    @property
    def shape(self) -> tuple:
        return self._storage.shape
    
    def numpy(self) -> np.ndarray:
        return self._storage.numpy()
    
    def __repr__(self):
        return f"Tensor({self.numpy()}, requires_grad={self.requires_grad}, device={self.device})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Mul.apply(self, other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Sub.apply(self, other)
    
    def __neg__(self):
        return Neg.apply(self)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Div.apply(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rsub__(self, other):
        return Sub.apply(Tensor(other), self)
    
    def __rtruediv__(self, other):
        return Div.apply(Tensor(other), self)
    
    def relu(self):
        return ReLU.apply(self)
    
    def __pow__(self, exponent):
        return Pow.apply(self, exponent)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
    def __getitem__(self, key):
        return Slice.apply(self, key)
    
    def reshape(self, *shape):
        return Tensor(self._storage.numpy().reshape(*shape), requires_grad=self.requires_grad)
    
    def sum(self, dim=None, keepdim=False):
        return Sum.apply(self, dim, keepdim)
    
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
                        child._use_count += 1
                        if id(child) not in visited:
                            queue.append(child)
    
    def backward(self):
        """Compute gradient using BFS with static use_count."""
        if self._ctx is None:
            self.grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)
            return
        
        # 预处理：计算 use_count
        self._compute_use_counts()
        
        queue = deque([self])
        self.grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)
        
        while queue:
            v = queue.popleft()
            
            if v._ctx is None:
                continue  # leaf tensor，不传播
            
            # Ensure v.grad is set for backward
            if v.grad is None:
                v.grad = Tensor(np.ones_like(v._storage.numpy()), copy=False)
            
            grads = v._ctx._grad_fn.backward(v._ctx, v.grad)
            
            for t, g in zip(v._ctx.saved_tensors, grads):
                if g is None or not isinstance(t, Tensor):
                    continue
                
                # Only accumulate gradient for leaf tensors
                if t.requires_grad and t._ctx is None:
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad = Tensor(t.grad._storage.numpy() + g._storage.numpy(), copy=False)
                
                # 传播给子节点后 use_count 减一，归零时入队
                t._use_count -= 1
                if t._use_count == 0:
                    queue.append(t)
            
            # 处理完 non-leaf 删除 grad（leaf 保留）
            v.grad = None
```

- [ ] **Step 2: 暂不提交 (等待整合)**

---

### Task 7: 更新 __init__.py 和删除 engine.py

**Files:**
- Modify: `micrograd/__init__.py`
- Delete: `micrograd/engine.py`

- [ ] **Step 1: 更新 __init__.py 导出**

```python
# micrograd/__init__.py
from .tensor import Tensor
from .function import Function
from .dispatch import registry
from .storage import Storage, CPUStorage
from . import ops
from . import kernels

__all__ = ["Tensor", "Function", "registry", "Storage", "CPUStorage", "ops", "kernels"]
```

- [ ] **Step 2: 删除 engine.py**

Run: `rm micrograd/engine.py`

- [ ] **Step 3: 暂不提交**

---

### Task 8: 更新 nn.py 导入路径

**Files:**
- Modify: `micrograd/nn.py`

- [ ] **Step 1: 更新 nn.py 导入**

```python
# micrograd/nn.py
import random
from .tensor import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Tensor(random.uniform(-1,1) * nin**(-0.5)) for _ in range(nin)]
        self.b = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
```

- [ ] **Step 2: 暂不提交**

---

### Task 9: 运行全部测试验证重构成功

**Files:**
- Test: 运行所有现有测试

- [ ] **Step 1: 运行测试验证**

Run: `pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 2: 如果测试失败，修复问题**

检查导入路径和 Storage 类型匹配问题。

- [ ] **Step 3: 提交所有改动**

```bash
git add micrograd/storage.py micrograd/dispatch.py micrograd/kernels/ micrograd/function.py micrograd/ops.py micrograd/tensor.py micrograd/__init__.py micrograd/nn.py
git rm micrograd/engine.py
git commit -m "$(cat <<'EOF'
refactor: add Storage abstraction and Dispatch layer for CPU

- Add Storage abstract class and CPUStorage implementation
- Add OpRegistry for kernel dispatch by device
- Create kernels/cpu.py with CPU kernels using decorator registration
- Split engine.py into tensor.py, function.py, ops.py
- Update nn.py imports

All existing tests pass with CPU backend.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Verification

运行测试确保重构成功：

```bash
pytest tests/ -v
```

预期结果：所有现有测试通过，CPU 功能不变。