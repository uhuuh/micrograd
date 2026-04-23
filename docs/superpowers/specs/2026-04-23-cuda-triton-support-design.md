# CUDA/Triton 支持架构设计

## 背景

micrograd 是一个教学性质的 autograd 引擎，当前仅支持 CPU (numpy backend)。目标是添加 CUDA 设备支持和 Triton 算子，用于研究 Triton GPU 编程。

## 目标

- Tensor 支持多设备 (cpu/cuda)
- 使用 NVIDIA 官方 cuda-python 管理 GPU 内存
- Triton kernel 实现算子的 forward 和 backward
- 渐进式实现：先完成基础设施，再逐个添加 Triton 算子

## 文件结构

```
micrograd/
├── tensor.py          # Tensor 类 + backward (原 engine.py)
├── function.py        # Function 基类
├── ops.py             # Add, Mul, Sub, Div, ReLU, MatMul, Sum 等算子
├── storage.py         # Storage, CPUStorage, CUDAStorage (新增)
├── triton_ops.py      # Triton kernels (新增)
├── cuda_utils.py      # CUDA 设备管理 (新增)
└── nn.py              # 神经网络模块 (保持不变)
```

---

## 1. Storage 抽象层 (`storage.py`)

Tensor 的数据存储抽象，分离 CPU 和 GPU 实现。

### Storage 基类

```python
class Storage:
    """抽象存储层，统一 CPU/GPU 数据访问接口"""
    device: str      # "cpu" 或 "cuda"
    shape: tuple
    dtype: np.dtype

    def numpy(self) -> np.ndarray:
        """获取数据：CPU 返回视图，GPU 返回复制到 host 的数组"""
        raise NotImplementedError

    def copy_to(self, other: "Storage"):
        """数据传输：CPU ↔ CUDA"""
        raise NotImplementedError

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def ptr(self) -> int:
        """内存指针：CPU 返回 numpy array 的指针，GPU 返回 cuda.core.Buffer 的指针"""
        raise NotImplementedError
```

### CPUStorage

```python
class CPUStorage(Storage):
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float64)
        self.device = "cpu"
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def numpy(self) -> np.ndarray:
        return self._data

    def ptr(self) -> int:
        return self._data.ctypes.data

    def copy_to(self, other: Storage):
        if other.device == "cuda":
            # CPU → GPU
            other._buffer.copy_from(self._data)
        else:
            other._data[:] = self._data
```

### CUDAStorage

```python
class CUDAStorage(Storage):
    def __init__(self, shape: tuple, dtype: np.dtype, buffer):
        self.device = "cuda"
        self._buffer = buffer  # cuda.core.Buffer
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def numpy(self) -> np.ndarray:
        host = np.empty(self.shape, dtype=self.dtype)
        self._buffer.copy_to(host)
        return host

    def ptr(self) -> int:
        return self._buffer.ptr

    def copy_to(self, other: Storage):
        if other.device == "cpu":
            # GPU → CPU
            self._buffer.copy_to(other._data)
        else:
            # GPU → GPU (同设备复制)
            other._buffer.copy_from(self._buffer)
```

---

## 2. CUDA 工具 (`cuda_utils.py`)

封装 cuda-python 的设备管理和内存分配。

```python
from cuda.core import Device, Buffer, Stream
import numpy as np

# 全局状态
_device: Device = None
_stream: Stream = None

def init_cuda(device_id: int = 0):
    """初始化 CUDA 设备"""
    global _device, _stream
    _device = Device(device_id)
    _stream = _device.create_stream()

def get_device() -> Device:
    return _device

def get_stream() -> Stream:
    return _stream

def allocate_gpu(shape: tuple, dtype: np.dtype = np.float64) -> "CUDAStorage":
    """分配 GPU 内存"""
    from .storage import CUDAStorage
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buffer = _device.create_buffer(size)
    return CUDAStorage(shape, dtype, buffer)

def is_cuda_available() -> bool:
    try:
        init_cuda()
        return True
    except:
        return False
```

---

## 3. Triton 算子 (`triton_ops.py`)

Triton kernel 实现，逐个添加。

### Add

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def triton_add(a: CUDAStorage, b: CUDAStorage) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    out = allocate_gpu(a.shape, a.dtype)
    grid = lambda META: (triton.cdiv(a.numel, META['BLOCK_SIZE']),)
    add_kernel[grid](a.ptr, b.ptr, out.ptr, a.numel, BLOCK_SIZE=1024)
    return out
```

### Mul

```python
@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)

def triton_mul(a: CUDAStorage, b: CUDAStorage) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    out = allocate_gpu(a.shape, a.dtype)
    grid = lambda META: (triton.cdiv(a.numel, META['BLOCK_SIZE']),)
    mul_kernel[grid](a.ptr, b.ptr, out.ptr, a.numel, BLOCK_SIZE=1024)
    return out

# Mul backward: grad_a = grad_out * b, grad_b = grad_out * a
@triton.jit
def mul_backward_kernel(grad_out_ptr, a_ptr, b_ptr, grad_a_ptr, grad_b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(grad_a_ptr + offsets, grad_out * b, mask=mask)
    tl.store(grad_b_ptr + offsets, grad_out * a, mask=mask)

def triton_mul_backward(grad_out: CUDAStorage, a: CUDAStorage, b: CUDAStorage) -> tuple[CUDAStorage, CUDAStorage]:
    from .cuda_utils import allocate_gpu
    grad_a = allocate_gpu(a.shape, a.dtype)
    grad_b = allocate_gpu(b.shape, b.dtype)
    grid = lambda META: (triton.cdiv(a.numel, META['BLOCK_SIZE']),)
    mul_backward_kernel[grid](grad_out.ptr, a.ptr, b.ptr, grad_a.ptr, grad_b.ptr, a.numel, BLOCK_SIZE=1024)
    return grad_a, grad_b

# === ReLU ===
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(0.0, x)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu(a: CUDAStorage) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    out = allocate_gpu(a.shape, a.dtype)
    grid = lambda META: (triton.cdiv(a.numel, META['BLOCK_SIZE']),)
    relu_kernel[grid](a.ptr, out.ptr, a.numel, BLOCK_SIZE=1024)
    return out

@triton.jit
def relu_backward_kernel(grad_out_ptr, x_ptr, grad_x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_x = grad_out * (x > 0).to(tl.float64)
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

def triton_relu_backward(grad_out: CUDAStorage, x: CUDAStorage) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    grad_x = allocate_gpu(x.shape, x.dtype)
    grid = lambda META: (triton.cdiv(x.numel, META['BLOCK_SIZE']),)
    relu_backward_kernel[grid](grad_out.ptr, x.ptr, grad_x.ptr, x.numel, BLOCK_SIZE=1024)
    return grad_x

# === Sum ===
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 单 block reduce
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    tl.store(out_ptr, sum_val)

def triton_sum(a: CUDAStorage, dim=None, keepdim=False) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    if dim is None:
        # 全元素 sum
        out = allocate_gpu((1,), a.dtype)
        grid = (1,)
        sum_kernel[grid](a.ptr, out.ptr, a.numel, BLOCK_SIZE=min(a.numel, 1024))
        if not keepdim:
            return allocate_gpu((), a.dtype)  # scalar
        return out
    else:
        # 按维度 sum (更复杂，需要多 block)
        # ... 暂时省略，Phase 3 详细实现
        raise NotImplementedError("dim sum not yet implemented")

def triton_sum_backward(grad_out: CUDAStorage, x: CUDAStorage, dim, keepdim) -> CUDAStorage:
    from .cuda_utils import allocate_gpu
    if dim is None:
        # grad_x = grad_out broadcast to x.shape
        grad_x = allocate_gpu(x.shape, x.dtype)
        grid = lambda META: (triton.cdiv(x.numel, META['BLOCK_SIZE']),)
        # fill kernel: grad_x[i] = grad_out[0]
        @triton.jit
        def fill_kernel(grad_out_ptr, grad_x_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            val = tl.load(grad_out_ptr)
            tl.store(grad_x_ptr + offsets, val, mask=mask)
        fill_kernel[grid](grad_out.ptr, grad_x.ptr, x.numel, BLOCK_SIZE=1024)
        return grad_x
    else:
        raise NotImplementedError("dim sum backward not yet implemented")
```

### MatMul (矩阵乘法)

MatMul 需要更复杂的 2D kernel，使用 block tiling 优化：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, out_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D block indexing
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    # K-dimension loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + (rm[:, None] * K + (k + rk[None, :])), mask=(rm[:, None] < M) & ((k + rk[None, :]) < K))
        b = tl.load(b_ptr + ((k + rk[:, None]) * N + rn[None, :]), mask=((k + rk[:, None]) < K) & (rn[None, :] < N))
        acc += tl.dot(a, b)

    # Store result
    tl.store(out_ptr + (rm[:, None] * N + rn[None, :]), acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def triton_matmul(a: CUDAStorage, b: CUDAStorage) -> CUDAStorage:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    from .cuda_utils import allocate_gpu
    out = allocate_gpu((M, N), a.dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](a.ptr, b.ptr, out.ptr, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return out

# MatMul backward: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
@triton.jit
def matmul_backward_a_kernel(grad_out_ptr, b_ptr, grad_a_ptr, M, K, N, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    # grad_a = grad_out @ b.T  (M x N) @ (N x K) = M x K
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = tl.arange(0, 64)  # 固定 N 维度块大小
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    for n in range(0, N, 64):
        grad_out = tl.load(grad_out_ptr + (rm[:, None] * N + rn[None, :]), mask=(rm[:, None] < M) & (rn[None, :] < N))
        b_T = tl.load(b_ptr + (rk[None, :] * N + rn[:, None]), mask=(rk[None, :] < K) & (rn[:, None] < N))  # b transposed
        acc += tl.dot(grad_out, b_T)
    tl.store(grad_a_ptr + (rm[:, None] * K + rk[None, :]), acc, mask=(rm[:, None] < M) & (rk[None, :] < K))

def triton_matmul_backward(grad_out: CUDAStorage, a: CUDAStorage, b: CUDAStorage) -> tuple[CUDAStorage, CUDAStorage]:
    M, K = a.shape
    _, N = b.shape
    from .cuda_utils import allocate_gpu
    grad_a = allocate_gpu((M, K), a.dtype)
    grad_b = allocate_gpu((K, N), b.dtype)
    # grad_a kernel
    grid_a = (triton.cdiv(M, 64), triton.cdiv(K, 32))
    matmul_backward_a_kernel[grid_a](grad_out.ptr, b.ptr, grad_a.ptr, M, K, N, BLOCK_M=64, BLOCK_K=32)
    # grad_b kernel (类似实现 a.T @ grad_out)
    # ...
    return grad_a, grad_b
```

---

## 4. Tensor 类 (`tensor.py`)

改造后的 Tensor，支持设备属性。

```python
from .storage import Storage, CPUStorage
from .cuda_utils import allocate_gpu, is_cuda_available

class Tensor:
    def __init__(self, data, requires_grad=False, device="cpu"):
        if isinstance(data, Storage):
            self._storage = data
        elif device == "cpu":
            self._storage = CPUStorage(np.array(data, dtype=np.float64))
        elif device == "cuda":
            arr = np.array(data, dtype=np.float64)
            self._storage = allocate_gpu(arr.shape, arr.dtype)
            # 复制初始数据到 GPU
            cpu_storage = CPUStorage(arr)
            cpu_storage.copy_to(self._storage)
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

    def cuda(self) -> "Tensor":
        if self.device == "cuda":
            return self
        gpu_storage = allocate_gpu(self.shape, self._storage.dtype)
        self._storage.copy_to(gpu_storage)
        return Tensor(gpu_storage, self.requires_grad)

    def cpu(self) -> "Tensor":
        if self.device == "cpu":
            return self
        return Tensor(CPUStorage(self._storage.numpy()), self.requires_grad)

    # backward 方法保持不变，使用 BFS + use_count
```

---

## 5. Function 基类 (`function.py`)

```python
class Function:
    """Base class for autograd operations"""

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
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        ctx = cls()
        output = cls.forward(ctx, *inputs)
        output._ctx = ctx
        ctx._grad_fn = cls
        return output
```

---

## 6. 算子改造 (`ops.py`)

每个算子检查设备并选择实现。

```python
from .function import Function
from .tensor import Tensor
from .storage import CPUStorage

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if a.device == "cuda":
            from .triton_ops import triton_add
            storage = triton_add(a.data, b.data)
            return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)
        else:
            return Tensor(CPUStorage(a.data.numpy() + b.data.numpy()),
                          requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        # Add backward: identity
        return grad_output, grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if a.device == "cuda":
            from .triton_ops import triton_mul
            storage = triton_mul(a.data, b.data)
            return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)
        else:
            return Tensor(CPUStorage(a.data.numpy() * b.data.numpy()),
                          requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if grad_output.device == "cuda":
            from .triton_ops import triton_mul_backward
            grad_a, grad_b = triton_mul_backward(grad_output.data, a.data, b.data)
            return Tensor(grad_a), Tensor(grad_b)
        else:
            return (Tensor(CPUStorage(b.data.numpy() * grad_output.data.numpy())),
                    Tensor(CPUStorage(a.data.numpy() * grad_output.data.numpy())))

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        if a.device == "cuda":
            from .triton_ops import triton_relu
            storage = triton_relu(a.data)
            return Tensor(storage, requires_grad=a.requires_grad)
        else:
            return Tensor(CPUStorage(np.maximum(0, a.data.numpy())), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        if grad_output.device == "cuda":
            from .triton_ops import triton_relu_backward
            grad_a = triton_relu_backward(grad_output.data, a.data)
            return (Tensor(grad_a),)
        else:
            return (Tensor(CPUStorage(grad_output.data.numpy() * (a.data.numpy() > 0).astype(float))),)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if a.device == "cuda":
            from .triton_ops import triton_matmul
            storage = triton_matmul(a.data, b.data)
            return Tensor(storage, requires_grad=a.requires_grad or b.requires_grad)
        else:
            return Tensor(CPUStorage(a.data.numpy() @ b.data.numpy()),
                          requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if grad_output.device == "cuda":
            from .triton_ops import triton_matmul_backward
            grad_a, grad_b = triton_matmul_backward(grad_output.data, a.data, b.data)
            return Tensor(grad_a), Tensor(grad_b)
        else:
            return (Tensor(CPUStorage(grad_output.data.numpy() @ b.data.numpy().swapaxes(-1, -2))),
                    Tensor(CPUStorage(a.data.numpy().swapaxes(-1, -2) @ grad_output.data.numpy())))

class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(dim, keepdim)
        if a.device == "cuda":
            from .triton_ops import triton_sum
            storage = triton_sum(a.data, dim, keepdim)
            return Tensor(storage, requires_grad=a.requires_grad)
        else:
            out = np.sum(a.data.numpy(), axis=dim, keepdims=keepdim)
            return Tensor(CPUStorage(out), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.saved_data
        if grad_output.device == "cuda":
            from .triton_ops import triton_sum_backward
            grad_a = triton_sum_backward(grad_output.data, a.data, dim, keepdim)
            return (Tensor(grad_a),)
        else:
            if dim is None:
                return (Tensor(CPUStorage(np.ones_like(a.data.numpy()) * grad_output.data.numpy())),)
            else:
                shape = list(a.data.numpy().shape)
                shape[dim] = 1
                grad_a = np.ones(shape) * grad_output.data.numpy()
                if not keepdim:
                    grad_a = np.squeeze(grad_a, axis=dim)
                return (Tensor(CPUStorage(grad_a)),)
```

---

## 7. 实现顺序

渐进式实现，分阶段完成：

### Phase 1: 文件拆分 + Storage 基础设施
1. 拆分 engine.py → tensor.py, function.py, ops.py
2. 实现 storage.py (CPUStorage)
3. 修改 Tensor 使用 Storage
4. 测试 CPU 功能不变

### Phase 2: CUDA 内存管理
1. 实现 cuda_utils.py
2. 实现 CUDAStorage
3. Tensor.cuda() / .cpu() 设备转换
4. 测试 CPU ↔ CUDA 数据传输

### Phase 3: Triton 算子 (逐个)
1. Add (forward + backward)
2. Mul (forward + backward)
3. ReLU
4. MatMul
5. Sum
6. 其他算子

---

## 8. 测试策略

每个阶段独立测试：

- Phase 1: 现有 pytest 测试应全部通过
- Phase 2: CPU ↔ CUDA 数据传输正确性
- Phase 3: Triton 算子 forward/backward 与 CPU 结果一致 (数值精度 tolerance)

```python
def test_cuda_cpu_consistency():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    x_cuda = x.cuda()

    y = x * 2
    y_cuda = x_cuda * 2

    assert np.allclose(y.numpy(), y_cuda.cpu().numpy())

    z = y.sum()
    z_cuda = y_cuda.sum()

    z.backward()
    z_cuda.backward()

    assert np.allclose(x.grad.numpy(), x_cuda.grad.cpu().numpy())
```

---

## 依赖

```
# requirements.txt 新增
cuda-python >= 12.0
triton >= 3.0
```