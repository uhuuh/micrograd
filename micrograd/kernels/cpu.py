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
def sum_forward(a: CPUStorage, dim: int | None = None, keepdim: bool = False) -> CPUStorage:
    out = np.sum(a.numpy(), axis=dim, keepdims=keepdim)
    return CPUStorage(out)

@registry.register_op("sum", "cpu", is_backward=True)
def sum_backward(grad_output: CPUStorage, a: CPUStorage, dim: int | None, keepdim: bool) -> CPUStorage:
    grad_np = grad_output.numpy()
    if dim is None:
        return CPUStorage(np.ones_like(a.numpy()) * grad_np)
    if not keepdim:
        grad_np = np.expand_dims(grad_np, axis=dim)
    return CPUStorage(np.broadcast_to(grad_np, a.numpy().shape).copy())


# === Slice ===
@registry.register_op("slice", "cpu")
def slice_forward(a: CPUStorage, key) -> CPUStorage:
    return CPUStorage(a.numpy()[key])

@registry.register_op("slice", "cpu", is_backward=True)
def slice_backward(grad_output: CPUStorage, a: CPUStorage, key) -> CPUStorage:
    out = np.zeros_like(a.numpy())
    out[key] = grad_output.numpy()
    return CPUStorage(out)