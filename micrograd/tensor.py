# micrograd/tensor.py
import numpy as np
from .storage import Storage, CPUStorage

# Ops imported at end of file to avoid circular dependency


class Tensor:
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
        self.grad_fn = None

    @property
    def data(self) -> Storage:
        return self._storage

    @data.setter
    def data(self, value: Storage):
        self._storage = value

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

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("cannot call backward on tensor with requires_grad=False")
        if self.grad_fn is None:
            raise RuntimeError("cannot call backward on leaf tensor (no grad_fn)")
        
        grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)
        self.grad_fn.backward_loop(grad)
        self.grad_fn = None


# Import ops at end to avoid circular dependency (ops.py imports Tensor)
from .ops import Add, Mul, Sub, Div, Neg, ReLU, Pow, MatMul, Sum, Slice