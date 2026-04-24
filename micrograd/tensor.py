# micrograd/tensor.py
import numpy as np
from collections import deque
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
        self._ctx = None
        self._use_count = 0

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

    def _compute_use_counts(self):
        """Preprocess: from root, count how many child nodes depend on each tensor."""
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

        # Preprocess: compute use_count
        self._compute_use_counts()

        queue = deque([self])
        self.grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)

        while queue:
            v = queue.popleft()

            if v._ctx is None:
                continue  # leaf tensor, no propagation

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

                # After propagating to child, decrement use_count; enqueue when zero
                t._use_count -= 1
                if t._use_count == 0:
                    queue.append(t)

            # Clear grad for non-leaf after processing (keep for leaf)
            v.grad = None


# Import ops at end to avoid circular dependency (ops.py imports Tensor)
from .ops import Add, Mul, Sub, Div, Neg, ReLU, Pow, MatMul, Sum, Slice