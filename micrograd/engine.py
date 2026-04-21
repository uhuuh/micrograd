import numpy as np
from collections import deque


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


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(b.data * grad_output.data, copy=False), Tensor(a.data * grad_output.data, copy=False)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, Tensor(-grad_output.data, copy=False)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(grad_output.data / b.data, copy=False), Tensor(-a.data * grad_output.data / (b.data ** 2), copy=False)


class Slice(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(key)
        return Tensor(a.data[key], requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        key, = ctx.saved_data
        out = np.zeros_like(a.data)
        out[key] = grad_output.data
        return Tensor(out, copy=False)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(-a.data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return Tensor(-grad_output.data, copy=False)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return Tensor(grad_output.data * (a.data > 0).astype(float), copy=False)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(exponent)
        return Tensor(a.data ** exponent, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        exponent, = ctx.saved_data
        return Tensor(exponent * (a.data ** (exponent - 1)) * grad_output.data, copy=False), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(grad_output.data @ b.data.swapaxes(-1, -2), copy=False), Tensor(a.data.swapaxes(-1, -2) @ grad_output.data, copy=False)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(dim, keepdim)
        out = np.sum(a.data, axis=dim, keepdims=keepdim)
        return Tensor(out, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.saved_data
        if dim is None:
            return Tensor(np.ones_like(a.data) * grad_output.data, copy=False)
        else:
            shape = list(a.data.shape)
            shape[dim] = 1
            grad_a = np.ones(shape) * grad_output.data
            if not keepdim:
                grad_a = np.squeeze(grad_a, axis=dim)
            return Tensor(grad_a, copy=False)


class Tensor:
    def __init__(self, data, requires_grad=False, copy=True):
        if copy:
            self.data = np.array(data, dtype=np.float64)
        else:
            self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None

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

    def sum(self, dim=None, keepdim=False):
        return Sum.apply(self, dim, keepdim)

    def backward(self):
        """Compute gradient using BFS with reference counting from saved_tensors."""
        if self._ctx is None:
            self.grad = Tensor(np.ones_like(self.data), copy=False)
            return

        # Build num_outputs dict from saved_tensors with cycle detection
        num_outputs = {}
        visiting = set()  # For cycle detection (currently being visited)

        def build_num_outputs(v):
            if id(v) in num_outputs:
                return
            if id(v) in visiting:
                raise RuntimeError("Cycle detected in computation graph")
            visiting.add(id(v))
            num_outputs[id(v)] = 0
            if v._ctx is not None:
                for child in v._ctx.saved_tensors:
                    if isinstance(child, Tensor):
                        build_num_outputs(child)
                        num_outputs[id(child)] += 1
            visiting.remove(id(v))

        build_num_outputs(self)

        # Initialize gradients
        self.grad = Tensor(np.ones_like(self.data), copy=False)

        # BFS: compute gradients as we traverse
        queue = deque([self])

        while queue:
            v = queue.popleft()
            if v._ctx is None:
                continue

            grads = v._ctx._grad_fn.backward(v._ctx, v.grad)

            for t, g in zip(v._ctx.saved_tensors, grads):
                if g is not None and isinstance(t, Tensor):
                    # Accumulate gradient (handle multiple outputs using same input)
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad = Tensor(t.grad.data + g.data, copy=False)

                    # Decrement output count and enqueue when all outputs processed
                    num_outputs[id(t)] -= 1
                    if num_outputs[id(t)] == 0:
                        queue.append(t)
