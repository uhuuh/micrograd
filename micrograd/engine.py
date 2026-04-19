import numpy as np


class Function:
    """Base class for autograd operations. Subclass with forward/backward static methods."""

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
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
        a, b = ctx.saved_tensors
        grad_a = grad_output.data * np.ones_like(a.data)
        grad_b = grad_output.data * np.ones_like(b.data)
        return Tensor(grad_a), Tensor(grad_b)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data
        return Tensor(grad_a), Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, Tensor(-grad_output.data)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output.data / b.data
        grad_b = -a.data * grad_output.data / (b.data ** 2)
        return Tensor(grad_a), Tensor(grad_b)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return Tensor(-a.data, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return Tensor(-grad_output.data)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return Tensor(grad_output.data * (a.data > 0).astype(float))


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a, exponent)
        return Tensor(a.data ** exponent, requires_grad=a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        a, exponent = ctx.saved_tensors
        return Tensor(exponent * (a.data ** (exponent - 1)) * grad_output.data), None


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
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

    def backward(self):
        """Compute gradient of this tensor with respect to leaf tensors."""
        if self._ctx is None:
            # This tensor is a leaf (no ops applied), just set grad to 1
            self.grad = Tensor(np.ones_like(self.data))
            return

        # Build topological order of the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                if v._ctx is not None:
                    for child in v._ctx.saved_tensors:
                        build_topo(child)
                topo.append(v)
        build_topo(self)

        # Initialize gradient at output = 1
        self.grad = Tensor(np.ones_like(self.data))
        grad_table = {id(self): self.grad}

        # Process in reverse topological order
        for v in reversed(topo):
            if v._ctx is not None:
                grad_fn = v._ctx._grad_fn
                inputs = v._ctx.saved_tensors
                grad_output = grad_table.get(id(v), Tensor(np.zeros_like(v.data)))

                # Call the op's backward
                grads = grad_fn.backward(v._ctx, grad_output)
                if not isinstance(grads, tuple):
                    grads = (grads,)

                # Accumulate gradients for each input
                for inp, g in zip(inputs, grads):
                    if inp.requires_grad and g is not None:
                        if id(inp) in grad_table:
                            grad_table[id(inp)].data += g.data
                        else:
                            grad_table[id(inp)] = g

        # Propagate gradients from grad_table to actual tensors
        for tensor_id, grad in grad_table.items():
            # Find the tensor by iterating (in practice, tensors are leaf nodes)
            pass  # grad_table already has correct gradients

        # Final pass: set .grad on all visited tensors that require grad
        visited_ids = set()
        def mark_visited(v):
            if id(v) not in visited_ids:
                visited_ids.add(id(v))
                if v._ctx is not None:
                    for child in v._ctx.saved_tensors:
                        mark_visited(child)
        mark_visited(self)

        # Now set .grad from grad_table
        for v in topo:
            if v.requires_grad and id(v) in grad_table:
                v.grad = grad_table[id(v)]
