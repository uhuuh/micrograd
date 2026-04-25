# micrograd/ops.py
from .function import Function
from .dispatch import registry


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("add", a.device)
        return forward_fn(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("mul", a.device)
        return forward_fn(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("mul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        from .tensor import Tensor
        return Tensor(grad_a), Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("sub", a.device)
        return forward_fn(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        from .tensor import Tensor
        return grad_output, Tensor(-grad_output.data.numpy(), copy=False)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("div", a.device)
        return forward_fn(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("div", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        from .tensor import Tensor
        return Tensor(grad_a), Tensor(grad_b)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("neg", a.device)
        return forward_fn(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        from .tensor import Tensor
        return Tensor(-grad_output.data.numpy(), copy=False)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("relu", a.device)
        return forward_fn(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        _, backward_fn = registry.dispatch("relu", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data)
        from .tensor import Tensor
        return (Tensor(grad_a),)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(exponent)
        forward_fn, _ = registry.dispatch("pow", a.device)
        return forward_fn(a.data, exponent)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        exponent, = ctx.saved_data
        _, backward_fn = registry.dispatch("pow", grad_output.device)
        grad_a, _ = backward_fn(grad_output.data, a.data, exponent)
        from .tensor import Tensor
        return Tensor(grad_a), None


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("matmul", a.device)
        return forward_fn(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("matmul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        from .tensor import Tensor
        return Tensor(grad_a), Tensor(grad_b)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None, keepdim=False):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(dim, keepdim)
        forward_fn, _ = registry.dispatch("sum", a.device)
        return forward_fn(a.data, dim, keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        dim, keepdim = ctx.saved_data
        _, backward_fn = registry.dispatch("sum", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, dim, keepdim)
        from .tensor import Tensor
        return (Tensor(grad_a),)


class Slice(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(key)
        forward_fn, _ = registry.dispatch("slice", a.device)
        return forward_fn(a.data, key)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        key, = ctx.saved_data
        _, backward_fn = registry.dispatch("slice", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data, key)
        from .tensor import Tensor
        return (Tensor(grad_a),)