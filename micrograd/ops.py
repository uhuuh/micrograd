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


# Tensor forward reference - import at end to avoid circular dependency
from .tensor import Tensor