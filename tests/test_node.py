import pytest
from micrograd.function import Function
from micrograd.tensor import Tensor
from micrograd.no_grad import no_grad
from micrograd.storage import CPUStorage
import numpy as np


def test_node_initialization():
    fn = Function()
    assert fn.saved_tensors == ()
    assert fn.saved_data == []
    assert fn.prev == set()
    assert fn.next == set()
    assert fn.leaf == set()

def test_node_save_for_backward():
    from micrograd.tensor import Tensor
    fn = Function()
    t1 = Tensor([1.0])
    t2 = Tensor([2.0])
    fn.save_for_backward(t1, t2)
    assert fn.saved_tensors == (t1, t2)

def test_node_save_data_for_backward():
    fn = Function()
    fn.save_data_for_backward(2, 3)
    assert fn.saved_data == [2, 3]


class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        from micrograd.dispatch import registry
        forward_fn, _ = registry.dispatch("add", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class MulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        from micrograd.dispatch import registry
        forward_fn, _ = registry.dispatch("mul", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        from micrograd.dispatch import registry
        _, backward_fn = registry.dispatch("mul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        from micrograd.tensor import Tensor
        return Tensor(grad_a), Tensor(grad_b)


def test_apply_creates_output_tensor():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    
    out = AddFunction.apply(a, b)
    
    assert isinstance(out, Tensor)
    assert np.allclose(out.data.numpy(), [3.0])
    assert out.requires_grad == True
    assert out.grad_fn is not None
    assert isinstance(out.grad_fn, AddFunction)

def test_apply_no_grad_mode():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    with no_grad():
        out = AddFunction.apply(a, b)
    
    assert out.requires_grad == False

def test_apply_no_grad_decorator():
    @no_grad()
    def compute():
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        return AddFunction.apply(a, b)
    
    out = compute()
    assert out.requires_grad == False

def test_apply_sets_grad_fn():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddFunction.apply(a, b)
    
    assert out.grad_fn is not None
    assert isinstance(out.grad_fn, AddFunction)

def test_apply_tracks_leaf_tensors():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddFunction.apply(a, b)
    
    assert a in out.grad_fn.leaf
    assert b in out.grad_fn.leaf

def test_apply_tracks_previous_nodes():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out1 = MulFunction.apply(a, b)
    out2 = AddFunction.apply(out1, a)
    
    assert out1.grad_fn in out2.grad_fn.prev
    assert out2.grad_fn in out1.grad_fn.next

def test_backward_simple():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddFunction.apply(a, b)
    grad_out = Tensor([1.0])
    
    out.grad_fn.backward_loop(grad_out)
    
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_array_almost_equal(a.grad.numpy(), [1.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [1.0])

def test_backward_mul():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = MulFunction.apply(a, b)
    grad_out = Tensor([1.0])
    
    out.grad_fn.backward_loop(grad_out)
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [3.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0])

def test_backward_chain():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    c = MulFunction.apply(a, b)
    out = AddFunction.apply(c, a)
    
    out.grad_fn.backward_loop(Tensor([1.0]))
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [4.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0])

def test_backward_accumulates_grad():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out1 = MulFunction.apply(a, b)
    out2 = MulFunction.apply(a, b)
    out = AddFunction.apply(out1, out2)
    
    out.grad_fn.backward_loop(Tensor([1.0]))
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [6.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [4.0])