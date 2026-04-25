import pytest
from micrograd.node import Node
from micrograd.tensor import Tensor
from micrograd import no_grad
from micrograd.storage import CPUStorage
import numpy as np


def test_node_initialization():
    node = Node()
    assert node.saved_tensors == ()
    assert node.saved_data == []
    assert node.prev == set()
    assert node.next == set()
    assert node.leaf == set()


def test_node_save_for_backward():
    t1 = Tensor([1.0])
    t2 = Tensor([2.0])
    node = Node()
    node.save_for_backward(t1, t2)
    assert node.saved_tensors == (t1, t2)


def test_node_save_data_for_backward():
    node = Node()
    node.save_data_for_backward(2, 3)
    assert node.saved_data == [2, 3]


class AddNode(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return CPUStorage(a.data.numpy() + b.data.numpy())
    
    @staticmethod
    def _backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(grad_output.data), Tensor(grad_output.data)


class MulNode(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return CPUStorage(a.data.numpy() * b.data.numpy())
    
    @staticmethod
    def _backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return Tensor(grad_output.data.numpy() * b.data.numpy()), Tensor(grad_output.data.numpy() * a.data.numpy())


def test_apply_creates_output_tensor():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddNode.apply(a, b)
    
    assert isinstance(out, Tensor)
    assert out.requires_grad == True
    np.testing.assert_array_almost_equal(out.numpy(), [5.0])


def test_apply_no_grad_mode():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    with no_grad.no_grad():
        out = AddNode.apply(a, b)
    
    assert out.requires_grad == False


def test_apply_no_grad_decorator():
    @no_grad.no_grad()
    def compute():
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        return AddNode.apply(a, b)
    
    out = compute()
    assert out.requires_grad == False


def test_apply_sets_grad_fn():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddNode.apply(a, b)
    
    assert out.grad_fn is not None
    assert isinstance(out.grad_fn, AddNode)


def test_apply_tracks_leaf_tensors():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddNode.apply(a, b)
    
    assert a in out.grad_fn.leaf
    assert b in out.grad_fn.leaf


def test_apply_tracks_previous_nodes():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out1 = MulNode.apply(a, b)
    out2 = AddNode.apply(out1, a)
    
    assert out1.grad_fn in out2.grad_fn.prev
    assert out2.grad_fn in out1.grad_fn.next


def test_backward_simple():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = AddNode.apply(a, b)
    grad_out = Tensor([1.0])
    
    out.grad_fn.backward(grad_out)
    
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_array_almost_equal(a.grad.numpy(), [1.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [1.0])


def test_backward_mul():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out = MulNode.apply(a, b)
    grad_out = Tensor([1.0])
    
    out.grad_fn.backward(grad_out)
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [3.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0])


def test_backward_chain():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    c = MulNode.apply(a, b)
    out = AddNode.apply(c, a)
    
    out.grad_fn.backward(Tensor([1.0]))
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [4.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [2.0])


def test_backward_accumulates_grad():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    out1 = MulNode.apply(a, b)
    out2 = MulNode.apply(a, b)
    out = AddNode.apply(out1, out2)
    
    out.grad_fn.backward(Tensor([1.0]))
    
    np.testing.assert_array_almost_equal(a.grad.numpy(), [6.0])
    np.testing.assert_array_almost_equal(b.grad.numpy(), [4.0])