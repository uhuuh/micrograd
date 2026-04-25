import pytest
from micrograd.node import Node

def test_node_initialization():
    node = Node()
    assert node.saved_tensors == ()
    assert node.saved_data == []
    assert node.prev == set()
    assert node.next == set()
    assert node.leaf == set()

def test_node_save_for_backward():
    from micrograd.tensor import Tensor
    node = Node()
    t1 = Tensor([1.0])
    t2 = Tensor([2.0])
    node.save_for_backward(t1, t2)
    assert node.saved_tensors == (t1, t2)

def test_node_save_data_for_backward():
    node = Node()
    node.save_data_for_backward(2, 3)
    assert node.saved_data == [2, 3]