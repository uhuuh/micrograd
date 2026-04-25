# test/test_optim.py
import numpy as np
from micrograd.tensor import Tensor
from micrograd.optim import AdamW

def test_adamw_initialization():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0], requires_grad=True)
    optim = AdamW([p1, p2], lr=0.001, weight_decay=0.01)
    assert optim.lr == 0.001
    assert optim.weight_decay == 0.01
    assert len(optim.params) == 2

def test_adamw_step():
    p = Tensor([1.0], requires_grad=True)
    p.grad = Tensor([0.5])
    optim = AdamW([p], lr=0.01, weight_decay=0.0)
    old_val = p.data.copy()
    optim.step()
    # AdamW update should change the value (momentum-based)
    assert not np.allclose(p.data, old_val)
    # Value should decrease with positive gradient
    assert p.data < old_val

def test_adamw_zero_grad():
    p = Tensor([1.0], requires_grad=True)
    p.grad = Tensor([0.5])
    optim = AdamW([p])
    optim.zero_grad()
    assert p.grad is None
