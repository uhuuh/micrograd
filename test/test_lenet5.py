# test/test_lenet5.py
import numpy as np
from micrograd.engine import Tensor
from micrograd.lenet5 import Conv2d, AvgPool2d

def test_conv2d_forward():
    # Input: (batch=1, channels=1, H=4, W=4)
    x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
    conv = Conv2d(1, 1, kernel_size=2)
    out = conv(x)
    # Output: (1, 1, 3, 3) assuming no padding, stride=1
    assert out.data.shape == (1, 1, 3, 3)

def test_conv2d_parameters():
    conv = Conv2d(3, 16, kernel_size=5)
    params = conv.parameters()
    assert len(params) == 2  # w and b
    assert params[0].data.shape == (16, 3, 5, 5)
    assert params[1].data.shape == (16,)

def test_avgpool2d_forward():
    # Input: (batch=1, channels=1, H=4, W=4)
    x = Tensor(np.ones((1, 1, 4, 4)), requires_grad=True)
    pool = AvgPool2d(kernel_size=2, stride=2)
    out = pool(x)
    # Output: (1, 1, 2, 2)
    assert out.data.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 1.0)
