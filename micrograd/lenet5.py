# micrograd/lenet5.py
import numpy as np
from micrograd.engine import Tensor
from micrograd.nn import Module

def conv2d_forward(input, weight, bias, stride=1, padding=0):
    """2D convolution forward pass."""
    # input: (batch, in_channels, H, W)
    # weight: (out_channels, in_channels, kernel_h, kernel_w)
    batch, in_c, in_h, in_w = input.shape
    out_c, _, kernel_h, kernel_w = weight.shape

    out_h = (in_h - kernel_h + 2 * padding) // stride + 1
    out_w = (in_w - kernel_w + 2 * padding) // stride + 1

    output = np.zeros((batch, out_c, out_h, out_w))

    for b in range(batch):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    patch = input[b, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                    output[b, oc, oh, ow] = np.sum(patch * weight[oc]) + bias[oc]

    return output


class Conv2d(Module):
    """2D Convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0

        # Initialize weights: Gaussian with std=0.1
        self.w = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1,
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_channels), requires_grad=True)

    def __call__(self, x):
        return Tensor(conv2d_forward(x.data, self.w.data, self.b.data,
                                      self.stride, self.padding),
                      requires_grad=x.requires_grad)

    def parameters(self):
        return [self.w, self.b]


class AvgPool2d(Module):
    """Average pooling layer."""
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        batch, channels, h, w = x.data.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        output = np.zeros((batch, channels, out_h, out_w))

        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        patch = x.data[b, c, h_start:h_start+self.kernel_size,
                                                    w_start:w_start+self.kernel_size]
                        output[b, c, oh, ow] = np.mean(patch)

        return Tensor(output, requires_grad=x.requires_grad)

    def parameters(self):
        return []


class LeNet5(Module):
    """LeNet-5 for MNIST (28x28 input)."""
    def __init__(self):
        # Conv layers
        self.conv1 = Conv2d(1, 6, kernel_size=5)
        self.pool1 = AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)

        # FC layers (using existing Layer class)
        from micrograd.nn import Layer
        # After conv+pool: 16 channels * 4*4 = 256 features
        self.fc1 = Layer(256, 120)
        self.fc2 = Layer(120, 84)
        self.fc3 = Layer(84, 10, nonlin=False)  # No activation on output

    def __call__(self, x):
        # x: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = x.relu()
        x = self.pool1(x)

        x = self.conv2(x)
        x = x.relu()
        x = self.pool2(x)

        # Flatten
        batch = x.data.shape[0]
        x = x.reshape(batch, -1)

        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        return x

    def parameters(self):
        return (self.conv1.parameters() + self.pool1.parameters() +
                self.conv2.parameters() + self.pool2.parameters() +
                self.fc1.parameters() + self.fc2.parameters() +
                self.fc3.parameters())
