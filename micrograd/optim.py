# micrograd/optim.py
import random
import numpy as np
from micrograd.tensor import Tensor

class AdamW:
    def __init__(self, parameters, lr=0.001, weight_decay=0.01,
                 betas=(0.9, 0.999), eps=1e-8):
        self.params = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # Initialize first and second moment estimates
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data

            # Decoupled weight decay (AdamW)
            p.data = p.data - self.lr * self.weight_decay * p.data

            # Momentum update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Velocity update
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
