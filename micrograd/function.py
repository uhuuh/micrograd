# micrograd/function.py


class Function:
    """Base class for autograd operations. Subclass with forward/backward static methods."""

    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = []

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def save_data_for_backward(self, *data):
        self.saved_data = list(data)

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output) -> tuple["Tensor | None", ...]:
        """Backward pass. Must return a tuple of gradients (Tensor or None) for each input."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        """Apply the operation: creates ctx, runs forward, attaches grad_fn to output."""
        ctx = cls()
        output = cls.forward(ctx, *inputs)
        output._ctx = ctx
        ctx._grad_fn = cls
        return output