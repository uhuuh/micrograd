# AGENTS.md

## Commands

```bash
pip install -e .     # Install package in editable mode
python -m pytest     # Run tests
```

## Architecture

This is a tensor-based autograd engine (evolved from the original scalar version):

- `micrograd/tensor.py` - Core `Tensor` class with autograd (BFS-based backward)
- `micrograd/function.py` - Base `Function` class for defining ops
- `micrograd/ops.py` - Operations: Add, Mul, Sub, Div, Neg, ReLU, Pow, MatMul, Sum, Slice
- `micrograd/dispatch.py` - Operator registry for device dispatch (CPU only currently)
- `micrograd/storage.py` - Abstract storage layer (CPUStorage wraps numpy)
- `micrograd/kernels/cpu.py` - CPU kernel implementations
- `micrograd/nn.py` - Simple MLP layers (scalar-based, uses `Tensor`)
- `micrograd/optim.py` - AdamW optimizer
- `micrograd/data.py` - Dataset/DataLoader + MNIST loader
- `micrograd/lenet5.py` - LeNet-5 CNN for MNIST

## Known Issues

- **Import path**: Use `from micrograd import Tensor` or `from micrograd.tensor import Tensor`. Do NOT use `from micrograd.engine import Tensor` - `engine.py` does not exist.
- Several files still have incorrect imports (`optim.py`, `lenet5.py`, `test/*.py`, `train_mnist.py`) - they need `micrograd.engine` → `micrograd.tensor`.

## Tests

- Tests use PyTorch to verify gradient correctness
- Two test directories: `tests/` (leaf gradient tests) and `test/` (tensor, optim, lenet5, integration)
- Tests currently fail due to import errors - need to fix `micrograd.engine` imports first