# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

micrograd is a tiny autograd engine (~100 lines) implementing reverse-mode autodiff over a dynamically built DAG of scalar values, plus a small neural networks library (~50 lines) with a PyTorch-like API.

## Commands

```bash
pip install -e .     # Install package in editable mode for development
python -m pytest     # Run tests (requires PyTorch installed for gradient verification)
```

## Architecture

### Core: `micrograd/engine.py`
The `Value` class is the fundamental building block:
- Stores `data` (scalar) and `grad` (gradient)
- Maintains `_prev` (children nodes) and `_op` (operation that produced this node) for graph construction
- `backward()` performs topological sort, sets `self.grad = 1`, then applies chain rule in reverse order
- Supported ops: `+`, `*`, `**`, `relu` (also `__neg__`, `__sub__`, `__truediv__` derived from these)

### NN Library: `micrograd/nn.py`
Built on top of `Value`:
- `Module`: Base class with `zero_grad()` and `parameters()`
- `Neuron`: Single neuron with weights `w`, bias `b`, optional ReLU activation
- `Layer`: Collection of `Neuron`s
- `MLP`: Multi-layer perceptron stacking `Layer`s; last layer has no activation (linear output)

### Notebooks
- `demo.ipynb`: Trains a 2-layer MLP on the moon dataset for binary classification
- `trace_graph.ipynb`: Uses `draw_dot` to visualize the computation graph with `data` and `grad` values

## Design Notes

- The DAG is built dynamically during forward pass; each `Value` operation returns a new `Value` with `_prev` pointing to operands
- `backward()` is intentionally simple: it doesn't zero gradients (allows accumulation via `+=`)
- Tests validate gradients against PyTorch's autograd using tolerance `1e-6`
