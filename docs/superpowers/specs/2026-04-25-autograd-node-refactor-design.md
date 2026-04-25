# Autograd Node Refactor Design

## Overview

Refactor backward propagation with unified Node class that stores both graph structure and compute logic:
- Intermediate tensors released by Python GC when not needed for backward
- Single BFS traversal using prev/next graph structure
- Automatic graph cleanup after backward

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Computation Graph                       │
│                                                              │
│   Tensor (leaf) ←─── Node ←─── Node ←─── Tensor (output)    │
│       │              │          │              │            │
│    grad=None     backward    backward      grad_fn=Node     │
│                  (backward)  (backward)                     │
│                                                              │
│   Node: graph structure + forward/backward logic (unified)  │
└─────────────────────────────────────────────────────────────┘
```

**Key Design**:
- **Node**: Single class storing graph structure (prev/next/leaf/saved_tensors) AND forward/backward logic
- **Tensor**: Only stores data and grad, grad_fn points to Node instance

## Node Class

```python
class Node:
    def __init__(self):
        self.saved_tensors = ()      # tensors NECESSARY for backward only
        self.saved_data = []         # non-tensor data (exponent, dim, etc.)
        self.prev: set[Node] = set() # upstream nodes (input grad_fn)
        self.next: set[Node] = set() # downstream nodes (output grad_fn)
        self.leaf: set[Tensor] = set() # leaf tensors (inputs without grad_fn)
    
    def save_for_backward(self, *tensors):
        """Save only tensors NEEDED for backward computation"""
        self.saved_tensors = tensors
    
    def save_data_for_backward(self, *data):
        self.saved_data = list(data)
    
    @staticmethod
    def forward(ctx, *inputs) -> Storage:
        """ctx auto-created (self), save necessary data for backward"""
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Tensor, ...]:
        """ctx auto-passed (self), return gradients for saved tensors"""
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        """Auto-create Node instance, call forward, build graph"""
        ctx = cls()  # Node instance
        
        needs_grad = any(t.requires_grad for t in inputs if isinstance(t, Tensor)) and not no_grad.enabled
        
        output_storage = cls.forward(ctx, *inputs)
        output = Tensor(output_storage, requires_grad=needs_grad)
        
        if needs_grad:
            output.grad_fn = ctx
            
            for t in inputs:
                if not isinstance(t, Tensor) or not t.requires_grad:
                    continue
                if t.grad_fn is None:
                    ctx.leaf.add(t)
                else:
                    ctx.prev.add(t.grad_fn)
                    t.grad_fn.next.add(ctx)
        
        return output
    
    def backward(self, grad_output):
        """BFS traversal, cleaning references"""
        queue = deque([(self, grad_output)])
        
        while queue:
            node, grad = queue.popleft()
            
            grads = type(node).backward(node, grad)
            
            for i, t in enumerate(node.saved_tensors):
                if not t.requires_grad or grads[i] is None:
                    continue
                
                if t.grad_fn is None:  # leaf
                    if t.grad is None:
                        t.grad = grads[i]
                    else:
                        t.grad = t.grad + grads[i]
                else:
                    t.grad_fn.next.discard(node)
                    if len(t.grad_fn.next) == 0:
                        queue.append((t.grad_fn, grads[i]))
            
            node.prev.clear()
            node.leaf.clear()
            node.saved_tensors = ()
```

**prev vs leaf**:
- `prev`: Input tensor has `grad_fn` → points to that Node
- `leaf`: Input tensor has no `grad_fn` (leaf tensor with requires_grad=True)

**next**: Points to downstream nodes, used to determine when node can execute (when next is empty)

**saved_tensors principle**: Only save tensors ACTUALLY used in backward:
- Add.backward: doesn't need a, b (grad = grad_output)
- Mul.backward: needs a, b (grad_a = b * grad_output)
- ReLU.backward: needs a (mask where a > 0)

## no_grad Implementation

```python
class no_grad:
    enabled = False  # global state
    
    def __init__(self):
        self._prev = None
    
    def __enter__(self):
        self._prev = no_grad.enabled
        no_grad.enabled = True
        return self
    
    def __exit__(self, *args):
        no_grad.enabled = self._prev
    
    def __call__(self, func):
        """Decorator usage"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
```

**Usage**:
```python
with no_grad():
    y = model(x)  # no graph built

@no_grad()
def inference(x):
    return model(x)
```

**Function.apply checks**: `needs_grad = ... and not no_grad.enabled`

## ops.py Examples

**Add - backward doesn't need inputs**:
```python
class Add(Node):
    @staticmethod
    def forward(ctx, a, b):
        # nothing saved - backward doesn't need inputs
        forward_fn, _ = registry.dispatch("add", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
```

**Mul - backward needs inputs**:
```python
class Mul(Node):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("mul", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("mul", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)
```

**ReLU - backward needs input**:
```python
class ReLU(Node):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        forward_fn, _ = registry.dispatch("relu", a.device)
        return forward_fn(a.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        _, backward_fn = registry.dispatch("relu", grad_output.device)
        grad_a = backward_fn(grad_output.data, a.data)
        return Tensor(grad_a)
```

**Pow - non-tensor input**:
```python
class Pow(Node):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.save_data_for_backward(exponent)
        forward_fn, _ = registry.dispatch("pow", a.device)
        return forward_fn(a.data, exponent)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        exponent = ctx.saved_data[0]
        _, backward_fn = registry.dispatch("pow", grad_output.device)
        grad_a, _ = backward_fn(grad_output.data, a.data, exponent)
        return Tensor(grad_a)
```

## Tensor Class Changes

```python
class Tensor:
    def __init__(self, data, requires_grad=False, device="cpu", copy=True):
        # ... storage init ...
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None  # points to Node or None (leaf)
    
    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("backward() called on tensor with requires_grad=False")
        
        if self.grad_fn is None:
            raise RuntimeError("backward() called on leaf tensor. "
                              "Call backward on output of computation.")
        
        grad = Tensor(np.ones_like(self._storage.numpy()), copy=False)
        self.grad_fn.backward(grad)
        self.grad_fn = None  # cleanup
```

**Removed attributes**: `_ctx`, `_use_count`

**Kept attributes**: `requires_grad`, `grad`, `grad_fn`

## File Changes

| File | Change |
|------|--------|
| `micrograd/node.py` | New file, unified Node class |
| `micrograd/ops.py` | All ops extend Node, adjust forward/backward |
| `micrograd/tensor.py` | Remove `_ctx/_use_count`, add `grad_fn`, rewrite `backward` |
| `micrograd/__init__.py` | Export `Node`, `no_grad` |
| `micrograd/function.py` | Delete (merged into Node) |
| `test/*.py` | Update tests for new API |

## Key Improvements

1. Computation graph built from Nodes, tensors no longer mutually referenced
2. Automatic prev/next/leaf cleanup after backward, memory released
3. no_grad supports decorator and context manager
4. Single BFS backward, no pre-processing for use_count