# Autograd Node Refactor Design

## Overview

Refactor the backward propagation to separate computation graph nodes (Node) from computation logic (Function). This enables:
- Intermediate tensors to be released by Python GC when not needed for backward
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
│   Node stores graph structure, Function stores compute logic │
└─────────────────────────────────────────────────────────────┘
```

**Separation Principle**:
- **Node**: Only stores graph structure (prev, next, leaf, backward_fn) and saved tensors for backward
- **Function**: Only stores forward/backward compute logic, no graph structure
- **Tensor**: Only stores data and grad, grad_fn points to Node (if operation output)

## Node Class

```python
class Node:
    def __init__(self):
        self.saved_tensors = ()      # tensors needed for backward
        self.saved_data = []         # other data (exponent, dim, etc.)
        self.prev: set[Node] = set() # upstream nodes (input grad_fn)
        self.next: set[Node] = set() # downstream nodes (output grad_fn)
        self.leaf: set[Tensor] = set() # leaf tensors (inputs without grad_fn)
        self.backward_fn = None      # Function class (Add, Mul, etc.)
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    
    def save_data_for_backward(self, *data):
        self.saved_data = list(data)
    
    def backward(self, grad_output):
        """Single BFS traversal, cleaning references after execution"""
        # See implementation details below
```

**prev vs leaf**:
- `prev`: Input tensor has `grad_fn` → points to that Node
- `leaf`: Input tensor has no `grad_fn` (leaf tensor with requires_grad=True)

**next**: Points to downstream nodes, used to determine when a node can execute (when next is empty)

## Function Class

```python
class Function:
    @staticmethod
    def forward(ctx: Node, *inputs) -> Storage:
        """ctx (Node) auto-created, save necessary data to ctx"""
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Node, grad_output) -> tuple[Tensor | None, ...]:
        """ctx auto-passed, backward retrieves saved data from ctx"""
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        """Auto-create Node (ctx), call forward, build computation graph"""
        ctx = Node()  # auto-create ctx
        
        needs_grad = any(t.requires_grad for t in inputs if isinstance(t, Tensor)) and not no_grad.enabled
        
        output_storage = cls.forward(ctx, *inputs)
        output = Tensor(output_storage, requires_grad=needs_grad)
        
        if needs_grad:
            output.grad_fn = ctx
            ctx.backward_fn = cls
            
            for t in inputs:
                if not isinstance(t, Tensor) or not t.requires_grad:
                    continue
                if t.grad_fn is None:
                    ctx.leaf.add(t)
                else:
                    ctx.prev.add(t.grad_fn)
                    t.grad_fn.next.add(ctx)
        
        return output
```

**ops.py Example (Add)**:
```python
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        forward_fn, _ = registry.dispatch("add", a.device)
        return forward_fn(a.data, b.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        _, backward_fn = registry.dispatch("add", grad_output.device)
        grad_a, grad_b = backward_fn(grad_output.data, a.data, b.data)
        return Tensor(grad_a), Tensor(grad_b)
```

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

## Node.backward Implementation

```python
def backward(self, grad_output):
    """Single BFS traversal with reference cleanup"""
    queue = deque([(self, grad_output)])
    
    while queue:
        node, grad = queue.popleft()
        
        grads = node.backward_fn.backward(node, grad)
        
        saved = list(node.saved_tensors)
        leaf_tensors = list(node.leaf)
        prev_nodes = list(node.prev)
        
        grad_idx = 0
        for t in saved:
            if t.requires_grad:
                g = grads[grad_idx]
                if g is not None:
                    if t.grad_fn is None:  # leaf
                        if t.grad is None:
                            t.grad = g
                        else:
                            t.grad = t.grad + g
                grad_idx += 1
        
        for prev_node in prev_nodes:
            prev_node.next.discard(node)
            if len(prev_node.next) == 0:
                queue.append((prev_node, grads[grad_idx]))
            grad_idx += 1
        
        node.prev.clear()
        node.leaf.clear()
```

**Key points**:
- Single BFS using `next` set to determine when node can execute
- Clean prev/next/leaf references after execution
- Gradients accumulate to leaf tensor's grad attribute

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
| `micrograd/node.py` | New file, Node class |
| `micrograd/function.py` | Rewrite Function.apply, adjust forward/backward signatures |
| `micrograd/ops.py` | All ops: forward returns Storage, backward signature adjusted |
| `micrograd/tensor.py` | Remove `_ctx/_use_count`, add `grad_fn`, rewrite `backward` |
| `micrograd/__init__.py` | Export `Node`, `no_grad` |
| `test/*.py` | Update tests for new API |

## Key Improvements

1. Computation graph built from Nodes, tensors no longer mutually referenced
2. Automatic prev/next/leaf cleanup after backward, memory released
3. no_grad supports decorator and context manager
4. Single BFS backward, no pre-processing for use_count