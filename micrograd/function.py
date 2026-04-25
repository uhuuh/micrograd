from collections import deque

class Function:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = []
        self.prev: set[Function] = set()
        self.next: set[Function] = set()
        self.leaf: set['Tensor'] = set()
    
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    
    def save_data_for_backward(self, *data):
        self.saved_data = list(data)
    
    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        from .tensor import Tensor
        from .no_grad import no_grad
        
        ctx = cls()
        
        needs_grad = any(isinstance(t, Tensor) and t.requires_grad for t in inputs) and not no_grad.enabled
        
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
    
    def backward_loop(self, grad_output):
        from .tensor import Tensor
        
        queue = deque([(self, grad_output)])
        
        while queue:
            fn, grad = queue.popleft()
            
            grads = type(fn).backward(fn, grad)
            
            for i, t in enumerate(fn.saved_tensors):
                if not t.requires_grad:
                    continue
                g = grads[i] if i < len(grads) else None
                if g is None:
                    continue
                
                if t.grad_fn is None:
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad = Tensor(t.grad.data.numpy() + g.data.numpy(), copy=False)
                else:
                    t.grad_fn.next.discard(fn)
                    if len(t.grad_fn.next) == 0:
                        queue.append((t.grad_fn, g))
            
            fn.prev.clear()
            fn.leaf.clear()
            fn.saved_tensors = ()