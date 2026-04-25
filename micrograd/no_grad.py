class no_grad:
    enabled = False
    
    def __init__(self):
        self._prev = None
    
    def __enter__(self):
        self._prev = no_grad.enabled
        no_grad.enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        no_grad.enabled = self._prev
        return False
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper