class no_grad:
    enabled = False
    
    def __enter__(self):
        no_grad.enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        no_grad.enabled = False
        return False
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper