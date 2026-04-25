from collections import deque

class Node:
    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = []
        self.prev: set[Node] = set()
        self.next: set[Node] = set()
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
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError