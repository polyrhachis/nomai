from . import Tensor
import numpy as _np





class SGD:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for tensor in self.parameters:
            tensor.data = tensor.data - self.lr * tensor.grad
    
    def zero_grad(self):
        for tensor in self.parameters:
            tensor.grad = _np.zeros_like(tensor.data)