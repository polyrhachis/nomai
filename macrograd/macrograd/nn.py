from .core import Tensor
import numpy as _np
from . import functional as F

class Module:

    
    def parameters(self):
        params = []
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                params.append(value)

            elif isinstance(value,(list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())

                    elif isinstance(item, Tensor):
                        params.append(item)

            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params


class Linear(Module):
    def __init__(self, dim):
        self.dim = dim
        self.w = None
        self.b = None  
        self.initialized = False

    def __call__(self, x):
        if not self.initialized:
            self.w = Tensor(_np.random.normal(size=(x.shape[-1], self.dim))/100)
            self.b = Tensor(_np.zeros((self.dim,)))
            self.initialized = True
        return x@self.w + self.b

    
class Sequential(Module):

    """
    a Sequential class just like the PyTorch one,
    it needs to be initialized by passing a dummy input before the training, to create the parameters
    """
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    




class Softmax:
    def __call__(self, tensor : Tensor):
        return F.Softmax(tensor)



class ReLU:
    def __call__(self, tensor : Tensor):
        return F.ReLU(tensor)
