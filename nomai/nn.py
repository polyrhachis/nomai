from .core import Tensor
import numpy as _np
from . import functional as F
from . import numpy as nnp
import cupy as _cp

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
    

    def to(self, device):
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                value.to(device)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.to(device)
                        
            elif isinstance(value, Module):
                value.to(device)



class Linear(Module):
    def __init__(self, dim, use_bias=True):
        self.dim = dim
        self.w = None
        self.b = None  
        self.initialized = False
        self.use_bias = use_bias

    def __call__(self, x):
        if not self.initialized:

            self.w = Tensor(_cp.random.normal(size=(x.shape[-1], self.dim))/10)
            self.b = Tensor(_cp.zeros((self.dim,)))
            self.initialized = True
        out = x@self.w
        if self.use_bias:
            out += self.b
        return out

    
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
    

class LayerNorm(Module):
    def __init__(self, dim):
        self.gamma = Tensor(_cp.ones((dim)))
        self.beta = Tensor(_cp.zeros((dim)))

    def __call__(self, x):
        eps = 1e-5
    
        mean = nnp.mean(x, axis=-1, keepdims=True)
        var = nnp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / nnp.sqrt(var + eps)
        
        return self.gamma * x_hat + self.beta



class SelfAttention(Module):
    def __init__(self, dim):
        self.Qw = Tensor(_cp.random.normal(size=(dim, dim))/10)
        self.Kw = Tensor(_cp.random.normal(size=(dim, dim))/10)
        self.Vw = Tensor(_cp.random.normal(size=(dim, dim))/10)
        self.mask = None
    
    def get_QKV(self, x):
        Q = x@self.Qw
        K = x@self.Kw
        V = x@self.Vw
        return Q, K, V
    
    
    def __call__(self, x):
        xp = x.xp

        mask_matrix = xp.tril(xp.ones((x.shape[-2], x.shape[-2])))
        

        self.mask = Tensor(xp.where(mask_matrix == 0, float('-inf'), 0.0), requires_grad=False)
        
        Q, K, V = self.get_QKV(x)
        scores = Q@K.swapaxes(-1, -2)/nnp.sqrt(Tensor(xp.array(x.shape[-1])))
        scores = scores + self.mask  
        return F.Softmax(scores) @ V + x
    


class FFN(Module):
    def __init__(self, dim_in, dim, activation):
        self.linear1 = Linear(dim=dim)
        self.linear2 = Linear(dim=dim_in)
        self.activation = activation
    
    def __call__(self, x):
        return x + self.linear2(self.activation(self.linear1(x)))

class Embedding(Module):
    def __init__(self, vocab_size, dim):
        self.emb = Tensor(_cp.random.normal(size=(vocab_size, dim))/10)

    def __call__(self, x):
        return self.emb[x]



class GPT(Module):

    def __init__(self, Sequential, embedding_dim, vocab_size, block_size):
        self.emb = Embedding(vocab_size=vocab_size, dim=embedding_dim)
        self.positional_emb = Embedding(vocab_size=block_size, dim=embedding_dim)
        self.Sequential = Sequential
     
    
    def __call__(self, x):
        x = self.emb(x)
        pos = _cp.arange(x.shape[-2])
        x = x + self.positional_emb(pos)
        return self.Sequential(x)




class Softmax:
    def __call__(self, tensor : Tensor):
        return F.Softmax(tensor)



class ReLU:
    def __call__(self, tensor : Tensor):
        return F.ReLU(tensor)
