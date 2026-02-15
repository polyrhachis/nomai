from . import Tensor
from . import numpy as nnp
import numpy as _np
import cupy as _cp





@_cp.fuse()
def Softmax(tensor : Tensor):
    xp = tensor.xp
    out = tensor.data
    normalizer = xp.max(out, axis=-1, keepdims=True)
    out = xp.exp(out - normalizer)
    out = out/xp.sum(out, axis=-1, keepdims=True)
    out = Tensor(out, parents=(tensor, ), op="softmax")
    def _backward():
        sum_g_sigma = xp.sum(out.grad * out.data, axis=-1, keepdims=True)
    
        tensor.grad += out.data * (out.grad - sum_g_sigma)

    out._backward = _backward
        
    return out

def ReLU(tensor : Tensor):
    xp = tensor.xp
    out = Tensor(xp.maximum(tensor.data, 0), parents=(tensor, ), op="ReLU") 
    def _backward():
           tensor.grad += xp.where(tensor.data > 0, 1, 0) * out.grad
    out._backward = _backward
    return out

def cross_entropy(logits, targets):
    xp = logits.xp
    probs = Softmax(logits)  
    
    N = targets.shape[0]
    
    correct_probs = probs[
        xp.arange(N),
        targets.data.astype(xp.int32)
    ]
    
    log_probs = nnp.log(correct_probs)
    
    return -log_probs.sum() / N
