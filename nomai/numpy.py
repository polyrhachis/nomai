from . import Tensor
import numpy as _np
import cupy as _cp



def one_hot(tensor : Tensor, num_classes):
    xp = tensor.xp
    out = xp.zeros((num_classes))
    out[tensor.data.astype(xp.int32)] = 1
    return Tensor(out, parents=tensor.parents, op="one_hot")


def exp(tensor : Tensor):
    xp = tensor.xp
    out = Tensor(xp.exp(tensor.data), parents=(tensor, ), op="exp")
    def _backward():
        tensor.grad += xp.exp(tensor.data) * out.grad
    out._backward = _backward
    return out

def log(tensor : Tensor):
    xp = tensor.xp
    out = Tensor(xp.log(tensor.data), parents=(tensor,), op="log")

    def _backward():
        tensor.grad += 1/tensor.data * out.grad
    out._backward = _backward
    return out

def cos(tensor : Tensor):
    xp = tensor.xp
    out = Tensor(xp.cos(tensor.data), parents=(tensor,), op="cos")
    def _backward():
        tensor.grad += -xp.sin(tensor.data) * out.grad
            
    out._backward = _backward
    return out

def sqrt(tensor : Tensor):
    xp = tensor.xp
    out = Tensor(xp.sqrt(tensor.data), parents=(tensor,), op="sqrt")
    def _backward():
        tensor.grad += (1/(2*xp.sqrt(tensor.data))) * out.grad
    out._backward = _backward
    return out


def mean(tensor: Tensor, axis=None, keepdims=False):
    xp = tensor.xp
    out = Tensor(xp.mean(tensor.data, axis=axis, keepdims=keepdims), parents=(tensor,), op="mean")

    def _backward():
        if axis is None:
            grad = xp.ones_like(tensor.data) * (out.grad / tensor.data.size)
        else:
            n = tensor.data.shape[axis]
            grad = out.grad / n
            
           
            if not keepdims:
                grad = xp.expand_dims(grad, axis)
                
        tensor.grad += grad

    out._backward = _backward
    return out


def var(tensor : Tensor, axis=None, keepdims=False):
    xp = tensor.xp
    out = Tensor(xp.var(tensor.data, axis=axis, keepdims=keepdims), parents=(tensor, ), op="var")
    
    def _backward():
        if axis is None:
            n = tensor.data.size
            
            grad = (2/n) * (tensor.data - xp.mean(tensor.data)) * out.grad
            tensor.grad += grad
        else:
            n = tensor.data.shape[axis]
        
            mean = xp.mean(tensor.data, axis=axis, keepdims=True)
            
            upstream_grad = out.grad
            
            if not keepdims:
                upstream_grad = xp.expand_dims(upstream_grad, axis)
                
            
            grad = (2/n) * (tensor.data - mean) * upstream_grad
            tensor.grad += grad
            
    out._backward = _backward
    return out
