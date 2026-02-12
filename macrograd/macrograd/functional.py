from . import Tensor
import numpy as _np

def Softmax(tensor : Tensor):
    out = tensor.data
    normalizer = _np.max(out, axis=-1, keepdims=True)
    out = _np.exp(out - normalizer)
    out = out/_np.sum(out, axis=-1, keepdims=True)
    out = Tensor(out, parents=(tensor, ), op="softmax")
    def _backward():
        sum_g_sigma = _np.sum(out.grad * out.data, axis=-1, keepdims=True)
    
        tensor.grad += out.data * (out.grad - sum_g_sigma)

    out._backward = _backward
        
    return out

def ReLU(tensor : Tensor):
    out = Tensor(_np.maximum(tensor.data, 0), parents=(tensor, ), op="ReLU") 
    def _backward():
           tensor.grad += _np.where(tensor.data > 0, 1, 0) * out.grad
    out._backward = _backward
    return out


def _exp(tensor : Tensor):
    out = Tensor(_np.exp(tensor.data), parents=(tensor, ), op="exp")
    def _backward():
        tensor.grad += _np.exp(tensor.data) * out.grad
    out._backward = _backward
    return out

def _log(tensor : Tensor):
    out = Tensor(_np.log(tensor.data), parents=(tensor,), op="log")

    def _backward():
        tensor.grad += 1/tensor.data * out.grad
    out._backward = _backward
    return out

def _cos(tensor : Tensor):
    out = Tensor(_np.cos(tensor.data), parents=(tensor,), op="cos")
    def _backward():
        tensor.grad += -_np.sin(tensor.data) * out.grad
            
    out._backward = _backward
    return out

def _sqrt(tensor : Tensor):
    out = Tensor(_np.sqrt(tensor.data), parents=(tensor,), op="sqrt")
    def _backward():
        tensor.grad += (1/(2*_np.sqrt(tensor.data))) * out.grad
    out._backward = _backward
    return out

_func_map = {_np.exp : _exp,
            _np.log : _log,
            _np.cos : _cos,
            _np.sqrt : _sqrt}