import numpy as _np
import cupy as _cp

class Tensor:

    """
    a Tensor who carries a xp.array (NumPy or CuPy), and tracks all the computation for 
    computing the backward pass
    """

    def __init__(self, data, requires_grad=True, parents=(), op=""):
        if isinstance(data, _cp.ndarray):
            self.xp = _cp
        else:
            self.xp = _np
        self.data = self.xp.array(data, dtype=self.xp.float32)
        self.shape = self.data.shape
        self.grad = self.xp.zeros_like(data, dtype=self.xp.float32)
        self.parents = parents
        self._backward = lambda: None
        self.op = op
        self.requires_grad = requires_grad

    def __getitem__(self, key):
        if type(key) == Tensor:
            key = key.data.astype(self.xp.int32)
        out = Tensor(self.data[key], parents=(self, ), op="indexing")
        
        def _backward():
            if self.requires_grad:
                self.xp.add.at(self.grad, key, out.grad)
        
        out._backward = _backward
        
        return out

    def __repr__(self):
        return f"Tensor: {self.data}"

    def _type_control(self, obj):
        obj = obj if isinstance(obj, Tensor) else Tensor(self.xp.array(obj))
        return obj
    
    def to(self, device):
        if device == 'cuda':
            self.data = _cp.array(self.data)
            self.grad = _cp.array(self.grad)
            self.xp = _cp
        if device == 'cpu':
            self.data = _np.array(self.data)
            self.grad = _np.array(self.grad)
            self.xp = _np
    


    def __add__(self, other):
        other = self._type_control(other)
        out = Tensor(self.data + other.data, parents=(self, other), op="add")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(1.0 * out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(1.0 * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = self._type_control(other)
        out = Tensor(self.data - other.data, parents=(self, other), op="sub")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(1.0 * out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(-1.0 * out.grad, other.shape)
        out._backward = _backward
        return out    
    
    def __rsub__(self, other):
        other = self._type_control(other)
        return other - self
    
    def __mul__(self, other):
        other = self._type_control(other)
        out = Tensor(self.data * other.data, parents=(self, other), op="mul")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out    
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self._type_control(other)
        out = Tensor(self.data / other.data, parents=(self, other), op="div")
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(1/other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad +=  self._unbroadcast((-self.data/ other.data**2 ) * out.grad, other.shape) 
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = self._type_control(other)
        out = Tensor(self.data ** other.data, parents=(self, other), op="pow")
        def _backward():
            self.grad += self._unbroadcast((other.data * self.data ** (other.data - 1)) * out.grad, self.shape)
            #other.grad += self._unbroadcast((self.data ** other.data) * self.xp.log(self.data) * out.grad, other.shape)
        out._backward = _backward
        return out

    
    def __matmul__(self, other): 
        other = self._type_control(other) 
        out = Tensor(self.data @ other.data, parents=(self, other), op="matmul") 
        def _backward(): 
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad @ self.xp.swapaxes(other.data, axis1=-1, axis2=-2), self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(self.xp.swapaxes(self.data, axis1=-1, axis2=-2)  @ out.grad, other.shape)
        out._backward = _backward 
        return out


    def reshape(self, *new_shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(new_shape), parents=(self, ), op="reshape")
        def _backward():
            self.grad += out.grad.reshape(old_shape)
        out._backward = _backward
        return out
    
    def swapaxes(self, axis1, axis2):
        out = Tensor(_np.swapaxes(self.data, axis1=axis1, axis2=axis2), parents=(self, ), op="swapaxes")
        def _backward():
            self.grad += _np.swapaxes(out.grad, axis1=axis1, axis2=axis2)
        out._backward = _backward
        return out
    
    def _unbroadcast(self, grad: _np.ndarray, target_shape: tuple):

        """
        here we handle the broadcasting for the reverse mode backpropagation
        """

        if grad.shape == target_shape: #if the shape isn't changed, we simply return the gradient
            return grad
        
        ndimdiff = grad.ndim - len(target_shape) #here we check how many dimension the gradient was expanded

        axis_to_sum = [i for i in range(ndimdiff)] 

        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i + ndimdiff] > 1: #here we check the dimension were the gradient was expanded
                axis_to_sum.append(i + ndimdiff) 
        grad = grad.sum(axis=tuple(axis_to_sum), keepdims=True)

        return grad.reshape(target_shape)


    def backward(self) -> None:
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                
                for parent in v.parents:
                    build_topo(parent)
                
                topo.append(v)

        build_topo(self)

        self.grad = self.xp.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
        for node in topo:
            node.parents = ()
            node._backward = lambda: None


    def sum(self):

        out = Tensor(self.xp.sum(self.data), parents=(self,), op="sum")
        def _backward():

            
            self.grad += self._unbroadcast(self.xp.ones_like(self.data) * out.grad, self.shape)
            
        out._backward = _backward
        return out
    

    def size(self):
        return self.data.size
    
    def __neg__(self):
        return self * -1



