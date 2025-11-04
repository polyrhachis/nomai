"""

module with all the layers to use in nomai.Sequential


"""

from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx






def default_forward(net, x):
    for layer in net.layers:
        x = layer(x)
    return x

def default_eval(net, x):
    for layer in net.layers:
        x = layer.eval(x)
    return x    

class Sequential(eqx.Module):

    """
    for combining layers, the layers must be put in a sequential way in the layers tuple.


    Args:
        layers (tuple) : layers of the net
        forward_func (function) : optional, to have custom forward, default is sequential
        eval_func (function) : optional, to have custom eval, default is sequential

    Examples:
     >>> nomai.Sequential(
        layers=(
           Linear(128),
           ReLU(),
           Linear(10))
           )


     >>> def forward(net, x):
            l1, r, l2, r2, l3 = net.layers
            return l3(r2(l2(r(l1(x)))))


     >>> def eval(net, x):
            l1, r, l2, r2, l3 = net.layers
            return l3.eval(r2.eval(l2.eval(r.eval(l1.eval(x)))))


     >>> nomai.Sequential(
           layers=(
           Linear(128),
           ReLU(),
           Linear(10)), forward_func=forward, eval_func=eval_func
           )
    """
    
    layers : tuple
    forward_func : callable = eqx.field(static=True)
    eval_func : callable = eqx.field(static=True)

    def __init__(self, layers, forward_func=default_forward, eval_func=default_eval):
        self.layers = layers
        self.forward_func = forward_func
        self.eval_func = eval_func

    def __call__(self, x):
        
        return self.forward_func(self, x)
    
    def materialize(self, dummy, w_init, key):

        initiliazed_layers = ()

        for layer in self.layers:
            layer, dummy, key = layer.materialize(dummy, w_init, key)
            initiliazed_layers += (layer,)

        return Sequential(layers=initiliazed_layers, forward_func=self.forward_func, eval_func=self.eval_func), key
    
    def eval(self, x):
        return self.eval_func(self, x)
    


class Linear(eqx.Module):

    """
    a standard Linear/Dense layer, it implements y = Xw+b (Xw convention)
    you only need to specify the dim out, (lazily initializated),

    Args:
        dim (int): output dimension

    
    """

    dim : int = eqx.field(static=True)
    w : jnp.ndarray
    b : jnp.ndarray

    def __init__(self, dim, w=None, b=None): #Lazy init
        self.dim = dim
        self.w = w
        self.b = b

    def materialize(self, dummy, w_init, key):
        dim_in = dummy.shape[1]

        n_in = dim_in #we get the input shape for the init


        w, new_key = w_init(shape=(dim_in, self.dim), dtype=dummy.dtype,key=key, n_in=n_in) #params init
        b = jnp.zeros(shape=(self.dim,), dtype=dummy.dtype)

        new_layer = Linear(dim=self.dim, w=w, b=b)
        return new_layer, new_layer(dummy), new_key #the new dummy to pass to the other layers
    
    def __call__(self, x):
        return jnp.add(jnp.matmul(x, self.w), self.b) #matmul + bias
    
    def eval(self, x):
        return self(x)



class Conv2D(eqx.Module):

    """
    a convolutional 2D layer, it use lax.conv_general_dilated, NCHW convention (lazily initializated).

    Args:
        channels (int) : channels out.
        kernel_size (tuple) : size of the kernel.
        padding (any) : padding, it can be "SAME" or some 2D tuple.
        stride (tuple) : stride, one per dimension.

    Examples:
       >>> nn.Conv2D(channels=32, kernel_size=(3,3), padding="SAME", stride=(1,1)).

    """

    channels : int = eqx.field(static=True)
    kernel_size : tuple = eqx.field(static=True)  #the channels_in are lazy initializated
    padding : tuple = eqx.field(static=True)
    stride : tuple = eqx.field(static=True)
    w : jnp.ndarray
    b : jnp.ndarray

    def __init__(self, channels, kernel_size, padding, stride, w=None, b=None):
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.w = w
        self.b = b

    def materialize(self, dummy, w_init,  key):
        channels = dummy.shape[1]
        kel_size = (self.channels, channels) + self.kernel_size
        

        n_in = dummy.shape[2] * dummy.shape[3] 



        w, new_key = w_init(shape=kel_size, dtype=dummy.dtype, key=key, n_in=n_in,)
        b = jnp.zeros(shape=(1,self.channels,1,1), dtype=dummy.dtype)

        new_layer = Conv2D(  
            channels=self.channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, w=w, b=b)
        
        return new_layer, new_layer(dummy), new_key
    
    def __call__(self, x): #Conv2D with NCHW convention
        return lax.conv_general_dilated(lhs=x, rhs=self.w, window_strides=self.stride,padding=self.padding) + self.b
    
    def eval(self, x):
        return self(x)   



class Flatten(eqx.Module):

    "flattens the input in shape (B, N)"
   
    def materialize(self, dummy, w_init, key):
        return Flatten(), self(dummy), key
    
    def __call__(self, x):
        return x.reshape(x.shape[0],-1) 

    def eval(self, x):
        return self(x)



class ReLU(eqx.Module):

    """ReLU activation"""

    def materialize(self, dummy, w_init, key):
        return self, dummy, key
    
    def __call__(self, x):
        return jnp.maximum(x,0)   

    def eval(self, x):
        return self(x)



class Sigmoid(eqx.Module): 
    """Sigmoid activation"""

    def materialize(self, dummy, w_init, key):
        return Sigmoid(), dummy, key
    
    def __call__(self, x):
        return 1/(1+jnp.exp(-(x)))  

    def eval(self, x):
        return self(x)
    

class Tanh(eqx.Module): 
    """Tanh activation"""

    def materialize(self, dummy, w_init, key):
        return Tanh(), dummy, key
    
    def __call__(self, x):
        return jnp.tanh(x)

    def eval(self, x):
        return self(x)
    

class Swish(eqx.Module):

    """Swish activation""" 
    
    def materialize(self, dummy, w_init, key):
        return Swish(), dummy, key
    
    def __call__(self, x):
        return x/(1+jnp.exp(-(x)))  

    def eval(self, x):
        return self(x)

    
class Rnn(eqx.Module):

    """
    a standard recurrent layer, it needs to be inside a Recurrent_Block (lazily initialized).

    Args:
     
      dim (int) : dim out of the layer.
      activation (int) : an activation from nomai.nn.

    Examples:
    >>> nomai.Sequential(
        layers=(
         
    >>>  Recurrent_block(
            layers=(
            nn.Rnn(100, activation=nn.ReLU()),
            nn.Rnn(10, activation=nn.Tanh()),
            ), return_seq=False)
            ))
    """

    dim : int = eqx.field(static=True)
    activation : object = eqx.field(static=True)
    w_x : jnp.ndarray
    w_h : jnp.ndarray
    b : jnp.ndarray


    def __init__(self, dim, activation, w_x=None, w_h=None, b=None):
        self.dim = dim
        self.activation = activation
        self.w_x = w_x
        self.w_h = w_h
        self.b = b

    
    def materialize(self, dummy, w_init, key):
        n_in = dummy.shape[-1]
        w_x, key = w_init(shape=(n_in, self.dim), dtype=dummy.dtype, key=key, n_in=n_in)
        w_h, key = w_init(shape=(self.dim, self.dim), dtype=dummy.dtype, key=key, n_in=n_in)
        b = jnp.zeros(shape=(self.dim,), dtype=dummy.dtype)

        dummy_h = jnp.zeros((1, self.dim), dtype=dummy.dtype)
        new_layer = Rnn(dim=self.dim, activation=self.activation, w_x=w_x, w_h=w_h, b=b)
        return new_layer, new_layer(dummy, dummy_h), key
    
    def __call__(self, x, h):
        return self.activation(x@self.w_x + h@self.w_h + self.b)
    

    def eval(self, x, h):
        return self(x, h)
    


class Recurrent_Block(eqx.Module):
 
    """
    wrapper for recurrent layers, it can output the final h (classification)
    or the entire sequence (specify with return_seq : bool).

    Args:
      layers (tuple) : recurrent Layers (like nn.Rnn).
      return_seq (bool) : self described.

    Examples:
    >>> nomai.Sequential(
        layers=(
         
    >>>  Recurrent_block(
            layers=(
            nn.Rnn(100, activation=nn.ReLU()),
            nn.Rnn(10, activation=nn.Tanh()),
            ), return_seq=False)
            ))

    """


    layers : tuple
    return_seq : bool = eqx.field(static=True)
    h : list

    def __init__(self, layers, return_seq, h=None):
        self.layers = layers
        self.return_seq = return_seq
        self.h = h


    def materialize(self, dummy, w_init, key):
        initiliazed_layers = ()
        h = []
        batch_size = dummy.shape[0]
        for layer in self.layers:
            layer, dummy, key = layer.materialize(dummy, w_init, key)
            initiliazed_layers += (layer,)
            h.append(jnp.zeros(shape=(batch_size, layer.dim), dtype=dummy.dtype))
        
        return Recurrent_Block(layers=initiliazed_layers, return_seq=self.return_seq, h=h), dummy[-1], key
    

    def __call__(self, x):
        x = jnp.transpose(x, axes=[1,0,2])

        def f(h, xs):
            for i, layer in enumerate(self.layers):
                xs = layer(xs, h[i])
                h[i] = xs
            return h, xs
        
        final_carry, last_output = lax.scan(f, init=self.h, xs=x)

        if self.return_seq:
            return final_carry
        return final_carry[-1]
    
    
    
    def eval(self, x):

        batch_size = x.shape[0]
        x = jnp.transpose(x, axes=[1,0,2])

        h = [jnp.zeros((batch_size, layer.dim), dtype=x.dtype) for layer in self.layers]


        def f(h, xs):
            for i, layer in enumerate(self.layers):
                xs = layer(xs, h[i])
                h[i] = xs
            return h, xs
        
        final_carry, sequence = lax.scan(f, init=h, xs=x)
        
        if self.return_seq:
            return sequence
        
        return final_carry[-1]  
    




class Softmax(eqx.Module):
    """Softmax activation""" 
    
    def materialize(self, dummy, w_init, key):
        return Softmax(), dummy, key
    
    def __call__(self, x):
        normalizer = jnp.max(x, axis=-1, keepdims=True)
        x = x-normalizer
        return jnp.exp(x)/jnp.sum(jnp.exp(x), axis=-1, keepdims=True)  

    def eval(self, x):
        return self(x)    
