from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx
import jax




class SGD(eqx.Module):
    """
    a classical stochastic gradient descent algorithm.

    Args:
      lr (float) : learning rate.
      decay (float) : decay factor (per iteration, not epoch).
    """

    lr : float 
    decay : any = eqx.field(static=True)

    def materialize(self, net):
        return SGD(lr=self.lr, decay=self.decay)
    
    def __call__(self, model, grad):
        new_model = jax.tree.map(lambda model, grad: model-grad*self.lr, model, grad)
        return new_model
    
    def update(self, grad):
        return eqx.tree_at(lambda opt: opt.lr, self, self.lr*self.decay)





class RMSprop(eqx.Module):



    """
    RMSprop optimization.

    Args:
      lr (float) : learning rate.
      gamma (float) : memory of previous gradients (should be 0<gamma<1).
      epsilon (float) : optional, to avoid division by zero (default = 1e-6).
    """



    lr : float = eqx.field(static=True)
    gamma : float = eqx.field(static=True)
    epsilon : float = eqx.field(static=True)
    memory : eqx.Module


    def __init__(self, lr, gamma, epsilon=1e-6, memory=None):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = memory

    def materialize(self, net):
        memory = jax.tree.map(lambda net: jnp.zeros_like(net), net)
        return RMSprop(lr=self.lr, gamma=self.gamma, epsilon=self.epsilon, memory=memory)
    
    def __call__(self, model, grad):
        grad = jax.tree.map(
            lambda model, grad, memory: 
            model - (self.lr/(jnp.sqrt(memory)+self.epsilon)) * grad,
            model, grad, self.memory)
        return grad
        
    def update(self, grad):
        new_memory = jax.tree.map(
            lambda opt, grad: self.gamma*opt+(1-self.gamma) * grad**2, self.memory, grad)
        return eqx.tree_at(lambda opt: opt.memory, self, new_memory)
