from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx
import jax
from . import nn



class CrossEntropyLoss(eqx.Module):

    def __call__(self, pred, y):
        logits = nn.Softmax()(pred)
        logits = y*jnp.log(logits)
        return -jnp.mean(jnp.sum(logits, axis=-1))
    


    
class MSE(eqx.Module):

    def __call__(self, p, y):
        return jnp.mean(jnp.square(p-y))