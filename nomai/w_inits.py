from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx




class RandNinit:

    """
    random normal initialization (divided by 40).
    """

    def __call__(self, shape, dtype, key, n_in):

        new_key, key = jrand.split(key)
        w = jrand.normal(key, shape=shape, dtype=dtype)/40
        return w, new_key

class He:
    """
    He initialization.
    """
    def __call__(self, shape, dtype, key, n_in):
        new_key, key = jrand.split(key)
        std = 2/n_in
        w = jrand.normal(key, shape=shape, dtype=dtype) * std
        return w, new_key

