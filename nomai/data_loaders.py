from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import jax


class supervised_loader:

    """
    a data_loader similiar to the pytorch one, it has drop_last
    """

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size



    def __call__(self):
        n = len(self.x)
        drop_last = (n // self.batch_size) * self.batch_size 

        
        for i in range(0, drop_last, self.batch_size):
            x = jax.device_put(self.x[i: i+self.batch_size], device=jax.devices()[0])
            y = jax.device_put(self.y[i: i+self.batch_size], device=jax.devices()[0])
            yield x, y
        
