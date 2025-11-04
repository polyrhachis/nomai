from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx




class Classification:

    """
    it outputs the accuracy of the model on the eval step.
    """    

    def __call__(self, preds, y):
        total_preds = len(y)
        preds = jnp.argmax(preds, axis=-1)
        preds = jnp.sum(preds==y)

        return ('Accuracy: ' + str(preds/total_preds*100) + '%')
    


class MSE:
    """
    it outputs the MSE loss of the model on the eval step.
    """

    def __call__(self, pred, y):
        return 'loss: ' + str(jnp.mean(jnp.square(pred-y)))