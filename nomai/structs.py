"""
module with all the Trainers
"""







from jax import (
    numpy as jnp,
    random as jrand,
    lax as lax,
)
import equinox as eqx
import jax



def default_train_step_func(net, x, y, loss):

    pred = net(x) 
    return loss(pred, y)


class Supervised_Trainer(eqx.Module):
    """

    a Trainer of a nomai.Sequential net.

    Args:

        net (nomai.Sequential) : a nomai.Sequential net.
        optimizer (eqx.Module) : an optimizer from nomai.optimizers.
        loss (eqx.Module) : a loss from nomai.losses.
        train_loader (object) : a loader from nomai.data_loaders.
        test_loader (object) : a loader from nomai.data_loaders.
        eval_metric (object) : an eval metric from nomai.eval_metrics.
        key (int) : a integer for defining model's randomness.
        train_step_func (callable) : optional, a function for defining a custom train_step (the parameters are: (net, x, y, loss)).

    Examples:
    >>> model = nomai.structs.Supervised_Trainer(
    net=net, optimizer=nomai.optimizers.RMSprop(lr=0.001, gamma=0.95), 
    loss=nomai.losses.CrossEntropyLoss(), train_loader=train_loader, test_loader=test_loader,
    eval_metric=nomai.metrics.Classification(), key=0
    )
    >>> model = model.materialize(dummy=dummy, w_init=w_init)
    >>> model = model.train(epochs=epochs)
    
    """


    net : eqx.Module  #our sequential
    optimizer : eqx.Module  #an optimizer from nomai.optimizers
    loss : eqx.Module = eqx.field(static=True) #a loss from nomai.losses
    train_loader : object = eqx.field(static=True) #a loader from nomai.data_loaders
    test_loader : object = eqx.field(static=True)
    eval_metric : object = eqx.field(static=True)
    train_step_func : callable = eqx.field(static=True)
    key : int

    def __init__(self, net, optimizer, loss, train_loader, test_loader, eval_metric,
                  key, train_step_func=default_train_step_func):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eval_metric = eval_metric
        self.train_step_func = train_step_func
        self.key = jrand.PRNGKey(key)

    def materialize(self, dummy, w_init):

        """
        creates the network's layers and initialize the optimizer.

        Args:
           dummy (jnp.ndarray) : a batch of array, like the ones the model will see in training.
           w_inits (object) : a w_init from nomai.w_inits.

        Examples:
          >>> model = nomai.structs.Supervised_Trainer(
           net=net, optimizer=nomai.optimizers.RMSprop(lr=0.001, gamma=0.95), 
           loss=nomai.losses.CrossEntropyLoss(), train_loader=train_loader, test_loader=test_loader,
           eval_metric=nomai.metrics.Classification(), key=0
    )
        >>> model = model.materialize(dummy=dummy, w_init=w_init)
        """


        new_net, key = self.net.materialize(dummy, w_init, self.key) #we init sequential and the optimizer
        new_opt = self.optimizer.materialize(new_net)
        return eqx.tree_at(lambda model: (model.net, model.optimizer, model.key), self, (new_net, new_opt, key))
    
    def __call__(self, x):
        return self.net(x)
    
    
    @jax.jit
    def train_step(self, x, y):

        grad = eqx.filter_grad(self.train_step_func)(self.net, x, y, self.loss) #function to autodiff respect to the net
        new_opt = self.optimizer.update(grad)  #updates
        new_net = new_opt(self.net, grad)
        return eqx.tree_at(lambda model: (model.net, model.optimizer), self, (new_net, new_opt))
    
    
    def train(self, epochs):

        """
        trains the model for n epochs.

        Examples:
           >>> model = model.train(epochs=epochs)
        """
        model = self
        for epoch in range(epochs):

            for x,y in model.train_loader():
                model = model.train_step(x,y)
                
            preds = []
            for x,y in model.test_loader():
                preds.append(model.eval(x))
                   
            preds = jnp.concatenate(preds)
            
            print(self.eval_metric(preds, model.test_loader.y))
                
        return model


    
    def eval(self, x):
        return self.net.eval(x)
    

    def save(self, size, name):

        """
        saves the model in a .onnx format, (it uses jax2onnx).
        """

        from jax2onnx import to_onnx

        if not isinstance(size, (tuple, list)):
            size = (size,)


        to_onnx(
        self.net,
        inputs=[("B",*size)],
        return_mode='file',
        output_path=name + '.onnx',
        )




    

    

