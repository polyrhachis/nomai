"""
training of a simple autoencoder on the mnist dataset.
"""

import nomai
from nomai import nn
import jax.numpy as jnp
import numpy as np



"""
mnist preprocessing, shapes = X_train : (60000, 784), y_train : (60000, 10), X_test : (10000, 784), y_test : (10000,).
"""
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = jnp.array(mnist["data"])/255.0, jnp.array(np.array(mnist["target"], dtype=np.int32))
X_train, y_train = X[:60000].astype(jnp.float32), y[:60000].astype(jnp.int16)
X_test, y_test = X[60000:].astype(jnp.float32), y[60000:]
del X, y
y_train = jnp.eye(10)[y_train]

###################


net = nomai.Sequential(
    layers=(
        
        nn.Linear(258),
        nn.ReLU(),
        nn.Linear(784),
        nn.Sigmoid()
    )
)

batch_size = 500

train_loader = nomai.data_loaders.supervised_loader(x=X_train, y=y_train, batch_size=batch_size)
test_loader = nomai.data_loaders.supervised_loader(x=X_test, y=X_test, batch_size=10000)

"""
you can override the standard train_step function to be anyone you want.
"""

def train_step_func(net, x, y, loss): #non-standard dynamic
    preds = net(x)
    return loss(preds, x)



model = nomai.structs.Supervised_Trainer(
    net=net, optimizer=nomai.optimizers.RMSprop(lr=0.001, gamma=0.95), 
    loss=nomai.losses.MSE(), train_loader=train_loader, test_loader=test_loader,
    eval_metric=nomai.metrics.MSE(), key=0, train_step_func=train_step_func
    )


dummy = jnp.ones((batch_size, 784),dtype=jnp.float32) #the floating point precision is determined by the dummy.

model = model.materialize(dummy=dummy, w_init=nomai.w_inits.He())

model = model.train(epochs=10)


model.save(size=784, name='autoencoder') #I will develop a tool for only saving the encoder.
