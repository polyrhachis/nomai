
"""
training of a simple MLP on the mnist dataset
"""

import nomai
from nomai import nn
import jax.numpy as jnp
import numpy as np



"""
mnist preprocessing, shapes = X_train : (60000, 784), y_train : (60000, 10), X_test : (10000, 784), y_test : (10000,)
"""
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = jnp.array(mnist["data"])/255.0, jnp.array(np.array(mnist["target"], dtype=np.int32))
X_train, y_train = X[:60000].astype(jnp.float32), y[:60000].astype(jnp.int16)
X_test, y_test = X[60000:].astype(jnp.float32), y[60000:]
del X, y
y_train = jnp.eye(10)[y_train]


inn = nomai.Sequential(
    layers=(
        
        nn.Linear(512),
        nn.ReLU(),
        nn.Linear(128),
        nn.ReLU(),
        nn.Linear(10)
    )
)

batch_size = 32

train_loader = nomai.data_loaders.supervised_loader(x=X_train, y=y_train, batch_size=batch_size)
test_loader = nomai.data_loaders.supervised_loader(x=X_test, y=y_test, batch_size=10000)

model = nomai.structs.Supervised_Trainer(
    net=inn, optimizer=nomai.optimizers.RMSprop(lr=0.001, gamma=0.95), 
    loss=nomai.losses.CrossEntropyLoss(), train_loader=train_loader, test_loader=test_loader,
    eval_metric=nomai.metrics.Classification(), key=0
    )


dummy = jnp.ones((batch_size, 784),dtype=jnp.float32) #the floating point precision is determined by the dummy

model = model.materialize(dummy=dummy, w_init=nomai.w_inits.He())

model = model.train(epochs=10)

model.save(size=784, name='mlp')
