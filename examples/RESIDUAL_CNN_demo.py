
"""
training of a simple Residual CNN on the mnist dataset
"""
import os

import nomai
from nomai import nn
import jax.numpy as jnp
import numpy as np



"""
mnist preprocessing, shapes = X_train : (60000, 1, 28, 28), y_train : (60000, 10), X_test : (10000, 1, 28, 28), y_test : (10000,)
"""

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = jnp.array(mnist["data"])/255.0, jnp.array(np.array(mnist["target"], dtype=np.int32))
X_train, y_train = X[:60000].astype(jnp.float32), y[:60000].astype(jnp.int16)
X_test, y_test = X[60000:].astype(jnp.float32), y[60000:]
del X, y
y_train = jnp.eye(10)[y_train]

X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)




"""
you can override the normal sequential function to be anyone you want.
"""

def forward_func(net, x):
    conv1, relu1, conv2, relu2, flatten, linear = net.layers

    x = relu1(conv1(x))
    x = relu2(conv2(x) + x)  #residual connection
    x = flatten(x)
    return linear(x)

def eval_func(net, x):
    conv1, relu1, conv2, relu2, flatten, linear = net.layers

    x = relu1.eval(conv1.eval(x))
    x = relu2.eval(conv2.eval(x) + x)  #residual connection
    x = flatten.eval(x)
    return linear.eval(x)

"""
is a little verbose the need to rewrite the function for eval, i will fix this.
"""





inn = nomai.Sequential(
    layers=(
        
       nn.Conv2D(channels=16, kernel_size=(3,3), padding="SAME", stride=(1,1)),
       nn.ReLU(),
       nn.Conv2D(channels=16, kernel_size=(3,3), padding="SAME", stride=(1,1)),
       nn.ReLU(),
       nn.Flatten(),
       nn.Linear(10)
    ), forward_func=forward_func, eval_func=eval_func
)

batch_size = 32

train_loader = nomai.data_loaders.supervised_loader(x=X_train, y=y_train, batch_size=batch_size)
test_loader = nomai.data_loaders.supervised_loader(x=X_test, y=y_test, batch_size=10000)

model = nomai.structs.Supervised_Trainer(
    net=inn, optimizer=nomai.optimizers.RMSprop(lr=0.005, gamma=0.95), 
    loss=nomai.losses.CrossEntropyLoss(), train_loader=train_loader, test_loader=test_loader,
    eval_metric=nomai.metrics.Classification(), key=0
    )


dummy = jnp.ones((batch_size, 1, 28, 28),dtype=jnp.float32) #the floating point precision is determined by the dummy

model = model.materialize(dummy=dummy, w_init=nomai.w_inits.He())

model = model.train(epochs=10)

model.save(size=784, name='CNN') #.ONNX
