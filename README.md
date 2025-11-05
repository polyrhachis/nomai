# nomai-jax
A minimal deep learning framework built from scratch in JAX with Equinox, made for extreme speed and simplicity. (_If you know PyTorch, you can use nomai in 5 minutes_)

# Installation
**pip install git+https://github.com/polyrhachis/nomai.git**


# Quick Example
```python
#the net (lazy init)
net = nomai.Sequential(
    layers=(
        
        nn.Linear(512),
        nn.ReLU(),
        nn.Linear(128),
        nn.ReLU(),
        nn.Linear(10)
    )
)

batch_size = 32

#data loaders
train_loader = nomai.data_loaders.supervised_loader(x=X_train, y=y_train, batch_size=batch_size)
test_loader = nomai.data_loaders.supervised_loader(x=X_test, y=y_test, batch_size=10000)

#model
model = nomai.structs.Supervised_Trainer(
    net=net, optimizer=nomai.optimizers.RMSprop(lr=0.001, gamma=0.95), 
    loss=nomai.losses.CrossEntropyLoss(), train_loader=train_loader, test_loader=test_loader,
    eval_metric=nomai.metrics.Classification(), key=0
    )
dummy = jnp.ones((batch_size, 784),dtype=jnp.float32) 

model = model.materialize(dummy=dummy, w_init=nomai.w_inits.He())

#training
model = model.train(epochs=10)
```


# What Is nomai?
nomai is a deep learning library designed to have a simple and concise syntax while still respecting JAX constraints, allowing for a very fast, **fully jitted** training step with just a few lines of code, while remaining highly customizable when desired.

# How It's Made?
**Tech used:** Python, JAX, Equinox.

nomai was created using only JAX and Equinox. I tried to make the creation of models as explicit as possible thanks to eqx.Module, vaguely reminiscent of the basic style of **Keras and Pytorch**. I found that it might be a good idea to have a class such as 'nomai. Sequential' as the ‘backbone’ of the library. It is very **versatile** as it allows you to write even very deep networks in an explicit and easy way, giving beginners a simple way to approach the fantastic world of jax, but allowing more experienced users to completely redefine the internal dynamics of both ‘nomai.Sequential’ and the Trainer, allowing residual connections, recurrent networks, and many other non-sequential architectures.

# Optimizations
nomai is based on a **lazy init** system, which allows the user to define only the output dimensions of each layer without worrying about calculating the shapes themselves. Simply provide the **materialize** function with an input equal to the batches that the network will see, and the library will calculate the correct size of each layer.
The library was created with the desire to get **maximum performance** from JAX in mind. In fact, **the entire train step function is entirely jitted**, and at each iteration it returns a new model with updated parameters, respecting the functional style of JAX.

# What I Learned

The creation of nomai allowed me to familiarize myself with the world of **compiled deep learning**, showing me how, at the cost of a few constraints, it is possible to have models that are **extremely faster than the classic models created with Pytorch**.


# Known Issues And Future Plans
I am aware that the library has some issues, one of which is having to explicitly specify an identical eval function in the case of a custom forward function.
Many fundamental layers and components are also missing, such as additional metrics (confusion matrix, f1, recall, etc.), fundamental layers (BatchNorm, Attention, GRU, LSMT, etc.), the Adam optimizer, and above all, other structures for unsupervised training or perhaps reinforcement learning. The library is still under development and I will introduce everything gradually.




# Thanks For Reading!  
If you have feedback, suggestions, or just want to share your thoughts, feel free to reach out at **polyrhachiss@gmail.com**
