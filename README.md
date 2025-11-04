# nomai-jax
A minimal deep learning framework built from scratch in JAX with Equinox, made for extreme speed and simplicity.

# Installation
**pip install git+https://github.com/polyrhachis/nomai.git**

# What Is nomai?
nomai is a deep learning library designed to have a simple and concise syntax while still respecting JAX constraints, allowing for a very fast, fully jitted training step with just a few lines of code, while remaining highly customizable when desired.

# How It's Made?
**Tech used:** Python, JAX, Equinox.

nomai was created using only JAX and Equinox. I tried to make the creation of models as explicit as possible thanks to eqx.Module, vaguely reminiscent of the basic style of Keras and Pytorch. I found that it might be a good idea to have a class such as 'nomai. Sequential' as the ‘backbone’ of the library. It is very versatile as it allows you to write even very deep networks in an explicit and easy way, giving beginners a simple way to approach the fantastic world of jax, but allowing more experienced users to completely redefine the internal dynamics of both ‘nomai.Sequential’ and the Trainer, allowing residual connections, recurrent networks, and many other non-sequential architectures.

# Optimizations
nomai is based on a lazy init system, which allows the user to define only the output dimensions of each layer without worrying about calculating the shapes themselves. Simply provide the materialize function with an input equal to the batches that the network will see, and the library will calculate the correct size of each layer.
The library was created with the desire to get maximum performance from JAX in mind. In fact, the entire train step function is entirely jitted, and at each iteration it returns a new model with updated parameters, respecting the functional style of JAX.

# What I Learned

The creation of nomai allowed me to familiarize myself with the world of compiled deep learning, showing me how, at the cost of a few constraints, it is possible to have models that are extremely faster than the classic models created with Pytorch.


# Known Issues And Future Plans
