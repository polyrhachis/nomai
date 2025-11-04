# nomai-jax
A minimal deep learning framework built from scratch in JAX with Equinox, made for extreme speed and simplicity.

# Installation
**pip install git+https://github.com/polyrhachis/nomai.git**

# What Is nomai?
nomai is a deep learning library designed to have a simple and concise syntax while still respecting JAX constraints, allowing for a very fast, fully jitted training step with just a few lines of code, while remaining highly customizable when desired.

# How It's Made?
**Tech used:** Python, JAX, Equinox.

nomai was created using only JAX and Equinox. I tried to make the creation of models as explicit as possible thanks to eqx.Module, vaguely reminiscent of the basic style of Keras and Pytorch. I found that it might be a good idea to have a class such as 'nomai. Sequential' as the ‘backbone’ of the library. It is very versatile as it allows you to write even very deep networks in an explicit and easy way, giving beginners a simple way to approach the fantastic world of jax, but allowing more experienced users to completely redefine the internal dynamics of both ‘nomai.Sequential’ and the Trainer, allowing residual connections, recurrent networks, and many other non-sequential architectures.
