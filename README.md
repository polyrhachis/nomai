# ⚡Nomai
Nomai is a small framework inspired by micrograd (by Andrej Karpathy), but designed to work with tensors instead of scalar values (with NumPy and even CuPy). Its main purpose is educational—although it’s surprisingly fast.
I wanted to truly understand how PyTorch works under the hood, so I built my own mini PyTorch implementation.
# Installation
```bash
pip install git+https://github.com/polyrhachis/nomai.git
```
# Example usage
```python
import nomai as nm
import numpy as np

# Create input and weight tensors
x = nm.Tensor(np.random.normal(size=(10, 10)))
w = nm.Tensor(np.random.normal(size=(10, 10)))

# Forward pass: matrix multiplication + ReLU
z = nm.Functional.ReLU(x @ w)

# Backward pass: compute gradients
z.backward()

# Access gradients
print("Gradient of x:", x.grad)
print("Gradient of w:", w.grad)

```
(for more examples, see the examples folder, there is a full MLP trained on the MNIST dataset)
# 🔥 Features
✅ Automatic differentiation for Tensor operations  

✅ Common functions like ReLU, Softmax, etc.

✅ A `nn` module for PyTorch-like training 

✅ Simple and readable code, implemented in a very explicit way

✅ CuPy integration

✅ Self-Attention

 # ⚠️ Still work in progress

 ❌ performance issues
 
 ❌ Out of memory bugs
 
 ❌ Other optimizers
 
For any questions or feedback, feel free to reach out at: **polyrhachiss@gmail.com**
