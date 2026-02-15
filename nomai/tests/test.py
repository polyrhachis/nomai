import torch
import numpy as np
import nomai as nm

def gradient_test_against_torch():
    # random data
    x = np.random.normal(size=(10, 10)).astype(np.float32)
    w = np.random.normal(size=(10, 10)).astype(np.float32)
    b = np.random.normal(size=(10, )).astype(np.float32)

    # --- Nomai ---
    xnm = nm.Tensor(x)
    wnm = nm.Tensor(w)
    bnm = nm.Tensor(b)
    z = nm.functional.ReLU(xnm @ wnm + bnm)
    z.sum().backward()
    
    # --- PyTorch ---
    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    zt = torch.nn.functional.relu(xt @ wt + bt)
    zt.sum().backward()


    print("x grad close:", np.allclose(xnm.grad, xt.grad.numpy(), rtol=1e-5, atol=1e-8))
    print("w grad close:", np.allclose(wnm.grad, wt.grad.numpy(), rtol=1e-5, atol=1e-8))
    print("b grad close:", np.allclose(bnm.grad, bt.grad.numpy(), rtol=1e-5, atol=1e-8))

gradient_test_against_torch()
