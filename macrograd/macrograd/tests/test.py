import torch
import numpy as np
import macrograd as mg

def gradient_test_against_torch():
    # random data
    x = np.random.normal(size=(10, 10)).astype(np.float32)
    w = np.random.normal(size=(10, 10)).astype(np.float32)
    b = np.random.normal(size=(10, )).astype(np.float32)

    # --- MacroGrad ---
    xmg = mg.Tensor(x)
    wmg = mg.Tensor(w)
    bmg = mg.Tensor(b)
    z = mg.functional.ReLU(xmg @ wmg + bmg)
    z.sum().backward()
    
    # --- PyTorch ---
    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    zt = torch.nn.functional.relu(xt @ wt + bt)
    zt.sum().backward()


    print("x grad close:", np.allclose(xmg.grad, xt.grad.numpy(), rtol=1e-5, atol=1e-8))
    print("w grad close:", np.allclose(wmg.grad, wt.grad.numpy(), rtol=1e-5, atol=1e-8))
    print("b grad close:", np.allclose(bmg.grad, bt.grad.numpy(), rtol=1e-5, atol=1e-8))

gradient_test_against_torch()
