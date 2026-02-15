import cupy as cp
import numpy as np
import nomai as nm
import nomai.nn as nn


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

encode = lambda string: [stoi[i] for i in string]
decode = lambda num: ''.join([itos[i] for i in num])

x = np.array(encode(text))
dataset_len = len(x)

batch_size = 128
block_size = 8

class DataLoader:
    def __init__(self, data_np, batch_size, block_size):
        self.data = cp.array(data_np, dtype=cp.int32)
        self.batch_size = batch_size
        self.block_size = block_size
        self.dataset_len = len(data_np)

    def __call__(self):

        ix = np.random.randint(0, self.dataset_len - self.block_size, size=self.batch_size)
        offsets = cp.arange(self.block_size)
        batch_indices = cp.array(ix)[:, None] + offsets
        
        x_batch = self.data[batch_indices]
        y_batch = self.data[batch_indices + 1]
        
        yield nm.Tensor(x_batch), nm.Tensor(y_batch)


loader = DataLoader(x, batch_size=128, block_size=8)


model = nn.GPT(
    Sequential= nn.Sequential(
        nn.SelfAttention(512),
        nn.LayerNorm(512),
        nn.FFN(dim_in=512, dim=2048, activation=nm.functional.ReLU),
        nn.LayerNorm(512),
        nn.SelfAttention(512),
        nn.LayerNorm(512),
        nn.FFN(dim_in=512, dim=2048, activation=nm.functional.ReLU),
        nn.LayerNorm(512),
        nn.SelfAttention(512),
        nn.LayerNorm(512),
        nn.FFN(dim_in=512, dim=2048, activation=nm.functional.ReLU),
        nn.LayerNorm(512),
        nn.SelfAttention(512),
        nn.LayerNorm(512),
        nn.FFN(dim_in=512, dim=2048, activation=nm.functional.ReLU),
        nn.LayerNorm(512),
        nn.Linear(65)
    ), 
    embedding_dim=512, 
    vocab_size=65, 
    block_size=block_size
)


def train_loop(steps):
    dummy = cp.ones((1, 1, block_size)).astype(cp.int32)
    model(dummy)
    model.to('cuda')
    
    optimizer = nm.optim.SGD(model.parameters(), lr=0.01)
    
    for step in range(steps):
        for i in range(100):
            optimizer.zero_grad()
            x, y = next(loader())
            
            z = model(x)
            z = z.reshape(-1, 65)
            y = y.reshape(-1)
            loss = nm.functional.cross_entropy(z, y)
            
            loss.backward()
            optimizer.step()
            
            

        print(f"step: {step}, loss:{loss} ===")

        
train_loop(100)  

def generate(model, start_text="", max_new_tokens=100, temperature=1.0):
    if start_text:
        context = encode(start_text)
    else:
        context = [0]
    
    context = cp.array(context).reshape(1, -1).astype(cp.int32)
    
    for _ in range(max_new_tokens):
        context_crop = context[:, -block_size:]
        logits = model(nm.Tensor(context_crop))
        logits = logits[:, -1, :] / temperature
        
        probs = nm.functional.Softmax(logits)
        probs_np = cp.asnumpy(probs.data).flatten()
        next_token = np.random.choice(len(probs_np), p=probs_np)
        
        next_token_tensor = cp.array([[next_token]]).astype(cp.int32)
        context = cp.concatenate([context, next_token_tensor], axis=1)
    
    generated = cp.asnumpy(context[0]).tolist()
    return decode(generated)


generated_text = generate(model, start_text="Hello", max_new_tokens=200, temperature=0.8)
print(generated_text)