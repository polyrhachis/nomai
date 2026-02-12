import macrograd.nn as nn
import numpy as np
import macrograd as mg
import macrograd.functional as F


'''
importing mnist
'''
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy()/255.0  
y = mnist.target.to_numpy().astype(int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


'''
the model
'''
model = mg.nn.Sequential(
    nn.Linear(512),
    nn.ReLU(),                 
    nn.Linear(10),

)

'''
loss fn
'''
def cross_entropy(pred, y):
    pred = F.Softmax(pred)
    return -(np.log(pred) * y).sum()/pred.shape[0]


class data_loader:
    
    def __init__(self, data, target, batch_size):
        self.data = data
        self.target = target
        self.batch_size = batch_size
    
    def __call__(self):
        lenght = self.data.shape[0]
        for i in range(0, lenght, self.batch_size):
            yield self.data[i: i + self.batch_size], self.target[i: i + self.batch_size]


y_train = np.eye(10)[y_train]

train_loader = data_loader(data=mg.Tensor(X_train), target=mg.Tensor(y_train), batch_size=50)
test_loader = data_loader(data=mg.Tensor(X_test), target=mg.Tensor(y_test), batch_size=10000)

dummy = mg.Tensor(np.random.normal(size=(1, 784)))

def train_loop(epochs):
    model(dummy) #warm up for the lazy initialization BEFORE the optim init
    optim = mg.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(epochs):
        for data, target in train_loader():


            optim.zero_grad()


            z = model(data)  #just like PyTorch


            loss = cross_entropy(z, target)
            loss.backward()


            optim.step()
        
        correct = 0
        total = 0 
        for data, target in test_loader():
            z = model(data)
            preds = np.argmax(z.data, axis=-1)
            correct += np.sum(preds==target.data)
            total += data.shape[0]
        print(f"Accuracy: {correct/total}, loss: {loss}")

train_loop(10) #~97% accuracy


