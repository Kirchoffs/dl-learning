import torch
from torch import nn

class CustomizedSequentialModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

X = torch.rand(3, 20)
net  = CustomizedSequentialModule(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
y = net(X)
print(y.shape)
print(y)
    