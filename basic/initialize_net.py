import torch
from torch import nn

def init_normal(m):
    if type(m) == torch.nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)
        nn.init.zeros_(m.bias)
        
net = nn.Sequential(nn.Linear(3, 1), nn.ReLU(), nn.Linear(1, 3))
net.apply(init_normal)
print(net[0].weight.data)
print(net[0].bias.data)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data)
print(net[0].bias.data)
