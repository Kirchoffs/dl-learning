import torch
from torch import nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

class LayerWithParameter(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.zeros(units,))
    
    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)
    
dense = LayerWithParameter(5, 3)
print(dense.weight)
