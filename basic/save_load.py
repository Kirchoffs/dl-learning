import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 15)
        self.output = nn.Linear(15, 5)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
def init_normal(m):
    if type(m) == torch.nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)
        nn.init.zeros_(m.bias)

net = MLP()
net.apply(init_normal)
print(net.state_dict())

torch.save(net.state_dict(), 'mlp.params')

clone_net = MLP()
clone_net.load_state_dict(torch.load('mlp.params'))

X = torch.randn(size = (2, 5))
Y = net(X)
Y_clone = clone_net(X)
print(Y == Y_clone)
