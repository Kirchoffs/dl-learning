# Command to run: 
# KMP_DUPLICATE_LIB_OK=TRUE python RNN/markov_model.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


T = 3000
time = torch.arange(1, T + 1, dtype = torch.float32)
y = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

plt.figure(figsize = (12, 8))
plt.plot(time.numpy(), y.numpy(), label = 'sine signal with noise', color = 'blue')
plt.xlabel('time')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = y[i: T - tau + i]
labels = y[tau:].reshape((-1, 1))

batch_size, n_train = 48, 2400
dataset = TensorDataset(features[:n_train], labels[:n_train])
data_loader = DataLoader(dataset, batch_size, shuffle = True)

def build_network():
    hidden_size = 12
    network = nn.Sequential(nn.Linear(tau, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
    network.apply(init_weight)
    return network

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def train(network, train_iter, loss_fn, epochs, lr):
    trainer = torch.optim.Adam(network.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            loss = loss_fn(network(X), y)
            loss.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss {loss:f}')

network = build_network()
loss_fn = nn.MSELoss()
train(network, data_loader, loss_fn, 15, 0.01)
