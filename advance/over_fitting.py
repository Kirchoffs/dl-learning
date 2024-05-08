import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

max_degree = 25
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size = (n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= np.math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale = 0.1, size = labels.shape)

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        y_hat = net(X)
        y = y.reshape(y_hat.shape)
        l = loss(y_hat, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def load_array(data_arrays, batch_size, is_train = False):
    data_tensors = [torch.from_numpy(data_array) for data_array in data_arrays]
    dataset = TensorDataset(*data_tensors)
    return DataLoader(dataset, batch_size, shuffle = is_train)

def train_epoch(net, train_iter, loss, trainer):
    for X, y in train_iter:
        trainer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        trainer.step()

def train(train_features, test_features, train_labels, test_labels, num_epochs = 500):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias = False)).double()
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train = False)
    trainer = torch.optim.SGD(net.parameters(), lr = 0.01)
    
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)
        train_loss = evaluate_loss(net, train_iter, loss)
        test_loss = evaluate_loss(net, test_iter, loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
    print('weight:', net[0].weight.data.numpy())
    plt.loglog(train_losses, label = 'train loss')
    plt.loglog(test_losses, label = 'test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
