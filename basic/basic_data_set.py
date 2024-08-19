import torch
from torch.utils.data import TensorDataset, DataLoader

features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
labels = torch.tensor([0, 1, 0, 1])

dataset = TensorDataset(features, labels)

dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

for batch_features, batch_labels in dataloader:
    print(f'Features: {batch_features}, Labels: {batch_labels}')
