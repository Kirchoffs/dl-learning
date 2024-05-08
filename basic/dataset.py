import torch
from torch.utils.data import TensorDataset

features = torch.tensor([[1, 2], [3, 4], [5, 6]])
labels = torch.tensor([0, 1, 0])

dataset = TensorDataset(features, labels)

for feature, label in dataset:
    print(f"Feature: {feature}, Label: {label}")
