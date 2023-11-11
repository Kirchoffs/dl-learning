import numpy as np

def identity(x):
    return x

def softmax_naive(a):
    exp_a = np.exp(a)
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum
    return y

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis = 1).reshape(-1, 1)
        y = np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1, 1)
        return y
        
        # x = x.T
        # x = x - np.max(x, axis = 0)
        # y = np.exp(x) / np.sum(np.exp(x), axis = 0)
        # return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
