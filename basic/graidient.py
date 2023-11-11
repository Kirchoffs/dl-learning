import numpy as np

def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = float(x[idx])
        
        x[idx] = tmp_val + h
        fxh_forward = f(x)
        
        x[idx] = tmp_val - h 
        fxh_backward = f(x)
        
        grad[idx] = (fxh_forward - fxh_backward) / (2 * h)
        
        x[idx] = tmp_val
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        tmp_val = float(x[idx])
        
        x[idx] = tmp_val + h
        fxh_forward = f(x)
        
        x[idx] = tmp_val - h 
        fxh_backward = f(x)
        
        grad[idx] = (fxh_forward - fxh_backward) / (2 * h)
        
        x[idx] = tmp_val
        
        it.iternext()   
        
    return grad
