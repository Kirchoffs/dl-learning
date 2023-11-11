import numpy as np

def numerical_gradient(f, x, h = 1e-4):
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh_forward = f(x)
        
        x[idx] = tmp_val - h
        fxh_backward = f(x)
        
        grad[idx] = (fxh_forward - fxh_backward) / (2 * h)
        
        x[idx] = tmp_val
        
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = [x.copy()]
    for i in range(step_num):
        grad = numerical_gradient(f, x) 
        x -= lr * grad
        if i % 5 == 0:
            x_history.append(x.copy())
    return x, np.array(x_history)
