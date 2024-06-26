import numpy as np
from loss_function import cross_entropy_error
from activation_function import softmax

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout
        dy = dout

        return dx, dy
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx

def test():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()
    
    #
    # forward
    #
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    final_price = mul_tax_layer.forward(all_price, tax)
    
    #
    # backward
    #
    dfinal_price = 1
    dall_price, dtax = mul_tax_layer.backward(dfinal_price)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    
    print(final_price)
    print(dapple, dapple_num, dorange, dorange_num, dtax)

if __name__ == '__main__':
    test()
