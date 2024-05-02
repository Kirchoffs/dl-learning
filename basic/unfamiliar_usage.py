import numpy as np

def multi_index_iter():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    it = np.nditer(x, flags = ['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        val = x[idx]
        print(idx, val)
        it.iternext()

if __name__ == '__main__':
    multi_index_iter()
