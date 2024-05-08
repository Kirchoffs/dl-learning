import numpy as np

x = np.array([1, 2, 3])
powers = np.power(x, np.arange(5).reshape(-1, 1))
print(powers)
