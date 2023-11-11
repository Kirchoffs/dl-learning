# Notes
## Run
### Disable Bytecode Generation
```
>> python -B gradient-simple-demo.py
```

## Debug
```
>> python -m pdb train_neural_net.py
>> n

>> b train_neural_net.py:28
>> c
```

## NumPy
### Axis
```
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.max(x, axis = 0)
z = np.max(x, axis = 1)

print(y)
# [4 5 6]
print(z)
# [3 6]
```

For x[2][3], if we want to get the max value based on axis = 0, which has length 2 here, we need to compare x[0][0] and x[1][0], x[0][1] and x[1][1], x[0][2] and x[1][2], respectively. So the result is [4, 5, 6].
