# Notes

```
x = np.array([-1, -1, 1, 1])
y = (x <= 0) # array([True,  True, False, False])
z = np.array([1, 2, 3, 4])
z[y] = 0 # array([0, 0, 3, 4])
```
