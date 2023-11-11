import numpy as np
import matplotlib.pyplot as plt
from simple_gradient import gradient_descent

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x = init_x, lr = lr, step_num = step_num)
plt.plot([-5, 5], [0, 0], 'b')
plt.plot([0, 0], [-5, 5], 'b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")

num_points = 100
theta = np.linspace(0, 2 * np.pi, num_points)
for i in range(0, len(x_history), 2):
    radius = np.sqrt(function_2(x_history[i]))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    plt.plot(x, y, linestyle = '--', markersize = 2, color = 'blue')

ax = plt.gca()
ax.set_aspect('equal', adjustable = 'box')

plt.show()
