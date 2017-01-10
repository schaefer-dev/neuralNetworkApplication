from math import pow
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


epsilon = 0.1
iterations = 300
delta = 1e-9

def f(x):
    return 0.001 * x[0, 0] * x[0, 0] - 0.001 * x[1, 0] * x[1, 0] 

def fpx(x):
    return matrix([[0.002 * x[0, 0]], [-0.002 * x[1, 0]]])

## alternative plotting Begin ---------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xh = yh = np.arange(-10.0, 10.0, 0.05)
Xh, Yh = np.meshgrid(xh, yh)
zs = np.array([f(matrix([[xh],[yh]])) for xh,yh in zip(np.ravel(Xh), np.ravel(Yh))])
Zh = zs.reshape(Xh.shape)

# Drawing of the function surface
#ax.plot_surface(Xh, Yh, Zh)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


## alternative plotting End ---------------

# Initial point
x = matrix([[3], [-1]])
r = np.zeros(x.shape[0])

for i in range(0, iterations):
    coordX = x.item(0)
    coordY = x.item(1)
    df = fpx(x)
    fxy = f(x)
    # Drawing of the adagrad in yellow
    ax.scatter(coordX,coordY,fxy, c='yellow')
    print(r,x)
    r = r + np.tensordot(df,df)
    x = x - epsilon * df/(delta+np.sqrt(r))

x2 = matrix([[3], [-1]])
# in red gradient descent
for i in range(0, iterations):
    coordX = x2.item(0)
    coordY = x2.item(1)
    fxy = f(x2)

    # Drawing of the gradient descent
    ax.scatter(coordX,coordY,fxy, c='red')

    x2 = x2 - epsilon * fpx(x2)

print(x)
print(x2)
plt.show()
