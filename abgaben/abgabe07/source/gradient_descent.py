from math import pow
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


epsilon = 0.01
iterations = 30

def f(x):
    return 3 * x[0, 0] * x[0, 0] - x[1, 0] * x[1, 0] 

def fpx(x):
    return matrix([[6 * x[0, 0]], [-2 * x[1, 0]]])

# Initial point
x = matrix([[5], [-1]])


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

for i in range(0, iterations+1):
    coordX = x.item(0)
    coordY = x.item(1)
    fxy = f(x)

    # Drawing of the gradient descent
    ax.scatter(coordX,coordY,fxy, c='yellow')

    print("X = " + str(coordX))
    print("Y = " + str(coordY))
    print ("f(x,y) = " + str(fxy))
    print("---------------------------")
    # print(x, f(x))
    x = x - epsilon * fpx(x)
    # ax.scatter(x, f(x), zdir='z', c='yellow')

plt.show()
