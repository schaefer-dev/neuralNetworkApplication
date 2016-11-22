from math import pow
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def fun(x,y):
    return 20 * x * x + 1/4 * y * y


def f(x):
    return 20 * x[0, 0] * x[0, 0] + x[1, 0] * x[1, 0] / 4


def fpx(x):
    return matrix([[20 * x[0, 0]], [x[1, 0] / 2]])
# Initial point
x = matrix([[-2], [4]])

epsilon = 0.04

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = f(X,Y)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# ax.set_zlim(-1, 600)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)
# colors = ('r', 'g', 'b', 'k')


## alternative plotting Begin ---------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xh = yh = np.arange(-6.0, 6.0, 0.05)
Xh, Yh = np.meshgrid(xh, yh)
zs = np.array([fun(xh,yh) for xh,yh in zip(np.ravel(Xh), np.ravel(Yh))])
Zh = zs.reshape(Xh.shape)

ax.plot_surface(Xh, Yh, Zh)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


## alternative plotting End ---------------



for i in range(0, 1000):
    coordX = x.item(0)
    coordY = x.item(1)
    fxy = f(x)
    ax.scatter(coordX,coordY,fxy, c='yellow')
    print("X = " + str(coordX))
    print("Y = " + str(coordY))
    print ("f(x,y) = " + str(fxy))
    print("---------------------------")
    # print(x, f(x))
    x = x - epsilon * fpx(x)
    # ax.scatter(x, f(x), zdir='z', c='yellow')

plt.show()
