from math import pow
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def f( x , y ):
	return 20*x*x + y*y/4


def fpx( x , y ):
	return 40*x

def fpy( x , y ):
	return y/2
# Initial point
x=-2
y=4

epsilon=0.04

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
#Z = f(X,Y)
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax.set_zlim(-1, 600)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#fig.colorbar(surf, shrink=0.5, aspect=5)
#colors = ('r', 'g', 'b', 'k')

for i in range(0, 100):
	#print x, y, f(x,y)
	x1 = x - epsilon * fpx(x,y)
	y1 = y - epsilon * fpy(x,y)
	x = x1
	y = y1
	#ax.scatter(x, y, f(x,y), zdir='z', c='yellow')

print int(x), int(y)

#plt.show()