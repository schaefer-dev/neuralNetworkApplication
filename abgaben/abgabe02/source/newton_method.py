from math import pow
import numpy as np
from pylab import *


def f(x):
    return 20 * x[0, 0] * x[0, 0] + x[1, 0] * x[1, 0] / 4


def fp(x):
    return matrix([[20 * x[0, 0]], [x[1, 0] / 2]])


def hfi(x):
    H = matrix([[40, 0], [0, 0.5]])
    Hi = H.I
    return Hi

# Initial point
x = matrix([[-2], [4]])

# Learning rate
epsilon = 1

for i in range(0, 30):
    print(x, f(x))
    x = x - epsilon * hfi(x) * fp(x)
