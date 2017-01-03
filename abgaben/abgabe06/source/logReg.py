import tensorflow as tf
import numpy as np
import pylab as plt

reader = open("logReg.csv", "r")
data = [];
for row in reader:
    d = row.split(',')
    data.append(d)   

I,Y=[],[]
del data[0]
for d in data:
    I.append([float(d[0]),float(d[1])])
    if  0 == int(d[2]):
        Y.append([0,1])
        # plot for a
        plt.scatter(float(d[0]), float(d[1]), marker='o', c='b')
    else:
        Y.append([1,0])
        # plot for a
        plt.scatter(float(d[0]), float(d[1]), marker='x', c='r')

# plot for a
plt.xlabel('u')
plt.ylabel('v')
plt.show()

x,X=[],[]
for d in I:
    x = 0
    for i in range(0,6):
        for j in range(0,6):
            x= x + pow(d[0],i)*pow(d[1],j)
    X.append(x)
print(X)



