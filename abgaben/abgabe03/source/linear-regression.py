#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

##### Ex1.1 ######
reader = csv.reader(open("auto-mpg.data", "rt"))
cars = [];
for row in reader:
	car = row[0].split()
	cars.append(car)

X,Y=[],[]

for car in cars:
	X.append(float(car[0]))
	Y.append(float(car[2]))

trX, trY, teX, teY = X[0:50],Y[0:50],X[50:100],Y[50:100]

##plot##
plt.plot(trX, trY, 'ro')
plt.xlabel('x-points')
plt.ylabel('y-points')
#plt.set_title('DataSet (a)')
#plt.plot(x,y, 'bo')
plt.savefig('train_distr.png')
plt.close()

plt.plot(teX, teY, 'ro')
plt.xlabel('x-points')
plt.ylabel('y-points')
#plt.set_title('DataSet (a)')
#plt.plot(x,y, 'bo')
plt.savefig('text_distr.png')
plt.close()

##end plot ##

print(trX)
print(len(trX))
print(trY)
print(len(trY))
print(teX)
print(teY)

##### End Ex1.1 ######


X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
#with tf.Session() as sess:
    # you need to initialize all variables
#    tf.initialize_all_variables().run()

 #   for i in range(100):
  #      for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
   #         sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})