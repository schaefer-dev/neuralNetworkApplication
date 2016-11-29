#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

rng = np.random


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
#plt.plot(trX, trY, 'ro')
#plt.xlabel('x-points')
#plt.ylabel('y-points')
#plt.set_title('DataSet (a)')
#plt.plot(x,y, 'bo')
#plt.savefig('train_distr.png')
#plt.close()

#plt.plot(teX, teY, 'ro')
#plt.xlabel('x-points')
#plt.ylabel('y-points')
#plt.set_title('DataSet (a)')
#plt.plot(x,y, 'bo')
#plt.savefig('test_distr.png')
#plt.close()


##end plot ##

#Debug
#print(trX)
#print(len(trX))
#print(trY)
#print(len(trY))
#print(teX)
#print(teY)

##### End Ex1.1 ######


##### new code ######

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")


learning_rate = 0.00001
training_epochs = 5000
display_step = 1
#initW=-20.0
#initb=619.0

# Set model weights
W_1 = tf.Variable(rng.randn())
W_2 = tf.Variable(rng.randn())
b = tf.Variable(rng.randn())
#W_1 = tf.Variable(tf.random_normal(1, stddev=0.01))
#W_2 = tf.Variable(tf.random_normal(1, stddev=0.01))
#b = tf.Variable(tf.random_normal(1, stddev=0.01))
#W=tf.Variable(initW)
#b=tf.Variable(initb)

# Construct a linear model
second_order= tf.mul(W_2, tf.mul(X, X))
pred = tf.add(second_order, tf.add(tf.mul(W_1, X), b))

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*len(trX))
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
costs=[]
predictions=[]
with tf.Session() as sess:
    sess.run(init)
   
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(trX, trY):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: trX, Y:trY})
            costs.append(c)
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(c), \
                "W_1=", sess.run(W_1),"W_2=", sess.run(W_2), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})
    print("Training cost=", training_cost, "W_1=", sess.run(W_1), "W_2=", sess.run(W_2), "b=", sess.run(b), '\n')

	# plot predicted training data
    for (x, y) in zip(trX, trY):
    	classification = sess.run(pred, feed_dict={X:x})
	predictions.append(classification)
	print(predictions)

    plt.plot(trX, predictions, 'ro')
    plt.xlabel('x')
    plt.ylabel('predicted value')
    plt.savefig('costplots/twodimprediction.png')
    plt.close();

# plot costs for each epoch 
epochlist=range(1,training_epochs+1)
plt.plot(epochlist, costs, 'r')
plt.xlabel('epoche')
plt.ylabel('cost')
plt.axis([1,training_epochs+1, 1500,3500])
#plt.set_title('Epochs')
#plt.savefig('costplots/cost'+str(training_epochs)+ '_second_dim' +'_w'+str(initW)+'_b'+str(initb)+'_learn'+str(learning_rate)+'.png')
plt.close()


#X = tf.placeholder("float", [None, 784]) # create symbolic variables
#Y = tf.placeholder("float", [None, 10])

#w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

#py_x = model(X, w)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
#train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
#predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
#with tf.Session() as sess:
    # you need to initialize all variables
#    tf.initialize_all_variables().run()

 #   for i in range(100):
  #      for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
   #         sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
