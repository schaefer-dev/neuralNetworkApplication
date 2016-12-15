#!/usr/bin/env python

import tensorflow
import tensorflow as tf
import pylab as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))

def sigmaprime(x):
    return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))

def predict(x):
    layer1 = tf.add(tf.matmul(x, wh), bh)
    layer1 = sigma(layer1)
    output = tf.add(tf.matmul(layer1, wo), bo)
    output = sigma(output)
    diff = tf.sub(output, y)



def train(X,epochs,epsilon):
    layer1 = tf.add(tf.matmul(x, wh), bh)
    layer1 = sigma(layer1)
    output = tf.add(tf.matmul(layer1, wo), bo)
    output = sigma(output)
    diff = tf.sub(output, y)
    acct_mat = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    d_output = tf.mul(diff, sigmaprime(output))
    d_bo = d_output
    d_wo = tf.matmul(tf.transpose(layer1), d_output)

    d_layer1 = tf.matmul(d_output, tf.transpose(wo))
    d_layer1 = tf.mul(d_layer1, sigmaprime(layer1))
    d_bh = d_layer1
    d_wh = tf.matmul(tf.transpose(x), d_layer1)
    step = [
        tf.assign(wh,
                tf.sub(wh, tf.mul(epsilon, d_wh)))
      , tf.assign(bh,
                tf.sub(bh, tf.mul(epsilon,
                                   tf.reduce_mean(d_bh, reduction_indices=[0]))))
      , tf.assign(wo,
                tf.sub(wo, tf.mul(epsilon, d_wo)))
      , tf.assign(bo,
                tf.sub(bo, tf.mul(epsilon,
                                   tf.reduce_mean(d_bo, reduction_indices=[0]))))]

    for i in xrange(epochs):
        batch_xs, batch_ys = X.next_batch(10)
        sess.run(step, feed_dict = {x: batch_xs,
                                    y : batch_ys})
        differ = sess.run(diff, feed_dict = {x: batch_xs,
                                    y : batch_ys})
        MSE = 0
        for d in differ:
            MSE += np.dot(d,d)
        MSE = MSE/2
        plt.scatter(i,MSE)
        if i % 1000 == 0:
            res = sess.run(acct_res, feed_dict =
                           {x: mnist.test.images[:1000],
                            y : mnist.test.labels[:1000]})
            print res

#parameter
hidden_1 = 256
learning_rate = tf.constant(0.5)
epochs=10000

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

wh = tf.Variable(tf.truncated_normal([784, hidden_1]))
bh = tf.Variable(tf.truncated_normal([1, hidden_1]))
wo = tf.Variable(tf.truncated_normal([hidden_1, 10]))
bo = tf.Variable(tf.truncated_normal([1, 10]))

train(mnist.train,epochs,learning_rate)
plt.show()