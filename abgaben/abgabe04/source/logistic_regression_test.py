import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

reader = open("iris.data", "r")
data = [];
for row in reader:
    d = row.split(',')
    data.append(d)    

X,Y=[],[]
for d in data:
    x=[]
    X.append([float(d[0]),float(d[3])])
    if 'Iris-setosa' in str(d[4]):
        Y.append([0])
    else:
        Y.append([1])

X = np.asarray(X)
Y = np.asarray(Y)
label=np.array([[0,1]])
trX, trY, teX, teY = X,Y,X,Y

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 1
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 2]) # mnist data image of shape 50*2=100
y = tf.placeholder(tf.float32, [None, 2]) # 0-1 digits recognition => 2 classes

# Set model weights
W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.matmul(x, W)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        cos = 1.
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            batch_xs, batch_ys = trX[start:end],trY[start:end]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            print(sess.run(c))
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cos))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: teX, y: label}))    
    # plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))

    Z = np.matmul(np.c_[xx.ravel(), yy.ravel()],sess.run(W))
    Z = np.argmax(Z,1)
    Z = Z.reshape(xx.shape)
    print Z
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    pos = np.where(teY == 0.0)
    neg = np.where(teY == 1.0)
    plt.scatter(teX[pos,0], teX[pos,1], marker='o', c='b')
    plt.scatter(teX[neg,0], teX[neg,1], marker='x', c='r')
    plt.xlabel('sepal length')
    plt.ylabel('ppetal width')
    plt.show()