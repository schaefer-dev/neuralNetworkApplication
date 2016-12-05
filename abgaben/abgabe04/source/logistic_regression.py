
import tensorflow as tf
import numpy as np
import pylab as plt

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


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

X = tf.placeholder("float", [None, 2]) # create symbolic variables
Y = tf.placeholder("float", [None, 2])

w = init_weights([2, 2]) # like in linear regression, we need a shared variable weight matrix for logistic regression
b = tf.Variable(tf.zeros([2]))

py_x = model(X, w)+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(1000):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        #print(sess.run(w))
        print(i, np.mean(np.argmax(teY, axis=1)==sess.run(predict_op, feed_dict={X: teX})))
    W = sess.run(w)
    B = sess.run(b)
    # plot
    x_min, x_max = trX[:, 0].min() - 1, trX[:, 0].max() + 1
    y_min, y_max = trX[:, 1].min() - 1, trX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))

    Z = np.matmul(np.c_[xx.ravel(), yy.ravel()],sess.run(w))
    print Z
    Z = np.argmax(Z,1)
    Z = Z.reshape(xx.shape)
    print Z
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.BrBG)
    setosa = np.where(np.argmax(np.matmul(teX,W)+B,1) == 0)
    virginica = np.where(np.argmax(np.matmul(teX,W)+B,1) == 1)
    plt.scatter(teX[setosa,0], teX[setosa,1], marker='o', c='b')
    plt.scatter(teX[virginica,0], teX[virginica,1], marker='x', c='r')
    plt.xlabel('sepal length')
    plt.ylabel('ppetal width')
    plt.show()