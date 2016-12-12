import tensorflow as tf
import input_data
import numpy as np
import pylab as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def sigmaprime(x):
    return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))

def sigmoid(x):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0),tf.exp(-x)))


# Parameters
learning_rate = 0.001
training_epochs = 3
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def predict(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = sigmoid(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    out_layer = sigmoid(out_layer)
    return out_layer

def train(X,epochs,epsilon):
    with tf.Session() as sess:
        sess.run(init)
        loss = []
        for epoch in range(0,epochs):
            i = 0
            MSE = 0
            y1 = sess.run(predict(X.images))
            y = X.labels[i]
            diff = y1-y
            for d in diff:
                MSE = np.dot(d,d)
            MSE = MSE/2
            plt.scatter(epoch,MSE)
            print("Epoch:", '%04d' % (epoch+1), "cost=",MSE)
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        plt.show()  


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = predict(x)

# Initializing the variables
init = tf.initialize_all_variables()

X = mnist.train
train(X,training_epochs,learning_rate)
