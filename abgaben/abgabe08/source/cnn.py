from __future__ import print_function
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Parameters
learning_rate = 0.0001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = conv2d(x,weights['h1'])+biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = max_pool_2x2(layer_1)
    # Hidden layer with RELU activation
    layer_1_flat = tf.reshape(layer_1, [-1, 14*14*32])
    layer_2 = tf.add(tf.matmul(layer_1_flat, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([14 * 14 * 32, 1024], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[32])),
    'b2': tf.Variable(tf.constant(0.1, shape=[1024])),
    'out': tf.Variable(tf.constant(0.1, shape=[10]))
}

#reshape images
x_reshape = tf.reshape(x, [-1,28,28,1])

# Construct model
pred = multilayer_perceptron(x_reshape, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            batch_x = batch[0]
            batch_y = batch[1]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))


        ### Test model on training data begin
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy on test-set:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        ### Test model on training data end

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy on test-set:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
