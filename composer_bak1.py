import time
import tensorflow as tf
import numpy as np 
from tensorflow.contrib import rnn
from random import shuffle


def load_input(file):
	infile = open(file, "r")
	lines = infile.readlines()
	labels = []
	vectors = []

	for line in lines:
		temp_list = []
		[label, vector] = line.split(":")
		waves = vector.split("\t")
		for wave in waves:
			if wave == "\n":
				break
			for token in wave.split(","):
				temp_list.append(int(token))

		labels.append([int(label) == 0, int(label) == 1])
		vectors.append(np.array([temp_list]))

	return {'labels': labels, 'vectors': vectors}


input_data = load_input("train_data/data_1152_1.txt")
labels = input_data['labels']
vectors = input_data['vectors']
print "labels:%d, vectors:%d, %d" % (len(labels), len(vectors), len(vectors[0]))
print vectors[0]

train_input = vectors[:15000]
train_output = labels[:15000]
test_input = vectors[15000:]
test_output = labels[15000:]


# Parameters
learning_rate = 0.01
training_iters = 100000000
batch_size = 256
display_step = 10

# Network Parameters
n_input = len(vectors[0][0])
n_steps = 1
n_hidden = 64
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights, biases
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
	# Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

 	# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    ptr = 0
    start = time.time()
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = train_input[ptr : ptr + batch_size], train_output[ptr : ptr + batch_size]
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    end = time.time()
    print("Optimization Finished! Time used: ", end - start)

    # Calculate accuracy for 
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_input, y: test_output}))




