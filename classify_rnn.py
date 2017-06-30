import sys
import time
import tensorflow as tf
import numpy as np 
from tensorflow.contrib import rnn
from random import shuffle


def load_input(file, n_classes):
	infile = open(file, "r")
	lines = infile.readlines()
	labels = []
	vectors = []

	for line in lines:
		vector_list = []
		[label, vector] = line.split(":")
		waves = vector.split("\t")
		for wave in waves:
			if wave == "\n":
				break
			#for token in wave.split(","):
			#	temp_list.append(float(token))
			
			wave_list = []
			for token in wave.split(","):
				wave_list.append(float(token))
			vector_list.append(wave_list)

		label_list = [0] * int(n_classes)
		label_list[int(label)] = 1
		labels.append(np.array(label_list))
		vectors.append(np.array(vector_list))

	infile.close()
	return {'labels': labels, 'vectors': vectors}


def RNN(x, weights, biases, n_steps):
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


def classify_rnn(input_data, n_classes, n_hidden, batch_size, learning_rate, training_iters):
	labels = input_data['labels']
	vectors = input_data['vectors']

	print "labels:%d, vectors:%d, %d" % (len(labels), len(vectors), len(vectors[0]))
	print vectors[0]
	print labels[0]

	n_train_data = len(labels) * 3 / 4
	train_input = vectors[:n_train_data]
	train_output = labels[:n_train_data]
	test_input = vectors[n_train_data:]
	test_output = labels[n_train_data:]

	# Parameters
	#learning_rate = 0.01
	#training_iters = 2000000
	#batch_size = 128
	display_step = 100

	# Network Parameters
	#n_input = len(vectors[0][0])
	#n_steps = 1
	n_steps = len(vectors[0])
	n_input = len(vectors[0][0])
	
	print "n_steps: ", n_steps, " n_input: ", n_input
	#n_hidden = 128
	#n_classes = 2

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


	pred = RNN(x, weights, biases, n_steps)

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
	    loss = 0.0
	    # Keep training until reach max iterations
	    while step * batch_size < training_iters:
	        batch_x, batch_y = train_input[ptr : ptr + batch_size], train_output[ptr : ptr + batch_size]
	        # Reshape data to get 28 seq of 28 elements
	        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
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

	    # Calculate accuracy
	    acc = sess.run(accuracy, feed_dict={x: test_input, y: test_output})
	    print("Optimization Finished! Time used: ", end - start, " Accuracy: ", acc)
	    return {'time': end - start, 'acc': acc, 'loss': loss}



if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "usage: input_data result_file n_classes\n"
		sys.exit(1)
	
	data_file = sys.argv[1]
	n_classes = sys.argv[3]

	result_file = open(sys.argv[2], "w")

	input_data = load_input(data_file, n_classes)
	

	learning_rate = 0.01
	#training_iters = 1000000
	hidden_array = [256]
	batch_array = [32, 64, 128, 256, 1024]

	for n_hidden in hidden_array:
		result_file.write("n_hidden: %d\n" % n_hidden)
		for batch_size in batch_array:
			training_iters = 1000 * batch_size
			with tf.variable_scope("hidden_%s_%s" % (n_hidden, batch_size)):
				
				result = classify_rnn(input_data, int(n_classes), int(n_hidden), batch_size, learning_rate, training_iters)
				line = "%d, %f, %f, %f\n" % (batch_size, result['acc'], result['time'], result['loss'])
				result_file.write(line)

	result_file.close()






