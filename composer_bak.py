import time
import tensorflow as tf
import numpy as np 
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

		labels.append([int(label)])
		vectors.append(np.array([temp_list]))

	return {'labels': labels, 'vectors': vectors}


input_data = load_input("train_data/data_1152_1.txt")
labels = input_data['labels']
vectors = input_data['vectors']
print "labels:%d, vectors:%d, %d" % (len(labels), len(vectors), len(vectors[0]))
print vectors[0]

train_input = vectors[:5000]
train_output = labels[:5000]
test_input = vectors[5000:]
test_output = labels[5000:]



sequence_length = 1
input_dimension = len(vectors[0][0])
print "Sequence_length: %d, Input_dimension: %d" % (sequence_length, input_dimension)
data = tf.placeholder(tf.float32, [None, sequence_length, input_dimension])
target = tf.placeholder(tf.float32, [None, sequence_length])

num_hidden = 256
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

#weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
#bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
weight = tf.Variable(tf.random_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.random_normal([int(target.get_shape()[1])]))

print target.get_shape(), val.get_shape(), "last:", last.get_shape(), "W:", weight.get_shape(), "b:", bias.get_shape()

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
print prediction.get_shape()

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
#cross_entropy = -tf.reduce_sum(tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))


optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)


mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)


batch_size = 100
no_of_batches = int(len(train_input)/batch_size)
epoch = 100
start = time.time()
for i in range(epoch):
	ptr = 0
	for j in range(no_of_batches):
		inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
		ptr += batch_size
		sess.run(minimize, {data: inp, target: out})
	print "Epoch - ", str(i)
end = time.time()

incorrect = sess.run(error, {data: test_input, target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
print "time used: %ds, incorrect: %d" % (end - start, incorrect)


sess.close()















