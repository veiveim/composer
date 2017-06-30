import sys
import time
import tensorflow as tf
import numpy as np 
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

width = 256
height = 3
n_channels = 1
n_classes = 2

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
				wave_list.append(np.float32(token))
			vector_list.append(wave_list)

		#label_list = [0] * int(n_classes)
		#label_list[int(label)] = 1
		#labels.append(np.array(label_list))
		labels.append(np.array(np.int(label)))
		vectors.append(np.array(vector_list))

	infile.close()
	#return {'labels': labels, 'vectors': vectors}
	return {'labels': np.array(labels), 'vectors': np.array(vectors)}


def cnn_model_fn(features, labels, mode):

 	# Input Layer
 	# output: shape of [batch_size, width:kwaves, height:1, channels:3]
	input_layer = tf.reshape(features, [-1, width, height, n_channels])

	# Convolutional Layer #1
	# output: shape of [batch_size, width, height, filters]
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 128,
		kernel_size = [5, 1],
		padding = "same",
		activation = tf.nn.relu)

	# Pooling Layer #1
	# output: [batch_size, 128, 1, 128]: the 2x1 filter reduces width by 50%.
	pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 1], strides = 2)
	print "conv1, pool1: ", conv1.get_shape(), pool1.get_shape()

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 256,
		kernel_size = [5, 1],
		padding = "same",
		activation = tf.nn.relu)
	
	# Pooling Layer #2
	# output: [batch_size, 64, 1, 256]: the 2x1 filter reduces width by 50%.
	pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 1], strides = 2)
	print "conv2, pool2: ", conv2.get_shape(), pool2.get_shape()
	
	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(
		inputs = pool2,
		filters = 512,
		kernel_size = [5, 1],
		padding = "same",
		activation = tf.nn.relu)
	
	# Pooling Layer #3
	# output: [batch_size, 32, 1, 512]: the 2x1 filter reduces width by 50%.
	pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2, 1], strides = 2)
	print "conv3, pool3: ", conv3.get_shape(), pool3.get_shape()
	

	# Dense Layer
	pool3_flat = tf.reshape(pool3, [-1, 32 * 1 * 512])
	dense = tf.layers.dense(inputs = pool3_flat, units = 1024, activation = tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs = dense, 
		rate = 0.4,
		training = mode == learn.ModeKeys.TRAIN)
	print "pool4_flat, dense, dropout: ", pool3_flat.get_shape(), dense.get_shape(), dropout.get_shape()




	# Logits Layer
	logits = tf.layers.dense(inputs = dropout, units = n_classes)

	loss = None
	train_op = None

	# Calculate Loss (for both TRAIN and EVAL modes)
 	if mode != learn.ModeKeys.INFER:
 		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=int(n_classes))
 		print "logits, onehot_labels: ", logits.get_shape(), onehot_labels.get_shape()
 		loss = tf.losses.softmax_cross_entropy(
 			onehot_labels=onehot_labels, logits=logits)

  	# Configure the Training Op (for TRAIN mode)
  	if mode == learn.ModeKeys.TRAIN:
  		train_op = tf.contrib.layers.optimize_loss(
        	loss=loss,
        	global_step=tf.contrib.framework.get_global_step(),
        	learning_rate=0.001,
        	optimizer="SGD")

  	# Generate Predictions
  	predictions = {
      	"classes": tf.argmax(
          	input=logits, axis=1),
      	"probabilities": tf.nn.softmax(
          	logits, name="softmax_tensor")
  	}


	# Return a ModelFnOps object
  	return model_fn_lib.ModelFnOps(
      	mode=mode, predictions=predictions, loss=loss, train_op=train_op)




if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "usage: input_data result_file n_classes\n"
		sys.exit(1)
	
	data_file = sys.argv[1]
	n_classes = sys.argv[3]
	result_file = open(sys.argv[2], "w")


	input_data = load_input(data_file, n_classes)

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
	
	# Create the Estimator
	classifier = learn.Estimator(
     	model_fn=cnn_model_fn, model_dir="/tmp/composer_convnet_model_conv3_class2_256_3_1")

	# Set up logging for predictions
  	tensors_to_log = {"probabilities": "softmax_tensor"}
  	logging_hook = tf.train.LoggingTensorHook(
      	tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	classifier.fit(
    	x=train_input,
    	y=train_output,
    	batch_size=100,
    	steps=20000,
    	monitors=[logging_hook])

	# Configure the accuracy metric for evaluation
	metrics = {
	    "accuracy":
	        learn.MetricSpec(
	            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
	}

	# Evaluate the model and print results
	eval_results = classifier.evaluate(
    	x=test_input, y=test_output, metrics=metrics)
	print(eval_results)
	result_file.write(eval_results)


