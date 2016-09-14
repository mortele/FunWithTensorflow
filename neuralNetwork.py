import tensorflow as tf
import numpy as np


def nn_1layer(inputData, inputs, nodesPerLayer, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	y_ = tf.add(tf.matmul(l1, output['weights']), output['biases'])

	return y_

def nn_2layer(inputData, inputs, nodesPerLayer, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	y_ = tf.add(tf.matmul(l2, output['weights']), output['biases'])

	return y_

def nn_3layer(inputData, inputs, nodesPerLayer, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer3 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.sigmoid(l3)

	y_ = tf.add(tf.matmul(l3, output['weights']), output['biases'])

	return y_


def nn_4layer(inputData, inputs, nodesPerLayer, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer3 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer4 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.sigmoid(l3)

	l4 = tf.add(tf.matmul(l3, layer4['weights']), layer4['biases'])
	l4 = tf.nn.sigmoid(l4)

	y_ = tf.add(tf.matmul(l4, output['weights']), output['biases'])

	return y_


def nn_5layer(inputData, inputs, nodesPerLayer, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer3 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer4 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	layer5 = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, nodesPerLayer])),
			  'biases':  tf.Variable(tf.random_normal([nodesPerLayer]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesPerLayer, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.sigmoid(l3)

	l4 = tf.add(tf.matmul(l3, layer4['weights']), layer4['biases'])
	l4 = tf.nn.sigmoid(l4)

	l5 = tf.add(tf.matmul(l4, layer5['weights']), layer5['biases'])
	l5 = tf.nn.sigmoid(l5)

	y_ = tf.add(tf.matmul(l5, output['weights']), output['biases'])

	return y_


