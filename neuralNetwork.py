import tensorflow as tf
import numpy as np


def nn_1layer(inputData, inputs, nodesLayer1, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesLayer1])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer1]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesLayer1, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	y_ = tf.add(tf.matmul(l1, output['weights']), output['biases'])

	return y_

def nn_2layer(inputData, inputs, nodesLayer1, nodesLayer2, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesLayer1])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer1]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesLayer1, nodesLayer2])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer2]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesLayer2, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	y_ = tf.add(tf.matmul(l2, output['weights']), output['biases'])

	return y_

def nn_3layer(inputData, inputs, nodesLayer1, nodesLayer2, nodesLayer3, outputs) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesLayer1])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer1]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesLayer1, nodesLayer2])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer2]))}
	layer3 = {'weights': tf.Variable(tf.random_normal([nodesLayer2, nodesLayer3])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer3]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesLayer3, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.sigmoid(l3)

	y_ = tf.add(tf.matmul(l2, output['weights']), output['biases'])

	return y_










