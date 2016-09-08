import tensorflow as tf
import numpy as np
import generateData as gen

inputs 		= 1
nodesLayer1 = 10
nodesLayer2 = 10
outputs		= 2

x = tf.placeholder('float', [None, 1], 	name='x')
y = tf.placeholder('float', 			name='y')


def neuralNetwork(inputData) :
	layer1 = {'weights': tf.Variable(tf.random_normal([inputs, nodesLayer1])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer1]))}
	layer2 = {'weights': tf.Variable(tf.random_normal([nodesLayer1, nodesLayer2])),
			  'biases':  tf.Variable(tf.random_normal([nodesLayer2]))}
	output = {'weights': tf.Variable(tf.random_normal([nodesLayer2, outputs])),
			  'biases':  tf.Variable(tf.random_normal([outputs]))}

	l1 = tf.add(tf.matmul(inputData, layer1['weights']), layer1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.relu(l2)

	y_ = tf.add(tf.matmul(l2, output['weights']), output['biases'])

	return y_

def trainNetwork(x) :
	prediction = neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	alpha			= 0.5
	numberOfEpochs 	= 10
	epochDataSize 	= int(1e5)
	batchSize		= int(1e2)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.uniformGreaterThan(batchSize, alpha)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec
			print "Epoch #: ", epoch+1, " epoch loss: ", epochLoss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		x,y = gen.uniformGreaterThan(1e3, alpha, training=False)
		print('Accuracy:',accuracy.eval({x: x, y: y}))

trainNetwork(x)







