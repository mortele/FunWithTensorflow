import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateData as gen


inputs 		= 1
nodesLayer1 = 10
nodesLayer2 = 10
outputs		= 2

x = tf.placeholder('float', [None, 1], 	name='x')
y = tf.placeholder('float')


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

def trainNetwork(x, plotting=False) :
	y = tf.placeholder('float')
	prediction = neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	alpha			= 0.5
	numberOfEpochs 	= 20
	epochDataSize 	= int(1e6)
	batchSize		= int(1e4)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.uniformGreaterThan(batchSize, alpha)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.uniformGreaterThan(testSize, alpha, training=False)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
			print "Epoch #: ", epoch+1, \
				  " epoch loss/data size: ", epochLoss/epochDataSize, \
				  " test set loss/test size: ", testCost/testSize 

		if plotting :
			N  = 1000
			xx = np.linspace(alpha*0.95, alpha*1.05, N)
			ee = xx > alpha
			ee = ee.astype(float)
			xx = xx.reshape([N,1])
			yy = sess.run(prediction, feed_dict={x: xx})
			ff = yy[:,0] > yy[:,1]
			ff = ff.astype(float)
			plt.figure(1)
			plt.plot(xx[:,0],ff, 'r-')
			plt.hold('on')
			plt.plot(xx[:,0],ee, 'k-')
			plt.xlabel('input data x')
			plt.ylabel('function output, neuralNetwork(x)')
			plt.legend(['NeuralNetwork','exact'])
			plt.axis([min(xx), max(xx), -0.1, 1.1])
			plt.show()

trainNetwork(x, plotting=True)








