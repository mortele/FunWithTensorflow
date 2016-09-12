import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateData as gen

inputs 		= 2
nodesLayer1 = 10
nodesLayer2 = 10
outputs		= 2

x = tf.placeholder('float', [None, 2], 	name='x')
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

	numberOfEpochs 	= 10
	epochDataSize 	= int(1e6)
	batchSize		= int(1e4)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.uniformOrdering(batchSize)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.uniformOrdering(testSize, training=False)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
			print "Epoch #: ", epoch+1, \
				  " epoch loss: ", epochLoss, \
				  " test set loss: ", testCost 

		if plotting :
			N  = 1000
			x1 = np.random.uniform(.45,.55,N)
			x2 = np.random.uniform(.45,.55,N)
			x1 = x1.reshape([N, 1])
			x2 = x2.reshape([N, 1])
			greater = (x1 > x2)
			less    = (x1 < x2)
			xx = np.concatenate((x1, x2), axis=1)
			y_ = np.concatenate((greater, less), axis=1)
			y_ = y_.astype(float)
			y  = sess.run(prediction, feed_dict={x: xx})
			netGreater = y[:,0] > y[:,1];
			netGreater = netGreater.astype(float)	
			greater = greater.T
			greater = greater[0]

			plt.figure(1)
			plt.plot(xx[:,0], 'ro')
			plt.hold('on')
			plt.plot(xx[:,1], 'bo')

			plt.plot(netGreater, 'k-x')
			plt.plot(greater.astype(float), 'y-o')

			print "Number of misses in N=%d tries: %f" % (N, sum(abs(netGreater-greater.astype(float))))
			plt.xlabel('test #')
			plt.ylabel('function output, neuralNetwork(x)')
			plt.legend(['x1', 'x2', 'NN(x1,x2)', 'x1>x2?'])
			plt.axis([0, N, -0.1, 1.1])
			plt.show()

trainNetwork(x, plotting=True)








