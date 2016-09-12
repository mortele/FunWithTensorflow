import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateData as gen

inputs 		= 1
nodesLayer1 = 10
nodesLayer2 = 10
outputs		= 1

function = lambda z: 2*z;

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
	l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.sigmoid(l2)

	y_ = tf.add(tf.matmul(l2, output['weights']), output['biases'])

	return y_

def trainNetwork(x, plotting=False) :
	y = tf.placeholder('float')
	prediction = neuralNetwork(x)
	cost = tf.nn.l2_loss(tf.sub(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	numberOfEpochs 	= 50
	epochDataSize 	= int(1e6)
	batchSize		= int(1e4)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.functionData(batchSize, function)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.functionData(testSize, function, training=False)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
			print "Epoch #: ", epoch+1, \
				  " epoch loss: ", epochLoss, \
				  " test set loss: ", testCost 

		if plotting :
			N   = 25
			xx  = np.linspace(0,1,N)
			xx  = xx.reshape([N,1])
			yy_ = sess.run(prediction, feed_dict={x: xx})
			yy  = function(xx)

			plt.figure(1)
			plt.plot(xx, yy_, 'ro')
			plt.hold('on')
			plt.plot(xx, yy, 'b--')
			plt.legend(['nn(x)', 'f(x)'])
			plt.show()


trainNetwork(x, plotting=True)








