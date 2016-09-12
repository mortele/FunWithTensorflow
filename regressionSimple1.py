import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateData as gen
import neuralNetwork as nn

inputs 	   = 1
layerNodes = 20
outputs	   = 1

function = lambda z: np.exp(-z);
a = 0
b = 5

x = tf.placeholder('float', [None, 1], 	name='x')
y = tf.placeholder('float')


#neuralNetwork = lambda inputData : nn.nn_1layer(inputData, inputs, layerNodes, outputs)
#neuralNetwork = lambda inputData : nn.nn_2layer(inputData, inputs, layerNodes, layerNodes, outputs)
neuralNetwork = lambda inputData : nn.nn_3layer(inputData, inputs, layerNodes, layerNodes, layerNodes, outputs)

def trainNetwork(x, plotting=False) :
	y = tf.placeholder('float')
	prediction = neuralNetwork(x)
	cost = tf.nn.l2_loss(tf.sub(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	numberOfEpochs 	= 25
	epochDataSize 	= int(1e6)
	batchSize		= int(1e4)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.functionData(batchSize, function, a, b)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.functionData(testSize, function, a, b, training=False)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
			print "Epoch #: %3d  epoch loss: %15f  test set loss: 15%f" % (epoch+1, epochLoss, testCost)

		if plotting :
			N     = testSize
			xx,yy = gen.functionData(testSize, function, a, b, training=False)
			#xx  = np.linspace(a,b,N)
			xx  = xx.reshape([N,1])
			yy_ = sess.run(prediction, feed_dict={x: xx})
			#yy  = function(xx)

			plt.figure(1)
			plt.plot(xx, yy_, 'ro')
			plt.hold('on')
			plt.plot(xx, yy, 'bx')
			plt.legend(['nn(x)', 'f(x)'])
			plt.show()


trainNetwork(x, plotting=True)








