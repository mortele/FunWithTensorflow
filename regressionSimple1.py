import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateData as gen
import neuralNetwork as nn

inputs 	   = 1
layerNodes = 5
layers     = 4
outputs	   = 1

function = lambda z: 1/z**12 - 1/z**6
a = 0.9
b = 1.6

x = tf.placeholder('float', [None, 1], 	name='x')
y = tf.placeholder('float')

#neuralNetwork = lambda inputData : nn.nn_1layer(inputData, inputs, layerNodes, outputs)
#neuralNetwork = lambda inputData : nn.nn_2layer(inputData, inputs, layerNodes, outputs)
neuralNetwork = lambda inputData : nn.nn_3layer(inputData, inputs, layerNodes, outputs)
#neuralNetwork = lambda inputData : nn.nn_4layer(inputData, inputs, layerNodes, outputs)
#neuralNetwork = lambda inputData : nn.nn_5layer(inputData, inputs, layerNodes, outputs)

def trainNetwork(x, plotting=False) :
	y = tf.placeholder('float')
	prediction = neuralNetwork(x)
	cost = tf.nn.l2_loss(tf.sub(prediction, y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost) 
	#optimizer = tf.train.AdagradOptimizer(learning_rate=0.3).minimize(cost) 

	numberOfEpochs 	= 500
	epochDataSize 	= int(1e7)
	batchSize		= int(1e5)
	testSize		= int(1e4)

	saver = tf.train.Saver()

	with tf.Session() as sess :
		#model = tf.initialize_variables(tf.all_variables, name='init') 
		model = tf.initialize_all_variables()
		sess.run(model)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				newEpoch = False
				if i==0 :
					newEpoch = True
				#xBatch, yBatch = gen.functionData(batchSize, function, a, b, normal=True, sigma=0.1, mu=1.12, newEpoch=newEpoch)
				xBatch, yBatch = gen.functionData(batchSize, function, a, b, linspace=True, newEpoch=newEpoch)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.functionData(testSize, function, a, b, linspace=True, training=False)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
			print "Epoch #: %3d  epoch loss/epoch size: %10g  test set loss/test size: %10g" % (epoch+1, epochLoss/float(epochDataSize), testCost/float(testSize))
			#saver.save(sess, 'LJ-netStates/LJ-state', global_step=epoch)


		if plotting :
			N     = testSize
			xx,yy = gen.functionData(testSize, function, a, b, normal=True, sigma=0.15, mu=1.12, training=False)
			xx  = xx.reshape([N,1])
			yy_ = sess.run(prediction, feed_dict={x: xx})
			xxx = np.linspace(a,b,testSize)
			yyy = function(xxx)

			plt.figure(1)
			plt.plot(xx, yy_, 'r.')
			plt.hold('on')
			plt.plot(xxx, yyy, 'b--')
			plt.legend(['NN(r)', 'LJ(r)'])
			plt.grid('on')
			plt.xlabel('r')
			plt.ylabel('V(r)')
			plt.axis([a, b, -0.5, function(a)+0.1])
			plt.show()


trainNetwork(x, plotting=True)








