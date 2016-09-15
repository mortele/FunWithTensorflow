import os
import sys
import shutil 
import tensorflow 			as 		tf
import numpy 				as 		np
import matplotlib.pyplot 	as 		plt
import generateData 		as 		gen
import datetime 			as 		time

# Chech for input on the command line.
loadFlag 		= False
loadFileName 	= ''
saveFlag 		= False
saveDirName		= ''
saveMetaName	= ''
trainingDir		= 'TrainingData'

if len(sys.argv) > 1 :
	i = 1
	while i < len(sys.argv) :
		if sys.argv[i] == '--load' :
			i = i + 1
			loadFlag 		= True
			loadFileName 	= sys.argv[i]
		elif sys.argv[i] == '--save' :
			i = i + 1
			saveFlag 		= True
			now 			= time.datetime.now().strftime("%d.%m-%H.%M.%S")
			saveDirName 	= trainingDir + '/' + now
			saveMetaName	= saveDirName + '/' + 'meta.dat'

			# If this directory exists
			if os.path.exists(saveDirName) :
				print "Attempted to place data in existing directory, %s. Exiting." % \
						(saveDirName)
				exit(1)
			else :
				os.makedirs(saveDirName)

			# Copy the python source code used to run the training, to preserve
			# the tf graph (which is not saved by tf.nn.Saver.save()).
			shutil.copy2(sys.argv[0], saveDirName + '/')

		else :
			i = i + 1


# Constants.
nInputs 	= 1			# Number of (float) intputs.
nNodes 	    = 10		# Number of nodes in each hidden layer.
nLayers  	= 4			# Number of hidden layers. 
nOutputs	= 1			# Number of (float) outputs.
a 			= 0.87		# Lower cut-off for input values to the NN.
b 			= 1.6		# Upper cut-off for input values to the NN.

# Lennard-Jones functional form (normalized, epsilon=sigma=1).
function = lambda z: 1/z**6 * (1/z**6 - 1)

# Tensorflow variables defined for later use.
x 		= tf.placeholder('float', [None, 1], 	name='x')
y 		= tf.placeholder('float', 				name='y')

## Helper function for setting up the weights and biases for each hidden layer
## of the network. 
#def initialize(nodesPrevious, nodesCurrent) :
#	return {'w': tf.Variable(tf.random_normal([nodesPrevious, nodesCurrent])),
#			'b': tf.Variable(tf.random_normal([nodesCurrent]))}
#
## Definition of the actual network, which does all the heavy lifting.
#def neuralNetwork(inp) :
#	nodes  = [nInputs] + [nNodes for i in xrange(nLayers)] + [nOutputs]
#	layers = [initialize(nodes[i-1], nodes[i]) for i in xrange(1,nLayers+1)]
#
#	# Propagate the input, inp, through the all but the last layer,
#	# using a rectified linear activation function.
#	for i in xrange(nLayers) :
#		inp = tf.matmul(inp, layers[i]['w'])
#		inp = tf.nn.relu(tf.add(inp, layers[i]['b']))
#
#	# For the last hidden layer, we use a sigmoid activation function,
#	inp = tf.add(tf.matmul(inp, layers[-2]['w']), layers[-2]['b'])
#	inp = tf.nn.sigmoid(inp)
#
#	# ...and for the output layer we employ no activation at all.
#	return tf.add(tf.matmul(inp, layers[-1]['w']), layers[-1]['b'])

def nn(inputData, inputs, nodesPerLayer, outputs) :
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

neuralNetwork = lambda inputData : nn(inputData, nInputs, nLayers, nOutputs)

def trainNetwork(x, plotting=False) :
	y 			= tf.placeholder('float')
	prediction 	= neuralNetwork(x)
	cost 		= tf.nn.l2_loss(tf.sub(prediction, y))
	optimizer 	= tf.train.AdamOptimizer().minimize(cost)
	saver 		= tf.train.Saver(max_to_keep=None)
	
	numberOfEpochs 	= 1
	epochDataSize 	= int(1e7)
	batchSize		= int(1e5)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		if loadFlag :
			saver.restore(sess, loadFileName)

		for epoch in xrange(numberOfEpochs) :
			epochLoss = 0
			for i in xrange(epochDataSize/batchSize) :
				xBatch, yBatch = gen.functionDataLinspace(batchSize, function, a, b)
				eo, ec = sess.run([optimizer, cost], feed_dict={x: xBatch, y: yBatch})
				epochLoss += ec

			xTest, yTest = gen.functionDataLinspace(testSize, function, a, b)
			to, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})

			print "Epoch #: %3d  epoch loss/size: %10g  test set loss/size: %10g" % \
					(epoch+1, epochLoss/float(epochDataSize), testCost/float(testSize))

			# If saving is enabled, save the graph variables ('w', 'b') and dump
			# some info about the training so far to SavedModels/<this run>/meta.dat.
			if saveFlag :
				if epoch == 0 :
					saveEpochNumber = 0
					with open(saveMetaName, 'w') as outFile :
						outStr = '# epochs: %d (size: %d), batch: %d, test: %d, nodes: %d, layers: %d' % \
								 (numberOfEpochs, epochDataSize, batchSize, testSize, nNodes, nLayers)
						outFile.write(outStr + '\n')
				else :
					with open(saveMetaName, 'a') as outFile :
						outStr = '%g %g' % (epochLoss/float(epochDataSize), testCost/float(testSize))
						outFile.write(outStr + '\n')

				if epoch % 10 == 0 :
					saveFileName = saveDirName + '/' 'ckpt'
					saver.save(sess, saveFileName, global_step=saveEpochNumber)
					saveEpochNumber = saveEpochNumber + 1

		# Return some values to plot.
		np  		= 10000
		xp,yp  		= gen.functionDataLinspace(np,function,a,b)
		yp_ 		= sess.run(prediction, feed_dict={x:xp, y:yp})
		return np, xp[:,0], yp[:,0], yp_[:,0]
		


np,xp,yp,yp_ = trainNetwork(x)


# Plot the function and the NN approximation.
plt.figure(1)
plt.plot(xp, yp, 'b--')
plt.hold('on')
plt.plot(xp, yp_, 'r-')
plt.legend(['LJ(r)', 'NN(r)'])
plt.xlabel('r')
plt.ylabel('V(r)')
plt.title('Comparison: Neural Net and L-J')

# Plot the log10 of the absolute error.
plt.figure(2)
plt.semilogy(xp, abs(yp-yp_), 'b-')
plt.xlabel('r')
plt.ylabel('|error(r)|')
plt.title('Absolute error in neural net approximation')

# Show the plots.
plt.show()





