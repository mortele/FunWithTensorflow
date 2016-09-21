import os
import sys
import shutil 
import tensorflow 			as 		tf
import numpy 				as 		np
import matplotlib.pyplot 	as 		plt
import generateData 		as 		gen
import datetime 			as 		time

# Constants.
nInputs 	= 1			# Number of (float) intputs.
nNodes 	    = 5			# Number of nodes in each hidden layer.
nLayers  	= 5			# Number of hidden layers. 
nOutputs	= 1			# Number of (float) outputs.
a 			= 0.87		# Lower cut-off for input values to the NN.
b 			= 1.6		# Upper cut-off for input values to the NN.

# Lennard-Jones functional form (normalized, epsilon=sigma=1).
function = lambda z: 1/z**6 * (1/z**6 - 1)

# Chech for input on the command line.
loadFlag 		= False
loadFileName 	= ''
saveFlag 		= False
saveDirName		= ''
saveMetaName	= ''
trainingDir		= 'TrainingData'
plotFlag		= False
forcePlotFlag	= False
epochInput		= -1

# Checks whether or not the input string is an integer.
def isInt(input):
    try: 
        int(input)
        return True
    except ValueError:
        return False

def findLastTrainingDir(dirList) :
	# First, remove any directory which is not named as a date 
	# according to 'day.month-hour.minute.second'.
	N = len(dirList)
	j = 0
	for i in xrange(N) :
		if dirList[j].startswith('.') :
			dirList.pop(j);
		else :
			j = j + 1
	N = len(dirList)

	# Extract the month, day, hour, minute, second for each 
	# directory.
	lst  = [[0 for i in xrange(N)] for i in xrange(5)]
	for i in xrange(N) :
		lst[0][i] = int(dirList[i].split('.')[1].split('-')[0])
		lst[1][i] = int(dirList[i].split('.')[0])
		lst[2][i] = int(dirList[i].split('-')[1].split('.')[0])
		lst[3][i] = int(dirList[i].split('-')[1].split('.')[1])
		lst[4][i] = int(dirList[i].split('-')[1].split('.')[2])

	# Find the index of the directory list corresponding to the 
	# last date and time.
	index = -1
	for i in xrange(5) :
		m = max(lst[i][:])
		for j in xrange(N) :
			if not lst[i][j] == m :
				for k in xrange(i,5) :
					lst[k][j] = 0
			else :
				index = j
	return dirList[index]

def findLastCheckpoint(lastTrainingDir) :
	fileList = os.listdir(lastTrainingDir)
	N = len(fileList)
	j = 0
	for i in xrange(N) :
		if not fileList[j].startswith('ckpt') or \
			fileList[j].endswith('.meta') :
			fileList.pop(j)
		else :
			j = j + 1
	N = len(fileList)
	ckptNumbers = [int(fileList[i].split('-')[1]) for i in xrange(N)]
	maxCkpt = max(ckptNumbers)
	return lastTrainingDir + '/' + 'ckpt-' + str(maxCkpt)

if len(sys.argv) > 1 :
	i = 1
	while i < len(sys.argv) :
		if sys.argv[i] == '--load' :
			i 	 	 = i + 1
			loadFlag = True
			# Check if a filename is given after --load.
			if len(sys.argv) > i and (not sys.argv[i].startswith('--')) \
					and (not isInt(sys.argv[i])):
					# If the next command line argument exists and does not 
					# start with '--', then we assume it is the file name.
					loadFileName 	= sys.argv[i] 
			else :
				# If not, we default to the last checkpoint saved by the 
				# last training run performed.
				dirs 			= os.listdir(trainingDir)
				lastTrainingDir = findLastTrainingDir(dirs)
				lastCkpt		= findLastCheckpoint(trainingDir + '/' + lastTrainingDir)
				loadFileName	= lastCkpt

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

		elif sys.argv[i] == '--plot' :
			i = i + 1
			plotFlag = True

		elif sys.argv[i] == '--fplot' :
			i = i + 1
			plotFlag  	  = True
			forcePlotFlag = True

		elif isInt(sys.argv[i]) :
			i = i + 1
			epochInput = int(sys.argv[i-1]) 

		else :
			i = i + 1

if saveFlag and loadFlag :
	loadDirName = loadFileName.split('ckpt')[0]
	shutil.copy2(loadDirName + 'meta.dat', saveMetaName)

# Tensorflow variables defined for later use.
x 		= tf.placeholder('float', [None, 1], 	name='x')
y 		= tf.placeholder('float', 				name='y')

def nn(inputData, inputs, nodesPerLayer, outputs) :
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
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, layer2['weights']), layer2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, layer3['weights']), layer3['biases'])
	l3 = tf.nn.relu(l3)
	
	l4 = tf.add(tf.matmul(l3, layer4['weights']), layer4['biases'])
	l4 = tf.nn.sigmoid(l4)

	y_ = tf.add(tf.matmul(l4, output['weights']), output['biases'])

	return y_

neuralNetwork = lambda inputData : nn(inputData, nInputs, nLayers, nOutputs)

def trainNetwork(x, plotting=False) :
	y 			= tf.placeholder('float')
	prediction 	= neuralNetwork(x)
	cost 		= tf.nn.l2_loss(tf.sub(prediction, y))
	optimizer 	= tf.train.AdamOptimizer().minimize(cost)
	saver 		= tf.train.Saver(max_to_keep=None)

	global plotFlag
	global forcePlotFlag
	numberOfEpochs = int(1e10)
	if epochInput == -1 and plotFlag == True :
		numberOfEpochs = 0
	elif not epochInput == -1 and plotFlag == True :
		numberOfEpochs = epochInput

	if (numberOfEpochs > 100) and (forcePlotFlag == False) and (plotFlag == True):
		plotFlag = False
		print "Number of epochs > 100, ignoring --plot argument."
		print "Use --fplot instead to force plotting for epochs > 100."
	epochDataSize 	= int(1e7)
	batchSize		= int(1e5)
	testSize		= int(1e4)

	with tf.Session() as sess :
		model = tf.initialize_all_variables()
		sess.run(model)

		if loadFlag :
			print "Loading file: ", loadFileName
			saver.restore(sess, loadFileName)

		print "Running for: ", numberOfEpochs, " epochs"
		if (saveFlag) :
			print "Saving to directory: ", saveDirName
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
				if epoch == 0 and loadFlag == False:
					saveEpochNumber = 0
					with open(saveMetaName, 'w') as outFile :
						outStr = '# epochs: %d (size: %d), batch: %d, test: %d, nodes: %d, layers: %d' % \
								 (numberOfEpochs, epochDataSize, batchSize, testSize, nNodes, nLayers)
						outFile.write(outStr + '\n')
				else :
					with open(saveMetaName, 'a') as outFile :
						outStr = '%g %g' % (epochLoss/float(epochDataSize), testCost/float(testSize))
						outFile.write(outStr + '\n')

				if epoch % 50 == 0 :
					saveFileName = saveDirName + '/' 'ckpt'
					saver.save(sess, saveFileName, global_step=saveEpochNumber)
					saveEpochNumber = saveEpochNumber + 1

		# Return some values to plot.
		np  		= 10000
		xp,yp  		= gen.functionDataLinspace(np,function,a,b)
		yp_ 		= sess.run(prediction, feed_dict={x:xp, y:yp})
		return np, xp[:,0], yp[:,0], yp_[:,0]
		


np,xp,yp,yp_ = trainNetwork(x)

if plotFlag :
	print "Plotting function and approximation error."

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



