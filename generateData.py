import numpy as np

def uniformGreaterThan(numberOfSamples, k, training=True) :
	if not training :
		oldState = np.random.get_state()
		np.random.seed(0)

	x = np.random.uniform(0,1,numberOfSamples)
	x = x.reshape([numberOfSamples, 1])
	greater = (x > k)
	less    = (x < k)
	y = np.concatenate((greater, less), axis=1)

	if not training :
		np.random.set_state(oldState)

	return x, y.astype(float)


def uniformOrdering(numberOfSamples, training=True) :
	if not training :
		oldState = np.random.get_state()
		np.random.seed(0)

	x1 = np.random.uniform(0,1,numberOfSamples)
	x2 = np.random.uniform(0,1,numberOfSamples)
	x1 = x1.reshape([numberOfSamples, 1])
	x2 = x2.reshape([numberOfSamples, 1])
	greater = (x1 > x2)
	less    = (x1 < x2)
	x = np.concatenate((x1, x2), axis=1)
	y = np.concatenate((greater, less), axis=1)

	if not training :
		np.random.set_state(oldState)

	return x, y.astype(float)

def functionData(numberOfSamples, 
				 function, 
				 a=0, 
				 b=1, 
				 training=True, 
				 normal=False,
				 mu=0.0,
				 sigma=0.1,
				 newEpoch=False) :

	if newEpoch :
		np.random.seed(1)

	if not training :
		oldState = np.random.get_state()
		np.random.seed(0)

	if normal :
		x = np.random.normal(mu, sigma, numberOfSamples)
	else :
		x = np.random.uniform(a,b,numberOfSamples)

	x = x.reshape([numberOfSamples, 1])
	y = function(x);

	if not training :
		np.random.set_state(oldState)

	return x, y


def functionDataLinspace(	numberOfSamples, 
				 			function, 
							a=0, 
				 			b=1) :

	x = np.linspace(a, b, numberOfSamples)
	x = x.reshape([numberOfSamples, 1])
	y = function(x);
	return x, y
