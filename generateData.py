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

if __name__ == '__main__' :
	n 		= 6

	x,y = uniformOrdering(n)
	print x
	print y

	print x.shape
	print y.shape