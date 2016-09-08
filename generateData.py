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


if __name__ == '__main__' :
	n 		= 6
	alpha 	= 0.5

	x,y = uniformGreaterThan(n, alpha)
	print x
	print y
