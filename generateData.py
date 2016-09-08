import numpy as np

def uniformGreaterThan(numberOfSamples, k, training=True) :
	if not training :
		oldState = np.random.get_state()
		np.random.seed(0)

	x = np.random.uniform(0,1,numberOfSamples)
	y = (x > k)

	if not training :
		np.random.set_state(oldState)
	return x, y.astype(float)


if __name__ == '__main__' :
	x,y = uniformGreaterThan(6,0.5)
	print x
	print y