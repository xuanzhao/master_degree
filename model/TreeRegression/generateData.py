import numpy as np
import matplotlib.pyplot as plt

def generateCircle_byPolar(N_i = 50, N_ii = 100):
	'''
	This version use polar coordinates.
	'''
	
	theta_i = np.arange(0., 2 * np.pi, 2 * np.pi / N_i)
	radi = 10 * np.random.rand(N_i)
	
	theta_ii = np.arange(0., 2 * np.pi, 2 * np.pi / N_ii)
	radii = 30 * np.random.rand(N_ii) +10

	ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)
	ax.scatter(theta_i,radi,c='b')
	ax.scatter(theta_ii,radii,c='r')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.show()

def generateCircle_byCartesian(r=1.0,c='b'):
	'''
	THis version use Cartesian coordinates
	'''
	theta = np.arange(0, np.pi*2.0, 0.01)
	theta = theta.reshape((theta.size, 1))
	x = r * np.cos(theta).T + np.random.randn(theta.size) *0.55
	y = r * np.sin(theta).T + np.random.randn(theta.size) *0.55

	plt.scatter(x,y,color=c)
	plt.axis('equal')
	plt.show()
	return x, y

def generate_Circle(n_samples=300,shuffle=False,factor=.4, noise=.2):
	from sklearn import datasets
	X, y = datasets.make_circles(n_samples,shuffle,factor, noise)
	plt.scatter(X[y==0,0],X[y==0,1], c='b')
	plt.scatter(X[y==1,0],X[y==1,1], c='r')
