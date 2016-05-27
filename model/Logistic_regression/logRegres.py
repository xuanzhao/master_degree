import numpy as np
import matplotlib.pylab as plt

def loadData(filename):
	data = np.loadtxt(filename)
	X0 = np.ones((len(data),1))
	X, y = np.split(data, [2], axis=1)
	return np.c_[X0,X], y

def sigmoid(Z):
	return 1.0 / (1+np.exp(-Z))


class logRegression(object):

	def fit_gradDesc(self, X, y, alpha=0.001, maxCycle=1000):
		X = np.mat(X); y = np.mat(y)
		m,n = X.shape
		self.weights = np.random.random((n,1)) #(n,1)

		for n in range(maxCycle):
			h = sigmoid(X*self.weights)   # (m,n) * (n,1) = (m,1)
			error = y - h  # (m,1)
			self.weights = self.weights + alpha * X.T * error   # (n,m) * (m,1)= (n,1)

		return self

	def fit_stocGradDesc(self, X, y, alpha=0.001, maxCycle=100):
		X = np.mat(X); y = np.mat(y)
		m,n = X.shape
		weights = np.random.random((n,1))

		for n in range(maxCycle):
			np.random.seed(n)
			np.random.permutation(np.c_[X,y])   # could avoid circle volatility
			for i in range(m):
				alpha = 4 / (1.0+ n+i) +0.01  # could accelerate converge
				h = sigmoid(X[i] * weights)  # (1,n) * (n,1) = (1,1)
				error = y[i] - h
				weights = weights + alpha * (error * X[i]).T     # (1,1) * (1,n) = (1,n).T

		self.weights = weights

		return self


	def plotBestFit(self, X, y):
		m, n = X.shape
		weights = self.weights.getA()

		ind_T = np.where(y==1)[0]
		X_T = np.take(X,ind_T, axis=0)
		ind_F = np.where(y==0)[0]
		X_F = np.take(X,ind_F, axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(X_T[:,1], X_T[:,2], s=30, c='red', marker='s', label='True')
		ax.scatter(X_F[:,1], X_F[:,2], s=30, c='green', label='False')
		x1 = np.linspace(-3.0, 3.0, 100)
		y1 = (-weights[0] - weights[1]*x1) / weights[2]
		ax.plot(x1, y1, label='classify boundary')
		ax.legend(loc='best')
		ax.set_xlabel('X feature 1')
		ax.set_ylabel('X feature 2')
		fig.show()

	def predict(self, inX):
		prob = sigmoid(inX * self.weights)
		return 1.0 if prob > 0.5 else 0.0


def visualize_Weight_GD(filename='testSet.txt', maxCycle=50):
	X, y = loadData(filename)
	m, n = X.shape
	alpha=0.01
	weights = np.random.random((n,1))
	X = np.mat(X); y = np.mat(y)
	weights_history = np.mat(np.zeros((maxCycle*m,n)))

	for j in range(maxCycle):
		np.random.seed(j)
		np.random.permutation(np.c_[X,y]) 
		for i in range(m):
			alpha = 4 / (1.0+ j+i) +0.01  # could accelerate converge
			h = sigmoid(X[i] * weights)  # (1,n) * (n,1) = (1,1)
			error = y[i] - h
			weights = weights + alpha * (error * X[i]).T     # (1,1) * (1,n) = (1,n).T
			weights_history[j*m + i, :] = weights.T
	
	fig, axes = plt.subplots(3,1)
	axes[0].plot(weights_history[:,0])
	axes[0].set_ylabel('X0')
	axes[1].plot(weights_history[:,1])
	axes[1].set_ylabel('X1')
	axes[2].plot(weights_history[:,2])
	axes[2].set_ylabel('X2')
	plt.xlabel('iteration #')
	fig.show()


def visualize_GD():
	'''
	Created on Oct 28, 2010

	@author: Peter
	'''
	import matplotlib
	import numpy as np
	import matplotlib.cm as cm
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt

	leafNode = dict(boxstyle="round4", fc="0.8")
	arrow_args = dict(arrowstyle="<-")

	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	delta = 0.025
	x = np.arange(-2.0, 2.0, delta)
	y = np.arange(-2.0, 2.0, delta)
	X, Y = np.meshgrid(x, y)
	Z1 = -((X-1)**2)
	Z2 = -(Y**2)
	#Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
	#Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
	# difference of Gaussians
	Z = 1.0 * (Z2 + Z1)+5.0

	# Create a simple contour plot with labels using default colors.  The
	# inline argument to clabel will control whether the labels are draw
	# over the line segments of the contour, removing the lines beneath
	# the label
	plt.figure()
	CS = plt.contour(X, Y, Z)
	plt.annotate('', xy=(0.05, 0.05),  xycoords='axes fraction',
	             xytext=(0.2,0.2), textcoords='axes fraction',
	             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
	plt.text(-1.9, -1.8, 'P0')
	plt.annotate('', xy=(0.2,0.2),  xycoords='axes fraction',
	             xytext=(0.35,0.3), textcoords='axes fraction',
	             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
	plt.text(-1.35, -1.23, 'P1')
	plt.annotate('', xy=(0.35,0.3),  xycoords='axes fraction',
	             xytext=(0.45,0.35), textcoords='axes fraction',
	             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
	plt.text(-0.7, -0.8, 'P2')
	plt.text(-0.3, -0.6, 'P3')
	plt.clabel(CS, inline=True, fontsize=10)
	plt.title('Gradient Ascent')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

