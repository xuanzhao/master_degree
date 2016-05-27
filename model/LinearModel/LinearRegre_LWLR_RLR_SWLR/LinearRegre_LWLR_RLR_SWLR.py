#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearReg(object):

	def fit(self, X, y):
		'''
		The formular is : w = (X.T*X).inv * X.T * y
		'''
		xTx = X.T * X  # (2*200) * (200*2) = 2*2
		if np.linalg.det(xTx) == 0.0:
			print 'This matrix is singular , cannot do inverse'
			return None
		self.w_ = xTx.I * X.T * y  # (2*2) * (2*200) * (200*1) = 2*1

		return self 


class localWeigt_LR(object):

 	def fit_P(self, testP, X, y, k):
 		'''
 		The formular is : wHat = (X.T*W*X).I * X.T * W * y
 		The kernel function is : w(i,j) = exp(abs(xi-xj) / -2* k**2) 
 		'''
 		X = np.mat(X); y = np.mat(y)
 		m = np.shape(X)[0]
 		weights = np.mat(np.eye( (m) ))

 		# get the kernel matrix
 		for j in np.arange(m):
 			
 			diffArray = testP - X[j,:]   # (1,n)
 			diffVal = np.sum(np.fabs(diffArray))
 			#diffVal = np.sqrt(diffArray * diffArray.T)
 			weights[j,j] = np.exp(diffVal / (-2.0 * k**2))
 		# 这里可以再优化，不用循环

 		xTWx = X.T * (weights * X)   # (n,m) * (m,m) * (m,n) = (n,n)
 		if np.linalg.det(xTWx) == 0.0 :
 			print 'This matrrix is singular, cannot do inverse'
 			return None
 		ws = xTWx.I * (X.T * (weights * y))  # (n,n) * (n,m) * (m,m) * (m,1) = (n,1)
 		return testP * ws  # (1,n) * (n,1) = (1,1)

 	def fit(self, testX, X, y, k=1.0):
 		'''
 		This function call fit_P to get a regression boundary.
 		k = 1.0 means no weights.
 		It is good for k=0.1
 		'''
 		self.k_ = k
 		m = np.shape(testX)[0]
 		self.yHat = np.zeros(m)
 		for i in range(m):
 			self.yHat[i] = self.fit_P(testX[i], X, y, k)
 		return self

		
class ridge_LR(object):
	
	def fit(self, X, y, lam=0.2):
		X = np.mat(X); y = np.mat(y)
		xTx = X.T*X
		denom = xTx + np.eye(X.shape[1]) * lam   #(n,m)*(m,n)+(n,n)=(n,n)
		if np.linalg.det(denom) == 0.0:
			print 'This matrix is singular, cannot do inverse'
			return None
		ws = denom.I * (X.T * y)  # (n,n) * (n*m) * (m,1) = (n*1)
		return ws

	def ridgeTest(self, X, y):
		# standize the data
		X = np.mat(X); y = np.mat(y)
		yMean = np.mean(y,0)
		y = y - yMean
		xMeans = np.mean(X, 0)
		xVars = np.var(X, 0)
		X = (X - xMeans) / xVars

		numTestPts = 30
		wMat = np.zeros((numTestPts, X.shape[1]))  # (30,n)
		for i in np.arange(numTestPts):
			ws = self.fit(X, y, np.exp(i-10))	# (n*1)
			wMat[i,:] = ws.T
		return wMat

class stageWise_LR(object):

	def fit(self, X, y, eps=0.01, numIt=100):
		# standize the data
		X = np.mat(X); y = np.mat(y)
		yMean = np.mean(y,0)
		y = y - yMean
		xMeans = np.mean(X,0)
		xVars = np.var(X,0)
		X = (X - xMeans) / xVars

		m,n = X.shape
		returnMat = np.zeros((numIt,n))
		ws = np.zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()

		for i in np.arange(numIt):
			print 'current weights is:', ws.T
			lowestError = np.inf
			for j in range(n):
				
				for sign in [-1,1]:
					wsTest = ws.copy()
					wsTest[j] += eps*sign
					yTest = X * wsTest
					rssE = self.rssError(y.A, yTest.A)
					if rssE < lowestError:
						lowestError = rssE
						wsMax = wsTest
			
			ws = wsMax.copy()
			returnMat[i,:] = ws.T

		return returnMat
		#plt.plot(returnMat)可以看每个特征的系数变化趋势

	def rssError(self, y, yHat):
		return ((y - yHat)**2).sum()



def generateData():
	x = np.linspace(0,1,num=50)
	y = 3.0 + 1.7*x + 0.1*np.sin(30*x) + 0.06 * np.random.randn(len(x))
	x = x.reshape(len(x),1)
	y = y.reshape(len(y),1)
	x0 = np.ones(x.shape)
	X = np.concatenate((x0,x),axis=1)

	# 返回的数据类型若是ndarray，则不能进行矩阵运算
	# 返回前需要转换成mat类型
	X = np.asmatrix(X)
	y = np.asmatrix(y)
	return X, y

def loadDataSet(fileName):
	df = pd.read_csv(fileName, sep='\t', header=None)
	dataSet = np.mat(df.iloc[:,:])
	X = dataSet[:, :-1]
	y = dataSet[:, -1]

	return X, y


def plot_LR(X, y, classifier):
	# 因为得到的X和y的数据类型都是都是matrix，要先处理成ndarray类型，
	# np.A 等同与 X = np.asarray(X)
	plt.scatter(X[:,1].flatten().A[0], y[:].flatten().A[0])
	#plt.scatter(X[:,1], y[:])
	# 为了不使得绘制的回归线出现问题，需要先将数据点按照升序排列
	X_Copy = X.copy()
	X_Copy.sort(0)  # 以行为单位进行对比排序
	yHat = X_Copy * classifier.w_    # (200,2) * (2,1) = (200,1)
	plt.plot(X_Copy[:,1], yHat)
	plt.show()

def plot_LWLR(X, y, classifier):
	XMat = np.mat(X)
	sortInd = XMat[:,1].argsort(0)
	# 这里Xmat[sortInd]会变成一个三维的矩阵
	XSorted = XMat[sortInd].reshape(X.shape)

	plt.scatter(X[:,1], y[:], s=2, c='red')
	plt.plot(XSorted[:,1], classifier.yHat[sortInd])
	plt.show()






















