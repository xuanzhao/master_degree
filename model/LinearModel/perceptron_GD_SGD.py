#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

class Perceptron(object):
	def __init__(self, eta=0.01, n_iter=1, shuffle=True):
		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle
		
	def activeF(self, X):
		yHat = X.dot(self.w_)
		return np.where(yHat >= 0.0, 1, -1)

	def predict(self, X):
		yHat = X.dot(self.w_)
		return np.where(yHat >= 0.0, 1, -1)

	def _shuffle(self, X, y):
		r = np.random.permutation(X.shape[0])
		return X[r,:], y[r]

	def fit(self, X, y):
		self.w_ = np.ones((X.shape[1],1))
		self.J_ = []

		for n in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
			print '------------------------------'
			print 'This is number ', n 
			#J = []
			for i in range(X.shape[0]):
				print 'each X[i] is', X[i,:], 'each y[i] is', y[i]
				cost = float(y[i] - self.activeF(X[i,:])) # cost is 1*1 matrix
				print 'cost is', cost
				
				self.J_.append(cost)
				#delta_w = y[i] - self.activeF(X[i,:])  # 1*1
				# below is get from formular, need standard data as input
				self.w_[0] += float(self.eta * X[i,0])   
				self.w_[1] += float(self.eta * X[i,1])
				print 'each w is', self.w_
			print '------------------------------'
			#self.J_.append(J)
		
		print 'type of ppn.J_',type(self.J_)
		print 'value of ppn.J_', self.J_
		print 'shape of ppn.J_', np.shape(self.J_)
		return self

def show_Perceptron_learnCost(X, y):
	ppn = Perceptron().fit(X, y)
	plt.plot(range(1, 1+len(ppn.J_)), ppn.J_, marker='o')
	plt.xlabel('epochs')
	plt.ylabel('# of iterations')
	plt.show()

def loadDataSet_noBias():
	df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data', header=None)

	dataSet = np.mat(df.iloc[:100,:])

	X = dataSet[:, [0, 2]]    
	X_std = (X - np.mean(X,axis=0))/ np.power(np.var(X,axis=0),.5)
	y = dataSet[:, 4]

	y = np.where(y=='Iris-setosa', 1, -1)
	return X_std, y

def visualDataSet(X, y):

	plt.scatter(X[y==1][0], X[y==1][2], color='red', 
	                marker='o', label='setosa')
	plt.scatter(X[y==-1][0], X[y==-1][2], color='blue', 
	                marker='x', label='versicolor')
	plt.legend()

def plot_decision_regions(X, y, classifier, res=0.02):
	'''
	The X is no bias data.
	'''
	from matplotlib.colors import ListedColormap

	markers = ('s', 'o', 'x', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	colormap = ListedColormap(colors[:len(np.unique(y))])

	# xx1 = (243,312), xx2 = (243,312)
	# x = 0 - 312 , y = 0 - 243
	# xx1是横坐标，xx2是纵坐标
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), 
	                    np.arange(x2_min, x2_max, res))
	# 将横坐标展开得到(75816长度)的向量，同样的纵坐标。
	# 转置前矩阵为(2,75816)，第一层是横坐标，第二层是纵坐标
	# 转置后矩阵为(75816,2)，意思为75816个数据点
	# Z为75816个数据点的预测值{-1,1}，矩阵为(75816,1)
	# Z得到对整个plot范围内对所有可能存在的点进行预测
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	# 将Z变为矩阵(243,312),这样就正好对应到了plot中点的坐标
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=.4, cmap=colormap)
	# 画出等高线上Z的值，这部分多余的
	C = plt.contour(xx1, xx2, Z, colors='black', linewidth=0.5)
	plt.clabel(C, inline=1, fontsize=10)
	# 得到分类边界
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)

	# 画出plot中的数据点
	for i, cl in enumerate(np.unique(y)):
		# where得到的是输入矩阵的满足条件的每个维度的位置索引。
		idx = np.where(y==cl)[0]
		plt.scatter(X[idx, 0], X[idx, 1], marker=markers[i],
					alpha=.8, cmap=colormap(i), 
	                label=np.where(cl==1, 'setosa', 'versicolor'))
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc='upper right')

	plt.show()



class GD(object):
	def __init__(self, eta=0.0001, n_iter=10, shuffle=True):
		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle

	def _shuffle(self, X, y):
		r = np.random.permutation(X.shape[0])
		return X[r, :], y[r]

	def activeF(self, X):
		yHat = X.dot(self.w_)   # 100*3 * 3*1
		return yHat

	def predict(self, X):
		return np.where(self.activeF(X) >= 0., 1, -1)

	def fit(self, X, y):
		self.w_ = np.ones((X.shape[1],1))
		self.J_ = []

		for n in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)
			error = y - self.activeF(X)   # 100*1
			 
			self.w_ = self.w_ + self.eta * X.T.dot(error)  # 3*100 * 100 *1

			J = float((error.T.dot(error)/2.0))   #The np.dot for python is matrix multiply
			self.J_.append(J)
			print '----------------------------'
			print 'This is number ', n
			print 'w is ', self.w_
			print 'J is ', J
			print '----------------------------'
		return self

def loadDataSet_withBias():
	df = pd.read_csv('https://archive.ics.uci.edu/ml/'
               'machine-learning-databases/iris/iris.data', header=None)

	dataSet = np.mat(df.iloc[:100,:])
	#np.random.shuffle(dataSet) # the shuffle can't give me the right result
	x0 = np.ones((dataSet.shape[0],1))

	X = dataSet[:, [0, 2]]
	# If X is not standarized, then when eta is trend to large, can't converge.
	X = (X - np.mean(X,axis=0))/ np.power(np.var(X,axis=0),.5)
	X_std = np.concatenate((x0, X),axis=1)  
	y = dataSet[:, 4]
	y = np.where(y=='Iris-setosa', 1, -1)

	return X_std, y


def show_GD_learnCost(X, y):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
	
	gd1 = GD(eta=0.01, n_iter=10).fit(X, y)
	#ax[0].plot(range(1, 1+len(gd1.J_)), np.log10(gd1.J_), marker='o')
	ax[0].plot(range(1, 1+len(gd1.J_)), gd1.J_, marker='o')
	ax[0].set_xlabel('epoches')
	ax[0].set_ylabel('$sum-squared-error$')
	ax[0].set_title('learning rate: 0.01')

	gd2 = GD(eta=0.001, n_iter=10).fit(X,y)
	ax[1].plot(range(1, 1+len(gd2.J_)), gd2.J_, marker='o')
	ax[1].set_xlabel('epoches')
	ax[1].set_ylabel('sum-squared-error')
	ax[1].set_title('learning rate: 0.001')

	plt.show()


class SGD(object):

	def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.shuffle = shuffle
		if self.shuffle:
			np.random.seed(random_state)

	def activeF(self, X):
		yHat = X.dot(self.w_)
		return yHat

	def _shuffle(self, X, y):
		r = np.random.permutation(X.shape[0])
		return X[r,:], y[r]

	def predict(self, X):
		return np.where(self.activeF(X) >= 0.0, 1, -1)

	def fit(self, X, y):
		self.w_ = np.ones( (X.shape[1],1))
		self.J_ = []

		for n in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y)

			J = 0
			for i in range(X.shape[0]):
				error = y[i] - self.activeF(X[i])   # 1*1  - 1*1
			 
				self.w_ = self.w_ + self.eta * X[i].T.dot(error)  # 3*1 * 1*1

				J += float((error.T.dot(error)/2.0))   #The np.dot for python is matrix multiply

			self.J_.append(J/len(y))

		return self


def show_SGD_learnCost(X, y):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
	
	sgd1 = SGD(eta=0.01, n_iter=10).fit(X, y)
	#ax[0].plot(range(1, 1+len(gd1.J_)), np.log10(gd1.J_), marker='o')
	ax[0].plot(range(1, 1+len(sgd1.J_)), sgd1.J_, marker='o')
	ax[0].set_xlabel('epoches')
	ax[0].set_ylabel('$sum-squared-error$')
	ax[0].set_title('learning rate: 0.01')

	sgd2 = SGD(eta=0.001, n_iter=10).fit(X,y)
	ax[1].plot(range(1, 1+len(sgd2.J_)), sgd2.J_, marker='o')
	ax[1].set_xlabel('epoches')
	ax[1].set_ylabel('sum-squared-error')
	ax[1].set_title('learning rate: 0.001')

	plt.show()

















