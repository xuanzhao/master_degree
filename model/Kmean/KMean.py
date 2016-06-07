import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(file='testSet.txt'):
	data = np.loadtxt(file, delimiter='\t')

	return data  #np.array

def distEclud(vecA, vecB):
	return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCenter(X, k):
	X = np.array(X)
	m,n = X.shape
	centroids = np.zeros((k,n))

	X[:,0].min(), X[:,0].max()
	X[:,1].min(), X[:,1].max()
	n_range_max = np.max(X, axis=0).reshape(1,-1)  # (1,n)
	n_range_min = np.min(X, axis=0).reshape(1,-1)

	centroids = np.random.rand(k, n) * n_range_max + \
				np.random.rand(k, n) * n_range_min  

	return centroids  # (k,n)


class KMean(object):
	"""docstring for KMean"""
	def __init__(self):
		super(KMean, self).__init__()


	def fit(self, X, k, dist=distEclud, createCent=randCenter):
		X = np.array(X)
		m,n = X.shape
		clusterAssment = np.zeros((m,2)) # col0=clusterInd, col1=distance
		centroids = createCent(X, k)   # (k,n)
		clusterChanged = True

		while clusterChanged:
			clusterChanged = False
			print 'centroids is :\n', centroids 

			for i in range(m):
				minDist = np.inf; minInd = -1
				for j in range(k):
					dist_ij = dist(X[i], centroids[j])
					if dist_ij < minDist:
						minDist = dist_ij; minInd = j
				if clusterAssment[i,0] != minInd:
					clusterChanged = True
					clusterAssment[i] = minInd, minDist**2
		
			for cent in range(k):
				centInd = (clusterAssment[:,0] == cent)
				centroids[cent] = np.mean(X[centInd],axis=0)

		self.centroids = centroids
		self.clusterAssment = clusterAssment
		self.X = X

		#return centroids, clusterAssment

	def plotData(self):
		X = self.X
		centroids = self.centroids
		clusterAssment = self.clusterAssment


		plt.scatter(X[:,0], X[:,1], c=clusterAssment[:,0])
		plt.scatter(centroids[:,0], centroids[:,1], s=80)


class Bi_KMeans(object):


	def twoMeans(self, X, k=2, dist=distEclud, createCent=randCenter):
		X = np.array(X)
		m,n = X.shape
		clusterAssment = np.zeros((m,2)) # col0=clusterInd, col1=distance
		centroids = createCent(X, k)   # (k,n)
		clusterChanged = True

		while clusterChanged:
			clusterChanged = False
			print 'centroids is :\n', centroids 

			for i in range(m):
				minDist = np.inf; minInd = -1
				for j in range(k):
					dist_ij = dist(X[i], centroids[j])
					if dist_ij < minDist:
						minDist = dist_ij; minInd = j
				if clusterAssment[i,0] != minInd:
					clusterChanged = True
					clusterAssment[i] = minInd, minDist**2
		
			for cent in range(k):
				centInd = (clusterAssment[:,0] == cent)
				centroids[cent] = np.mean(X[centInd],axis=0)

		return centroids, clusterAssment


	def fit(self, X, k, dist=distEclud):
		X = np.array(X)
		m,n = X.shape
		clusterAssment = np.zeros((m,2))
		centroid_0 = np.mean(X, axis=0)  # array(n)
		centList = [centroid_0]

		for i in range(m):
			clusterAssment[i,1] = dist(centroid_0, X[i]) ** 2

		while len(centList) < k:
			lowestSSE = np.inf
			
			for i in range(len(centList)):
				ind = clusterAssment[:,0] == i
				CurrentCluster = X[ind]
				centroidMat, splitClustAss = self.twoMeans(CurrentCluster)

				sseSplited = np.sum(splitClustAss[:,1])
				sseNotSplit = np.sum(X[np.logical_not(ind)][:,1])
				print 'splited data and not splited data sse error is', sseSplited, sseNotSplit

				if (sseSplited+sseNotSplit) < lowestSSE:
					bestCentToSplit = i
					bestNewTwoCenters = centroidMat
					bestNewTwoClustAss = splitClustAss
					lowestSSE = sseSplited + sseNotSplit

			bestSplitInd = bestNewTwoClustAss[:,0] == 1
			bestNewTwoClustAss[bestSplitInd] = len(centList)
			bestNewTwoClustAss[np.logical_not(bestSplitInd)] = bestCentToSplit
			print 'the best center to split is ', bestCentToSplit
			print 'the length of bestNewTwoClustAss is ', len(bestNewTwoClustAss)

			centList[bestCentToSplit] = bestNewTwoCenters[0,:]
			centList.append(bestNewTwoCenters[1,:])
			clusterAssment[clusterAssment[:,0]==bestCentToSplit,:] = \
							bestNewTwoClustAss

		self.centroids = np.array(centList)
		self.clusterAssment = clusterAssment
		self.X = X

		return np.array(centList), clusterAssment


	def plotData(self):
		X = self.X
		centroids = self.centroids
		clusterAssment = self.clusterAssment


		plt.scatter(X[:,0], X[:,1], c=clusterAssment[:,0])
		plt.scatter(centroids[:,0], centroids[:,1], s=80)










































		
