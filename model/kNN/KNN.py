import numpy as np
import operator

def createDataSet():
	group  = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def loadData(filename):
	X = np.loadtxt(filename, delimiter='\t', usecols=[0,1,2])
	y = np.loadtxt(filename, dtype='int',delimiter='\t', usecols=[3] )
	return X, y 

def autoNorm(data):
	minVals = data.min(axis=0)
	maxVals = data.max(axis=0)
	rangeVals = maxVals - minVals
	m,n = data.shape

	normData = (data - minVals[np.newaxis,:]) / rangeVals[np.newaxis,:]

	return normData, rangeVals, minVals

class KNN(object):

	def fit(self, inX, data, labels, k):
		#first, calculate the distance between inX with data
		m, n = data.shape
		diffMat = np.tile(inX, (m,1)) - data
		sqDiffMat = diffMat**2
		sqDistances = sqDiffMat.sum(axis=1)

		#second, get the k number nearest data
		sortedDistIndices = sqDistances.argsort()
		classCount = {}
		for i in range(k):
			voteLabel = labels[sortedDistIndices[i]]
			classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

		#Third, sort the label frequent
		sortedClassCount = sorted(classCount.iteritems(),
								key=operator.itemgetter(1),
								reverse=True)
		return sortedClassCount[0][0]

	def test(self, filename):
		X, y = loadData(filename)
		data = np.random.permutation(np.c_[X, y])
		X = data[:,:3]; y = data[:,-1]
		# np.split(data, [3], axis=1)
		ratio = 0.1
		norm_X, rangeVals, minVals = autoNorm(X)
		m, n = norm_X.shape
		numTest_X = int(m*ratio)

		errorCount = 0.0
		for i in range(numTest_X):
			result = self.fit(norm_X[i,:], norm_X[numTest_X:m, :], y[numTest_X:m], 3)
			print "the classifier result is: %d, the real answer is: %d" \
					% (result, y[i])
			if (result != y[i]): errorCount += 1.0
		print "The total error rate is: %f" % (errorCount/ float(numTest_X)) 

	def predict(self):
		resultList = ['not at all', 'in small dose', 'in large dose']
		percentTats = float(raw_input(\
						"percentage of time spent playing video games?"))
		ffMiles = float(raw_input("frequent flier miles earned per year?"))
		iceCream = float(raw_input("liters of iceCream consumed per year?"))

		X, y = loadData('datingTestSet2.txt')
		norm_X, rangeVals, minVals = autoNorm(X)
		inArr = np.array([ffMiles, percentTats, iceCream])
		norm_inArr = (inArr - minVals) / rangeVals

		predict_result = self.fit(norm_inArr, norm_X, y, 3)
		print 'You will probably like this person: ',\
				resultList[predict_result - 1]

















