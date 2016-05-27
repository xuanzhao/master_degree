import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(data):
	"""
	return a set of the vocabulary.
	"""
	vocabSet = set([])
	for document in data:
		vocabSet = vocabSet.union(set(document))
	vocabList = list(vocabSet)
	return vocabList


def setOfwords2Vec(vocabList, document):
	vocabVec = np.array(vocabList)
	docVec = np.array(document)

	ind = []
	wordVec = np.zeros_like(vocabVec, dtype=int)
	for word in docVec:
		if word in vocabVec:
			wordVec[vocabList.index(word)] = 1
	return wordVec

def bagOfwords2Vec(vocabList, document):
	vocabVec = np.array(vocabList)
	docVec = np.array(document)

	ind = []
	wordVec = np.zeros_like(vocabVec, dtype=int)
	for word in docVec:
		if word in vocabVec:
			wordVec[vocabList.index(word)] += 1
	return wordVec

def createTrainMat(X):
	vocabList = createVocabList(X)
	trainMat = np.zeros((len(X),len(vocabList)))
	for i,doc in enumerate(X):
		trainMat[i] = setOfwords2Vec(vocabList, doc)
	return trainMat

def testNB():
	X, y = loadDataSet()
	vocabList = createVocabList(X)
	trainMat = createTrainMat(X)
	p0Vec, p1Vec, p1 = bayes().train_NB(trainMat, y)
	
	testEntry = ['love', 'my', 'dalmation'] 
	thisDoc = np.array(setOfwords2Vec(vocabList, testEntry))
	print testEntry,'classified as: ', bayes().predict(thisDoc, p0Vec, p1Vec, p1)
	
	testEntry = ['stupid','garbage'] 
	thisDoc = np.array(setOfwords2Vec(vocabList, testEntry))
	print testEntry,'classified as: ', bayes().predict(thisDoc, p0Vec, p1Vec, p1)


class bayes(object):

	def train_NB(self, X, y):
		"""
		y is {0,1}, y=1 is spam email
		"""
		m, n = X.shape
		p1 = sum(y) / float(m)

		# below calculate the condition probability
		p0Num = np.ones((1,n))   # avoid for the zero probability
		p1Num = np.ones((1,n))
		p0Total = 2.0
		p1Total = 2.0

		for i in range(m):
			if y[i] == 1:
				p1Num += X[i]
				p1Total += sum(X[i])
			else:
				p0Num += X[i]
				p0Total += sum(X[i])

		p1Vect = np.log(p1Num / p1Total)   # avoid multiply floor to zero
		p0Vect = np.log(p0Num / p0Total)

		return p0Vect, p1Vect, p1

	def predict(self, x, p0Vect, p1Vect, p1):
		"""
		"""
		p1 = np.sum(x * p1Vect, axis=1) + np.log(p1)
		p0 = np.sum(x * p0Vect, axis=1) + np.log(1.0-p1)

		return 1 if p1>p0 else 0

































