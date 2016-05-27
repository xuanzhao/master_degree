def createDataSet():
	'''
	This function is used to create test data for decision tree.
	'''
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]

	labels = ['no surfacing', 'flippers']

	return dataSet, labels


def calcShannonEnt(dataSet):
	'''
	This function is used to calculate shannon's entropy, which is defined as 
	$ H = sum( pr(x_i)* -lg(pr(x_i)) )

	Input : DataSet 

	Output : the Entropy of the input.
	'''
	from math import log

	numEntries = len(dataSet)
	# Calculate the count for each catagory of input data
	labelCount = {}
	for data in dataSet:
		currentLabel = data[-1]
		if currentLabel not in labelCount.keys():
			labelCount[currentLabel] = 1
		else:labelCount[currentLabel] += 1
	
	# Calculate the shannon entropy for above catagories.
	shannonEnt = 0.0
	for label in labelCount:
		prob = float(labelCount[label]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	
	return shannonEnt


def splitDataSet(dataSet, axis, value):
	'''
	This function is used to split dataSet by special axis with special value.

	Input : dataSet, axis(data feature), value(the feature value)

	Output : special dataSet which is split by the value of the axis
	'''
	
	resDataSet = []

	for data in dataSet:
		if data[axis] == value:
			resData = data[:axis] + data[axis+1:]
			resDataSet.append(resData)

	return resDataSet

def chooseBestFeatureToSplit(dataSet):
	'''
	This function is used to choose best feature to split the dataSet. This 
	function will call 'splitDataSet' and 'calcShannonEnt'.
	This function will loop all the data feature, each time use one feature to 
	split the data, then calculate the Entropy. After that, the function output
	the best feature which can decrease the infoGain as much as possible one.
	
	Input : dataSet

	Output : the best feature's order.
	'''

	bestFeature = -1
	bestInfoGain = 0.0
	numFeatures = len(dataSet[0]) - 1
	origEntropy = calcShannonEnt(dataSet)

	for featOrd in range(numFeatures):
		featVals = [data[featOrd] for data in dataSet]
		uniqFeatVals = set(featVals)  # get all possible val for each feature
		# using all value of one feature to split the dataSet while 
		# every subdataSet calculate its entropy,
		# finally get the E[newEntropy]
		newEntropy = 0.0
		for fVal in uniqFeatVals:    
			subDataSet = splitDataSet(dataSet, featOrd, fVal)
			prob_subDataSet = len(subDataSet) / float(len(dataSet))
			# because last version I lost the float of the denomitor,The program gose wrong.
			newEntropy += prob_subDataSet * calcShannonEnt(subDataSet)

		newInfoGain = origEntropy - newEntropy
		if newInfoGain > bestInfoGain :
			bestInfoGain = newInfoGain
			bestFeature = featOrd

	return bestFeature

def majorityVote(classList):
	'''
	This function is used to vote all the value of the classValue. 
	When all of the feature is already used to split the dataSet, then 
	use this function to determine the catagory of the input subdataSet.

	Input : classList
	Output : the catagory of the input dataSet.
	'''
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1

	sortedClassCount = sorted(classCount.iteritems(), key=lambda d:d[1])
	
	return sortedclassCount[0][0]
	#sortedClass = sorted(classCount)


def createTree(dataSet, labels):
	'''
	This function is used to create a decision Tree.
	This function has two stop condition. 
		1. each leaf only has one label.
		2. all of the feature is used out for splitting, Then call majorityVote.
	This function uses dictionary to denote the tree structure. For each level, the 
	node is the bestFeatureLabel and the branch is denoted by the featureValue. finally
	the leaves is the dataset labels.
	For the tree, the key are labels and the value of the key are the value of the feature.
	Input : dataSet, labels
	Output : a decision Tree struction.
	'''

	# write down the stop conditions
	classList = [data[-1] for data in dataSet]
	if len(set(classList)) == 1:
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityVote(classList)

	# recurse create each level of the tree.
	bestFeatOrd = chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel = labels[bestFeatOrd]
	myTree = {bestFeatureLabel: {}}
	del(labels[bestFeatOrd])

	bestFeatVals = set([data[bestFeatOrd] for data in dataSet] )

	for val in bestFeatVals:
		sublabels = labels[:]
		myTree[bestFeatureLabel][val] = createTree(splitDataSet( \
										dataSet, bestFeatOrd, val), sublabels)

	return myTree


def classify(inputTree, featLabels, testVec):
	'''
	This function is used to classify the input data(testVec). Because we only know
	the input data's feature value, don't know the feature's label which is used to
	do classify, so each time we must get the feature order first. 

	Input : a Tree structure, feature's labels, test data
	Output: the classify result.
	'''
	featLabel = inputTree.keys()[0]
	featVal = inputTree[featLabel]
	featIndex = featLabels.index(featLabel)

	for val in featVal:
		if testVec[featIndex] == val:
			if type(featVal[val]) == dict:
				classLabel = classify(featVal[val], featLabels, testVec)
			else:
				classLabel = featVal[val]

	return classLabel


def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)


def testTreeBy_lenseData():
	'''
	This function is used to test decision tree classifer
	'''
	fr = open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	fr.close()
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = createTree(lenses, lensesLabels)
	return lensesTree


def testTreeBy_yeastData():
	'''
	This function uses UCI dataSet 'yeast' to test decision tree classifer.
	'''

	fr = open('yeast.txt')
	yeastData = [inst.strip().split()[:-2] for inst in fr.readlines()]
	fr.seek(0)
	yeastLabels = [inst.strip().split()[-1] for inst in fr.readlines()]
	fr.close()

	import numpy.random
	simple_index =  numpy.random.choice(len(yeastData), 30, replace=False)
	yeastDataSimple = [yeastData[ind] for ind in simple_index]
	yeastLabelsSimple = [yeastLabels[ind] for ind in simple_index]
	yeastTree = createTree(yeastDataSimple, yeastLabelsSimple)
	
	return yeastTree
















