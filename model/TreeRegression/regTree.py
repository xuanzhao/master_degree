from numpy import *

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)

	return dataMat

def binSplitDataSet(dataSet, feature, value):
	'''
	split DataSet by the special feature with its value
	'''
	mat0 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :]
	mat1 = dataSet[nonzero(dataSet[:,feature] > value)[0], :]
	return mat0, mat1

def regLeaf(dataSet):
	'''
	This function is used to get the leafNode value, which is mean 
	of data value.
	'''
	return mean(dataSet[:,-1])

def regErr(dataSet):
	'''
	This function is the error function, which is calculate the total
	variance of the data value.
	'''
	return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	'''
	this function is used to choose the best split feature with value.
	If it can't find a good split then return None, call 'createTree()'
	to create a leafNode at the same time and the leafNode also return 
	None.
	There are three cases that will not do a split, meanwhile directly create
	a leafNode. If it could find a good split, then return the feature idx 
	and the feature value.

	Output: feature idx and feature value.
	'''

	tolS = ops[0]; tolN = ops[1]
	# If all of data value is same then return None and the total variate
	if len( set(dataSet[:,-1].T.tolist()[0]) ) == 1:
		return None, leafType(dataSet) 

	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	for featIdx in range(n-1):
		for featVal in dataSet[:,featIdx]:
			mat0, mat1 = binSplitDataSet(dataSet, featIdx, featVal)

			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIdx
				bestValue = featVal
				bestS = newS

	if (S - bestS) < tolS:
		return None, leafType(dataSet)

	mat0 ,mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	'''
	This function will use recurse to create a tree. First, it will
	call 'chooseBestSplit()' to choose the best feature to split. If 
	it doesn't satisfy with terminal condition, then it will
	call 'binSplitDataSet()' to split the dataSet to subSet. 

	Input: 
	dataSet:
	leafType: determine what type of tree to create.
	errType: determine the error function.
	ops: customized tuple that used for tree create. tolS=1 is the 
	error value that can decrease mostly; tolN=4 is the smallest num
	of data of splitting. 

	Output: a tree structer.
	'''
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None: return val 

	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)

	return retTree


def regression(inputTree, data):
	'''
	This function is used to get the result for regression after
	the data through out inputTree.
	'''

	spInd = 0; spVal = 0
	leftTree = None; rightTree = None
 	

	for key in inputTree.keys():
		if key == 'spInd':
			spInd = inputTree[key]
		elif key == 'spVal':
			spVal = inputTree[key]
		elif key == 'left':
			leftTree = inputTree[key]
		elif key == 'right':
			rightTree = inputTree[key]

	# mat0, mat1 = binSplitDataSet(data, spInd, spVal)

	# if mat0 is not None:
	# 	result = regression(leftTree, mat0)
	# if mat1 is not None:
	# 	result = regression(rightTree, mat1)

	if data[:,spInd] <= spVal:
		if leftTree is dict:
			regression(leftTree, data)
		else: 
			print leftTree
			return leftTree
	
	if data[:,spInd] > spVal: 
		if rightTree is dict:
			regression(rightTree, data)
		else:
			print rightTree
			return rightTree

	# if isinstance(inputTree.values(),float):
 # 		return result[-1]

def plotTree(myTree, dataSet):
	'''
	This function is to visualize the dataSet after regTree.
	this is 2D plot.
	'''
	import matplotlib.pyplot as plt
	fig = plt.figure()
	fig.clf()

	y = zeros(len(dataSet),dtype=float)      # y is ndarray
	x = array(dataSet[:,1])       # x is ndarray
	#x = range(len(y))
	for i, data in enumerate(dataSet):
		y[i] = regression(myTree, data)  # because regression return dict.
										 # maybe the tree is wrong.

	plt.plot(x,y,'o')
	plt.show()


# the above for the Tree prune.

def isTree(obj):
	'''
	This function is used to justify whether this obj is tree or not.
	'''
	return type(obj).__name__ == 'dict'

def getMean(tree):
	'''
	This function will use recurse to through to leaves node then calculate
	the mean of the two leafNode.
	'''
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right']) / 2.0


def prune(tree, testData):
	'''
	the merged error value compares with the fore one. if less than before then merge.
	'''
	if shape(testData)[0] == 0:
		return getMean(tree)
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) 

	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)

	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2) ) + \
						sum(power(rSet[:,-1] - tree['right'],2) )
		treeMean = (tree['left']+tree['right']) / 2.0
		errorMerge = sum(power(testData[:,-1] - treeMean, 2))
		if errorMerge < errorNoMerge:
			print 'merging'
			return treeMean
		else:
			return tree
	else:
		return tree


# above function is used for create model tree

def modelLeaf(dataSet):
	'''
	This function will get a linear model which is ploy on leafNode.
	'''
	ws, X, Y = linearSolve(dataSet)
	return ws

def modelErr(dataSet):
	'''
	This function is error function for linear model on the leafNode.
	'''
	ws, X, Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y - yHat, 2))

def linearSolve(dataSet):
	'''
	use linear algebra formular to get the linear model solution.
	'''
	m, n = shape(dataSet)
	X = mat(ones((m,n))); Y = mat(ones((m,1)))
	X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1] # X[:,0] is bias value

	xTx = X.T*X
	if linalg.det(xTx) == 0.0:
		raise NameError('This matrix is singular, cannnot do inverse, \n\
			try increasing the second value of ops')
	ws = xTx.I * (X.T * Y)
	return ws, X, Y

def plotModelTree(myTree, dataSet):
	m, n = shape(dataSet)
	X = mat(ones((m,n))); Y = mat(ones((m,1)))
	X[:,1:n] = dataSet[:,0:n-1] # X[:,0] is bias value

	featInd = myTree['spInd']
	featVal = myTree['spVal']

	#XL, XR = binSplitDataSet(dataSet, featInd, featVal)
	ind = X[:,1] < featVal
	XL = concatenate( (X[:,0][ind],X[:,1][ind]), axis=0) 
	wsL = myTree['left']
	YL = XL.T * wsL
	
	ind = X[:,1] >= featVal
	XR = concatenate( (X[:,0][ind],X[:,1][ind]), axis=0) 
	wsR = myTree['right']
	YR = XR.T * wsR

	import matplotlib.pyplot as plt
	fig = plt.figure()
	fig.clf()

	X1 = concatenate( (XL.T[:,1], XR.T[:,1]), axis=0 )
	Y1 = concatenate( (YL, YR), axis=0)
	print shape(XL.T[:,1]), shape(YL)
	plt.plot(X1, Y1,'o')
	plt.show()

# The above function is used to predict by using regTree.

def regTreeEval(model, inData):
	'''
	Input : model is the leafNode
	'''
	return float(model)

def modelTreeEval(model, inData):
	'''
	This function will first formularize the inData, add one column at the first.
	Then return predict value.

	Input : inDat is a 1 by n vector; model is the leafNode
	'''
	n = shape(inData)[1]
	X = mat(ones((1,n+1)))
	X[:,1:n+1] = inData
	return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
	'''
	This will use up to down method to look up the tree, when it to the leafNode, then
	it calls 'modelEval()', which will return the predict value.
	Input: 
	Output: predict value for single data
	'''
	if not isTree(tree):
		return modelEval(tree, inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['right']):
			return treeForeCast(tree['right'], inData, modelEval)
		else:
			return modelEval(tree['right'], inData)
	else:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
	'''
	This function will call 'treeForeCast()' multiple times to get the predict value of
	the testDataSet.
	'''
	m = len(testData)
	yHat = mat(zeros((m,1)))
	for i in range(m):
		yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
	return yHat



























