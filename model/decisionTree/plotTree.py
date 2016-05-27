import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	'''
	This function is used to plot tree's node.
	nodeTxt: the text which is displayed on the node
	centerPt: the node's coordinates
	parentPt: the arrow's start point's coordinates
	noteType: node or leaves
	'''
	# createPlot.ax1 is a global variate which initial a plot zone.
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',\
							xytext=centerPt, textcoords='axes fraction',\
							va='center', ha='center', bbox=nodeType, \
							arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
	'''
	This function is used to calculate the center point between 
	parentNode and childrenNode for adding text.
	'''
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	'''
	The most of work to plot tree is done here.
	First, this function calculate the Tree's width and hight, then use these
	info to calculate the position of the tree.

	This function is also a recurse function.
	'''
	numLeaves = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]

	# get the childNode Position
	cntrPt = (plotTree.xOff + (1.0 + float(numLeaves)) /2.0/plotTree.totalW, plotTree.yOff)
	
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)

	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
	'''
	this function is used to create a plot on a new figure.
	This function is our main function, it calls plotTree() and plotTree()
	calls plotMidText() and above function.
	'''
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	# createPlot.ax1 = plt.subplot(111, frameon=False)

	# plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	# plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	# plt.show()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	
	# Set the global variate, total width and total depth of the tree,
	# the global variate, trace the position of plotted.
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5 / plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5,1.0), '')
	plt.show()



def retrieveTree(i):
	'''
	This function is used to get the tree struction as a output.
	'''

	listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]

	return listOfTrees[i]


def getNumLeafs(myTree):
	'''
	This function is used to get the number of leaves of the tree.
	The tree is stored in a dictionary which is consist of node' name and node's value.
	This function uses recurse method to get to the leaves node then sum up the number.
	if the key's value is still dict then recurse.
	if the key's value is not dict then it's a leaf.

	Input : a Tree structre which is dictionary.
	Output : the number of the leaves of the tree.
	'''
	numLeaves = 0
	split_label = myTree.keys()[0]
	subTree = myTree[split_label]

	for labelVal in subTree.keys():
		if type(subTree[labelVal]) is not dict:
			numLeaves += 1
		else: 
			numLeaves += getNumLeafs(subTree[labelVal])
	# for 
	# 	if 
	# 		numLeaves += getNumLeafs()
	# 	else:	numLeaves += 1

	return numLeaves


def getTreeDepth(myTree):
	'''
	This function is used to get the depth of the tree.
	The tree is stored in a dictionary which consist of nodes' name and nodes' value.
	This function uses recurse method to get the depth, when get to the split node then let
	the depth plus 1, the terminal condition is touch to the leaves node and set the depth 1.

	For the dictionary, the only count node is split_label.

	Input: a Tree structer which is dictionary.
	Output: the depth of the tree.
	'''

	maxDepth = 0
	split_label = myTree.keys()[0]
	subTree = myTree[split_label]


	for labelVal in subTree.keys():
		if type(subTree[labelVal]) is dict:
			thisDepth = 1 + getTreeDepth(subTree[labelVal])
		else: thisDepth = 1

		if thisDepth > maxDepth:
			maxDepth = thisDepth

	return maxDepth































