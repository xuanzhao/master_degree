




def get_decisionTree_code(tree, feature_names, target_names, spacer_base='    '):
	'''
	Produce psuedo-code for decision tree.

	Args
	-----
	tree -- scikit-learn get_decisionTree
	feature_names -- list of feature names.
	target_names -- list of target (class) names.
	spacer_base -- used for spacing code (default: '4 space')

	'''
	
	left      = tree.tree_.children_left
	right     = tree.tree_.children_right
	threshold = tree.tree_.threshold
	features  = [feature_names[i] for i in tree.tree_feature]
	value     = tree.tree_.value

	def recurse(left, right, threshold, features, node, depth):
		spacer = spacer_base * depth

		if (threshold[node] != -2): # if not leaf node
			print spacer + 'if ( ' + features[node] + ' <= ' + \
				  str(threshold[node]) + ' ) {')
			
			if left[node] != -1:	# if left child is true
				recurse(left, right, threshold, features, right[node], depth+1) 
			print spacer + '} \n' + spacer + 'else {'
			
			if right[node] != -1:	# if right child is true
				recurse(left, right, threshold, features, right[node], depth+1)
			print spacer + '}'

		else:
			target = value[node]
			for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
				target_name = target_names[i]
				target_count = int(v)
				print spacer + 'return' + str(target_name) + \
					  ' ( ' + str(target_count) + ' examples )'

	recurse(left, right, threshold, features, 0, 0)