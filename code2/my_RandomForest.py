from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

from sklearn import metrics
import my_DecTre_clf
import my_DecTre_reg

# ================================================
# Types and constants
# ================================================

# # RList = []
# LEAFTYPE = {'sigmoidRegr':gradDescent, 'sigmoidSTOCRegr':stocGradDescent ,
#              'ridgeRegr':ridgeRegr}
# ERRTYPE = {'sigmoidErr': sigmoidErr, 'lseErr': lseErr}



class RandomForestClassifier(object):
	"""A random forest classifier.

	A random forest is a meta estimator that fits a number of decision tree
	classifiers on various sub-samples of the dataset and use averaging to 
	improve the predictive accuracy and control over-fitting.
	The sub-sample size is always the same as the original input sample size
	but the samples are drawn with replacement if `bootstrap=True`(default).

	Parameters
	----------
	n_trees: integer, optional(default=10)
		The number of trees in the forest.

	errType: 

	leaftype:

	max_features:

	max_depth:

	min_samples_split:

	min_weight_fraction_leaf:

	random_state:

	class_weight:

	bootstrap : boolean, optional (default=True)
		Whether bootstrap samples are used when building trees.

	oob_score : bool
		Whether to use out-of-bag samples to estimate the generalization error.

	n_jobs : integer, optional (default=1)
		The number of jobs to run in parallel for both `fit` and `predict`.
		If -1, then the number of jobs is set to the number of cores.


	Attributes
	----------
	estimators : list of DecisionTreeClassifier
		The collection of fitted sub-estimators.

	n_features : int
		The number of features when ``fit`` is performed.

	n_outputs : int
		The number of outputs when ``fit`` is performed.

	classes : array of shape = [n_classes] or a list of such arrays.

	oob_score : float
		Score of the training dataset obtained using an out-of-bag estimate.
	"""

	def __init__(self,
				 n_trees=2,
				 errType='lseErr',
				 leafType='ridgeRegr',
                 max_depth=3,
                 min_samples_split=5,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 class_weight=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 ):

    # user's input attributes
		self.n_trees = n_trees
        self.errType = ERRTYPE.get(errType)
        self.leafType = LEAFTYPE.get(leafType)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight

    # decisionTree privite attributes, which is determinated by input data
        # self.n_features = None
        # self.n_outputs  = None
        # self.classes    = None

    	tree_params = { errType : ERRTYPE.get(errType),
				        leafType : LEAFTYPE.get(leafType),
				        max_depth : max_depth,
				        min_samples_split : min_samples_split,
				        min_weight_fraction_leaf : min_weight_fraction_leaf,
				        max_features : max_features,
				        random_state : random_state,
				        class_weight : class_weight} 

        self.tree = []
        for n in range(n_trees):
        	self.tree.append(DecisionTreeRegresion(tree_params))

 
	def fit(self, X_train, y_train):
		for tree in self.trees:
			tree.fit(X_train, y_train)
		
		return self

	def predict(self,X_test):
		predictions = []
		for tree in self.trees:
			y_pred = tree.predict(X_test)
			predictions.append(y_pred)
		
		avg_pred = np.mean(predictions, axis=1)
		return avg_pred































