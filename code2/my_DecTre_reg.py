import numpy as np
import matplotlib.pylab as plt

# ===============================================
# common function
# ===============================================
def stocGradDescent(X, y, alpha=0.01, maxCycle=500):
    X = np.mat(X); y = np.mat(y).T
    m,n = X.shape
    weights = np.random.random((n,1))

    for n in range(maxCycle):
        np.random.seed(n)
        np.random.permutation(np.c_[X,y])
        for i in range(m):
            alpha = 4 / (1.0 + n + i) +0.01
            h = sigmoid(X[i] *weights)
            error = y[i] - h
            weights = weights + alpha * (error * X[i]).T

    return weights


def gradDescent(X, y, alpha=0.01, maxCycle=500):
    X = np.mat(X); y = np.mat(y).T
    m, n = X.shape
    weights = np.random.random((n,1))   # (n,1)

    for n in range(maxCycle):
        alpha = 4 / (1.0 + n) +0.01
        h = sigmoid(X*weights)          # (m,n) * (n,1) = (m,1)
        error = y - h                   # (m,1)
        weights = weights + alpha * X.T * error    # (n,m)*(m,1)= (n,1)

    # print '------------this is leafType-----------\n'
    # print 'leafType will return :', weights
    return weights  # (n,1)


def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def sigmoidErr(X, y):
    X = np.mat(X); y = np.mat(y)
    ws = gradDescent(X, y)
    yHat = sigmoid(X*ws)    # (m,n) * (n,1) = (m,1)
    # error = np.sum(np.not_equal(y, yHat))
    error = np.sum(np.power(y - yHat, 2))
    return error

def ridgeRegr(X, y, lam=0.2):
    X = np.mat(X); y = np.mat(y).T
    xTx = X.T * X       # (n,m) * (m,n) = (n,n)
    denom = xTx + np.eye(X.shape[1])*lam     # (n,n)
    ws = np.linalg.pinv(denom) * (X.T * y)   # (n,n)*(n,m)*(m,1) = (n,1) 

    return ws

def lseErr(X, y):
    X = np.mat(X); y = np.mat(y)
    ws = ridgeRegr(X, y)
    yHat = X * ws   # (m,n) * (n,1) = (m,1)
    error = np.sum(np.power(y - yHat, 2))

    return error

# def get_RList(tree):

#     RList = []
#     def get_R(tree):
#         if tree.RInfo != None:
#             print '\ntree RInfo is:', tree.RInfo
#             RList.append(tree.RInfo)
#         else:
#             get_R(tree.leftChild)
#             get_R(tree.rightChild)
#     get_R(tree)
#     return RList

def bagForFeatures(max_features, n_features):
    """
    The number of features to consider when looking for the best split:
        - If int, then consider 'max_features' as number at each split.
        - If float, then 'max_features' is a percentage and
            'int(max_features * n_features)' features are considered at
            each split.
        - If 'sqrt', then 'max_features=sqrt(n_features)'.
        - If 'log2', then 'max_features=log2(n_features)'.
        - If None, then 'max_features=n_features'.
    """
    if max_features == None:
        return np.arange(n_features)
    elif isinstance(max_features, int):
        if max_features < n_features:
            return np.random.choice(n_features, max_features, replace=False)
        else: return np.arange(n_feautres)
    elif isinstance(max_features, float):
        if max_features < 1.0:
            return np.random.choice(n_features, int(max_features*n_features), replace=False)
        else: return n_features
    elif max_features == 'sqrt':
        return np.random.choice(n_features, int(np.sqrt(n_features)))
    elif max_features == 'log2':
        return np.random.choice(n_features, int(np.log2(n_features)))

# ================================================
# Types and constants
# ================================================

# RList = []
LEAFTYPE = {'sigmoidRegr': gradDescent, 'sigmoidSTOCRegr':stocGradDescent ,
             'ridgeRegr': ridgeRegr}
ERRTYPE = {'sigmoidErr': sigmoidErr, 'lseErr': lseErr}

# ===============================================
# Decision Tree Classifier
# ===============================================

class treeNode(object):

    def __init__(self, parent):
        self.parent = parent
        self.leftChild = None
        self.rightChild = None
        self.splitIndex = None
        self.splitValue = None  # if it is leafNode, splitValue = weights
        self.n_samples = 0
        self.n_features = 0
        self.RInfo = None


    def binSplitData(self, dataMat, featIdx, featVal):

        ind = dataMat[:, featIdx] <= featVal
        leftMat = dataMat[ind,:]
        rightMat = dataMat[np.logical_not(ind), :]

        return leftMat, rightMat


    def chooseBestSplit(self, dataMat, leafType, errType, max_depth,
                        min_samples_split, min_weight_fraction_leaf,
                        class_weight, max_features, n_features):


        # all data is same class
        yHat = dataMat[:,-1]
        if len(np.unique(yHat)) == 1:
            print 'before return leafType, let me check the value\n'
            print 'here all data is same class :',yHat
            print 'the leafType return value :',leafType(dataMat[:,:-1],dataMat[:,-1])
            return None, leafType(dataMat[:,:-1],dataMat[:,-1])
        # fit the max_depth
        if max_depth != None:
            if self.selfDepth > max_depth:
                print 'before return leafType, let me check the value\n'
                print 'here fit the max_depth:', self.selfDepth
                print 'the leafType return value :',leafType(dataMat[:,:-1],dataMat[:,-1])
                return None, leafType(dataMat[:,:-1],dataMat[:,-1])

        # get the feature index for split
        featIndexes = bagForFeatures(max_features, n_features)
        bestError = np.inf; bestIndex = 0; bestValue = 0

        for featIndex in featIndexes:
            for splitVal in np.unique(dataMat[:, featIndex]):
                leftMat, rightMat = self.binSplitData(dataMat, featIndex, splitVal)
                if (leftMat.shape[0] < min_samples_split) or \
                    (rightMat.shape[0] < min_samples_split): continue
                newError = errType(leftMat[:, :-1],leftMat[:, -1]) + \
                           errType(rightMat[:, :-1], rightMat[:, -1])

                if newError < bestError:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestError = newError
                print '-----------------iteration-----------------------\n'
                print 'featIndex : ',featIndex, 'FeatValue :', splitVal
                print 'newError  : ',newError, 'bestError : ', bestError
                print '-------------------------------------------------\n'

        # fit the min_samples_split
        leftMat, rightMat = self.binSplitData(dataMat, bestIndex, bestValue)
        if (leftMat.shape[0] < min_samples_split) or \
            (rightMat.shape[0] < min_samples_split):
            print 'before return leafType, let me check the value\n'
            print 'here fit the min_samples_split :',self.n_samples
            print 'the leafType return value :',leafType(dataMat[:,:-1],dataMat[:,-1])
            return None, leafType(dataMat[:,:-1],dataMat[:,-1])

        print '********************* return **********************\n'
        print 'bestIndex : ',bestIndex, 'bestValue :', bestValue
        print 'bestError : ', bestError
        print '---------------------------------------------------\n'
        
        #raw_input('let me see see first')

        return bestIndex, bestValue

    def createTree(self, dataMat, leafType, errType, max_depth,
                   min_samples_split, min_weight_fraction_leaf,
                   random_state, class_weight,
                   max_features):

        self.n_samples, self.n_features = dataMat[:,:-1].shape
        featId, featVal = self.chooseBestSplit(dataMat, leafType, errType, 
                        max_depth, min_samples_split, min_weight_fraction_leaf, 
                        class_weight, max_features, self.n_features)
        self.dataMat = dataMat
        if featId == None: 
            self.splitIndex = None
            self.splitValue = featVal # leaf node featVal is weights
            self.parent.RInfo = self.parent.calc_R(self.parent.dataMat)
        else:
            self.splitIndex = featId
            self.splitValue = featVal
            print '-------------------- get BestSplit --------------\n'
            print 'self.splitIndex :', self.splitIndex, 'self.splitValue : ', self.splitValue
            print '-------------------------------------------------\n'
            leftMat, rightMat = self.binSplitData(dataMat, featId, featVal)
            print '------------- after one split ----------------------\n'
            print 'leftMat shape  :', leftMat.shape, 'mat samples :\n', leftMat[:3,:]
            print 'rightMat shape :', rightMat.shape, 'mat samples :\n',rightMat[:3,:]
            print '---------------------------------------------------\n'

           
            self.leftChild = treeNode(self)
            self.rightChild = treeNode(self)
            print "\n***********this node's information:***************\n"
            print "self.parent     :", self.parent
            print "self.leftChild  :", self.leftChild
            print 'self.rightChild :', self.rightChild
            print 'self.splitIndex :', self.splitIndex
            print 'self.splitValue :', self.splitValue
            print 'self.n_samples  :', self.n_samples
            print 'self.n_features :', self.n_features
            print "\n**************************************************\n"
            self.leftChild.createTree(leftMat, leafType, errType, 
                            max_depth, min_samples_split, min_weight_fraction_leaf, 
                            random_state, class_weight, max_features)
            
            self.rightChild.createTree(rightMat, leafType, errType, 
                            max_depth, min_samples_split, min_weight_fraction_leaf, 
                            random_state, class_weight, max_features)
            #raw_input('let me see see again')
        
        #raw_input('let me see see again again ')
        return self

    def calc_R(self,dataMat):
        X = (dataMat[:,:-1])
        X_mean = np.mean(X, axis=0)
        X_radius = np.std(X, axis=0)
        RInfo = np.c_[X_mean, X_radius] 
        #RInfo = {'X_mean':X_mean, 'X_radius':X_radius} 
        return RInfo   # (n, 2), col0=center, col1=radius

    def get_RList(self):

        RList = []
        def get_R(self):
            if self.RInfo == None:
                get_R(self.leftChild)
                get_R(self.rightChild)
            else:
                RList.append(self.RInfo)
                if self.leftChild.splitIndex !=None:
                    get_R(self.leftChild)
                if self.rightChild.splitIndex !=None:
                    get_R(self.rightChild)
        get_R(self)
        return RList

    @property
    def selfDepth(self):
        """finally set the ``treeNode`` depth and ID"""
        depth = 0
        if self.parent:
            depth += 1
            return depth + self.parent.selfDepth

        return depth

    def getTreeStruc(self, indent=' '):
        if self.leftChild and self.rightChild:
            assert(len(indent) > 0)
            print indent + 'splitIndex [%d]<%f ' % (self.splitIndex, self.splitValue )
            self.leftChild.getTreeStruc(indent + indent[0])
            self.rightChild.getTreeStruc(indent + indent[0])
        else:
            splitValue = ' '.join(map(str, self.splitValue.tolist()))
            print indent + 'leaf node: ' + splitValue

    def isTree(self, obj):
        return type(obj).__name__ == 'treeNode'


    def treeForeCast(self, x_test, leafType):

        if self.splitIndex==None:
            if leafType.__name__ == 'gradDescent' or 'stocGradDescent':
                weights = np.mat(self.splitValue)
                sigmoidResult = sigmoid(x_test * weights)
                prediction = (1 if  sigmoidResult> 0.5 else 0)
                #print '---------the sigmoidResult value :', sigmoidResult
                return prediction
            else:
                print 'Forecast fault'

        if x_test[:,self.splitIndex] < self.splitValue :
            if self.isTree(self.leftChild):
                return self.leftChild.treeForeCast(x_test, leafType) 
        else:
            if self.isTree(self.rightChild):
                return self.rightChild.treeForeCast(x_test, leafType)

    def treeDecisionBoundary(self, x_test, leafType):


        if self.splitIndex==None:
            if leafType.__name__ == 'gradDescent' or 'stocGradDescent':
                weights = np.mat(self.splitValue)
                sigmoidResult = sigmoid(x_test * weights)
                #print '---------sigmoidResult :', 
                if (sigmoidResult>0.49 and sigmoidResult<.51):
                    return 0.5
            else:
                print 'get decisonBoundary value fault'

        if x_test[:,self.splitIndex] < self.splitValue :
            if self.isTree(self.leftChild):
                return self.leftChild.treeForeCast(x_test, leafType)
        else:
            if self.isTree(self.rightChild):
                return self.rightChild.treeForeCast(x_test, leafType) 


class DecisionTreeRegresion(object):


    """A decision tree classifier.

    Parameters
    ----------
    errType : string, (default='lss')
        The function to measure the quality of a split.
        traditional, 'gini' for the Gini impurity and 'entropy' for the
        information gain.

    splitter : string, (default='best')
        The strategy used to choose the split at each node. Suported
        strategies are 'best' to choose the best split and 'random' to
    ex    choose the best random set split.

    max_features : int, float, string oex None, (default=None) 
        The number of featInds to consider when looking for the best split:
            - If int, then consider 'max_features' as number at each split.
            - If float, then 'max_features' is a percentage and
                'int(max_features * n_features)' features are considered at
                each split.
            - If 'auto', then 'max_features=sqrt(n_features)'.
            - If 'sqrt', then 'max_features=sqrt(n_features)'.
            - If 'log2', then 'max_features=log2(n_features)'.
            - If None, then 'max_features=n_features'.

    max_depth : int or None, (default=None)
        The maximum depth of the tree. If None, then nodes are expand untill
        all leaves are pure or until all leaves contain less than 
        min_samples_split or unitll all leaves less than the error threshold.

    min_samples_split : int, (default=5)
        The minimum number of samples required to split an internal node.

    min_weight_fraction_leaf : float, (defualt=0.0)
        The minimum weighted fraction of the input samples required to be at
        a leaf node.

    class_weight : dict, list of dicts, 'balanced' or None. (default=None)
        Weights associated with classes in the form ``{class_label: weight}``,
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multipied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    random_state: int, RandomState instance or None, (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
           by `np.random`.


    Attributes
    ----------
    classes : array of shape = [n_classes] or a list of such arrays The 
        classes labels (single output problem).

    feature_importance : array of shape = [n_features]
        The feature importances. The higher, the more important the feature.

    max_features : int, The infered value of max_features.

    n_classes : int 

    n_features : int, The number of features when ``fit`` is performed.

    n_outputs : int, The number of outputs when ``fit`` is performed.

    tree : Tree object, The underlying Tree object.

    Examples
    -----------
    pass

    """

    def __init__(self,
                 errType='lseErr',
                 leafType='ridgeRegr',
                 max_depth=5,
                 min_samples_split=10,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 class_weight=None,
                 ):

    # user's input attributes
        self.errType = ERRTYPE.get(errType)
        self.leafType = LEAFTYPE.get(leafType)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight

    # decisionTree privite attributes, which is determinated by input data
        self.n_features = None
        self.n_outputs  = None
        self.classes    = None

        self.tree = treeNode(None)

    # ``fit`` method will train a DecisionTree.
    def fit(self, X, y, sample_weight=None):
        """
        Build a decision tree from the trainning set (X, y).

        Parameters
        ----------
        X :
        y :
        sample_weight :

        Returns
        -------
        self : object
            return self.
        """
        # --------------------- set tree attributes by input data--
        self.n_outputs, self.n_features = X.shape
        self.classes = np.unique(y)

        # ===================== Build tree =======================
        
        # --------------------- Set tree attributes --------------
        dataMat = np.c_[X, y]

        errType = self.errType
        leafType = self.leafType
        max_depth = self.max_depth
        min_samples_split = self.min_samples_split
        min_weight_fraction_leaf = self.min_weight_fraction_leaf
        random_state = self.random_state
        class_weight = self.class_weight

        max_features = self.max_features
        
        

        # --------------------- training special tree -----------
        self.tree.createTree(dataMat, 
                            leafType=leafType, 
                            errType=errType,
                            max_depth= max_depth, 
                            min_samples_split = min_samples_split,
                            min_weight_fraction_leaf= min_weight_fraction_leaf,
                            random_state=random_state,
                            class_weight=class_weight,
                            max_features=max_features)

        return self


    def predict(self, X):

        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. 
        For a regression model, the predicted value based on X is returned.

        Parameters
        ----------
        X :

        Returns
        -------
        y : array of shape = [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        self._validate_X_predict(X)
        X = np.mat(X)
        m,n = X.shape
        yHat = np.mat(np.zeros((m,1)))
        tree = self.tree
        leafType = self.leafType

        for i in range(m):
            yHat[i,0] = tree.treeForeCast(X[i], leafType)

        return yHat.A

    def getDecisionBoundary(self, X):
        self._validate_X_predict(X)
        X = np.mat(X)
        m,n = X.shape
        yBoundary = np.mat(np.zeros((m,1)))
        tree = self.tree
        leafType = self.leafType

        for i in range(m):
            yBoundary[i,0] = tree.treeDecisionBoundary(X[i], leafType)

        return yBoundary.A

    def _validate_X_predict(self, X):
        
        """
        Validate X whenever one tries to predict, apply, predict_proba
        """
        if self.tree is None:
            raise NotFittedError("Estimator not fitted,"
                                 "call 'fit' before explotting the model.")

        n_features = X.shape[1]
        if self.n_features != n_features:
            raise ValueError("Number of features of the model must"
                             " match the input. Model n_features is %s and"
                             " input n_features is %s"
                             % (self.n_features, n_features))
    

    def apply(self, X):

        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]

        Returns
        -------
        X_leaves : array_like, shape = [n_samples]
            For each datapoint x in X, return the index of the leaf x ends
            up in. Leaves are numbered within
            ``[0 : self.tree.node_count)``, possibly with gaps in the numbering.
        """
        X = self._validate_X_predict(X)
        return self.tree.apply(X)



# ============================RandomFrestRegression===========================
class RandomForestRegression(object):
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
                 n_trees=10,
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
        self.errType = errType
        self.leafType = leafType
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

        self.trees = []
        for n in range(n_trees):
            self.trees.append(DecisionTreeRegresion(
                            errType = ERRTYPE.get(errType), # here not pass para, ERRTYPE not defined
                            leafType = LEAFTYPE.get(leafType),
                            max_depth = max_depth,
                            min_samples_split = min_samples_split,
                            min_weight_fraction_leaf = min_weight_fraction_leaf,
                            max_features = max_features,
                            random_state = random_state,
                            class_weight = class_weight)
                            )

 
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


def RF_fit(X_train, y_train, n_trees=10, 
            max_depth=5, min_samples_split=10, max_features=None):

    from sklearn.utils import resample
    trees = []
    for n in range(n_trees):
        trees.append(DecisionTreeRegresion(
                     errType='lseErr',
                     leafType='ridgeRegr',
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_weight_fraction_leaf=0.0,
                     max_features=max_features,
                     random_state=None,
                     class_weight=None)
                    )

    for tree in trees:
        X_boot_train, y_boot_train = resample(X_train, y_train)
        tree.fit(X_boot_train, y_boot_train)

    return trees  # type is list

def RF_predict(X_test,trees):
    predictions = []
    for tree in trees:
        y_pred = tree.predict(X_test)
        predictions.append(y_pred)

    predictions = np.array(predictions)  # (n,m,1)
    avg_pred = np.mean(predictions, axis=0)
    sigmoid_pred = np.where(avg_pred>0.5, 1, 0)
    return sigmoid_pred


def get_RF_avgRList(trees):
    
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    # get_RF_RList
    RF_RList=[]
    for tree in trees:
        RF_RList.extend(tree.tree.get_RList())   # len = m

    RF_R_Mat = np.array(RF_RList)  #(m,n,2), col0=center, col1=radius
    RF_R_centers = RF_R_Mat[:,:,0]  # (m,n)
    RF_R_radius = RF_R_Mat[:,:,1]   # (m,n)

    # get the number of cluster
    avg_num_R = int( RF_R_Mat.shape[0] /len(trees))  # total R divided by number trees
    # get the connectivity graph of R_list
    connect_graph = kneighbors_graph(RF_R_centers, n_neighbors=len(trees)-1, include_self=False)
    # connect_graph shape = (m,m) , if neibor then value=1, else=0

    # compute clustering
    R_cluster = AgglomerativeClustering(n_clusters=avg_num_R, connectivity=connect_graph,
                                    linkage='ward').fit(RF_R_centers)

    #get_RF_avgRList(R_cluster):
    R_cluster_label = R_cluster.labels_
    RF_avgRList = []

    for label in np.unique(R_cluster_label):
        R_mean  = np.mean(RF_R_centers[R_cluster_label == label], axis=0)
        R_radius = np.mean(RF_R_radius[R_cluster_label == label], axis=0)

        R = np.c_[R_mean, R_radius] # shape (n,2)
        RF_avgRList.append(R)

    return RF_avgRList









