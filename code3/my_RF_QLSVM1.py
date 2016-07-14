from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from time import time
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
# ===============================================
# common function
# ===============================================

SGDClf = linear_model.SGDClassifier(loss='modified_huber',penalty='l1')

LogicReg = linear_model.LogisticRegression(penalty='l1', C=1.0, n_jobs=4)

RidgeReg = linear_model.Ridge(alpha=1.0)

KernelRidge = KernelRidge(alpha=1.0, kernel="linear", gamma=None)

RANSACReg = linear_model.RANSACRegressor(linear_model.LinearRegression())

BayesReg = linear_model.BayesianRidge(n_iter=300,alpha_1=1.e-6,alpha_2=1.e-6,
                                      lambda_1=1.e-6, lambda_2=1.e-6)

IsotonicReg = IsotonicRegression(y_min=None, y_max=None, increasing=True,
                                 out_of_bounds='nan')


def lseErr(X, y, leafType):

    if len(np.unique(y)) != 1:
        model = leafType
        model.fit(X, y)
        try:
            model.__getattribute__('predict_proba')
            #print 'current leaf model could predict_probability, which is \n',model
            yHat = model.predict_proba(X)
        except AttributeError, e:
            #print 'AttributeError, ', e
            model.__getattribute__('predict')
            #print 'now use predict method, leaf model is \n',model
            yHat = model.predict(X)
        yHat = yHat[:,1]

        error = np.sum(np.power(y - yHat, 2)) / len(yHat)

        #yHat = model.predict_log_proba(X)
        #error = metrics.log_loss(y, yHat)
        return error
    else:
        return 0.0


def lseErr_regul(X, y, leafType, k1=1,k2=1):
    if len(np.unique(y)) != 1:
        model = leafType
        model.fit(X, y)
        try:
            model.__getattribute__('predict_proba')
            # print 'current leaf model could predict_probability, which is \n',model
            yHat = model.predict_proba(X)
        except AttributeError, e:
            # print 'AttributeError, ', e
            model.__getattribute__('predict')
            # print 'now use predict method, leaf model is \n',model
            yHat = model.predict(X)

        yHat = yHat[:,1] # get predict is 1
        #neg_ratio = np.true_divide(np.sum(y==0), len(y))
        X1_mean = np.mean(X[y==1], axis=0)
        X0_mean = np.mean(X[y==0], axis=0)

        X1_delta = X[y==1] - X0_mean # (m,n)
        X0_delta = X[y==0] - X1_mean

        #X1_delta = X[y==1] - X1_mean # (m,n)
        #X0_delta = X[y==0] - X0_mean
        X_delta1 = np.r_[X1_delta, X0_delta]
        
        X_delta2 = X - np.mean(X, axis=0)

        error_mse = np.sum(np.power(y - yHat, 2)) / len(yHat) #+ neg_ratio
        error_reg = (k1 * np.sum(np.power(X_delta1, 2)) +
                     k2 * np.sum(np.power(X_delta2, 2)) )/len(yHat)

        #yHat = model.predict_log_proba(X)
        #error = metrics.log_loss(y, yHat)
        return (error_mse, error_reg)
    else:
        X_delta2 = X - np.mean(X, axis=0)
        error_reg = k2 * np.sum(np.power(X_delta2, 2)) / len(X_delta2)
        #print ' current split data is all same clss'
        return (0, error_reg)
        
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

def get_boundary(X, y,n_neighbors=8,radius=0.5):

    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, n_jobs=4)
    neigh.fit(X)

    boundary_points = []
    nonBoundary_points = []

    X = np.array(X); y = np.array(y)
    m,n = X.shape

    for i in np.arange(m):
        x = X[i,:].reshape(1,-1)
        neigh_ind = neigh.kneighbors(x, 8, return_distance=False)
        x = np.c_[x, y[i].reshape(-1,1)]
        if len(np.unique(y[neigh_ind])) > 1: # x is boundary point
            boundary_points.append(x)
        else: # x is not boundary point
            nonBoundary_points.append(x)

    data_bound = np.array(boundary_points).reshape(len(boundary_points),n+1)
    data_nonBound = np.array(nonBoundary_points).reshape(len(nonBoundary_points),n+1)

    return data_bound, data_nonBound



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
            return np.sort(np.random.choice(n_features, max_features, replace=False))
        else:
            print 'the max_features > data features' 
            return np.arange(n_feautres)
    elif isinstance(max_features, float):
        if max_features < 1.0:
            return np.sort(np.random.choice(n_features, int(max_features*n_features), replace=False))
        else: return n_features
    elif max_features == 'sqrt':
        return np.sort(np.random.choice(n_features, int(np.sqrt(n_features))))
    elif max_features == 'log2':
        return np.sort(np.random.choice(n_features, int(np.log2(n_features))))

# ================================================
# Types and constants
# ================================================

# RList = []
LEAFTYPE = {'SGDClf': SGDClf, 'LogicReg': LogicReg, 'RidgeReg': RidgeReg, 
            'RANSACReg': RANSACReg, 'BayesReg': BayesReg,
            'IsotonicReg': IsotonicReg, 'KernelRidge':KernelRidge
            }
ERRTYPE = {'lseErr': lseErr, 'lseErr_regul': lseErr_regul}

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
        self.dataMat = None


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
        m,n = dataMat.shape
        if len(np.unique(yHat)) == 1:
            #print 'before return leafType, let me check the value\n'
            # print '---------------------------------------------------\n'
            print 'here all data is same class '
            print 'the leafType return (class label)', int(np.unique(yHat))
            # print '---------------------------------------------------\n'
            if self.parent.rightChild is self:
                leftChild = self.parent.leftChild
                if isinstance(leftChild.splitValue, int):
                    if leftChild.splitValue != self.splitValue:
                        print 'two pair of leftNode is different class data, get RInfo...'
                        self.RInfo = self.calc_R(self.dataMat)
                        leftChild.Rinfo = leftChild.calc_R(leftChild.dataMat)
                elif isinstance(leftChild.splitValue, float):
                    print 'This is rightNode, leftChild is splitNode, get RInfo at rightNode...'
                    self.RInfo = self.calc_R(self.dataMat)
            return None, int(np.unique(yHat))
        # fit the max_depth
        if max_depth != None:
            if self.selfDepth > max_depth:
                #print 'before return leafType, let me check the value\n'
                # print '---------------------------------------------------\n'
                print 'here fit the max_depth:', self.selfDepth
                # print 'the leafType return model'
                # print '---------------------------------------------------\n'
                return None, leafType.fit(dataMat[:,:-1],dataMat[:,-1])

        # get the feature index for split
        featIndexes = bagForFeatures(max_features, n_features)
        bestError = np.inf; bestIndex = 0; bestValue = 0

        if len(featIndexes) > m:
            print ' number features > number data, shrink featIndexes'
            m = int(m)
            featIndexes = bagForFeatures(m, n_features)

        for featIndex in featIndexes:
            featVal = np.unique(dataMat[:, featIndex])
            for splitVal in np.random.choice(featVal, 0.5*len(featVal), replace=False):
                leftMat, rightMat = self.binSplitData(dataMat, featIndex, splitVal)
                if (leftMat.shape[0] < min_samples_split) or \
                    (rightMat.shape[0] < min_samples_split): 
                    # print 'fit oneside less than min_samples_split'
                    # print 'not split at current, do countinue...'
                    continue
                # fit dataMat is linear separable
                errorL_mse, errorL_reg = errType(leftMat[:, :-1],leftMat[:, -1], leafType)
                errorR_mse, errorR_reg = errType(rightMat[:, :-1], rightMat[:, -1], leafType)
                error_mse = errorL_mse + errorR_mse
                error_reg = errorL_reg + errorR_reg
                #print 'error_mse is ', error_mse
                newError = error_mse + error_reg
                if error_mse < .1:
                    Error_mes, Error_reg = errType(dataMat[:,:-1],dataMat[:,-1], leafType)
                    #print 'Error_mes is', Error_mes
                    if Error_mes < 0.1:
                        print 'current subDataSet is approxmiately linear separable, do not split'
                        return None, leafType.fit(dataMat[:,:-1],dataMat[:,-1])
                    #else:
                        #print 'oneside mse is less than threshold, but whole dataMat can not fit linear model well'

                if newError < bestError:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestError = newError
                #print '-----------------iteration-----------------------\n'
                #print 'featIndex : ',featIndex, 'FeatValue :', splitVal
                #print 'newError  : ',newError, 'bestError : ', bestError
                #print '-------------------------------------------------\n'

        # fit the min_samples_split
        leftMat, rightMat = self.binSplitData(dataMat, bestIndex, bestValue)
        if (leftMat.shape[0] < min_samples_split) or \
            (rightMat.shape[0] < min_samples_split):
           #print 'before return leafType, let me check the value\n'
            # print '---------------------------------------------------\n'
            print 'here fit oneside less than the min_samples_split :\n'
            print 'the total number of sample is ', self.n_samples
            # print 'the leafType return model'
            # print '---------------------------------------------------\n'
            return None, leafType.fit(dataMat[:,:-1],dataMat[:,-1])

        print '************ find bestSplit do return ***************\n'
        print 'bestIndex : ',bestIndex, 'bestValue :', bestValue
        print 'bestError : ', bestError
        print '---------------------------------------------------\n'
        
        #raw_input('let me see see first')
        try:
            if self.parent.rightChild is self:
                print 'This is right side which is a splitNode :', bestValue
                leftChild = self.parent.leftChild
                print 'And my left is :', leftChild.splitValue
                if isinstance(leftChild.splitValue, int):
                    print 'left Node is a class ,get RInfo at leftNode...'
                    leftChild.RInfo = leftChild.calc_R(leftChild.dataMat) 
        except AttributeError, e:
            print e

        return bestIndex, bestValue

    def createTree(self, dataMat, leafType, errType, max_depth,
                   min_samples_split, min_weight_fraction_leaf,
                   random_state, class_weight,
                   max_features):

        self.n_samples, self.n_features = dataMat[:,:-1].shape
        self.dataMat = dataMat
        featId, featVal = self.chooseBestSplit(dataMat, leafType, errType, 
                        max_depth, min_samples_split, min_weight_fraction_leaf, 
                        class_weight, max_features, self.n_features)
        if featId == None: 
            self.splitIndex = None
            self.splitValue = featVal # leaf node featVal is weights
            #self.parent.RInfo = self.parent.calc_R(self.parent.dataMat)
            if not isinstance(self.splitValue, int):
                self.RInfo = self.calc_R(self.dataMat)
        else:
            self.splitIndex = featId
            self.splitValue = featVal
            #print '-------------------- get BestSplit --------------\n'
            #print 'self.splitIndex :', self.splitIndex, 'self.splitValue : ', self.splitValue
            #print '-------------------------------------------------\n'
            leftMat, rightMat = self.binSplitData(dataMat, featId, featVal)
            #print '------------- after one split ----------------------\n'
            #print 'leftMat shape  :', leftMat.shape, 'mat samples :\n', leftMat[:3,:]
            #print 'rightMat shape :', rightMat.shape, 'mat samples :\n',rightMat[:3,:]
            #print '---------------------------------------------------\n'

           
            self.leftChild = treeNode(self)
            self.rightChild = treeNode(self)
            #print "\n***********this node's information:***************\n"
            #print "self.parent     :", self.parent
            #print "self.leftChild  :", self.leftChild
            #print 'self.rightChild :', self.rightChild
            #print 'self.splitIndex :', self.splitIndex
            #print 'self.splitValue :', self.splitValue
            #print 'self.n_samples  :', self.n_samples
            #print 'self.n_features :', self.n_features
            #print "\n**************************************************\n"
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
        X = (dataMat[:,:-1]); y = (dataMat[:, -1])
        clas = np.unique(y)
        if len(clas) != 1:
            print '---this Node is has two class to cluster :', clas
            print 'using weight data to calc_R...'
            pos_ratio = (y==0).sum() / len(y)
            neg_ratio = (y==1).sum() / len(y)
            ind = np.where(y==1)[0]
            w_X = np.r_[pos_ratio * X[ind], neg_ratio * X[~ind]]
            
            X_mean = np.mean(w_X, axis=0)
            X_radius = np.std(w_X, axis=0)
            RInfo = np.c_[X_mean, X_radius] 
            #RInfo = {'X_mean':X_mean, 'X_radius':X_radius} 
        else:
            print '---this Node is pure class :', clas
            X_mean = np.mean(X, axis=0)
            X_radius = np.std(X, axis=0)
            RInfo = np.c_[X_mean, X_radius] 
        return RInfo   # (n, 2), col0=center, col1=radius

    def get_RList(self):

        RList = []
        def get_R(self):
            if self.RInfo == None:
                if self.isTree(self.leftChild):
                    get_R(self.leftChild)
                    get_R(self.rightChild)
            else:
                RList.append(self.RInfo)
                if self.isTree(self.leftChild):
                    get_R(self.leftChild)
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

    def getTreeStruc(self, indent='    '):
        if self.leftChild and self.rightChild:
            assert(len(indent) > 0)
            print indent + 'splitIndex [%d]<%f ' % (self.splitIndex, self.splitValue )
            self.leftChild.getTreeStruc(indent + indent[:4])
            self.rightChild.getTreeStruc(indent + indent[:4])
        else:
            splitValue = self.splitValue
            if isinstance(splitValue, int):
                print indent + 'leaf node: ', splitValue
            else:
                print indent + 'leaf node: ', type(splitValue)



    def isTree(self, obj):
        return type(obj).__name__ == 'treeNode'


    def treeForeCast(self, x_test, leafType):

        if self.splitIndex==None:
            x_test = x_test.reshape(1,-1)  # (1,n)
            if isinstance(self.splitValue, int):
                prediction = self.splitValue
            else:
                prediction = self.splitValue.predict(x_test)
                prediction = (1 if prediction > 0.5 else 0)
            return prediction

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
                 leafType='RidgeReg',
                 max_depth=5,
                 min_samples_split=5,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=None,
                 class_weight=None,
                 ):
    # LEAFTYPE = {'SGDClf': SGDClf, 'LogicReg': LogicReg, 'RidgeReg': RidgeReg, 
                 # 'RANSACReg': RANSACReg, 'BayesReg': BayesReg,
                 # 'IsotonicReg': IsotonicReg, 'KernelRidge':KernelRidge
                 # }
    # ERRTYPE = {'lseErr': lseErr, 'lseErr_regul': lseErr_regul}

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



# ============================RandomForestClssification===========================

class RF_QLSVM_clf(object):


    def __init__(self, n_trees=10,
                errType='lseErr_regul',leafType='LogicReg',
                max_depth=5, min_samples_split=10, max_features=None,
                min_weight_fraction_leaf=0.0,
                random_state=None, class_weight=None,
                bootstrap_data=True):
    
        self.n_trees = n_trees
        self.errType = errType
        self.leafType = leafType
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight   
        self.bootstrap_data = bootstrap_data

        # LEAFTYPE = {'SGDClf': SGDClf, 'LogicReg': LogicReg, 'RidgeReg': RidgeReg, 
                     # 'RANSACReg': RANSACReg, 'BayesReg': BayesReg,
                     # 'IsotonicReg': IsotonicReg, 'KernelRidge':KernelRidge
                     # }
        # ERRTYPE = {'lseErr': lseErr, 'lseErr_regul': lseErr_regul}


    def fit(self, X_train, y_train):


        from sklearn.utils import resample
        trees = []
        for n in range(self.n_trees):
            trees.append(DecisionTreeRegresion(
                            errType = self.errType, # here not pass para, ERRTYPE not defined
                            leafType = self.leafType,
                            max_depth = self.max_depth,
                            min_samples_split = self.min_samples_split,
                            min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                            max_features = self.max_features,
                            random_state = self.random_state,
                            class_weight = self.class_weight)
                            )

        m,n = X_train.shape

        for i,tree in enumerate(trees):
            # get data samples
            if self.bootstrap_data:
                X_boot_train, y_boot_train = resample(X_train, y_train)
                # get oob data samples
                boot_ind = np.in1d(X_train[:,0], X_boot_train[:,0])
                X_oob_train = X_train[~boot_ind]
                y_oob_train = y_train[~boot_ind]
                tree.data_oob = np.c_[X_oob_train, y_oob_train]
                start = time()
                tree.fit(X_boot_train, y_boot_train)
                end = time() - start
                print 'done training number %d tree, ues time %f hours\n' % (i,(end/60/60))
            else:
                tree.fit(X_train, y_train)

        self.trees = trees
        self.X_train = X_train

        return self  # type is list


    def RF_predict(self, X_test):
        predictions = []  # len=m
        trees = self.trees

        for tree in trees:
            X_test_tree = X_test
            y_pred = tree.predict(X_test_tree)  #(m,1)
            predictions.append(y_pred)      

        predictions = np.array(predictions)  # (n,m,1) , n is number of trees
        avg_pred = np.mean(predictions, axis=0) #(m,1)
        sigmoid_pred = np.where(avg_pred>0.5, 1, 0)

        return sigmoid_pred


    def get_RF_avgRList_byAggloCluster(self, cluster_ratio):
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        trees = self.trees
        m,n = self.X_train.shape
        # get_RF_RList
        RF_RList=[]
        for tree in trees:

            tree_RList = tree.tree.get_RList()
            tree_RMat = np.array(tree_RList)
            # tree_new_RMat = np.zeros((tree_RMat.shape[0],n,2))
            # tree_new_RMat[:,tree.feat_ind] = tree_RMat
            RF_RList.extend(tree_RMat)   # len = m

        RF_R_Mat = np.array(RF_RList)  #(m,n,2), col0=center, col1=radius
        RF_R_centers = RF_R_Mat[:,:,0]  # (m,n)
        RF_R_radius = RF_R_Mat[:,:,1]   # (m,n)

        # get the number of cluster
        avg_num_R = int(RF_R_Mat.shape[0])  # total R divided by number trees
        # get the connectivity graph of R_list
        connect_graph = kneighbors_graph(RF_R_centers, n_neighbors=int(0.7*len(trees)), include_self=False)
        # connect_graph shape = (m,m) , if neibor then value=1, else=0
        
        if isinstance(cluster_ratio, float):
            try:
                R_cluster = AgglomerativeClustering(n_clusters=int(cluster_ratio*avg_num_R),
                                                    connectivity=connect_graph,
                                                    linkage='ward').fit(RF_R_centers)
            except ValueError,e:
                print 'ValueError ',e
                R_cluster = AgglomerativeClustering(n_clusters=int(cluster_ratio*avg_num_R)+1,
                                                    connectivity=connect_graph,
                                                    linkage='ward').fit(RF_R_centers)

        elif isinstance(cluster_ratio, int):
            R_cluster = AgglomerativeClustering(n_clusters=cluster_ratio,
                                    connectivity=connect_graph,
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







