from __future__ import division
import numpy as np
import matplotlib.pylab as plt

from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support as score
# ===============================================
# common function
# ===============================================

SGDClf = linear_model.SGDClassifier(loss='modified_huber',penalty='l1')

LogicReg = linear_model.LogisticRegression(penalty='l2', C=10.0)

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
            
        error = np.sum(np.power(y[:,np.newaxis] - yHat, 2)) / len(yHat)

        #yHat = model.predict_log_proba(X)
        #error = metrics.log_loss(y, yHat)
        return error
    else:
        return 0.0


def lseErr_regul(X, y, leafType, k=.5):
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
        
        X1_mean = np.mean(X[y==1], axis=0)
        X0_mean = np.mean(X[y==0], axis=0)
        X1_delta = X[y==1] - X0_mean # (m,n)
        X0_delta = X[y==0] - X1_mean
        X_delta = np.r_[X1_delta, X0_delta]
        error = (np.sum(np.power(y[:,np.newaxis] - yHat, 2))  + \
                k * np.sum(np.power(X_delta, 2)) ) / len(yHat)

        #yHat = model.predict_log_proba(X)
        #error = metrics.log_loss(y, yHat)
        return error
    else:
        return 0.0
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

def get_boundary(X, y):

    neigh = NearestNeighbors(n_neighbors=8, radius=1.0, n_jobs=4)
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
            #print 'before return leafType, let me check the value\n'
            print 'here all data is same class :',yHat
            print 'the leafType return dataMat[:,-1])', int(np.unique(yHat))
            return None, int(np.unique(yHat))
        # fit the max_depth
        if max_depth != None:
            if self.selfDepth > max_depth:
                #print 'before return leafType, let me check the value\n'
                print 'here fit the max_depth:', self.selfDepth
                print 'the leafType return value :',leafType.fit(dataMat[:,:-1],dataMat[:,-1])
                return None, leafType.fit(dataMat[:,:-1],dataMat[:,-1])

        # get the feature index for split
        featIndexes = bagForFeatures(max_features, n_features)
        bestError = np.inf; bestIndex = 0; bestValue = 0

        for featIndex in featIndexes:
            for splitVal in np.unique(dataMat[:, featIndex]):
                leftMat, rightMat = self.binSplitData(dataMat, featIndex, splitVal)
                if (leftMat.shape[0] < min_samples_split) or \
                    (rightMat.shape[0] < min_samples_split): continue
                newError = errType(leftMat[:, :-1],leftMat[:, -1], leafType) + \
                           errType(rightMat[:, :-1], rightMat[:, -1], leafType)

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
            print 'here fit the min_samples_split :',self.n_samples
            print 'the leafType return value :',leafType.fit(dataMat[:,:-1],dataMat[:,-1])
            return None, leafType.fit(dataMat[:,:-1],dataMat[:,-1])

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
            self.splitValue = featVal # leaf node featVal is model
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
                # if self.leftChild.splitIndex !=None:
                #     get_R(self.leftChild)
                # if self.rightChild.splitIndex !=None:
                #     get_R(self.rightChild)
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



# ============================RandomFrestClassification===========================

class QLSVM_clf_RF(object):

    def __init__(self, n_trees=10,
                errType='lseErr_regul',leafType='LogicReg',
                max_depth=5, min_samples_split=10, max_features=None,
                min_weight_fraction_leaf=0.0,
                random_state=None, class_weight=None):
    
        self.n_trees = n_trees
        self.errType = errType
        self.leafType = leafType
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight    


    def fit(self, X_train, y_train):
        
        # LEAFTYPE = {'SGDClf': SGDClf, 'LogicReg': LogicReg, 'RidgeReg': RidgeReg, 
                     # 'RANSACReg': RANSACReg, 'BayesReg': BayesReg,
                     # 'IsotonicReg': IsotonicReg, 'KernelRidge':KernelRidge
                     # }
        # ERRTYPE = {'lseErr': lseErr, 'lseErr_regul': lseErr_regul}

        from sklearn.utils import resample
        trees = []
        for n in range(self.n_trees):
            trees.append(DecisionTreeRegresion(
                         errType=self.errType,
                         leafType=self.leafType,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                         max_features=self.max_features,
                         random_state=self.random_state,
                         class_weight=self.class_weight)
                        )
        m,n = X_train.shape
        data_oob_List = []
        
        for tree in trees:

            # get random features
            feat_ind = np.sort(np.random.choice(n, int(np.log2(n)+1), replace=False))
            # get data samples
            X_boot_train, y_boot_train = resample(X_train[:,feat_ind], y_train)
            
            # get oob data samples
            boot_ind = np.in1d(X_train[:,0], X_boot_train[:,0])
            X_oob_train = X_train[~boot_ind][:,feat_ind]
            y_oob_train = y_train[~boot_ind]
            data_oob_List.append(np.c_[X_oob_train, y_oob_train])

            tree.feat_ind = feat_ind
            tree.fit(X_boot_train, y_boot_train)

        self.trees = trees
        self.data_oob_List = data_oob_List  # each element is a array
        return self  # type is list

    def RF_predict(self, X_test):
        predictions = []  # len=m
        trees = self.trees

        for tree in trees:
            X_test_tree = X_test[:,tree.feat_ind]
            y_pred = tree.predict(X_test_tree)  #(m,1)
            predictions.append(y_pred)      

        predictions = np.array(predictions)  # (n,m,1) , n is number of trees
        avg_pred = np.mean(predictions, axis=0) #(m,1)
        sigmoid_pred = np.where(avg_pred>0.5, 1, 0)
        
        return sigmoid_pred


    def get_QLSVM_RF(self, X_train, y_train, lamb=2):

        import get_Quasi_linear_Kernel
        from functools import partial
        from sklearn.svm import SVC
        import scipy as sp
        from sklearn.grid_search import RandomizedSearchCV

        trees = self.trees
        RBFinfo_list = []
        QLSVM_List = []
        f1_scores = []
        data_oob_List = self.data_oob_List
        QL_SVM_param_dist= {'kernel': ['precomputed'],
                        'C': sp.stats.expon(scale=1000)}

        RF_R_clus_List = []
        # get QLSVM_List and RF_weights
        for i,tree in enumerate(trees):

            # get tree's data with it's feature
            X_train_tree = X_train[:,tree.feat_ind]
            # get tree's cluster RMat
            R_clus_List = self.get_RList_byAggloCluster(tree.tree.get_RList())
            RF_R_clus_List.append(R_clus_List)
            RMat = np.array(R_clus_List) # (m,n,2)
            # get QL kernel matrix
            RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat,lamb=lamb)
            Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)
            K_train_tree = Quasi_linear_kernel(X_train_tree,X_train_tree) # for training SVM
            # run randomized search get best QL SVM
            clf = SVC(kernel='precomputed')
            n_iter_search = 100
            random_search = RandomizedSearchCV(clf, param_distributions=QL_SVM_param_dist,
                                           n_iter=n_iter_search)
            random_search.fit(K_train_tree, y_train)
            # print("Random_search Best estimator is :\n"), random_search.best_estimator_
            QLSVM_List.append(random_search.best_estimator_)


            # get oob test data, K_oob kernel matrix 
            data_oob = data_oob_List[i]
            X_oob = data_oob[:,:-1]; y_oob = data_oob[:,-1]
            K_oob = Quasi_linear_kernel(X_oob,X_train_tree)     # for get SVM weight
            oob_pred = random_search.best_estimator_.predict(K_oob) # (m,1)

            precision, recall, fscore, support = score(y_oob, oob_pred,average='binary')
            print '\nQLSVM number %d get training score:\n' % i
            print 'precision: {}'.format(precision)
            print 'recall: {}'.format(recall)
            print 'fscore: {}'.format(fscore)
            print '\n'
            #clf_weight = metrics.f1_score(y_oob, oob_pred)+0.01
            #raw_input('for the check')
            f1_scores.append(fscore)


        # standarize f1_scores which is RF_weights
        f1_scores = np.array(f1_scores)
        sum_f1_score = np.sum(f1_scores)
        weights = np.true_divide(f1_scores, sum_f1_score)
        RF_weights = np.nan_to_num(weights)
        self.RF_weights = RF_weights
        #self.RF_weights = np.ones(len(RF_weights)) / float(len(RF_weights))
        print '*'*100
        print 'done get trees_weights :', RF_weights # np.array,(m_tree) 
        print '*'*100

        self.QLSVM_List = QLSVM_List
        self.QLSVM_lamb = lamb
        self.QLSVM_X_train = X_train
        print '*'*100
        print 'done get QLSVM_List : '#, QLSVM_List 
        print '*'*100

        self.RF_R_clus_List = RF_R_clus_List
        print 'done get RF_R_clus_List shape is ' ,np.array(self.RF_R_clus_List).shape


    def QLSVM_predict(self, X_test, y_test):
        '''QLSVM_predict
        '''
        import get_Quasi_linear_Kernel
        from functools import partial
        from sklearn.svm import SVC

        QLSVM_List = self.QLSVM_List
        RF_weights = self.RF_weights
        X_train = self.QLSVM_X_train
        lamb = self.QLSVM_lamb
        RF_R_clus_List = self.RF_R_clus_List
        trees = self.trees

        y_pred = np.zeros((len(X_test),len(QLSVM_List)))

        for i,clf in enumerate(QLSVM_List):

            # get tree's test data with it's features
            X_test_tree = X_test[:,trees[i].feat_ind]
            X_train_tree = X_train[:,trees[i].feat_ind]

            RMat = np.array(RF_R_clus_List[i]) # (m,n,2)
            RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat,lamb=lamb)
            Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

            K_test_tree = Quasi_linear_kernel(X_test_tree,X_train_tree)
            y_pred[:,i] = clf.predict(K_test_tree)
            #print y_pred[:,i] 
            #y_pred[:, i] = QLSVM_List[i].predict(K_test_tree)
            precision, recall, fscore, support = score(y_test, y_pred[:,i],average='binary')
            print '\nQLSVM number %d get test score:\n' % i
            print 'precision: {}'.format(precision)
            print 'recall: {}'.format(recall)
            print 'fscore: {}'.format(fscore)
            print '\n'
            #raw_input('for the check')
            y_pred[:,i] = y_pred[:,i] * RF_weights[i] 

        final_y_pred_prob = np.sum(y_pred, axis=1) 
        print 'final_y_pred_prob is \n',final_y_pred_prob
        final_y_pred = np.where(final_y_pred_prob>=0.5, 1, 0)

        return final_y_pred

    def get_RList_byAggloCluster(self, RList):
    
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph


        R_Mat = np.array(RList)  #(m,n,2), col0=center, col1=radius
        R_centers = R_Mat[:,:,0]  # (m,n)
        R_radius = R_Mat[:,:,1]   # (m,n)

        # get the connectivity graph of R_list
        #connect_graph = kneighbors_graph(RF_R_centers, n_neighbors=int(np.sqrt(len(trees)-1)), include_self=False)
        # connect_graph shape = (m,m) , if neibor then value=1, else=0
        #0.55*R_Mat.shape[0]
        connect_graph = kneighbors_graph(R_centers, n_neighbors=int(np.log2(len(R_centers))), include_self=False)

        try:
            R_cluster = AgglomerativeClustering(n_clusters=int(R_Mat.shape[0]*np.random.rand()*0.8)-5,
                                                connectivity=connect_graph,
                                                linkage='ward').fit(R_centers)
        except ValueError,e:
            print 'ValueError ',e
            R_cluster = AgglomerativeClustering(n_clusters=int(np.log2(R_Mat.shape[0])),
                                                connectivity=connect_graph,
                                                linkage='ward').fit(R_centers)
        #get_RF_avgRList(R_cluster):
        R_cluster_label = R_cluster.labels_
        R_cluster_List = []

        for label in np.unique(R_cluster_label):
            R_mean  = np.mean(R_centers[R_cluster_label == label], axis=0)
            R_radi = np.mean(R_radius[R_cluster_label == label], axis=0)

            R = np.c_[R_mean, R_radi] # shape (n,2)
            R_cluster_List.append(R)

    
        return R_cluster_List   # type is list, len=m




