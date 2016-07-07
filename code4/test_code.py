import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

from sklearn import svm
from __future__ import division
# import my_DecTre_clf
import my_RF_QLSVM
import my_QLSVM_RF
import get_Quasi_linear_Kernel
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

from time import time
from operator import itemgetter
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata

from functools import partial

from time import time
from operator import itemgetter
import scipy as sp
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
# Utility function to report best scores
def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# ========================== import real data ===========================
data = scipy.io.loadmat('breastdata.mat')
X = data['X']; Y = data['Y']

# data = scipy.io.loadmat('sonar.mat')
# X = data['X']; Y = data['Y']
data = fetch_mldata('sonar')
X = data['data'] ; Y = data['target']
Y  = np.where(Y==1, 1, 0)

data = fetch_mldata('glass')
X = data['data'] ; Y = data['target']
Y  = np.where(Y==1, 1, 0)

# ========================== standardize data ===========================
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

X_mean = np.mean(X,axis=0)
X_std  = np.std(X,axis=0)
X = (X - X_mean) / X_std

# ==========================QLSVM_RF testing ===============
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


myFore.get_QLSVM_RF(X_train, y_train)
y_pred = myFore.QLSVM_predict(X_test,y_test)
print '*'*35, 'current CV :','*'*35,
#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'y_test is \n', y_test
print 'accuracy_score :', metrics.precision_score(y_test, y_pred)
print 'recall_score :', metrics.recall_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)
print '*'*150

# from sklearn.metrics import precision_recall_fscore_support
# y_true = np.array([1, 1, 1, 0, 0])
# y_pred = np.array([1, 0, 0, 0, 1])
# precision_recall_fscore_support(y_true, y_pred, average='binary')


# ========================= separate boundary data ========================
X_bound,X_nonB = my_DecTre_reg.get_boundary(X,Y)
Y_b = X_bound[:,-1]
X_b = X_bound[:,:-1]
Y_nb = X_nonB[:,-1]
X_nb = X_nonB[:,:-1]

plt.figure(1)
plt.scatter(X_b[:, 0], X_b[:, 1], marker='o', c=Y_b)
plt.figure(2)
plt.scatter(X_nb[:, 0], X_nb[:, 1], marker='o', c=Y_nb)

# ========================== RF_QLSVM training and testing =====================
start = time()
myFore = my_RF_QLSVM.RF_QLSVM_clf(n_trees=3, 
                    leafType='LogicReg', errType='lseErr_regul',
                    max_depth=None, min_samples_split=2,
                    max_features='log2')
myFore.fit(X,Y)
end = time() - start

y_pred = myFore.RF_predict(X_test)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'precision :', metrics.precision_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)

# cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=5, test_size=0.3,random_state=0)
# scores = cross_validation.cross_val_score(myFore,X,Y,cv=5)
# print 'accuracy_score: %0.2f (+/-) %.2f' % (scores.mean(), scores.std()*2)

RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.9))
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

# another way to pass the kernel matrix
# K_train = Quasi_linear_kernel(X_train,X_train)
# K_test = Quasi_linear_kernel(X_test,X_train)
# clf = svm.SVC(kernel='precomputed')
# clf.fit(K_train, y_train)
# y_pred = clf.predict(K_test)

# clf = svm.SVC(kernel=Quasi_linear_kernel)
# clf.fit(X, Y)

# scatter(X_test[:,0],X_test[:,1], c=y_test)
# y_pred = clf.predict(X_test)
# print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
# print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
# print 'f1_score :', metrics.f1_score(y_test, y_pred)


################### QLSVM_RF CV training and testing ====================
start = time()
myFore = my_QLSVM_RF.QLSVM_clf_RF(n_trees=20, 
                    leafType='LogicReg', errType='lseErr_regul',
                    max_depth=None, min_samples_split=2,
                    max_features=.3)
myFore.fit(X, Y)
end = time() - start

y_pred = myFore.RF_predict(X_test)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'precision :', metrics.precision_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


skf = cross_validation.StratifiedKFold(Y, n_folds=5,
                                      shuffle=True,random_state=13)

precision_score = []
recall_score = []
f1_score = []

for train_index, test_index in skf:
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = Y[train_index], Y[test_index]

  myFore.get_QLSVM_RF(X_train, y_train)
  y_pred = myFore.QLSVM_predict(X_test,y_test)
  print '*'*35, 'current CV :''*'*35,'\n'
  #print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
  print 'y_test is \n', y_test
  print 'precision_score :', metrics.precision_score(y_test, y_pred)
  print 'recall_score :', metrics.recall_score(y_test, y_pred)
  print 'f1_score :', metrics.f1_score(y_test, y_pred)
  print '*'*150
  precision_score.append(metrics.precision_score(y_test, y_pred))
  recall_score.append(metrics.recall_score(y_test, y_pred))
  f1_score.append(metrics.f1_score(y_test, y_pred))





# =================== Grid_search hyperparameters ================

        
# specify parameters and distributions to sample from
RF_param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, 9, None],
                 "max_features": [0.3, 0.4, 0.5, 0.6, 0.7],
                 "min_samples_split": [3, 5, 7, 9],
				}

# RBF_SVM_param_dist= {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5],
#                      'C': [0.01, 0.1, 1, 10, 100, 200, 400, 500, 1000]}

RBF_SVM_param_dist= {'kernel': ['rbf'],
					'gamma': sp.stats.expon(scale=.1),
					'C': sp.stats.expon(scale=1000)}
                    # 'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30, 40, 50, 
                    # 	  70, 90, 100, 150, 200, 250, 300, 350, 400, 450,
                    # 	  500, 550, 600, 700, 800, 900, 1000]}

# Linear_SVM_param_dist = {'kernel': ['linear'], 
# 						 'C': [0.01, 0.1, 1, 10, 100, 200, 400, 500, 1000]}
Linear_SVM_param_dist = {'kernel': ['linear'], 
						 'C': sp.stats.expon(scale=1000)}

QL_SVM_param_dist= {'kernel': ['precomputed'],
					# 'gamma': sp.stats.expon(scale=.1),
					'C': sp.stats.expon(scale=1000)}
                    # 'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30, 40, 50, 
                    # 	  70, 90, 100, 150, 200, 250, 300, 350, 400, 450,
                    # 	  500, 550, 600, 700, 800, 900, 1000]}


skf = cross_validation.StratifiedKFold(Y, n_folds=3,
                                      shuffle=False,random_state=13)
# K_train = Quasi_linear_kernel(X_train,X_train)
# K_test = Quasi_linear_kernel(X_test,X_train)
K_X = Quasi_linear_kernel(X,X)
clf = svm.SVC(kernel='precomputed')
# y_pred = clf.predict(K_test)

# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=QL_SVM_param_dist,
                                   n_iter=n_iter_search, cv=skf)
start = time()
random_search.fit(K_X, Y.ravel())
print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print("Random_search Best estimator is :\n"), random_search.best_estimator_
report(random_search.grid_scores_,n_top=5)
# print the classification_report
# y_test, y_pred = y_test, random_search.predict(K_test) 
#Call predict on the estimator with the best found parameters.
# print(classification_report(y_test, y_pred))
# print()

# run grid search
grid_search = GridSearchCV(clf, param_grid=QL_SVM_param_dist)
start = time()
grid_search.fit(K_X, Y.ravel())
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
print("Grid_search Best estimator is :\n"), grid_search.best_estimator_
report(grid_search.grid_scores_,n_top=10)
# print the classification_report
y_test, y_pred = y_test, grid_search.predict(K_test)
print(classification_report(y_test, y_pred))
print()


clf = svm.SVC(kernel='rbf')
# run randomized search
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=RBF_SVM_param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X, Y.ravel())
print("RBF_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print("Random_search Best estimator is :\n"), random_search.best_estimator_
report(random_search.grid_scores_,n_top=5)
# print the classification_report
# y_test, y_pred = y_test, random_search.predict(X_test) 
#Call predict on the estimator with the best found parameters.
# print(classification_report(y_test, y_pred))
# print()
