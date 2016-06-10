import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from __future__ import division
# import my_DecTre_clf
import my_RF_QLSVM
import my_QLSVM_RF
import get_Quasi_linear_Kernel
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn import cross_validation

from time import time
from operator import itemgetter
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

# ========================= generate data ============================

plt.subplot(311)
plt.title("One informative feature, one cluster per class", fontsize='small')
X, Y = make_classification(n_samples=2000,n_features=8, n_redundant=0, n_informative=8,
                             n_clusters_per_class=4,random_state=13)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

plt.subplot(312)
plt.title("Two informative features, one cluster per class", fontsize='small')
X, Y = make_classification(n_samples=300,n_features=3, n_redundant=0, n_informative=3,
                             n_clusters_per_class=2,random_state=13)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, cmap=plt.cm.Paired)

plt.subplot(313)
plt.title("Gaussian divided into three quantiles", fontsize='small')
X, Y = make_gaussian_quantiles(n_samples=500,n_features=2, n_classes=2, 
								mean=None,cov=1.0,random_state=13)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

# ========================== import real data ===========================
data = scipy.io.loadmat('breastdata.mat')
X = data['X']; Y = data['Y']

data = scipy.io.loadmat('sonar.mat')
X = data['X']; Y = data['Y']

# ========================== standardize data ===========================
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

X_mean = np.mean(X,axis=0)
X_std  = np.std(X,axis=0)
X = (X - X_mean) / X_std
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# ========================= separate boundary data ========================
X_bound,X_nonB = get_boundary(X,Y)
Y_b = X_bound[:,-1]
X_b = X_bound[:,:-1]
Y_nb = X_nonB[:,-1]
X_nb = X_nonB[:,:-1]

plt.figure(1)
plt.scatter(X_b[:, 0], X_b[:, 1], marker='o', c=Y_b)
plt.figure(2)
plt.scatter(X_nb[:, 0], X_nb[:, 1], marker='o', c=Y_nb)

#========================== plot data face ================================
plot_step = 0.1
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myTree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5],linewidths=2)
contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)

#==========================plot RList point =================================
plt.scatter(RMat[:,0,0], RMat[:,0,1], marker='o', c=Y_nb)

# ========================= training decision Tree ===========================
myTree = my_RF_QLSVM.DecisionTreeRegresion(leafType='LogicReg', 
											 errType='lseErr_regul',
											 max_depth=5,
											 min_samples_split=3)
myTree.fit(X, Y)
y_pred = myTree.predict(X_test)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'precision_score :', metrics.precision_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)

# ========== decision tree training and testingquasi_linear SVM ==========
RMat = np.array(myTree.tree.get_RList())
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)


clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X_train, y_train)

# scatter(X_test[:,0],X_test[:,1], c=y_test)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'precision_score :', metrics.precision_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


# ========================== training RF =====================================
myFore = my_DecTre_reg.RF_fit(X_train, y_train, n_trees=10, 
							  leafType='LogicReg', errType='lseErr_regul',
							  max_depth=5, min_samples_split=3,
							  max_features=.2)
y_pred = my_DecTre_reg.RF_predict(X_test, myFore)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


# =============== RF training and testing quasi_linear SVM ==============
RMat = np.array(my_DecTre_reg.get_RF_avgRList_byAggloCluster(myFore))
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

# another way to pass the kernel matrix
# K_train = Quasi_linear_kernel(X_train,X_train)
# K_test = Quasi_linear_kernel(X_test,X_train)
# clf = svm.SVC(kernel='precomputed')
# clf.fit(K_train, y_train)
# y_pred = clf.predict(K_test)

clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X_train, y_train)

# scatter(X_test[:,0],X_test[:,1], c=y_test)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


################### QLSVM_RF CV training and testing ====================
myFore = my_QLSVM_RF.QLSVM_clf_RF(n_trees=10, 
							  	  leafType='LogicReg', errType='lseErr_regul',
							  	  max_depth=5, min_samples_split=3,
							  	  max_features=.2)
myFore.fit(X, Y)

skf = cross_validation.StratifiedKFold(Y, n_folds=3)

for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	myFore.get_QLSVM_RF(X_train, y_train)
	y_pred = myFore.QLSVM_predict(X_test)
	print '*'*100, 'current CV :', '*'*100,'\n'
	#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
	print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
	print 'recall_score :', metrics.recall_score(y_test, y_pred)
	print 'f1_score :', metrics.f1_score(y_test, y_pred)
	print '*'*200

# ================= training and testing RBF SVM ========================

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)

# ================= training and testing poly SVM ========================

clf = svm.SVC(kernel='poly',degree=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)

# ========================= plot learning_curve ==========================


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

	from sklearn.learning_curve import learning_curve
	from sklearn import cross_validation

	if ylim is not None:
		plt.ylim(*ylim)

	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
	    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	
	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score")
	
	plt.legend(loc="best")
	return plt

title = "Learning Curves (SVM, RBF kernel)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='rbf')
plot_learning_curve(estimator, title, X, Y, (0.7, 1.01), cv=cv, n_jobs=2)
plt.show()


title = "Learning Curves (SVM, Quasi_linear kernel)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)
estimator = svm.SVC(kernel='precomputed')
K_Matrix = Quasi_linear_kernel(X,X)
plot_learning_curve(estimator, title, K_Matrix, Y, (0.7, 1.01), cv=cv, n_jobs=2)
plt.show()

#===================== plot validation_curve ==========================

def plot_validation_curve(estimator, title, X, y, param_name, param_range,
							cv=10, scoring='accuracy', n_jobs=2):
	from sklearn.learning_curve import validation_curve
	train_scores, test_scores = validation_curve(
	    estimator, X, y, param_name, param_range,
	    cv=cv, scoring=scoring, n_jobs=n_jobs)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.figure()
	plt.title(title)
	plt.xlabel(param_name)
	plt.ylabel("Score")
	plt.ylim(0.0, 1.1)
	plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
	             color="g")
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
	plt.legend(loc="best")
	plt.show()

title = "Validation Curve with RBF kernel SVM"
param_name="gamma"
param_range = np.logspace(-6, 0, 20)
estimator = svm.SVC(kernel='rbf')
plot_validation_curve(estimator, title, X, Y, param_name, param_range)


title = "Validation Curve with RBF kernel SVM"
param_name="C"
param_range = np.logspace(-3, 2.5, 20)
estimator = svm.SVC(kernel='rbf')
plot_validation_curve(estimator, title, X, Y, param_name, param_range)

title = 'Validation Curve with Quasi_linear kernel SVM'
param_name='C'
param_range = np.logspace(-3, 2.5, 20)
estimator = svm.SVC(kernel='precomputed')
K_X = Quasi_linear_kernel(X,X)
plot_validation_curve(estimator, title, K_X, Y, param_name, param_range)


# =================== Grid_search hyperparameters ================
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
        
# specify parameters and distributions to sample from
RF_param_dist = {"max_depth": [3, 4, 5, 6, 7, 8, 9, None],
                 "max_features": [0.3, 0.4, 0.5, 0.6, 0.7],
                 "min_samples_split": [3, 5, 7, 9],
				}

RBF_SVM_param_dist= {'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3,1e-4,1e-5],
                     'C': [0.01, 0.1, 1, 10, 100, 200, 400, 500, 1000]}

# Linear_SVM_param_dist = {'kernel': ['linear'], 
# 						 'C': [0.01, 0.1, 1, 10, 100, 200, 400, 500, 1000]}
Linear_SVM_param_dist = {'kernel': ['linear'], 
						 'C': sp.stats.expon(scale=1000)}


QL_SVM_param_dist= {'kernel': ['precomputed'],
					'C': sp.stats.expon(scale=1000)}
                    # 'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30, 40, 50, 
                    # 	  70, 90, 100, 150, 200, 250, 300, 350, 400, 450,
                    # 	  500, 550, 600, 700, 800, 900, 1000]}


K_train = Quasi_linear_kernel(X_train,X_train)
K_test = Quasi_linear_kernel(X_test,X_train)
K_X = Quasi_linear_kernel(X,X)
clf = svm.SVC(kernel='precomputed')
# y_pred = clf.predict(K_test)

# run randomized search
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=QL_SVM_param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(K_train, y_train)
print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print("Random_search Best estimator is :\n"), random_search.best_estimator_
report(random_search.grid_scores_,n_top=5)
# print the classification_report
y_test, y_pred = y_test, random_search.predict(K_test) 
#Call predict on the estimator with the best found parameters.
print(classification_report(y_test, y_pred))
print()

# run grid search
grid_search = GridSearchCV(clf, param_grid=QL_SVM_param_dist)
start = time()
grid_search.fit(K_X, Y)
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
n_iter_search = 500
random_search = RandomizedSearchCV(clf, param_distributions=RBF_SVM_param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X_train, y_train)
print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print("Random_search Best estimator is :\n"), random_search.best_estimator_
report(random_search.grid_scores_,n_top=5)
# print the classification_report
y_test, y_pred = y_test, random_search.predict(X_test) 
#Call predict on the estimator with the best found parameters.
print(classification_report(y_test, y_pred))
print()


#=================== plot_forest_feature importance ====================

from sklearn.ensemble import ExtraTreesClassifier


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()












