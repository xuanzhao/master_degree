import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics

from sklearn import cross_validation

from sklearn import svm
from __future__ import division
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
Y  = np.where(Y==6, 1, 0)

data = fetch_mldata('Pima')
X = data['data'] ; Y = data['target']


data = scipy.io.loadmat('yeast1.mat')
# X_train = data['X1'] ; y_train = data['Ytrain']
# X_test = data['Xt']; y_test = data['Ytest']
# X = np.r_[X_train, X_test]; Y = np.r_[y_train, y_test]
X = data['X']; Y = data['Y']
Y = Y[:,1]

#=========================== experiment RBF and linear SVM ======================
skf = cross_validation.StratifiedKFold(Y, n_folds=3, shuffle=True,random_state=13)

precision_score = []; recall_score = []; f1_score = []

for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	clf = svm.SVC(kernel='rbf')
	# run randomized search
	n_iter_search = 100
	random_search = RandomizedSearchCV(clf, param_distributions=RBF_SVM_param_dist,
	                                   n_iter=n_iter_search, n_jobs=4, scoring='f1')
	start = time()
	random_search.fit(X_train, y_train)
	print("RBF_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
	print("Random_search Best estimator is :\n"), random_search.best_estimator_
	report(random_search.grid_scores_,n_top=5)
	
	y_pred = random_search.predict(X_test)
	print '*'*100, 'current CV :', '*'*100,'\n'
	#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
	print 'precision_score :', metrics.precision_score(y_test, y_pred)
	print 'recall_score :', metrics.recall_score(y_test, y_pred)
	print 'f1_score :', metrics.f1_score(y_test, y_pred)
	print '*'*200
	precision_score.append(metrics.precision_score(y_test, y_pred))
	recall_score.append(metrics.recall_score(y_test, y_pred))
	f1_score.append(metrics.f1_score(y_test, y_pred))

print '====================== Final validation score ======================\n'
print("Mean validation precision_score: %0.3f (std: %0.03f)" % 
	 (np.mean(precision_score), np.std(precision_score)))
print("Mean validation recall_score: %0.3f (std: %0.03f)" % 
	 (np.mean(recall_score), np.std(recall_score)))
print("Mean validation f1_score: %0.3f (std: %0.03f)" % 
	 (np.mean(f1_score), np.std(f1_score)))


#=========================== experiment Quasi-linear SVM ======================
skf = cross_validation.StratifiedKFold(Y, n_folds=3, shuffle=True,random_state=13)
num_R = {}

for ratio in np.arange(0.1,0.9,0.1):

	precision_score = []; recall_score = []; f1_score = []

	for train_index, test_index in skf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]

		# training randomforest
		start = time()
		myFore = my_RF_QLSVM.RF_QLSVM_clf(n_trees=3, 
		                    leafType='LogicReg', errType='lseErr_regul',
		                    max_depth=None, min_samples_split=5,
		                    max_features='log2',bootstrap_data=True)
		myFore.fit(X_train, y_train)
		end = time() - start

		# get R information
		RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(ratio))
		RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
		Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

		# training QL_SVM
		clf = svm.SVC(kernel='precomputed')
		K_train = Quasi_linear_kernel(X_train,X_train)
		K_test = Quasi_linear_kernel(X_test,X_train)

		# run randomized search
		n_iter_search = 50
		random_search = RandomizedSearchCV(clf, param_distributions=QL_SVM_param_dist,
		                                   n_iter=n_iter_search, n_jobs=4, scoring='f1')
		start = time()
		random_search.fit(K_train, y_train)
		print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
	      " parameter settings." % ((time() - start), n_iter_search))
		print("Random_search Best estimator is :\n"), random_search.best_estimator_
		report(random_search.grid_scores_,n_top=5)
		
		# testing QL_SVM
		y_pred = random_search.predict(K_test)
		print '*'*100, 'current CV :', '*'*100,'\n'
		#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
		print 'precision_score :', metrics.precision_score(y_test, y_pred)
		print 'recall_score :', metrics.recall_score(y_test, y_pred)
		print 'f1_score :', metrics.f1_score(y_test, y_pred)
		print '*'*200

		precision_score.append(metrics.precision_score(y_test, y_pred))
		recall_score.append(metrics.recall_score(y_test, y_pred))
		f1_score.append(metrics.f1_score(y_test, y_pred))

	print '============= Final validation score with %d RInfo ======================\n' % len(RMat)
	print("Mean validation precision_score: %0.3f (std: %0.03f)" % 
		 (np.mean(precision_score), np.std(precision_score)))

	print("Mean validation recall_score: %0.3f (std: %0.03f)" % 
		 (np.mean(recall_score), np.std(recall_score)))

	print("Mean validation f1_score: %0.3f (std: %0.03f)" % 
		 (np.mean(f1_score), np.std(f1_score)))

	num_R[len(RMat)] = [[np.mean(precision_score), np.std(precision_score)],
						[np.mean(recall_score), np.std(recall_score)],
						[np.mean(f1_score), np.std(f1_score)]]


	
		
	









