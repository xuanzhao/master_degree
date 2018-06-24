import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

from sklearn import svm
from __future__ import division
import my_RF_QLSVM1
import my_RF_QLSVM2
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

# RBF_SVM_param_dist= {'kernel': ['rbf'],
# 					'gamma': sp.stats.expon(scale=.1),
# 					'C': sp.stats.expon(scale=1000)}

# Linear_SVM_param_dist = {'kernel': ['linear'], 
# 						 'C': sp.stats.expon(scale=1000)}

# QL_SVM_param_dist= {'kernel': ['precomputed'],
# 					'C': sp.stats.expon(scale=1000)}


RBF_SVM_param_dist= {'kernel': ['rbf'],
					'gamma': [0.1, 0.5, 1, 5],
          'C': [0.1, 0.5, 1, 5, 10, 20, 50]}

# Linear_SVM_param_dist = {'kernel': ['linear'], 
# 						 'C': [0.01, 0.1, 1, 10, 100, 200, 400, 500, 1000]}
Linear_SVM_param_dist = {'kernel': ['linear'], 
						 # 'C': sp.stats.expon(scale=1000)}
					'C': [0.1, 0.5, 1, 5, 10, 20, 50]}

QL_SVM_param_dist= {'kernel': ['precomputed'],
          'gamma': [0.1, 0.5, 1, 5],
          'C': [0.1, 0.5, 1, 5, 10, 20, 50]}

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

# ========================== import real data ===========================
data = scipy.io.loadmat('yeast.mat')
X_train = data['X1'] ; y_train = data['Ytrain'] ; y_train = y_train[:,6]
X_test = data['Xt']; y_test = data['Ytest']; y_test = y_test[:,6]
X = np.r_[X_train, X_test]; Y = np.r_[y_train, y_test]

X1 ,X2 = get_boundary(X_train, y_train,n_neighbors=6,radius=1)
X_train = X1[:,:-1] ; y_train = X1[:,-1]; y_train = np.array(map(int,y_train))
#X = data['X']; Y = data['Y']
#Y = Y[:,2]

# ========================== Normalize data ===========================

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                      test_size=0.50, random_state=None)

# Standard normalization
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# L2 normalization
X_train = Normalizer(norm='l2').fit_transform(X_train)
X_test = Normalizer(norm='l2').fit_transform(X_test)
X_train = X_train / np.tile(np.sqrt(np.sum(X_train*X_train,axis=1)),(436,1)).transpose()
X_test = X_test / np.tile(np.sqrt(np.sum(X_test*X_test,axis=1)),(436,1)).transpose()


#================= experiment RBF and linear SVM with 3-CV======================
accuracy_score = []; precision_score = []; recall_score = []; f1_score = []
for i in range(0,50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                          test_size=0.50, random_state=None)
    # L2 normalization
    X_train = Normalizer(norm='l2').fit_transform(X_train)
    X_test = Normalizer(norm='l2').fit_transform(X_test)
    skf = cross_validation.StratifiedKFold(y_train, n_folds=3, shuffle=True,random_state=13)
    grid_search = GridSearchCV(svm.SVC(decision_function_shape='ovr'), 
              param_grid=Linear_SVM_param_dist, 
                                       n_jobs=4, 
                                       cv=skf, scoring='accuracy', refit=True)
    start = time()
    grid_search.fit(X_train, y_train)
    print("RBF_linear kernel SVM RandomSearch took %.2f seconds"
      " parameter settings." % ((time() - start)))
    print("grid_search Best estimator is :\n"), grid_search.best_estimator_
    report(grid_search.grid_scores_,n_top=5)

    y_pred = grid_search.predict(X_test)

    accuracy_score.append(metrics.accuracy_score(y_test, y_pred, normalize=True))
    precision_score.append(metrics.precision_score(y_test, y_pred,average='binary'))
    recall_score.append(metrics.recall_score(y_test, y_pred,average='binary'))
    f1_score.append(metrics.f1_score(y_test, y_pred,average='binary'))

print '====================== Final Test score ======================\n'
print("Mean validation accuracy_score: %0.4f (std: %0.4f)" % 
   (np.mean(accuracy_score), np.std(accuracy_score)))
print("Mean validation precision_score: %0.4f (std: %0.4f)" % 
   (np.mean(precision_score), np.std(precision_score)))
print("Mean validation recall_score: %0.4f (std: %0.4f)" % 
   (np.mean(recall_score), np.std(recall_score)))
print("Mean validation f1_score: %0.4f (std: %0.4f)" % 
   (np.mean(f1_score), np.std(f1_score)))


#================= experiment Quasi-linear SVM without CV =================
RF_accuracy_score = []; RF_precision_score = []; RF_recall_score = []; RF_f1_score = []
SVM_accuracy_score = []; SVM_precision_score = []; SVM_recall_score = []; SVM_f1_score = []

for i in range(0, 2):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                          test_size=0.50, random_state=None)
    # L2 normalization
    X_train = Normalizer(norm='l2').fit_transform(X_train)
    X_test = Normalizer(norm='l2').fit_transform(X_test)
    # training randomforest
    print 'start training randomforest\n'
    start = time()
    myFore = my_RF_QLSVM2.RF_QLSVM_clf(n_trees=10, 
                        leafType='linear_SVC', errType='lseErr_regul',
                        max_depth=None, min_samples_split=10,
                        max_features='log2',bootstrap_data=True)
    myFore.fit(X_train, y_train)
    end = time() - start
    print 'done training randomforest, ues time %f hours\n' % (end/60/60)

    y_pred = myFore.RF_predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)
    precision = metrics.precision_score(y_test, y_pred,average='binary')
    recall = metrics.recall_score(y_test, y_pred,average='binary')
    f1 = metrics.f1_score(y_test, y_pred,average='binary')
    print '*'*100, 'current randomforest test result :', '*'*100,'\n'
    print 'accuracy_score :', accuracy
    print 'precision_score :', precision
    print 'recall_score :', recall
    print 'f1_score :', f1 
    print '*'*200,'\n'

    RF_accuracy_score.append(accuracy)
    RF_precision_score.append(precision)
    RF_recall_score.append(recall)
    RF_f1_score.append(f1)

    # num_R = {}
    accuracy_score = []; precision_score =[]; recall_score =[]; f1_score = []
    for i, ratio in enumerate(np.arange(.07,0.3,0.04)):
        RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(ratio))
        RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat, lamb=1)
        Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

        # training QL_SVM
        K_train = Quasi_linear_kernel(X_train,X_train)
        K_test = Quasi_linear_kernel(X_test,X_train)

        # run randomized search
        skf = cross_validation.StratifiedKFold(y_train, n_folds=3, shuffle=True,random_state=13)
        grid_search = GridSearchCV(svm.SVC(decision_function_shape='ovr'), 
                  param_grid=QL_SVM_param_dist, 
                                           n_jobs=4, 
                                           cv=skf, scoring='accuracy', refit=True)
        start = time()
        grid_search.fit(K_train, y_train)
        # print("QL_linear kernel SVM RandomSearch took %.2f seconds"
        #   " parameter settings." % ((time() - start)))
        # print("grid_search Best estimator is :\n"), grid_search.best_estimator_
        # report(grid_search.grid_scores_,n_top=5)

        y_pred = grid_search.predict(K_test)
        accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)
        precision = metrics.precision_score(y_test, y_pred,average='binary')
        recall = metrics.recall_score(y_test, y_pred,average='binary')
        f1 = metrics.f1_score(y_test, y_pred,average='binary')
        print '*'*100, 'current QL SVM test result :', '*'*100,'\n'
        print 'accuracy_score :', accuracy
        print 'precision_score :', precision
        print 'recall_score :', recall
        print 'f1_score :', f1 
        print '*'*200,'\n'
        accuracy_score.append(accuracy)
        precision_score.append(precision)
        recall_score.append(recall)
        f1_score.append(f1)

    accuracy = np.max(accuracy_score)
    precision = np.max(precision_score)
    recall = np.max(recall_score)
    f1 = np.max(f1_score)
    SVM_accuracy_score.append(accuracy)
    SVM_precision_score.append(precision)
    SVM_recall_score.append(recall)
    SVM_f1_score.append(f1)
        
        # num_R[i] = np.array([len(RMat), accuracy, precision, recall, f1])
  
print '====================== Final Test score ======================\n'
print("RF Mean test accuracy_score: %0.4f (std: %0.4f)" %
    (np.mean(RF_accuracy_score), np.std(RF_accuracy_score))) 
print("RF Mean test precision_score: %0.4f (std: %0.4f)" % 
   (np.mean(RF_precision_score), np.std(RF_precision_score)))
print("RF Mean test recall_score: %0.4f (std: %0.4f)" % 
   (np.mean(RF_recall_score), np.std(RF_recall_score)))
print("RF Mean test f1_score: %0.4f (std: %0.4f)" % 
   (np.mean(RF_f1_score), np.std(RF_f1_score)))

print("SVM Mean test accuracy_score: %0.4f (std: %0.4f)" % 
   (np.mean(SVM_accuracy_score), np.std(SVM_accuracy_score)))
print("SVM Mean test precision_score: %0.4f (std: %0.4f)" % 
   (np.mean(SVM_precision_score), np.std(SVM_precision_score)))
print("SVM Mean test recall_score: %0.4f (std: %0.4f)" % 
   (np.mean(SVM_recall_score), np.std(SVM_recall_score)))
print("SVM Mean test f1_score: %0.4f (std: %0.4f)" % 
   (np.mean(SVM_f1_score), np.std(SVM_f1_score)))



#================= experiment Quasi-linear SVM without CV =================
skf = cross_validation.StratifiedKFold(y_train, n_folds=3, shuffle=True,random_state=13)
num_R = {}

# training randomforest
print 'start training randomforest\n'
start = time()
myFore = my_RF_QLSVM2.RF_QLSVM_clf(n_trees=10, 
                    leafType='LogicReg', errType='lseErr_regul',
                    max_depth=None, min_samples_split=5,
                    max_features='log2',bootstrap_data=True)
myFore.fit(X_train, y_train)
end = time() - start
print 'done training randomforest, ues time %f hours\n' % (end/60/60)

y_pred = myFore.RF_predict(X_test)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print '*'*100, 'current randomforest test result :', '*'*100,'\n'
print 'precision_score :', precision
print 'recall_score :', recall
print 'f1_score :', f1 
print '*'*200,'\n'
RF_predict = np.array([precision, recall, f1])

for i, ratio in enumerate(np.arange(.5,1.0,0.1)):
	RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(ratio))
	RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
	Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

	# training QL_SVM
	K_train = Quasi_linear_kernel(X_train,X_train)
	K_test = Quasi_linear_kernel(X_test,X_train)

	# run randomized search
	n_iter_search = 100
	random_search = RandomizedSearchCV(svm.SVC(), 
							param_distributions=QL_SVM_param_dist,
	                        n_iter=n_iter_search, n_jobs=-1, 
	                        cv=skf, scoring='f1')
	start = time()
	random_search.fit(K_train, y_train)
	print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
	print("Random_search Best estimator is :\n"), random_search.best_estimator_
	report(random_search.grid_scores_,n_top=5)
	
	# testing QL_SVM
	C = random_search.best_params_['C']
	kernel = random_search.best_params_['kernel']
	clf = svm.SVC(kernel=kernel, C=C)
	clf.fit(K_train, y_train)
	y_pred = clf.predict(K_test)
	precision = metrics.precision_score(y_test, y_pred)
	recall = metrics.recall_score(y_test, y_pred)
	f1 = metrics.f1_score(y_test, y_pred)
	print '*'*100, 'current %d test result :' % i, '*'*100,'\n'
	#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
	print 'precision_score :', precision
	print 'recall_score :', recall
	print 'f1_score :', f1 
	print '*'*200,'\n'

	num_R[i] = np.array([len(RMat), precision, recall, f1])


#=========================== experiment RBF and linear SVM ======================
skf = cross_validation.StratifiedKFold(Y, n_folds=3, shuffle=True,random_state=13)

precision_score = []; recall_score = []; f1_score = []

for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	# run randomized search
	n_iter_search = 200
	random_search = RandomizedSearchCV(svm.SVC(), 
						param_distributions=Linear_SVM_param_dist,
	                                   n_iter=n_iter_search, n_jobs=4, 
	                                    scoring='f1')
	start = time()
	random_search.fit(X_train, y_train)
	print("RBF_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
	print("Random_search Best estimator is :\n"), random_search.best_estimator_
	report(random_search.grid_scores_,n_top=5)
	
	C = random_search.best_params_['C']
	gamma = random_search.best_params_['gamma']
	kernel = random_search.best_params_['kernel']

	clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
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
final_num_R = {}

# precision_score = []; recall_score = []; f1_score = []

for train_index, test_index in skf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	print 'down split dataSet to skf\n'
	# training randomforest
	print 'start training randomforest\n'
	start = time()
	myFore = my_RF_QLSVM.RF_QLSVM_clf(n_trees=30, 
	                    leafType='LogicReg', errType='lseErr_regul',
	                    max_depth=None, min_samples_split=5,
	                    max_features='log2',bootstrap_data=True)
	myFore.fit(X_train, y_train)
	end = time() - start
	print 'done training randomforest\n'

	for i, ratio in enumerate(np.arange(.1,.9,0.1)):
		RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(ratio))
		RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
		Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

		# training QL_SVM
		K_train = Quasi_linear_kernel(X_train,X_train)
		K_test = Quasi_linear_kernel(X_test,X_train)

		# run randomized search
		n_iter_search = 100
		random_search = RandomizedSearchCV(svm.SVC(), 
								param_distributions=QL_SVM_param_dist,
		                        n_iter=n_iter_search, n_jobs=-1, scoring='f1')
		start = time()
		random_search.fit(K_train, y_train)
		print("Quasi_linear kernel SVM RandomSearch took %.2f seconds for %d candidates"
	      " parameter settings." % ((time() - start), n_iter_search))
		print("Random_search Best estimator is :\n"), random_search.best_estimator_
		report(random_search.grid_scores_,n_top=5)
		
		# testing QL_SVM
		C = random_search.best_params_['C']
		kernel = random_search.best_params_['kernel']
		clf = svm.SVC(kernel=kernel, C=C)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(K_test)
		precision = metrics.precision_score(y_test, y_pred)
		recall = metrics.recall_score(y_test, y_pred)
		f1 = metrics.f1_score(y_test, y_pred)
		print '*'*100, 'current %d CV :' % i, '*'*100,'\n'
		#print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
		print 'precision_score :', precision
		print 'recall_score :', recall
		print 'f1_score :', f1 
		print '*'*200

		if i not in num_R:
			num_R[i] = np.array([len(RMat), precision, recall, f1])
		else:
			num_R[i] = np.vstack([num_R[i], 
							np.array([len(RMat), precision, recall,f1])])

print '================== Final validation score  =========================\n'
for i in num_R:
	final_num_R[i] = np.mean(num_R[i], axis=0)
	final_num_R[i] = np.vstack([final_num_R[i],
						np.std(num_R[i], axis=0)])


# print '============= Final validation score with %d RInfo ======================\n' % len(RMat)

# print("Mean validation precision_score: %0.3f (std: %0.03f)" % 
# 	 (np.mean(precision_score), np.std(precision_score)))

# print("Mean validation recall_score: %0.3f (std: %0.03f)" % 
# 	 (np.mean(recall_score), np.std(recall_score)))

# print("Mean validation f1_score: %0.3f (std: %0.03f)" % 
# 	 (np.mean(f1_score), np.std(f1_score)))
# print '\n======================================================================\n'

		# precision_score.append(metrics.precision_score(y_test, y_pred))
		# recall_score.append(metrics.recall_score(y_test, y_pred))
		# f1_score.append(metrics.f1_score(y_test, y_pred))
		# if len(RMat) in num_R:
		# 	if num_R[len(RMat)][2][0] < np.mean(f1_score):
		# 		num_R[len(RMat)] = [[np.mean(precision_score), np.std(precision_score)],
		# 							[np.mean(recall_score), np.std(recall_score)],
		# 							[np.mean(f1_score), np.std(f1_score)]]
		# else:
		# 	num_R[len(RMat)] = [[np.mean(precision_score), np.std(precision_score)],
		# 				[np.mean(recall_score), np.std(recall_score)],
		# 				[np.mean(f1_score), np.std(f1_score)]]


		
		
	









