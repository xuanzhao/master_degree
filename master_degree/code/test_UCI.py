import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

from sklearn import svm
from __future__ import division
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


# ========================== import real data ===========================
data = scipy.io.loadmat('./data/breastdata.mat')
X = data['X']; Y = data['Y'].ravel()

# data = scipy.io.loadmat('sonar.mat')
# X = data['X']; Y = data['Y']

data = scipy.io.loadmat('./data/uci-20070111-breast-w.mat')
X = data['int0'].transpose() ; Y = data['Class'].transpose()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['benign','malignant'])
Y = le.transform(Y)
Y = Y.ravel()

data = fetch_mldata('glass')
X = data['data'] ; Y = data['target']
Y  = np.where(Y==1, 1, 0)
# Y  = np.where(Y==3, 1, 0)

data = np.loadtxt('./data/banknote.txt', delimiter=',')
X = data[:,:-1] ; Y = data[:,-1]
Y = Y.astype(np.int)

data = pd.read_csv('./data/yeast.txt', delimiter='\s+', header=None)
data = data.iloc[:,1:]
data[9] =np.where(data[9] == 'MIT', 1, 0)
X = data.iloc[:,:-1] ; Y = data.iloc[:,-1]

data = pd.read_csv('./data/yeast.txt', delimiter='\s+', header=None)
data = data.iloc[:,1:]
data[9] =np.where(data[9] == 'NUC', 1, 0)
X = data.iloc[:,:-1] ; Y = data.iloc[:,-1]

data = pd.read_csv('./data/yeast.txt', delimiter='\s+', header=None)
data = data.iloc[:,1:]
data[9] =np.where(data[9] == 'CYT', 1, 0)
X = data.iloc[:,:-1] ; Y = data.iloc[:,-1]

data = pd.read_csv('./data/yeast.txt', delimiter='\s+', header=None)
data = data.iloc[:,1:]
data[9] =np.where(data[9] == 'ME3', 1, 0)
X = data.iloc[:,:-1] ; Y = data.iloc[:,-1]

data = fetch_mldata('sonar')
X = data['data'] ; Y = data['target']
Y  = np.where(Y==1, 1, 0)

data = pd.read_csv('./data/arrhythmia.txt', delimiter=',', header=None, na_values='?')
data = data.fillna(data.mean(axis=0),axis=0)
X = data.iloc[:,:-1] ; Y = data.iloc[:,-1]
Y  = np.where(Y==01, 1, 0)

data = np.loadtxt('./data/haberman.txt', delimiter=',')
X = data[:,:-1] ; Y = data[:,-1]
Y = Y.astype(np.int)
Y  = np.where(Y==1, 1, 0)

# data = fetch_mldata('Pima')
# X = data['data'] ; Y = data['target']
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
