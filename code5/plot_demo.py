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



#======================== make sin data ===================================
rng = np.random.RandomState(1)
X = np.sort(10 * rng.rand(200, 1), axis=0)
y1 = np.sin(X).ravel() + .25
y2 = np.sin(X).ravel() - .25
y1[::5] += 1 * (0.5 - rng.rand(40))
y2[::5] -= 1 * (0.5 - rng.rand(40))
y1 = y1[:, np.newaxis]
y2 = y2[:, np.newaxis]

X1 = np.c_[X, y1]
Y1 = np.ones((X1.shape[0],1))
X2 = np.c_[X, y2]
Y2 = np.zeros((X2.shape[0],1))


X = np.r_[X1,X2]
Y = np.r_[Y1,Y2]

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
Y = Y[idx]

fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(111)
pos = ax.scatter(X1[:,0], X1[:,1], c='red')
neg = ax.scatter(X2[:,0], X2[:,1], c='blue')
ax.scatter(X[:,0], X[:,1], c=Y)
ax.axis('tight')
#legend([pos, neg], ['positive sample', 'negative sample'])

#========================= make gaussian data ==============================
plt.title("Gaussian divided into three quantiles", fontsize='small')
X, Y = make_gaussian_quantiles(n_samples=500,n_features=2, n_classes=2, 
								mean=None,cov=1.0,random_state=13)

fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

#========================= training decision tree ==========================


myTree = my_RF_QLSVM.DecisionTreeRegresion(leafType='LogicReg', 
											 errType='lseErr_regul',
											 max_depth=5,
											 min_samples_split=10)
myTree.fit(X, Y)
myTree.tree.getTreeStruc()


# ========== decision tree training and testingquasi_linear SVM ==========
RMat = np.array(myTree.tree.get_RList())
RMat = np.array(myFore.trees[0].tree.get_RList())
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X, Y)


#========================== plot data face ================================
plot_step = 0.01
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

Z_svm = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
contours = plt.contourf(xx,yy,Z_svm, cmap=plt.cm.Paired)

# another way to pass the kernel matrix
K_train = Quasi_linear_kernel(X,X)
K_test = Quasi_linear_kernel(np.c_[xx.ravel(), yy.ravel()],X)
clf = svm.SVC(kernel='precomputed')
clf.fit(K_train, Y)
Z = clf.predict(K_test)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
#y_pred = clf.predict(K_test)

#==========================plot RList point =================================
kernel = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='y', s=80, label='kernel data')
legend([pos, neg, kernel], ['positive sample', 'negative sample', 'kernel data'])


# ========================== training RF =====================================
myFore = my_RF_QLSVM.RF_QLSVM_clf( n_trees=3, 
							  leafType='LogicReg', errType='lseErr_regul',
							  max_depth=5, min_samples_split=3,
							  max_features=None,
							  bootstrap_data=True)
myFore.fit(X, Y)
# =============== RF training and testing quasi_linear SVM ==============
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.8))
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)

clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X, Y)
















