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


import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll

#======================== make sin data ===================================
rng = np.random.RandomState(1)
X = np.sort(7 * rng.rand(200, 1), axis=0)
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
pos = ax.scatter(X1[:,0], X1[:,1], c='red' , label='pos data')
neg = ax.scatter(X2[:,0], X2[:,1], c='blue' , label='neg data')
ax.scatter(X[:,0], X[:,1], c=Y)
ax.axis('tight')
#legend([pos, neg], ['positive sample', 'negative sample'])


# =================== make imbalance data =================================
N2 = 200;
X2 = 10*np.random.rand(N2, 2) - 5;
y2 = np.zeros((N2, 1));
# y2[(X2[:, 0]**3 + X2[:, 0]**2 + X2[:, 0]+1)/40 > X2[:, 1]] = 1;
y2[ np.logical_and(X2[:,0]<-3, X2[:,1]<-3)] = 1
fig= plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X2[:,0], X2[:,1], c=y2)

pos_ratio = (y2==0).sum() / len(y2)
neg_ratio = (y2==1).sum() / len(y2)
ind = np.where(y2==1)[0]
w_X2 = np.r_[pos_ratio * X2[ind], neg_ratio * X2[~ind]]

X_mean = np.mean(w_X2, axis=0)
kernel = ax.scatter(X_mean[0], X_mean[1], c='y', s=80)

# =================== make cross data =================================
N = 400;
X = 10*np.random.rand(N, 2) - 5;
Y = np.zeros((N, 1));
# y2[(X2[:, 0]**3 + X2[:, 0]**2 + X2[:, 0]+1)/40 > X2[:, 1]] = 1;
down = -4; up = -2
while(up<7.0):
	left = -4; right = -2
	Y[ np.logical_and(np.logical_and(X[:,0]>left,X[:,0]<right),
					np.logical_and(X[:,1]>down,X[:,1]<up))] = 1
	while(right<7.0):
		Y[ np.logical_and(np.logical_and(X[:,0]>left,X[:,0]<right),
						np.logical_and(X[:,1]>down,X[:,1]<up))] = 1
		left +=4; right +=4
	down +=4; up +=4
fig= plt.figure()
ax = fig.add_subplot(111)
ind_pos = np.where(Y==1)[0]
ind_neg = np.where(Y==0)[0]
pos = ax.scatter(X[ind_pos,0], X[ind_pos,1], c='red' , label='pos data')
neg = ax.scatter(X[ind_neg,0], X[ind_neg,1], c='blue' , label='neg data')
ax.axis('tight')

# X_train, X_test, y_train, y_test = train_test_split(X, Y,
#                                             test_size=0.33, random_state=13)
myFore = my_RF_QLSVM1.RF_QLSVM_clf(n_trees=1, 
                    leafType='LogicReg', errType='lseErr_regul',
                    max_depth=None, min_samples_split=5,
                    max_features=None,bootstrap_data=False)
myFore.fit(X, Y)
y_pred = myFore.RF_predict(X)
y_pred = y_pred.reshape((-1,))

fig= plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], 
			  color='r', label='predict pos')
ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], 
			  color='b', label='predict neg')
ax.axis('tight')
plt.title('5 decision trees predict result')


#RMat = np.array(myFore.trees[0].tree.get_RList())
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.1))
kernel = ax.scatter(RMat[:,0,0], RMat[:,1,0],
		marker='.', c='y', s=140, label='clusterd kernel data', alpha=0.7)
ax.legend()

from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix_basic,RMat=RMat)
K_train = Quasi_linear_kernel(X,X)
K_test = Quasi_linear_kernel(X,X)
clf = svm.SVC(kernel='precomputed' , C=1000)
clf.fit(K_train, Y)
y_pred = clf.predict(K_test)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], 
			  color='r', label='predict pos')
ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], 
			  color='b', label='predict neg')
ax.axis('tight')
ax.legend()
plt.title('Quasi_Linear SVM predict result')

#========================= make gaussian data ==============================
plt.title("Gaussian divided into three quantiles", fontsize='small')
X, Y = make_gaussian_quantiles(n_samples=500,n_features=2, n_classes=2, 
								mean=None,cov=1.0,random_state=13)

fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
ax.axis('tight')


#========================= make swiss roll data ============================
# Generate data (swiss roll dataset)
n_samples = 1500
noise = 0.05
X, t = make_swiss_roll(n_samples, noise)
X[:, 1] *= .5

from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
label = ward.labels_
Y= label
plt.style.use('ggplot')
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
              'o', color=plt.cm.jet(float(l) / np.max(label + 1)))


Y  = np.where(Y==1, 1, 0)
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2],
          'o', color='r', alpha=0.5, label='pos data')
ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2],
          'x', color='b', alpha=0.5, label='neg data')
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                            test_size=0.33, random_state=13)
# ==================== trainning random forest ==========================
myFore = my_RF_QLSVM.RF_QLSVM_clf(n_trees=1, 
                    leafType='LogicReg', errType='lseErr_regul',
                    max_depth=None, min_samples_split=5,
                    max_features=None,bootstrap_data=False)
myFore.fit(X_train, y_train)
y_pred = myFore.RF_predict(X_test)
y_pred = y_pred.reshape((-1,))

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot3D(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], 
			X_test[y_pred == 1, 2], 'o', color='c', label='predict pos')
ax.plot3D(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], 
			X_test[y_pred == 0, 2], 'x', color='c', label='predict neg')

# RMat = np.array(myFore.trees[0].tree.get_RList())
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.5))
kernel = ax.scatter(RMat[:,0,0], RMat[:,1,0], RMat[:,2,0] ,
		marker='.', c='y', s=140, label='clusterd kernel data')

ax.plot3D(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2],
          'o', color='r', alpha=0.3)
ax.plot3D(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2],
          'x', color='b', alpha=0.3)
ax.plot3D(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2],
          'o', color='r', alpha=0.3)
ax.plot3D(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2],
          'x', color='b', alpha=0.3)
myFore.trees[0].tree.getTreeStruc()


from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix_basic,RMat=RMat)
K_train = Quasi_linear_kernel(X_train,X_train)
K_test = Quasi_linear_kernel(X_test,X_train)
clf = svm.SVC(kernel='precomputed' , C=100)
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot3D(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], 
			X_test[y_pred == 1, 2], 'o', color='c', label= 'predict pos')
ax.plot3D(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], 
			X_test[y_pred == 0, 2], 'x', color='c', label= 'predict neg')
#========================= training decision tree ==========================


myTree = my_RF_QLSVM.DecisionTreeRegresion(leafType='LogicReg', 
											 errType='lseErr_regul',
											 max_depth=3,
											 min_samples_split=5)
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
clf = svm.SVC(kernel='precomputed' , C=100)
clf.fit(K_train, Y )
Z = clf.predict(K_test)
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired, alpha=0.1)
ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5],linewidths=2, label='quasi_linear SVM')
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
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.5))
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix_basic,RMat=RMat)

clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X, Y)


# =========================================================================
myFore = my_RF_QLSVM.RF_QLSVM_clf( n_trees=3, 
                              leafType='LogicReg', errType='lseErr_regul',
                              max_depth=3, min_samples_split=5,
                              max_features=None,
                              bootstrap_data=True)
myFore.fit(X, Y)
fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(111)
pos = ax.scatter(X1[:,0], X1[:,1], c='red' , label='pos data')
neg = ax.scatter(X2[:,0], X2[:,1], c='blue' , label='neg data')
ax.scatter(X[:,0], X[:,1], c=Y)
ax.axis('tight')
RMat = np.array(myFore.trees[0].tree.get_RList())
kernel1 = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='y', s=60, alpha=0.5,label='tree 1 kernel')
RMat = np.array(myFore.trees[1].tree.get_RList())
kernel2 = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='g', s=60, alpha=0.5,label='tree 2 kernel')
RMat = np.array(myFore.trees[2].tree.get_RList())
kernel3 = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='r', s=60, alpha=0.5,label='tree 3 kernel')
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(1.0))
kernel = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='k', s=100, label='clustered kernel')
ax.legend()
plt.title('Extract kernel data based on RandomForest')









,