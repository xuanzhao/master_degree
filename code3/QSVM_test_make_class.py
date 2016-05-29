import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from __future__ import division
# import my_DecTre_clf
import my_DecTre_reg
import get_Quasi_linear_Kernel
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
X, Y = make_gaussian_quantiles(n_samples=1000,n_features=10, n_classes=2, 
								mean=None,cov=1.0,random_state=13)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)


X_mean = np.mean(X,axis=0)
X_std  = np.std(X,axis=0)
X = (X - X_mean) / X_std
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

#========================== plot data face ================================
plot_step = 0.01
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5])
contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)

# ========================= training decision Tree ===========================
myTree = my_DecTre_clf.DecisionTreeClassifier(max_depth=5)
myTree.fit(X_train, y_train)

myTree = my_DecTre_reg.DecisionTreeRegresion(leafType='LogicReg', 
											 errType='lseErr_regul',
											 max_depth=3)
myTree.fit(X_train, y_train)
y_pred = myTree.predict(X_test)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
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
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


# ========================== training RF =====================================
myFore = my_DecTre_reg.RF_fit(X_train, y_train, n_trees=10, 
							  leafType='LogicReg', errType='lseErr_regul',
							  max_depth=5)
y_pred = my_DecTre_reg.RF_predict(X_test, myFore)

print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)


# =============== RF training and testing quasi_linear SVM ==============
RMat = np.array(my_DecTre_reg.get_RF_avgRList(myFore))
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)


clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X_train, y_train)

# scatter(X_test[:,0],X_test[:,1], c=y_test)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)
print 'f1_score :', metrics.f1_score(y_test, y_pred)

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







