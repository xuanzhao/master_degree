import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
import my_DecTre
import get_Quasi_linear_Kernel
# ========================= generate data ============================

plt.subplot(311)
plt.title("One informative feature, one cluster per class", fontsize='small')
X, Y = make_classification(n_samples=1000,n_features=5, n_redundant=0, n_informative=5,
                             n_clusters_per_class=4)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

plt.subplot(312)
plt.title("Two informative features, one cluster per class", fontsize='small')
X, Y = make_classification(n_samples=500,n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, cmap=plt.cm.Paired)

plt.subplot(313)
plt.title("Gaussian divided into three quantiles", fontsize='small')
X, Y = make_gaussian_quantiles(n_samples=500,n_features=10, n_classes=2, 
								mean=None,cov=1.0)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)


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
myTree = my_DecTre.DecisionTreeClassifier(min_samples_split=20, max_features='log2')
myTree.fit(X_train, y_train)

y_pred = myTree.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)


# ========================== training quasi_linear SVM =======================
RMat = np.array(myTree.tree.get_RList())
from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix,RMat=RMat)


clf = svm.SVC(kernel=Quasi_linear_kernel)
clf.fit(X_train, y_train)


# ========================== testing model ===========================

# scatter(X_test[:,0],X_test[:,1], c=y_test)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)





