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
# ========================= generate data ============================

plt.subplot(211)
plt.title("One informative feature, one cluster per class", fontsize='small')
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

plt.subplot(212)
plt.title("Two informative features, one cluster per class", fontsize='small')
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, cmap=plt.cm.Paired)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

plot_step = 0.01
x_min = X1[:, 0].min() 
x_max = X1[:, 0].max() 
y_min = X1[:, 1].min() 
y_max = X1[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5])
contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)

# ========================= training decision Tree ===========================
myTree = my_DecTre.DecisionTreeClassifier(max_depth=5, min_samples_split=30)
myTree.fit(X_train, y_train)

# ========================== training quasi_linear SVM =======================

clf = svm.SVC(kernel=get_KernelMatrix)
clf.fit(X_train, y_train)


# ========================== testing model ===========================

# scatter(X_test[:,0],X_test[:,1], c=y_test)
y_pred = clf.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, y_pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, y_pred)





