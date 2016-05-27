import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn import metrics


# ========================= generate data ============================
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2, 1:3]
y = y[y != 2]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:.9 * n_sample]
y_train = y[:.9 * n_sample]
X_test = X[.9 * n_sample:]
y_test = y[.9 * n_sample:]


#========================== plot data ================================
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Circle out the test data
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, zorder=10)

plt.axis('tight')
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
# ========================= training model ===========================
import my_DecTre
myTree = my_DecTre.DecisionTreeClassifier( min_samples_split=3)
myTree.fit(X_train, y_train)



# ========================== testing model ===========================

scatter(X_test[:,0],X_test[:,1], c=y_test)
pred = myTree.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, pred)

plot_step = 0.05
x_min = X[:, 0].min() -1
x_max = X[:, 0].max() +1
y_min = X[:, 1].min() -1
y_max = X[:, 1].max() +1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myTree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5])

contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)
