import my_DecTre
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_circles


# ========================= generate data ============================

np.random.seed(0)
X, y = make_circles(n_samples=400, factor=.5, noise=.05)


n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order]

X_train = X[:.7 * n_sample]
y_train = y[:.7 * n_sample]
X_test = X[.7 * n_sample:]
y_test = y[.7 * n_sample:]

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
myTree = my_DecTre.DecisionTreeClassifier(max_depth=5, min_samples_split=30)
myTree.fit(X_train, y_train)

# ========================== testing model ===========================

scatter(X_test[:,0],X_test[:,1], c=y_test)
pred = myTree.predict(X_test)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, pred)
print 'accuracy_score :', metrics.accuracy_score(y_test, pred)

plot_step = 0.01
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myTree.getDecisionBoundary(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        levels=[-.5, 0, .5])

contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)