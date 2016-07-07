import my_DecTre_reg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_circles


# ========================= generate data ============================

np.random.seed(0)
X, Y = make_circles(n_samples=500, factor=.6, noise=.09)


n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
Y = Y[order]

X_train = X[:.7 * n_sample]
y_train = Y[:.7 * n_sample]
X_test = X[.7 * n_sample:]
y_test = Y[.7 * n_sample:]

#========================== plot data ================================
plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

# Circle out the test data
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, zorder=10)

plt.axis('tight')
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()


# ========================= training model ===========================
myFore = my_QLSVM_RF.RF_fit(X_train, y_train, n_trees=3, max_depth=3,
							 min_samples_split=30)


# ========================== testing model ===========================

scatter(X_test[:,0],X_test[:,1], c=y_test)
pred = my_QLSVM_RF.RF_predict(X_test,myFore)
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