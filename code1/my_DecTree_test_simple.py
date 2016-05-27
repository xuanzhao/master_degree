import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# ======================== generate training data ======================
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y1 = np.sin(X).ravel() + .5
y2 = np.sin(X).ravel() - .5
y1[::4] += 1 * (0.5 - rng.rand(20))
y2[::4] -= 1 * (0.5 - rng.rand(20))
y1 = y1[:, np.newaxis]
y2 = y2[:, np.newaxis]

X1 = np.c_[X, y1]
Y1 = -1 * np.ones((X1.shape[0],1))
X2 = np.c_[X, y2]
Y2 = np.ones((X2.shape[0],1))

scatter(X1[:,0], X1[:,1], c='b')
scatter(X2[:,0], X2[:,1], c='r')

X_train = np.r_[X1,X2]
y_train = np.r_[Y1,Y2]

# Shuffle
idx = np.arange(X_train.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]

# ========================= generate testing data ===================
X1_test = np.arange(0.0, 5.0, 0.1)[:, np.newaxis]
X2_test = np.sin(X1_test)
X_test = np.c_[X1_test, X2_test]
scatter(X_test[:,0],X_test[:,1], c='y')

# ========================= training model ===========================
import my_DecTre
myTree = my_DecTre.DecisionTreeClassifier(min_samples_split=5)
myTree.fit(X_train, y_train)

plot_step = 0.05
x1_min, x1_max = X_test[:,0].min()-1 ,X_test[:,0].max()+1
x2_min, x2_max = X_test[:,1].min()-1 ,X_test[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, plot_step), np.arange(x2_min, x2_max, plot_step))
Z = myTree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)
plt.imshow(Z, interpolation='nearest', cmap=plt.cm.PuOr_r)
contours = plt.contourf(xx,yy,Z, cmap=plt.cm.Paired)

z = myTree.getDecisionBoundary(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
contours = plt.contourf(xx,yy,z, cmap=plt.cm.Paired)

# ========================== testing model ===========================

pred = myTree.predict(X_test)
scatter(X_test[:,0],X_test[:,1], c=pred)
print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, predictions)
print 'accuracy_score :', metrics.accuracy_score(y_test, predictions)

