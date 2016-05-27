# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.1)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, y_1, "go", label="max_depth=2")
plt.plot(X_test, y_2, "ro", label="max_depth=5")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


d = np.array([0.0, 0.5139, 3.1328, 3.8502, 5])
d_std = np.diff(d)/2.0

d_mean = d[:-1] + d_std
d_mean = np.mat(d_mean).T
d_std = np.mat(d_std).T
# scatter(d_mean , d_y, s=100, c='y')

def get_RBFinfo(X, center=d_mean, theta=d_std, lamb=2):
	# because we have #k center, so the R should be size (m,k)
	X = np.mat(X)
	m,n =  X.shape
	k = center.shape[0]   # (k)
	R = mat(np.zeros((m,k)))  # (m,k)

	for j in range(k):
		for i in range(m):
			diff_row = X[i,:] - center[j,:] 	# (1,n) - (1,n) = (1,n)
			diff_dot = diff_row * diff_row.T 	# (1,n) * (n,1) = (1,1)
			R[i,j] = np.exp( - diff_dot / lamb * theta[j,:]*theta[j,:].T)
												# (1,1) * (1,n)*(n,1) = (1,1)
	# standardize R
	R_std = np.true_divide(R, np.sum(R,axis=1))

	print 'down get R, R shape is ',R.shape
	return R_std

def get_KernelMatrix(X_test, X_train):
	# Construct the semi-positive definite kernel matrix ,size is (m,m)
	X_train = np.mat(X_train)
	X_test  = np.mat(X_test)

	R_test = get_RBFinfo(X_test)
	R_train = get_RBFinfo(X_train)
	R_train = np.mat(R_train)
	R_test  = np.mat(R_test)

	# for the train_kernel, the matrix size is (m_d,m_d)
	# (m_d,n_d)*(n_d,m_d) = (m_d,m_d)
	# (m_d,k)*(k,m_d) = (m_d,m_d)

	# for the test_kernel, the matrix size is (m_t,m_d)
	# (m_t,n_t)*(n_d*m_d) = (m_t,m_d)
	# (m_t,k) * (k,m_d) = (m_t,m_d)
	K = np.multiply((1+ X_test*X_train.T),(R_test*R_train.T))

	print 'down get kernelMatrix, the shape is', K.shape

	return K



def train_Quasi_linear_SVM():
	from sklearn import svm
	clf = svm.SVC(kernel=get_KernelMatrix)
	X_train = np.r_[X1,X2]
	Y_train = np.r_[Y1,Y2]
	scatter(X[:,0],X[:,1],c='g')
	clf.fit(X_train, Y_train)

	y_pred = clf.predict(X_test)

	#scatter(X_test, y_pred)

	clf = svm.SVC(kernel=get_KernelMatrix)
	clf.fit(X, y)
	clf.predict(X_test1)

	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	h = 0.05
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

	# Plot also the training points
	plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
	plt.title('2-Class classification using Support Vector Machine with quasi-linear kernel')
	plt.axis('tight')
	plt.legend([Y_train[0], Y_train[-1]], ['negtive sample', 'postive sample'])
	plt.show()
		

from io import BytesIO as StringIO
from IPython.display import Image
from sklearn import tree
import pydotplus

# Export a decision tree in DOT format. then write into a unicode text object.
out = StringIO()
tree.export_graphviz(regr_1, out_file=out)
# Get graph and create a image.
graph = pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


from io import BytesIO as StringIO
from IPython.display import Image
from sklearn import tree
import pydotplus
dot_data = StringIO()  
tree.export_graphviz(regr_1, out_file=dot_data,  
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names, 
                     node_ids=True, 
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

tree_struc = zip(regr_1.tree_.feature, regr_1.tree_.threshold,
				regr_1.tree_.children_left,regr_1.tree_.children_right)








