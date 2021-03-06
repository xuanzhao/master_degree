from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def get_RMat(RList):
	RMat = np.array(RList)
	return RMAT


def get_RBFinfo(X, RMat, lamb=2):
	# because we have #k center, so the R should be size (m,k)
	X = np.mat(X)
	m,n =  X.shape
	k = RMat.shape[0]   # (k)
	R = mat(np.zeros((m,k)))  # (m,k)

	for j in range(k):
		for i in range(m):
			diff_row = X[i,:] - RMat[j,0] 	# (1,n) - (1,n) = (1,n)
			diff_dot = diff_row * diff_row.T 	# (1,n) * (n,1) = (1,1)
			R[i,j] = np.exp( - diff_dot / lamb * RMat[j,1]*RMat[j,1].T)
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



def train_Quasi_linear_SVM(X_train, y_train):

	clf = svm.SVC(kernel=get_KernelMatrix)

	clf.fit(X_train, Y_train)
	return clf

def evalue_Quasi_linear_SVM(X_test, y_test):
	y_pred = clf.predict(X_test)


