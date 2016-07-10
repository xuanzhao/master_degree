import numpy as np

def get_RBFinfo(X, RMat, lamb=2):
	# because we have #k center, so the R should be size (m,k)
	print 'RMat shape is ', RMat.shape
	X = np.mat(X)
	m,n =  X.shape
	k = RMat.shape[0]   # (k)
	R = np.mat(np.zeros((m,k)))  # (m,k)

	for j in range(k):
		for i in range(m):
			Rj = np.mat(RMat[j]).reshape((2,n))
			diff_row = X[i,:] - Rj[0] 	# (1,n) - (1,n) = (1,n)
			diff_dot = diff_row * diff_row.T 	# (1,n) * (n,1) = (1,1)
			R[i,j] = np.exp( - diff_dot / lamb * Rj[1]*Rj[1].T)
												# (1,1) * (1,n)*(n,1) = (1,1)
	# standardize R
	R_std = np.true_divide(R, np.sum(R,axis=1))
	New_R_std = np.nan_to_num(R_std)

	#New_R = np.nan_to_num(R)
	print 'down get R, R shape is ',R.shape
	return New_R_std   # (m,k)

def get_KernelMatrix(X_test, X_train, RMat):
	# Construct the semi-positive definite kernel matrix ,size is (m,m)
	T = (X_test is X_train)
	X_train = np.mat(X_train)   # (m_d,n)
	X_test  = np.mat(X_test)	# (m_t,n)

   	if T:
   		print 'create standarized train_kernel matrix...'
   		R_train = get_RBFinfo(X_train, RMat)   # (m_d, k)
		R_train = np.mat(R_train)

		K_train = np.multiply((1+ X_train*X_train.T),(R_train*R_train.T))  #(m_d,m_d)

		K_diag_train = np.diag(K_train)   # (m_d, ) (k(x,x))
		m,n = K_train.shape               # m = m_d, n = m_d
		Cor_val = np.sqrt(np.multiply(np.tile(K_diag_train,(m,1)), # (m_d,m_d)
							np.tile(K_diag_train,(n,1)).T))   # (m_d, m_d)
		K_train = np.true_divide(K_train, Cor_val)    # (m_d,m_d)

		print 'down get standardized train_kernel matrix..., the shape is', K_train.shape
		return K_train

	else:
		print 'create standarized test_kernel matrix...'
		R_train = get_RBFinfo(X_train, RMat)   # (m_d, k)
		R_train = np.mat(R_train)

		K_train = np.multiply((1+ X_train*X_train.T),(R_train*R_train.T))  #(m_d,m_d)
		K_diag_train = np.diag(K_train)   # (m_d, ) (k(x,x))

		R_test = get_RBFinfo(X_test, RMat) 	   # (m_t, k)
		R_test  = np.mat(R_test)
		K_test = np.multiply((1+ X_test*X_train.T),(R_test*R_train.T))  #(m_t,m_d)
		
		K_diag_test = np.diag(K_test)   # (m_t, ) (k(y,y))
		m,n = K_test.shape			   # m = m_t, n = m_d
		Cor_val = np.sqrt(np.multiply(np.tile(K_diag_test,(n,1)).T, # (m_t, m_d)
							np.tile(K_diag_train,(m,1))))  # (m_t, m_d)
		K_test = np.true_divide(K_test, Cor_val)  # (m_t,m_d)

		print 'down get standardized test_kernel matrix..., the shape is', K_test.shape
		return K_test



def get_KernelMatrix_basic(X_test, X_train, RMat):

	# Construct the semi-positive definite kernel matrix ,size is (m,m)
	T = (X_test is X_train)
	X_train = np.mat(X_train)   # (m_d,n)
	X_test  = np.mat(X_test)	# (m_t,n)

   	if T:
   		print 'create train_kernel matrix...'
   		R_train = get_RBFinfo(X_train, RMat)   # (m_d, k)
		R_train = np.mat(R_train)
		R_test = np.mat(R_train)
	else:
		print 'create test_kernel matrix...'
		R_train = get_RBFinfo(X_train, RMat)   # (m_d, k)
		R_train = np.mat(R_train)
		R_test = get_RBFinfo(X_test, RMat) 	   # (m_t, k)
		R_test  = np.mat(R_test)

	# for the train_kernel, the matrix size is (m_d,m_d)
	# (m_d,n)*(n,m_d) = (m_d,m_d) ----> X_train
	# (m_d,k)*(k,m_d) = (m_d,m_d) ----> R_train

	# for the test_kernel, the matrix size is (m_t,m_d)
	# (m_t,n) * (n,m_d) = (m_t,m_d) ----> X_test
	# (m_t,k) * (k,m_d) = (m_t,m_d) ----> R_test
	K = np.multiply((1+ X_test*X_train.T),(R_test*R_train.T))
	print 'down get kernelMatrix, the shape is', K.shape

	return K




