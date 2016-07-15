fig = plt.figure()
plt.style.use('ggplot')
ax = fig.add_subplot(111)
pos = ax.scatter(X1[:,0], X1[:,1], c='red' , label='pos data')
neg = ax.scatter(X2[:,0], X2[:,1], c='blue' , label='neg data')
ax.scatter(X[:,0], X[:,1], c=Y)
ax.axis('tight')
plt.xticks(np.linspace(0, 7, 15, endpoint=True))
plt.yticks(np.linspace(-1.5, 1.5, 9, endpoint=True))

myTree = my_RF_QLSVM1.DecisionTreeRegresion(leafType='LogicReg', 
                                             errType='lseErr_regul',
                                             max_depth=4,
                                             min_samples_split=5)
myTree.fit(X, Y)
plot_step = 0.01
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myTree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
tree = ax.contour(xx, yy, Z, colors=['g', 'g', 'g'], linestyles=['-', '-', '-'],
        levels=[-.5, 0, .5],linewidths=1, label='decision tree predict line')

myFore = my_RF_QLSVM2.RF_QLSVM_clf( n_trees=3, 
                              leafType='LogicReg', errType='lseErr_regul',
                              max_depth=5, min_samples_split=5,
                              max_features=None,
                              bootstrap_data=True)
myFore.fit(X, Y)
plot_step = 0.01
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myFore.RF_predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
tree = ax.contour(xx, yy, Z, colors=['g', 'g', 'g'], linestyles=['-', '-', '-'],
        levels=[-.5, 0, .5],linewidths=1, label='decision tree predict line')

RMat0 = np.array(myFore.trees[0].tree.get_RList())
RMat1 = np.array(myFore.trees[1].tree.get_RList())
RMat2 = np.array(myFore.trees[2].tree.get_RList())
RMat = np.array(myFore.get_RF_avgRList_byAggloCluster(0.4))
kernel1 = ax.scatter(RMat0[:,0,0], RMat0[:,1,0], marker='o', c='r', s=60, label='kernel data', alpha=0.3)
kernel2 = ax.scatter(RMat1[:,0,0], RMat1[:,1,0], marker='o', c='b', s=60, label='kernel data', alpha=0.3)
kernel3 = ax.scatter(RMat2[:,0,0], RMat2[:,1,0], marker='o', c='g', s=60, label='kernel data', alpha=0.3)
kernel = ax.scatter(RMat[:,0,0], RMat[:,1,0], marker='o', c='k', s=80, label='kernel data')

from functools import partial
RBFinfo = partial(get_Quasi_linear_Kernel.get_RBFinfo,RMat=RMat)
Quasi_linear_kernel = partial(get_Quasi_linear_Kernel.get_KernelMatrix_basic,RMat=RMat)
K_train = Quasi_linear_kernel(X,X)
K_test = Quasi_linear_kernel(np.c_[xx.ravel(), yy.ravel()],X)
clf = svm.SVC(kernel='precomputed' , C=100)
clf.fit(K_train, Y )
Z = clf.predict(K_test)
Z = Z.reshape(xx.shape)
svm = ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['-', '-', '-'],
        levels=[-.5, 0, .5],linewidths=2, label='Quasi-Linear SVM predict boundary')

ax.legend([pos,neg,kernel1, kernel2, kernel3, kernel, tree.collections[0], svm.collections[0]]
    ,['pos data','neg data','tree 1 kernel','tree 2 kernel','tree 3 kernel',
    'clustered kernel data','RF classification boundary',
    'Quasi-Linear SVM predict boundary'], fontsize='small')
plt.title('Extract kernel at leafNode which is LinearClassifier')


plt.plot([3.14,3.14], [-1.7,1.7], color='blue', linewidth=1.0, linestyle="--")
plt.plot([1.2,1.2], [-1.7,1.7], color='blue', linewidth=0.8, linestyle="--")
plt.plot([5.6,5.6], [-1.7,1.7], color='blue', linewidth=0.8, linestyle="--")
plt.plot([-0.5,1.2], [0.02,0.02], color='red', linewidth=0.6, linestyle="--")
plt.plot([1.2,3.14], [0.03,0.03], color='red', linewidth=0.6, linestyle="--")
plt.plot([3.14,5.65], [-0.33,-0.33], color='red', linewidth=0.6, linestyle="--")
plt.plot([5.65,7.5], [-0.33,-0.33], color='red', linewidth=0.6, linestyle="--")
plt.plot([5.65,7.5], [0.6,0.6], color='red', linewidth=0.4, linestyle="--")
plt.plot([0.13,0.13], [-1.5,0.02], color='blue', linewidth=0.4, linestyle="--")
plt.plot([2.89,2.89], [0.03,1.7], color='blue', linewidth=0.4, linestyle="--")
plt.plot([3.77,3.77], [-1.5,-0.33], color='blue', linewidth=0.4, linestyle="--")
plt.plot([6.04,6.04], [-0.33,0.6], color='blue', linewidth=0.2, linestyle="--")
plt.plot([4.40,4.40], [-1.5,-0.33], color='blue', linewidth=0.2, linestyle="--")
plt.plot([1.2,2.89], [1.22,1.22], color='red', linewidth=0.2, linestyle="--")

myTree.tree.getTreeStruc()
    splitIndex [0]<3.149385 
        splitIndex [0]<1.206384 
            splitIndex [1]<0.019991 
                splitIndex [0]<0.130531 
                    leaf node:  [[ 0.  0.]]
                    leaf node:  [[ 0.  0.]]
                leaf node:  [[-5.65484593  7.33001519]]
            splitIndex [1]<0.030807 
                leaf node:  0
                splitIndex [0]<2.898392 
                    splitIndex [1]<1.229508 
                        leaf node:  [[ 1.22039176  3.7005728 ]]
                        leaf node:  1
                    leaf node:  1
        splitIndex [0]<5.651739 
            splitIndex [1]<-0.339326 
                splitIndex [0]<3.771717 
                    leaf node:  0
                    splitIndex [0]<4.408023 
                        leaf node:  [[ 1.2208726  6.4050778]]
                        leaf node:  [[ 1.03074045  5.86170328]]
                leaf node:  1
            splitIndex [1]<-0.331500 
                leaf node:  0
                splitIndex [1]<0.602208 
                    splitIndex [0]<6.044793 
                        leaf node:  1
                        leaf node:  [[-0.15685533  4.04364542]]
                    leaf node:  1


RMat(label and clf) 
array([[[ 0.05577466,  0.05497404],
        [-0.16942653,  0.02994897]],

       [[ 0.16131892,  0.02215799],
        [-0.04722246,  0.06406533]],

       [[ 0.64605979,  0.34455101],
        [ 0.67474524,  0.29691971]],

       [[ 2.97332145,  0.0969651 ],
        [-0.10699842,  0.11188143]],

       [[ 2.11452396,  0.46073885],
        [ 0.87178637,  0.311668  ]],

       [[ 2.98786693,  0.09313093],
        [ 0.36168941,  0.13022345]],

       [[ 3.61369861,  0.13564568],
        [-0.66331451,  0.13556958]],

       [[ 4.12439024,  0.17994125],
        [-0.7153861 ,  0.26118737]],

       [[ 5.09250089,  0.30420659],
        [-0.77414366,  0.2437598 ]],

       [[ 3.66997678,  0.17270438],
        [-0.12434821,  0.21331832]],

       [[ 6.05186868,  0.16506   ],
        [-0.50998232,  0.13777473]],

       [[ 6.42317498,  0.21713378],
        [ 0.25030024,  0.22363044]],

       [[ 6.76277838,  0.16682866],
        [ 0.76084851,  0.07712944]]])


RMat(clf) 
array([[[ 0.64605979,  0.34455101],
        [ 0.67474524,  0.29691971]],

       [[ 2.11452396,  0.46073885],
        [ 0.87178637,  0.311668  ]],


       [[ 4.12439024,  0.17994125],
        [-0.7153861 ,  0.26118737]],

       [[ 5.09250089,  0.30420659],
        [-0.77414366,  0.2437598 ]],

       [[ 6.42317498,  0.21713378],
        [ 0.25030024,  0.22363044]]])

