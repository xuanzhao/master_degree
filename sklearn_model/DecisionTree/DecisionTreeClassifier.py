import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import cross_validation
from sklearn import datasets
import numpy as np
import os

#os.chdir('/Users/ken/IPython_Note/master_degree')


# load data and permutation the data
iris = datasets.load_iris()
perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Split into training and testing sets
skf = cross_validation.StratifiedKFold(iris.target,  n_folds=3)
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
print 'f1_score :', cross_validation.cross_val_score(clf, X, Y, cv=skf, scoring='f1_weighted')
print 'accuracy score :',cross_validation.cross_val_score(clf, X, Y, cv=skf, scoring='accuracy')
print 'recall score :',cross_validation.cross_val_score(clf, X, Y, cv=skf, scoring='recall')

for train_index, test_index in skf:
	X_train, X_test = iris.data[train_index], iris.data[test_index]
	y_train, y_test = iris.target[train_index], iris.target[test_index]
	print 'X_train shape is', X_train.shape
	print 'y_train shape is', y_train.shape
	print 'X_test shape is', X_test.shape
	print 'y_test shape is', y_test.shape



#X_train, X_test, y_train, y_test = train_test_split[iris.data, iris.target, test_size=0.33, random_state=42]

#Build model on training data

DecTre = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
DecTre = DecTre.fit(X_train, y_train)

predictions = DecTre.predict(X_test)

print 'confusion_matrix :\n', sklearn.metrics.confusion_matrix(y_test, predictions)
print 'accuracy_score :', sklearn.metrics.accuracy_score(y_test, predictions)

plot_step = 0.01
x_min = X[:, 0].min() 
x_max = X[:, 0].max() 
y_min = X[:, 1].min() 
y_max = X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
Z = myTree.getDecisionBoundary(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)


from io import BytesIO as StringIO
from IPython.display import Image
from sklearn import tree
import pydotplus

# Export a decision tree in DOT format. then write into a unicode text object.
out = StringIO()
tree.export_graphviz(DecTre, out_file=out)
# Get graph and create a image.
graph = pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


from sklearn.externals.six import StringIO  
import pydotplus 
dot_data = StringIO() 
tree.export_graphviz(DecTre, out_file=dot_data) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 


from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(DecTre, out_file=dot_data,  
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
