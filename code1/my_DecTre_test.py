import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split[X, y, 
									test_size=0.33, random_state=13]



myTree = DecisionTreeClassifier(max_depth=5,min_samples_split=5)
myTree.fit(X_train, y_train)

yHat = myTree.predict(X_test)


print 'confusion_matrix :\n', metrics.confusion_matrix(y_test, predictions)
print 'accuracy_score :', metrics.accuracy_score(y_test, predictions)



