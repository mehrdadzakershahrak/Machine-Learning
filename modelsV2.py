#coding=utf8
import pdb
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score as cv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC as SVM
from sklearn import tree

featurepath = 'output3.csv'
labelpath = 'labels.csv'

# load and format feature and label matrices
X = pd.read_csv(featurepath, header = None, skiprows = 1, usecols = range(0,69))
del X[0] # remove indices column
del X[1] # remove citation count column since we are testing the classes

y = pd.read_csv(labelpath, header = None, skiprows = 1) 
del y[0] # remove indices column

# random train and test set division
Xtest, Xtrain, ytest, ytrain = split(X, y, test_size = 0.9, random_state = 0) # 90/10 split
del X
del y
#pdb.set_trace()
# fit the model
classifier = tree.DecisionTreeClassifier()
#classifier = OneVsRestClassifier(SVM(random_state = 0))
classifier.fit(Xtrain, ytrain.values.ravel()) # change shape of y to (nsample,)

# predict sentiment for the test set
predicted = classifier.predict(Xtest)

# calculate the diagnostics
accuracy = sklearn.metrics.accuracy_score(ytest, predicted)
confusion = sklearn.metrics.confusion_matrix(ytest, predicted)

print '\nAccuracy = ', accuracy
print '\nConfussion matrix:\n', confusion
