import sklearn as sk
from sklearn import neural_network
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#Reading the main data set file
mat = scipy.io.loadmat('outputt.mat')
data = mat ['output']
print (data.shape)
#Reading the index file
mat = scipy.io.loadmat('test_ind.mat')
idx = mat['test_ind']
for x in range(0,10):
	#The values of indices are in matlab. So, the indices are decremented to be used in Python
	#The citation column is separated
	#The other columns are combined together
	
	test_idx = idx [:,x] - 1
	indices = np.arange(0,2244019,1)
	
	#The remaining indices are for training
	train_idx = np.setdiff1d(indices, test_idx,assume_unique=True)

	train_label = data[train_idx,2]
	train_data = data [train_idx,3:69]
	author_train = data [train_idx,1]
	test_data = data [test_idx,3:69]
	author_test = data[test_idx,1]
	test_label = data [test_idx,2]
	train_data = np.column_stack([author_train,train_data])
	test_data = np.column_stack([author_test,test_data])

	#Multi-layer perceptron used for regression
	#Adam is the fastest solver
	#Relu solver and tanh has approximately the same performance
	#The maximum iteration above 500 is not helpful
	
	mlp = neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(100),
							   max_iter=500, shuffle=True, random_state=4,
							   activation='tanh')

	print ('Modeling has been completed')
	model = mlp.fit(train_data, train_label)
	print ('Fitting has been completed')
	pred1 = model.predict(test_data)
	print ('Prediction has been completed')
	error = abs(test_label - pred1)
	error2 = pow(error,2)
	print (np.mean(error))
	print (np.mean(error2))


