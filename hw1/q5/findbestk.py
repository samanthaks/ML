import scipy.io as spio
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def read_file(filename):
	'''
	Returns the images matrix and the labels matrix
	'''
	data = spio.loadmat(filename)
	X = np.asarray(data['X'], dtype=np.int32)
	Y = np.asarray(data['Y'], dtype=np.int32)
	return X,Y

def create_folds(X,Y,k=5):
	'''
	Create folds for cross validation
	'''
	X_and_Y = np.append(X,Y,axis=1)
	random.shuffle(X_and_Y)
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	fold_size = int(len(X)/k)
	curr_ind = 0
	for fold in range(k):
		test_inds = np.arange(curr_ind,curr_ind+fold_size,1)
		train_inds = np.delete(range(len(X_and_Y)),test_inds)
		X_test.append(X_and_Y[test_inds,:-1])
		Y_test.append(X_and_Y[test_inds,-1])
		X_train.append(X_and_Y[train_inds,:-1])
		Y_train.append(X_and_Y[train_inds,-1])

	# PREPROCESS!!
	new_X_train = [[] for i in range(k)]
	new_X_test = [[] for i in range(k)]
	for fold in range(k):
		# Choose the top 200 features with the highest variance (choose features using training data but also apply this to testing data)
		vars = []
		for feature in X_train[fold].T:
	   		vars.append(np.var(feature))
		inds = np.argpartition(vars, -200)[-200:]
		new_X_train[fold] = X_train[fold][:,inds]
		new_X_test[fold] = X_test[fold][:,inds]
		# Normalize each of the features to have mean zero and variance one (do this separately for training data and testing data)
		new_X_train[fold] = (new_X_train[fold] - new_X_train[fold].mean(axis=0)) / new_X_train[fold].std(axis=0)
		new_X_test[fold] = (new_X_test[fold] - new_X_test[fold].mean(axis=0)) / new_X_test[fold].std(axis=0)
	return new_X_train,Y_train,new_X_test,Y_test

def euclidean(image1,image2):
	'''
	Computes the euclidean distance between two images
	'''
	diff = np.subtract(image1,image2)
	prod = np.matmul(diff.T,diff)
	return prod

def knn_cv_predict(X_train,Y_train,test_image,i,k,metric):
	'''
	Predicts label given image for cross-validation and kNN
	'''
	all_dist = []
	for train_image in X_train[i]:
		if metric is 'euclidean':
			curr_dist = euclidean(test_image,train_image)
		elif metric is 'manhattan':
			curr_dist = manhattan(test_image,train_image)
		elif metric is 'max':
			curr_dist = max(test_image,train_image)
		all_dist.append(curr_dist)
	all_dist = np.asarray(all_dist)
	inds = all_dist.argsort()[:k]
	k_labels = Y_train[i][inds]
	k_labels = k_labels.flatten()
	pred_label = np.argmax(np.bincount(k_labels))
	return pred_label

def knn_cv_eval(X_train,Y_train,X_test,Y_test,k=1,metric='euclidean'):
	'''
	Compare predicted labels with true labels for cross-validation and kNN
	'''
	accuracies = []
	for (fold,(fold_X_test,fold_Y_test)) in enumerate(zip(X_test,Y_test)):
		predictions = []
		for image in fold_X_test:
			label = knn_cv_predict(X_train,Y_train,image,fold,k,metric)
			predictions.append(label)
		count = 0
		for pred,act in zip(predictions,fold_Y_test):
			if pred == act:
				count += 1
		accuracies.append(count/len(predictions))
	return np.mean(accuracies)

def findbestk():
	'''
	Uses cross-validation to find most optimal k
	'''
	# Read in the dataset
	X,Y = read_file('hw1data.mat')
	print('Finished reading in the file')
	# Create folds for cross validation
	cv_X_train,cv_Y_train,cv_X_test,cv_Y_test = create_folds(X,Y)
	print("Finished creating folds for cross validation")

	# Search for the most optimal k
	accuracies = []
	for k in np.arange(1,6,1):
		accuracy = knn_cv_eval(cv_X_train,cv_Y_train,cv_X_test,cv_Y_test)
		print(k, accuracy)
		accuracies.append(accuracy)
	best_k = np.argmax(accuracies)+1
	print("Most optimal k is", best_k)

	plt.bar(np.arange(1,6,1), accuracies, align='center')
	plt.axis([0, 6, 0.9, 1.0])
	plt.xlabel('k')
	plt.ylabel('Accuracy')
	plt.title('Accuracy vs. k')
	plt.show()
	print("Finished finding best k")