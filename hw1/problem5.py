import scipy.io as spio
import numpy as np
from random import sample
import math

class ProbClassifier():
	def __init__(self,filename,ratio):
		'''
		Initializes the data
		'''
		self.X,self.Y = self.read_file(filename)
		self.X_train,self.X_test,self.Y_train,self.Y_test = self.separate_data(ratio)
		self.preprocess_data()

	def read_file(self,filename):
		'''
		Returns the images matrix and the labels matrix
		'''
		data = spio.loadmat(filename)
		X = np.asarray(data['X'], dtype=np.int32)
		Y = np.asarray(data['Y'], dtype=np.int32)
		return X,Y

	def separate_data(self,ratio):
		'''
		Create training and testing sets
		'''
		inds = sample(range(len(self.X)),int(ratio*len(self.X)))
		X_train = self.X[inds,]
		X_test = np.delete(self.X,inds,axis=0)
		Y_train = self.Y[inds]
		Y_test = np.delete(self.Y,inds,axis=0)
		return X_train,X_test,Y_train,Y_test

	def preprocess_data(self):
		'''
		Preprocess data to avoid underflow or overflow
		'''
		# Choose the top 200 features with the highest variance (choose features using training data but also apply this to testing data)
		vars = []
		for feature in self.X_train.T:
	   		vars.append(np.var(feature))
		inds = np.argpartition(vars, -200)[-200:]
		self.X_train = self.X_train[:,inds]
		self.X_test = self.X_test[:,inds]

		# Normalize each of the features to have mean zero and variance one (do this separately for training data and testing data)
		self.X_train = (self.X_train - self.X_train.mean(axis=0)) / self.X_train.std(axis=0)
		self.X_test = (self.X_test - self.X_test.mean(axis=0)) / self.X_test.std(axis=0)

	def evaluate(self):
		'''
		Predicts on testing data, compares to true labels
		'''
		predictions = []
		for image in self.X_test:
			label = self.predict(image)
			predictions.append(label)

		count = 0
		for pred,act in zip(predictions,self.Y_test):
			if pred == act:
				count += 1
		return count/len(predictions)

class MultiGauss(ProbClassifier):
	'''
	Multivariate Gaussian Classifier
	'''
	def __init__(self,filename,ratio):
		'''
		Initializes the data
		'''
		ProbClassifier.__init__(self,filename,ratio)
		self.separated_data = self.separate_by_label()
		self.means,self.covs = self.get_mles()

	def separate_by_label(self):
		'''
		Creates list of lists, each containing the data for one label
		'''
		separated_data = [[] for i in range(10)] # images for the label that corresponds to that index, i.e. [[images for label 0],[images with label 1],...]
		for vector,label in zip(self.X_train,self.Y_train):
			separated_data[int(label)].append(vector)
		return separated_data

	def get_mles(self):
		'''
		Creates list of means and covariance matrices, for all labels in the data
		'''
		means = [] # means for the label that corresponds to that index, i.e. [[mean image for label 0],[mean image for label 1],...]
		covs = [] # covariance matrices for the label that corresponds to that index, i.e. [[cov for label 0],[cov for label 1],...]
		offset = 0.1 * np.identity(200)
		for label_data in self.separated_data:
			label_data = np.asarray(label_data)
			means.append(label_data.mean(axis=0))
			covs.append(np.cov(label_data,rowvar=False)+offset)
		return means,covs

	def get_prior(self,label):
		'''
		Calculates prior which will be used to determine probability of label
		'''
		prior = len(self.separated_data[label])/len(self.X_train)
		return prior

	def get_conditional(self,label,image):
		'''
		Calculates conditional using Multivariate Gaussian formula, used to determine probability of label
		'''
		frac = 1/math.sqrt(np.linalg.det(self.covs[label]))
		diff = np.subtract(image,self.means[label])
		inverse = np.linalg.inv(self.covs[label])
		power = np.matmul(diff.T,np.matmul(inverse,diff))
		power = -0.5 * power
		conditional = frac * np.exp(power)
		return conditional

	def predict(self, image):
		'''
		Predicts label for image
		'''
		max_probability = 0
		max_label = -1
		for label in range(10):
			prior = self.get_prior(label)
			conditional = self.get_conditional(label,image)
			probability = prior*conditional
			if probability > max_probability:
				max_probability = probability
				max_label = label
		print(max_label)
		return max_label

class kNN(ProbClassifier):
	'''
	k-Nearest Neighbor Classifier
	'''
	def __init__(self,filename,ratio):
		'''
		Initializes the data
		'''
		ProbClassifier.__init__(self,filename,ratio)
		self.k = 1

	def euclidean(self,image1,image2):
		'''
		Computes the euclidean distance between two images
		'''
		diff = np.subtract(image1,image2)
		prod = np.matmul(diff.T,diff)
		return math.sqrt(prod)

	def predict(self,test_image):
		'''
		Predicts label given image
		'''
		all_dist = []
		label = -1
		for train_image in self.X_train:
			curr_dist = self.euclidean(test_image,train_image)
			all_dist.append(curr_dist)
		inds = np.argpartition(curr_dist, -self.k)[-self.k:]
		labels = self.Y_train[inds]
		counts = np.bincount(labels[0])
		return np.argmax(counts)
		
if __name__ == "__main__":
	mg_classifier = MultiGauss('hw1data.mat',0.8)
	print(mg_classifier.evaluate())

	#knn_classifier = kNN('hw1data.mat',0.8)
	#print(knn_classifier.evaluate())
