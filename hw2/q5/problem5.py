import numpy as np
from random import sample
from collections import Counter

train_file = "hw2data_1/reviews_te.csv"

class Unigram():
	'''
	Perceptron with unigram data representation
	'''

	def __init__(self,train_ratio):
		'''
		Initializes the data
		'''
		self.train_ratio = train_ratio

		self.X,self.Y = self.get_data()
		print("Finished separating documents from labels")

		self.X_train,self.X_test,self.Y_train,self.Y_test = self.split_data()
		print("Finished separating training from testing")

		self.vocabulary = self.unigrams()
		print("Finished getting unigrams")

		self.weights = self.perceptron()
		print("Finished running perceptron")

	def get_data(self):
		'''
		Get labels and documents
		'''
		X = []
		Y = []
		with open(train_file, "r") as f:
			for row in f:
				arr = row.split(",")
				X.append(arr[1])
				Y.append(arr[0])
		return X,Y

	def split_data(self):
		'''
		Separate into training and testing
		'''
		train_inds = sample(range(len(self.X)),int(self.train_ratio*len(self.X)))
		X_train = []
		Y_train = []
		for train_i in train_inds:
			X_train.append(self.X[train_i])
			Y_train.append(self.Y[train_i])
		test_inds = np.delete(range(len(self.X)),train_inds)
		X_test = []
		Y_test = []
		for test_i in test_inds:
			X_test.append(self.X[test_i])
			Y_test.append(self.Y[test_i])
		return X_train,X_test,Y_train,Y_test

	def unigrams(self):
		'''
		Represent data as unigrams
		'''
		all_tokens = []
		for document in self.X_train:
			tokens = document.split()
			all_tokens.extend(tokens)
		unigrams = Counter(all_tokens)
		return unigrams

	def perceptron(self):
		'''
		Run perceptron algorithm to get linear classifiers
		'''
		return None

if __name__ == "__main__":
	# Unigrams
	print("Starting with unigrams...")
	unigram_perceptron = Unigram(train_ratio=0.8)
	#unigram_accuracy = unigram_perceptron.evaluate()
	#print(0.8, "unigram_perceptron", unigram_accuracy)