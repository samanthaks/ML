import numpy as np
from nltk.corpus import stopwords
from random import sample
from collections import Counter
import pickle

train_file = "hw2data_1/reviews_tr.csv"
test_file = "hw2data_1/reviews_te.csv"

class Bigram():
	'''
	Perceptron with bigram data representation
	'''

	def __init__(self,train_ratio):
		'''
		Initializes the data
		'''
		self.vocab_size = 20000
		self.train_ratio = train_ratio

		self.X_train,self.X_test,self.Y_train,self.Y_test = self.get_data()
		# pickle.dump(self.X, open("bigram_X.pkl", "wb"))
		# pickle.dump(self.Y, open("bigram_Y.pkl", "wb"))
		# self.X = pickle.load(open("bigram_X.pkl", "rb"))
		# self.Y = pickle.load(open("bigram_Y.pkl", "rb"))
		# print("Finished separating documents from labels")

		self.X_train,self.Y_train = self.split_data()
		# pickle.dump(self.X_train, open("bigram_X_train.pkl", "wb"))
		# pickle.dump(self.X_test, open("bigram_X_test.pkl", "wb"))
		# pickle.dump(self.Y_train, open("bigram_Y_train.pkl", "wb"))
		# pickle.dump(self.Y_test, open("bigram_Y_test.pkl", "wb"))
		# self.X_train = pickle.load(open("bigram_X_train.pkl", "rb"))
		# self.X_test = pickle.load(open("bigram_X_test.pkl", "rb"))
		# self.Y_train = pickle.load(open("bigram_Y_train.pkl", "rb"))
		# self.Y_test = pickle.load(open("bigram_Y_test.pkl", "rb"))
		# print("Finished separating training from testing")
		
		self.bigrams = self.bigrams()
		# pickle.dump(self.bigrams, open("bigram_bigrams.pkl", "wb"))
		# self.bigrams = pickle.load(open("bigram_bigrams.pkl", "rb"))
		# print("Finished getting bigrams")

		self.weights = self.perceptron()
		# pickle.dump(self.weights, open("bigram_weights.pkl", "wb"))
		# self.weights = pickle.load(open("bigram_weights.pkl", "rb"))
		# print("Finished running perceptron")

		self.accuracy = self.evaluate()

	def get_data(self):
		'''
		Get labels and documents
		'''
		X_train = []
		Y_train = []
		with open(train_file, "r") as f:
			row_num = 1
			for row in f:
				if row_num > 1:
					arr = row.split(",")
					X_train.append(arr[1])
					if arr[0] is "0":
						Y_train.append(-1)
					else:
						Y_train.append(1)
				row_num += 1
		X_test = []
		Y_test = []
		with open(test_file, "r") as f:
			row_num = 1
			for row in f:
				if row_num > 1:
					arr = row.split(",")
					X_test.append(arr[1])
					if arr[0] is "0":
						Y_test.append(-1)
					else:
						Y_test.append(1)
				row_num += 1
		return X_train,X_test,Y_train,Y_test

	def split_data(self):
		'''
		Choose which subset of training data to use
		'''
		train_inds = sample(range(len(self.X_train)),int(self.train_ratio*len(self.X_train)))
		X_train = []
		Y_train = []
		for train_i in train_inds:
			X_train.append(self.X_train[train_i])
			Y_train.append(self.Y_train[train_i])
		return X_train,Y_train

	def bigrams(self):
		'''
		Represent data as bigrams
		'''
		bigrams = {}
		for document in self.X_train:
			document_tokens = document.split()
			for word1,word2 in zip(document_tokens[:-1],document_tokens[1:]):
				bigram = (word1,word2)
				if bigram in bigrams:
					bigrams[bigram] = bigrams[bigram] + 1
				else:
					bigrams[bigram] = 1
		bigrams = sorted(bigrams, key=bigrams.get, reverse=True)
		bigrams = bigrams[:self.vocab_size]
		index = 0
		bigrams_dict = {}
		for key in bigrams:
			bigrams_dict[key] = index
			index += 1
		return bigrams_dict

	def perceptron(self):
		'''
		Run perceptron algorithm to get linear classifiers
		'''
		# Initial classifier
		w = np.zeros(len(self.bigrams)+1)

		# First pass
		for document,label in zip(self.X_train,self.Y_train):
			# Represent document as vector
			document_tokens = document.split()
			document_bigrams = []
			for word1,word2 in zip(document_tokens[:-1],document_tokens[1:]):
				bigram = (word1,word2)
				document_bigrams.append(bigram)
			document_counter = Counter(document_bigrams)
			x = np.zeros(len(self.bigrams+1)
			x[len(self.bigrams)] = 1 # data lifting, homogeneous coordinates
			for pair in document_counter:
				if pair in self.bigrams:
					index = self.bigrams[pair]
					x[index] = document_counter[pair]
			# Update w
			if int(np.sign(np.dot(w,x))) is not label:
				w = np.add(w,label*x)
		print("Finished first pass of perceptron")

		# Second pass
		new_inds = list(range(len(self.X_train)))
		np.random.shuffle(new_inds)
		total_w = np.zeros(len(self.bigrams)+1)
		for ind in new_inds:
			document = self.X_train[ind]
			label = self.Y_train[ind]
			# Represent document as vector
			document_tokens = document.split()
			document_bigrams = []
			for word1,word2 in zip(document_tokens[:-1],document_tokens[1:]):
				bigram = (word1,word2)
				document_bigrams.append(bigram)
			document_counter = Counter(document_bigrams)
			x = np.zeros(len(self.bigrams+1)
			x[len(self.bigrams)] = 1 # data lifting, homogeneous coordinates
			for pair in document_counter:
				if pair in self.bigrams:
					index = self.bigrams[pair]
					x[index] = document_counter[pair]
			# Update w
			if int(np.sign(np.dot(w,x))) is not label:
				w = np.add(w,label*x)
			total_w = np.add(total_w,w)
		print("Finished second pass of perceptron")

		# Final classifier
		return 1/len(self.X_train) * total_w

	def evaluate(self):
		'''
		Predicts labels for test documents
		'''
		predictions = []
		for document in self.X_test:
			document_tokens = document.split()
			document_bigrams = []
			for word1,word2 in zip(document_tokens[:-1],document_tokens[1:]):
				bigram = (word1,word2)
				document_bigrams.append(bigram)
			document_counter = Counter(document_bigrams)
			x = np.zeros(len(self.bigrams+1)
			x[len(self.bigrams)] = 1 # data lifting, homogeneous coordinates
			for pair in document_counter:
				if pair in self.bigrams:
					index = self.bigrams[pair]
					x[index] = document_counter[pair]
			label = int(np.sign(np.dot(self.weights,x)))
			predictions.append(label)
		count = 0
		for pred,act in zip(predictions,self.Y_test):
			if pred == act:
				count += 1
		return count/len(predictions)