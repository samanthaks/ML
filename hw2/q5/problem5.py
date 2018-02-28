import numpy as np
from nltk.corpus import stopwords
from random import sample
from collections import Counter
import pickle

train_file = "hw2data_1/reviews_tr.csv"
test_file = "hw2data_1/reviews_te.csv"

class Unigram():
	'''
	Perceptron with unigram data representation
	'''

	def __init__(self,train_ratio):
		'''
		Initializes the data
		'''
		self.vocab_size = 20000
		self.train_ratio = train_ratio

		# self.X,self.Y = self.get_data()
		# pickle.dump(self.X, open("unigram_X.pkl", "wb"))
		# pickle.dump(self.Y, open("unigram_Y.pkl", "wb"))
		self.X = pickle.load(open("unigram_X.pkl", "rb"))
		self.Y = pickle.load(open("unigram_Y.pkl", "rb"))
		print("Finished separating documents from labels")

		# self.X_train,self.X_test,self.Y_train,self.Y_test = self.split_data()
		# pickle.dump(self.X_train, open("unigram_X_train.pkl", "wb"))
		# pickle.dump(self.X_test, open("unigram_X_test.pkl", "wb"))
		# pickle.dump(self.Y_train, open("unigram_Y_train.pkl", "wb"))
		# pickle.dump(self.Y_test, open("unigram_Y_test.pkl", "wb"))
		self.X_train = pickle.load(open("unigram_X_train.pkl", "rb"))
		self.X_test = pickle.load(open("unigram_X_test.pkl", "rb"))
		self.Y_train = pickle.load(open("unigram_Y_train.pkl", "rb"))
		self.Y_test = pickle.load(open("unigram_Y_test.pkl", "rb"))
		print("Finished separating training from testing")

		# self.vocabulary = self.unigrams()
		# pickle.dump(self.vocabulary, open("unigram_vocabulary.pkl", "wb"))
		self.vocabulary = pickle.load(open("unigram_vocabulary.pkl", "rb"))
		print("Finished getting unigrams")

		# self.weights = self.perceptron()
		# pickle.dump(self.weights, open("unigram_weights.pkl", "wb"))
		self.weights = pickle.load(open("unigram_weights.pkl", "rb"))
		print("Finished running perceptron")

		self.accuracy = self.evaluate()

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
				if arr[0] is "0":
					Y.append(-1)
				else:
					Y.append(1)
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
		all_stopwords = set(stopwords.words('english'))
		all_tokens = []
		for document in self.X_train:
			tokens = document.split()
			for word in tokens:
				if word not in all_stopwords:
					all_tokens.append(word)
		vocabulary = Counter(all_tokens)
		vocabulary = vocabulary.most_common(self.vocab_size)
		vocabulary = dict(vocabulary)
		index = 0
		for key in vocabulary:
			vocabulary[key] = index
			index += 1
		return vocabulary

	def perceptron(self):
		'''
		Run perceptron algorithm to get linear classifiers
		'''
		# Initial classifier
		w = np.zeros(len(self.vocabulary))

		# First pass
		for document,label in zip(self.X_train,self.Y_train):
			# Represent document as vector
			document_tokens = document.split()
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					x[index] = document_counter[word]
			# Update w
			if int(np.sign(np.dot(w,x))) is not label:
				w = np.add(w,label*x)
		print("Finished first pass of perceptron")

		# Second pass
		new_inds = list(range(len(self.X_train)))
		np.random.shuffle(new_inds)
		total_w = np.zeros(len(self.vocabulary))
		for ind in new_inds:
			document = self.X_train[ind]
			label = self.Y_train[ind]
			# Represent document as vector
			document_tokens = document.split()
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					x[index] = document_counter[word]
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
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					x[index] = document_counter[word]
			label = int(np.sign(np.dot(self.weights,x)))
			predictions.append(label)
		count = 0
		for pred,act in zip(predictions,self.Y_test):
			if pred == act:
				count += 1
		return count/len(predictions)

class Tfidf():
	'''
	Perceptron with tfidf data representation
	'''

	def __init__(self,train_ratio):
		'''
		Initializes the data
		'''
		self.vocab_size = 20000
		self.train_ratio = train_ratio

		# self.X,self.Y = self.get_data()
		# pickle.dump(self.X, open("tfidf_X.pkl", "wb"))
		# pickle.dump(self.Y, open("tfidf_Y.pkl", "wb"))
		self.X = pickle.load(open("tfidf_X.pkl", "rb"))
		self.Y = pickle.load(open("tfidf_Y.pkl", "rb"))
		print("Finished separating documents from labels")

		# self.X_train,self.X_test,self.Y_train,self.Y_test = self.split_data()
		# pickle.dump(self.X_train, open("tfidf_X_train.pkl", "wb"))
		# pickle.dump(self.X_test, open("tfidf_X_test.pkl", "wb"))
		# pickle.dump(self.Y_train, open("tfidf_Y_train.pkl", "wb"))
		# pickle.dump(self.Y_test, open("tfidf_Y_test.pkl", "wb"))
		self.X_train = pickle.load(open("tfidf_X_train.pkl", "rb"))
		self.X_test = pickle.load(open("tfidf_X_test.pkl", "rb"))
		self.Y_train = pickle.load(open("tfidf_Y_train.pkl", "rb"))
		self.Y_test = pickle.load(open("tfidf_Y_test.pkl", "rb"))
		print("Finished separating training from testing")

		# self.vocabulary,self.counts = self.tfidf()
		# pickle.dump(self.vocabulary, open("tfidf_vocabulary.pkl", "wb"))
		# pickle.dump(self.counts, open("tfidf_counts.pkl", "wb"))
		self.vocabulary = pickle.load(open("tfidf_vocabulary.pkl", "rb"))
		self.counts = pickle.load(open("tfidf_counts.pkl", "rb"))
		print("Finished getting tfidf")

		# self.weights = self.perceptron()
		# pickle.dump(self.weights, open("tfidf_weights.pkl", "wb"))
		self.weights = pickle.load(open("tfidf_weights.pkl", "rb"))
		print("Finished running perceptron")

		self.accuracy = self.evaluate()

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
				if arr[0] is "0":
					Y.append(-1)
				else:
					Y.append(1)
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

	def tfidf(self):
		'''
		Represent data as tfidf
		'''
		all_stopwords = set(stopwords.words('english'))
		all_tokens = []
		counts = {}
		for document in self.X_train:
			tokens = document.split()
			tokens_set = set(tokens)
			for word in tokens:
				if word not in all_stopwords:
					all_tokens.append(word)
			for word_set in tokens_set:
				if word_set not in all_stopwords:
					if word_set not in counts:
						counts[word_set] = 1
					else:
						counts[word_set] = counts[word_set] + 1
		vocabulary = Counter(all_tokens)
		vocabulary = vocabulary.most_common(self.vocab_size)
		vocabulary = dict(vocabulary)
		index = 0
		for key in vocabulary:
			vocabulary[key] = index
			index += 1
		return vocabulary,counts

	def perceptron(self):
		'''
		Run perceptron algorithm to get linear classifiers
		'''
		# Initial classifier
		w = np.zeros(len(self.vocabulary))

		# First pass
		for document,label in zip(self.X_train,self.Y_train):
			# Represent document as vector
			document_tokens = document.split()
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					tf = document_counter[word]
					idf = len(self.X_train)/self.counts[word]
					x[index] = tf * np.log10(idf)
			# Update w
			if int(np.sign(np.dot(w,x))) is not label:
				w = np.add(w,label*x)
		print("Finished first pass of perceptron")

		# Second pass
		new_inds = list(range(len(self.X_train)))
		np.random.shuffle(new_inds)
		total_w = np.zeros(len(self.vocabulary))
		for ind in new_inds:
			document = self.X_train[ind]
			label = self.Y_train[ind]
			# Represent document as vector
			document_tokens = document.split()
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					tf = document_counter[word]
					idf = len(self.X_train)/self.counts[word]
					x[index] = tf * np.log10(idf)
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
			document_counter = Counter(document_tokens)
			x = np.zeros(len(self.vocabulary))
			for word in document_counter:
				if word in self.vocabulary:
					index = self.vocabulary[word]
					tf = document_counter[word]
					idf = len(self.X_train)/self.counts[word]
					x[index] = tf * np.log10(idf)
			label = int(np.sign(np.dot(self.weights,x)))
			predictions.append(label)
		count = 0
		for pred,act in zip(predictions,self.Y_test):
			if pred == act:
				count += 1
		return count/len(predictions)

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

		# self.X,self.Y = self.get_data()
		# pickle.dump(self.X, open("bigram_X.pkl", "wb"))
		# pickle.dump(self.Y, open("bigram_Y.pkl", "wb"))
		self.X = pickle.load(open("bigram_X.pkl", "rb"))
		self.Y = pickle.load(open("bigram_Y.pkl", "rb"))
		print("Finished separating documents from labels")

		# self.X_train,self.X_test,self.Y_train,self.Y_test = self.split_data()
		# pickle.dump(self.X_train, open("bigram_X_train.pkl", "wb"))
		# pickle.dump(self.X_test, open("bigram_X_test.pkl", "wb"))
		# pickle.dump(self.Y_train, open("bigram_Y_train.pkl", "wb"))
		# pickle.dump(self.Y_test, open("bigram_Y_test.pkl", "wb"))
		self.X_train = pickle.load(open("bigram_X_train.pkl", "rb"))
		self.X_test = pickle.load(open("bigram_X_test.pkl", "rb"))
		self.Y_train = pickle.load(open("bigram_Y_train.pkl", "rb"))
		self.Y_test = pickle.load(open("bigram_Y_test.pkl", "rb"))
		print("Finished separating training from testing")
		
		# self.bigrams = self.bigrams()
		# pickle.dump(self.bigrams, open("bigram_bigrams.pkl", "wb"))
		self.bigrams = pickle.load(open("bigram_bigrams.pkl", "rb"))
		print("Finished getting bigrams")

		# self.weights = self.perceptron()
		# pickle.dump(self.weights, open("bigram_weights.pkl", "wb"))
		self.weights = pickle.load(open("bigram_weights.pkl", "rb"))
		print("Finished running perceptron")

		self.accuracy = self.evaluate()

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
				if arr[0] is "0":
					Y.append(-1)
				else:
					Y.append(1)
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
		w = np.zeros(len(self.bigrams))

		# First pass
		for document,label in zip(self.X_train,self.Y_train):
			# Represent document as vector
			document_tokens = document.split()
			document_bigrams = []
			for word1,word2 in zip(document_tokens[:-1],document_tokens[1:]):
				bigram = (word1,word2)
				document_bigrams.append(bigram)
			document_counter = Counter(document_bigrams)
			x = np.zeros(len(self.bigrams))
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
		total_w = np.zeros(len(self.bigrams))
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
			x = np.zeros(len(self.bigrams))
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
			x = np.zeros(len(self.bigrams))
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

if __name__ == "__main__":
	# Unigrams
	# print("Starting with unigrams...")
	# unigram_perceptron = Unigram(train_ratio=0.8)
	# print("Unigram accuracy", unigram_perceptron.accuracy)

	# Tfidf
	# print("Starting with tfidf...")
	# tfidf_perceptron = Tfidf(train_ratio=0.8)
	# print("Tfidf accuracy", tfidf_perceptron.accuracy)

	# Bigrams
	print("Starting with bigrams...")
	bigram_perceptron = Bigram(train_ratio=0.8)
	print("Bigram accuracy", bigram_perceptron.accuracy)