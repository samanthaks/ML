import numpy as np
from nltk.corpus import stopwords
from random import sample
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from unigram import Unigram
from tfidf import Tfidf
from bigram import Bigram

if __name__ == "__main__":
	# # Make sure everything is working !!
	# # Unigrams
	# print("Starting with unigrams...")
	# unigram_perceptron = Unigram(train_ratio=0.8)
	# print("Unigram accuracy", unigram_perceptron.accuracy)

	# # Tfidf
	# print("Starting with tfidf...")
	# tfidf_perceptron = Tfidf(train_ratio=0.8)
	# print("Tfidf accuracy", tfidf_perceptron.accuracy)

	# # Bigrams
	# print("Starting with bigrams...")
	# bigram_perceptron = Bigram(train_ratio=0.8)
	# print("Bigram accuracy", bigram_perceptron.accuracy)

	# PART C: Compare the data representations
	ratios = np.arange(0.1,0.95,0.05)
	unigram_accuracies = []
	tfidf_accuracies = []
	bigram_accuracies = []
	for r in ratios:
		unigram_perceptron = Unigram(train_ratio=r)
		unigram_accuracy = unigram_perceptron.accuracy
		unigram_accuracies.append(unigram_accuracy)
		print(r, "unigram_perceptron", unigram_accuracy)

		tfidf_perceptron = Tfidf(train_ratio=r)
		tfidf_accuracy = tfidf_perceptron.accuracy
		tfidf_accuracies.append(tfidf_accuracy)
		print(r, "tfidf_perceptron", tfidf_accuracy)

		bigram_perceptron = Bigram(train_ratio=r)
		bigram_accuracy = bigram_perceptron.accuracy
		bigram_accuracies.append(bigram_accuracy)
		print(r, "bigram_perceptron", bigram_accuracy)

	pickle.dump(unigram_accuracies, open("unigram_accuracies.pkl", "wb"))
	pickle.dump(tfidf_accuracies, open("tfidf_accuracies.pkl", "wb"))
	pickle.dump(bigram_accuracies, open("bigram_accuracies.pkl", "wb"))

	# unigram_accuracies = pickle.load(open("unigram_accuracies.pkl", "rb"))
	# tfidf_accuracies = pickle.load(open("tfidf_accuracies.pkl", "rb"))
	# bigram_accuracies = pickle.load(open("bigram_accuracies.pkl", "rb"))

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(ratios,unigram_accuracies,c='b',label='Unigrams')
	ax1.scatter(ratios,tfidf_accuracies,c='r',label='Tfidf')
	ax1.scatter(ratios,bigram_accuracies,c='y',label='Bigrams')
	plt.axis([0.0,1.0,0.85,0.9])
	plt.xlabel('Proportion of data set used for training')
	plt.ylabel('Accuracy of classifier')
	plt.title('Accuracy of classifier vs. Proportion of data set used for training')
	plt.legend(loc='lower left')
	plt.savefig('problem5.png')

	# PART D: Find the highest and lowest weights for the unigram representation