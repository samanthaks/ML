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
	ratios = np.arange(0.05,1.05,0.05)
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
	num_samples = ratios * 1000000
	ax1.scatter(num_samples,unigram_accuracies,c='b',label='Unigrams')
	ax1.scatter(num_samples,tfidf_accuracies,c='r',label='Tfidf')
	ax1.scatter(num_samples,bigram_accuracies,c='y',label='Bigrams')
	plt.axis([0,1000000,0.8,0.9])
	plt.xlabel('Number of training samples')
	plt.ylabel('Accuracy of classifier')
	plt.title('Accuracy of classifier vs. Number of training samples')
	plt.legend(loc='lower left')
	plt.savefig('problem5.png')


	# PART D: Find the highest and lowest weights for the unigram representation
	unigram_perceptron = Unigram(train_ratio=1.0)
	unigram_vocabulary = unigram_perceptron.vocabulary
	unigram_weights = unigram_perceptron.weights

	pickle.dump(unigram_vocabulary, open("final_unigram_vocabulary.pkl", "wb"))
	pickle.dump(unigram_weights, open("final_unigram_weights.pkl", "wb"))
	# unigram_weights = pickle.load(open("final_unigram_weights.pkl", "rb"))
	# unigram_vocabulary = pickle.load(open("final_unigram_vocabulary.pkl", "rb"))

	highest_inds = unigram_weights.argsort()[-10:][::-1]
	highest_words = []
	for high_i in highest_inds:
		for word,index in unigram_vocabulary.items():
			if index == high_i:
				highest_words.append(word)
				continue
	print("Words with highest weights", highest_words)
	lowest_inds = unigram_weights.argsort()[:10][::-1]
	lowest_words = []
	for low_i in lowest_inds:
		for word,index in unigram_vocabulary.items():
			if index == low_i:
				lowest_words.append(word)
				continue
	print("Words with lowest weights", lowest_words)