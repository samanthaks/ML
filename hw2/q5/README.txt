COMS 4771 Machine Learning (Spring 2018)
Problem Set #2

Emily Song, Kendal Sandridge, Samantha Stultz - eks2138, ks3311, sks2200
Due: March 3, 2018

Problem 5

ABOUT THE FILES
unigram.py, tfidf.py, and bigram.py contain the implementations for the linear classifiers based on unigrams, tfidf-weighting, and bigrams, respectively. problem5.py contains the main script that: (1) runs the classifiers for different splits of the training data set and saves a plot comparing the accuracies (Part C), and (2) for the classifier based on the unigram representation, finds the 10 words that have the highest weights and the 10 words that have the lowest weights (Part D). 

HOW TO RUN
In this directory, make sure there is a folder called "hw2data_1" and the files "reviews_te.csv" and "reviews_tr" inside that folder. Or, change the "train_file" and "test_file" paths at the top of unigram.py, tfidf.py, and bigram.py files to the location where the data files lie. Then, run "python3 problem5.py" to create the plots and find the highest/lowest weights.
Note: For the sake of running the classifiers in a reasonable amount of time, we have limited the vocabulary size to 20000 for all of the classifiers; if you wish to change this, simply change the "self.vocab_size" parameter in the "init" function for the classifier.