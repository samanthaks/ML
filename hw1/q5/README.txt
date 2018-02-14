COMS 4771 Machine Learning (Spring 2018)
Problem Set #1

Emily Song, Kendal Sandridge, Samantha Stultz - eks2138, ks3311, sks2200
Due: Feburary 13, 2018

Problem 5

How to run the scripts:

Run problem5.py (make sure to be using Python 3) to build the probabilistic classifier moded by the Multivariate Gaussian (part i) and the k-Nearest Neighbor classifier (part ii). The script will run the classifiers for various splits of the data and save a plot of the trends (part iii), and then run only the k-Nearest Neighbor classifier with different distance metrics for various splits of the data and save another plot of the trends (part iv).

If you want to find the k that generates the highest accuracy on different test sets using cross validation, uncomment line 193 to run the code in findbestk.py. We found that k=1-5 all produced similar accuracies around 0.90-0.95, with k=1 producing a slightly better accuracy than the others, so we chose to use k=1 in the classifiers built for the comparisons in parts (iii) and (iv).