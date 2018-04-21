COMS 4771 Machine Learning (Spring 2018)
Problem Set #4

Emily Song, Kendal Sandridge, Samantha Stultz - eks2138, ks3311, sks2200
Due: April 21, 2018

Problem 2

ABOUT THE FILES
problem2.py: This is the main script. It reads from the datasets and runs Lloyd's k-means algorithm as well as the kernelized k-means algorithm. It will generate plots of the correct cluster labeling, as well as plots for the clusters predicted by the various algorithms.
dataset1.pkl, dataset2.pkl, dataset3.pkl: These are the datasets, created using sklearn. They were chosen to show how Lloyd's algorithm can result in an undesirable clustering for certain datasets. We choose to look at concentric circles, moon shapes, and blobs with lots of noise.
actual1.png, actual2.png, actual3.png: These are the plots of the correct cluster labelings
predicted1.png, predicted2.png, predicted3.png: These are the plots of the clusters predicted by Lloyd's k-means algorithm.
predicted1-linear-kernel.png,predicted1-polynomial-kernel.png,predicted1-rbf-kernel.png,predicted2-linear-kernel.png,predicted2-polynomial-kernel.png,predicted2-rbf-kernel.png,predicted3-linear-kernel.png,predicted3-polynomial-kernel.png,predicted3-rbf-kernel.png: These are the plots of the clusters predicted by the kernelized k-means algorithm.

HOW TO RUN
Run "python3 problem2.py" to read from the datasets, run the algorithms, and create the plots.