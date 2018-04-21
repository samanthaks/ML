from random import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs

def euclidean(p1,p2):
	'''
	Computes euclidean distance between two points in R^2
	'''
	return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))

def lloyds(data,labels,k):
	'''
	k-means algorithm for data in R^2
	'''
	# Initialize k clusters randomly
	clusters = []
	for i in range(k):
		clusters.append([random(),random()])
	print("Finished initializing cluster centers")

	# Repeat till no more changes occur
	iteration = 1
	labeled_data =  [ [] for i in range(k) ]
	while True:
		for point in data:
			min_dist = float("inf")
			for i in range(k):
				d = euclidean(clusters[i],point)
				if d < min_dist:
					min_dist = d
					label = i
			labeled_data[label].append(point)
		new_clusters = []
		for i in range(k):
			new_clusters.append(np.mean(labeled_data[i],axis=0))
		if np.array_equal(clusters, new_clusters):
			break
		clusters = new_clusters
		labeled_data =  [ [] for i in range(k) ]
		print("End of iteration", iteration)
		iteration += 1

	return labeled_data, clusters

def plot_2(data, labels, labeled_data,clusters,n):
	'''
	Plot for examples where k=2
	'''
	# Plot actual centers and points
	orig_labeled_data = [[],[]]
	for (point,label) in zip(data,labels):
		orig_labeled_data[label].append(point)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	label0 = orig_labeled_data[0]
	label1 = orig_labeled_data[1]
	x_vals0 = []
	y_vals0 = []
	x_vals1 = []
	y_vals1 = []
	for (p0,p1) in zip(label0,label1):
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	orig_clusters = [np.mean(orig_labeled_data[0],axis=0),np.mean(orig_labeled_data[1],axis=0)]
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(orig_clusters[0][0],orig_clusters[0][1], c='r')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(orig_clusters[1][0],orig_clusters[1][1], c='b')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "actual" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting actual")

	# Plot predicted centers and points
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	label0 = labeled_data[0]
	label1 = labeled_data[1]
	x_vals0 = []
	y_vals0 = []
	x_vals1 = []
	y_vals1 = []
	for (p0,p1) in zip(label0,label1):
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(clusters[0][0],clusters[0][1], c='r')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(clusters[1][0],clusters[1][1], c='b')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "k-means algorithm on dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "predicted" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting predicted")

def plot_3(data, labels, labeled_data,clusters,n):
	'''
	Plot for examples where k=3
	'''
	# Plot actual centers and points
	orig_labeled_data = [[],[],[]]
	for (point,label) in zip(data,labels):
		orig_labeled_data[label].append(point)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	label0 = orig_labeled_data[0]
	label1 = orig_labeled_data[1]
	label2 = orig_labeled_data[2]
	x_vals0 = []
	y_vals0 = []
	x_vals1 = []
	y_vals1 = []
	x_vals2 = []
	y_vals2 = []
	for (p0,p1) in zip(label0,label1):
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	for p2 in label2:
		x_vals2.append(p2[0])
		y_vals2.append(p2[1])
	orig_clusters = [np.mean(orig_labeled_data[0],axis=0),np.mean(orig_labeled_data[1],axis=0),np.mean(orig_labeled_data[2],axis=0)]
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(orig_clusters[0][0],orig_clusters[0][1], c='r')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(orig_clusters[1][0],orig_clusters[1][1], c='b')
	ax1.scatter(x_vals2, y_vals2, c='g', label=2)
	ax1.scatter(orig_clusters[2][0],orig_clusters[2][1], c='g')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "actual" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting actual")

	# Plot predicted centers and points
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	label0 = labeled_data[0]
	label1 = labeled_data[1]
	label2 = labeled_data[2]
	x_vals0 = []
	y_vals0 = []
	x_vals1 = []
	y_vals1 = []
	x_vals2 = []
	y_vals2 = []
	for (p0,p1) in zip(label0,label1):
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	for p2 in label2:
		x_vals2.append(p2[0])
		y_vals2.append(p2[1])
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(clusters[0][0],clusters[0][1], c='r')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(clusters[1][0],clusters[1][1], c='b')
	ax1.scatter(x_vals2, y_vals2, c='g', label=2)
	ax1.scatter(clusters[2][0],clusters[2][1], c='g')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "k-means algorithm on dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "predicted" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting predicted")

if __name__ == "__main__":
	data1, labels1 = make_circles()
	labeled_data1, clusters1 = lloyds(data1,labels1,2)
	plot_2(data1, labels1, labeled_data1, clusters1,1)

	data2, labels2 = make_moons()
	labeled_data2, clusters2 = lloyds(data2,labels2,2)
	plot_2(data2, labels2, labeled_data2, clusters2,2)

	data3, labels3 = make_blobs(cluster_std=[4, 4, 4],random_state=8)
	labeled_data3, clusters3 = lloyds(data3,labels3,3)
	plot_3(data3,labels3, labeled_data3, clusters3,3)