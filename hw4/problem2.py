from random import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
import pickle

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

def plot_2(data, labels, labeled_data,clusters,n,plot_actual=False):
	'''
	Plot for examples where k=2
	'''
	# Plot actual centers and points (only if specified)
	if plot_actual:
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
		ax1.scatter(orig_clusters[0][0],orig_clusters[0][1], marker='v', c='y')
		ax1.scatter(x_vals1, y_vals1, c='b', label=1)
		ax1.scatter(orig_clusters[1][0],orig_clusters[1][1], marker='v', c='y',label='cluster center')
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
	for p0 in label0:
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
	for p1 in label1:
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(clusters[0][0],clusters[0][1], marker='v', c='y')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(clusters[1][0],clusters[1][1], marker='v', c='y',label='cluster center')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "k-means algorithm on dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "predicted" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting predicted")

def plot_3(data, labels, labeled_data,clusters,n,plot_actual=False):
	'''
	Plot for examples where k=3
	'''
	# Plot actual centers and points (only if specified)
	if plot_actual:
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
		ax1.scatter(orig_clusters[0][0],orig_clusters[0][1], marker='v', c='y')
		ax1.scatter(x_vals1, y_vals1, c='b', label=1)
		ax1.scatter(orig_clusters[1][0],orig_clusters[1][1], marker='v', c='y')
		ax1.scatter(x_vals2, y_vals2, c='g', label=2)
		ax1.scatter(orig_clusters[2][0],orig_clusters[2][1], marker='v', c='y',label='cluster center')
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
	for p0 in label0:
		x_vals0.append(p0[0])
		y_vals0.append(p0[1])
	for p1 in label1:
		x_vals1.append(p1[0])
		y_vals1.append(p1[1])
	for p2 in label2:
		x_vals2.append(p2[0])
		y_vals2.append(p2[1])
	ax1.scatter(x_vals0, y_vals0, c='r', label=0)
	ax1.scatter(clusters[0][0],clusters[0][1], marker='v',c='y')
	ax1.scatter(x_vals1, y_vals1, c='b', label=1)
	ax1.scatter(clusters[1][0],clusters[1][1], marker='v',c='y')
	ax1.scatter(x_vals2, y_vals2, c='g', label=2)
	ax1.scatter(clusters[2][0],clusters[2][1], marker='v', c='y',label='cluster center')
	plt.xlabel('x')
	plt.ylabel('y')
	title = "k-means algorithm on dataset" + str(n)
	plt.title(title)
	plt.legend(loc='upper left');
	name = "predicted" + str(n) + ".png"
	plt.savefig(name)
	print("Finished plotting predicted")

def kernel(data, labels, k, kernel):
	'''
	Kernelized k-means
	'''
	clusters = []
	for i in range(k):
		clusters.append([random(),random()])
	print("Finished initializing cluster centers")

	# Repeat till no more changes occur
	iteration = 1
	labeled_data = [ [] for i in range(k) ]
	for index, label in enumerate(labels):
		labeled_data[label].append(index)

	# Gram matrix for kernel
	kernel_vals = None
	if kernel is 'lin':
		kernel_vals = linear_kernel(data,data)
	if kernel is 'poly':
		kernel_vals = polynomial_kernel(data,data)
	if kernel is 'rbf':
		kernel_vals = rbf_kernel(data,data)
		
	# Repeat till no more changes occur
	while True:
		new_labeled_data =  [ [] for i in range(k) ]
		for n in range(len(data)):
			min_dist = float("inf")
			for i in range(k):

				# New distance calculation
				d1 = kernel_vals[n][n]
				d2 = 0
				for i2 in labeled_data[i]:
					d2 += kernel_vals[n][i2]
				d2 = d2 * 2 / len(labeled_data[i])
				d3 = 0
				for i3a in labeled_data[i]:
					for i3b in labeled_data[i]:
						d3 += kernel_vals[i3a][i3b]
				d3 = d3 / math.pow(len(labeled_data[i]),2)
				d = d1 - d2 + d3
				if d < min_dist:
					min_dist = d
					label = i

			new_labeled_data[label].append(n)

		new_clusters = []
		for i in range(k):
			labels = new_labeled_data[i]
			points = data[labels]
			new_clusters.append(np.mean(points,axis=0))

		first_set = set(map(tuple, clusters))
		secnd_set = set(map(tuple, new_clusters))
		if first_set == secnd_set:
			break
		clusters = new_clusters
		labeled_data = new_labeled_data
		new_labeled_data =  [ [] for i in range(k) ]
		print("End of iteration", iteration)
		iteration += 1

	return labeled_data, clusters

def ind_to_pt(data,labeled_data,k):
	'''
	Converts indices of points to actual data points
	'''
	new_labeled_data = [ [] for i in range(k)]
	for label in range(len(labeled_data)):
		for index in labeled_data[label]:
			new_labeled_data[label].append(data[index])
	return new_labeled_data


if __name__ == "__main__":
	## K-MEANS ALGORITHM
	# DATASET 1
	# data1, labels1 = make_circles()
	# dataset1 = [data1, labels1]
	# pickle.dump(dataset1, open("dataset1.pkl", "wb"))
	dataset1 = pickle.load(open("dataset1.pkl", "rb"))
	data1,labels1 = dataset1
	labeled_data1, clusters1 = lloyds(data1,labels1,2)
	plot_2(data1, labels1, labeled_data1, clusters1,1,True)

	# DATASET 2
	# data2, labels2 = make_moons()
	# dataset2 = [data2, labels2]
	# pickle.dump(dataset2, open("dataset2.pkl", "wb"))
	dataset2 = pickle.load(open("dataset2.pkl", "rb"))
	data2,labels2 = dataset2
	labeled_data2, clusters2 = lloyds(data2,labels2,2)
	plot_2(data2, labels2, labeled_data2, clusters2,2,True)

	# DATASET 2
	# data3, labels3 = make_blobs(cluster_std=[5, 5, 5],random_state=8)
	# dataset3 = [data3, labels3]
	# pickle.dump(dataset3, open("dataset3.pkl", "wb"))
	dataset3 = pickle.load(open("dataset3.pkl", "rb"))
	data3,labels3 = dataset3
	labeled_data3, clusters3 = lloyds(data3,labels3,3)
	plot_3(data3,labels3, labeled_data3, clusters3,3,True)

	## KERNELIZED K-MEANS ALGORITHM
	# DATASET 1 - linear, polynomial, rbf kernels
	labeled_data1_lin, clusters1_lin = kernel(data1,labels1,2,'lin')
	labeled_data1_lin1 = ind_to_pt(data1,labeled_data1_lin,2)
	plot_2(data1, labels1, labeled_data1_lin1, clusters1_lin, '1-linear-kernel')
	labeled_data1_poly, clusters1_poly = kernel(data1,labels1,2,'poly')
	labeled_data1_poly1 = ind_to_pt(data1,labeled_data1_poly,2)
	plot_2(data1, labels1, labeled_data1_poly1, clusters1_poly, '1-polynomial-kernel')
	labeled_data1_rbf, clusters1_rbf = kernel(data1,labels1,2,'rbf')
	labeled_data1_rbf1 = ind_to_pt(data1,labeled_data1_rbf,2)
	plot_2(data1, labels1, labeled_data1_rbf1, clusters1_rbf, '1-rbf-kernel')

	# DATASET 2 - linear, polynomial, rbf kernels
	labeled_data2_lin, clusters2_lin = kernel(data2,labels2,2,'lin')
	labeled_data2_lin2 = ind_to_pt(data2,labeled_data2_lin,2)
	plot_2(data2, labels2, labeled_data2_lin2, clusters2_lin, '2-linear-kernel')
	labeled_data2_poly, clusters2_poly = kernel(data2,labels2,2,'poly')
	labeled_data2_poly2 = ind_to_pt(data2,labeled_data2_poly,2)
	plot_2(data2, labels2, labeled_data2_poly2, clusters2_poly, '2-polynomial-kernel')
	labeled_data2_rbf, clusters2_rbf = kernel(data2,labels2,2,'rbf')
	labeled_data2_rbf2 = ind_to_pt(data2,labeled_data2_rbf,2)
	plot_2(data2, labels2, labeled_data2_rbf2, clusters2_rbf, '2-rbf-kernel')

	# DATASET 3 - linear, polynomial, rbf kernels
	labeled_data3_lin, clusters3_lin = kernel(data3,labels3,3,'lin')
	labeled_data3_lin3 = ind_to_pt(data3,labeled_data3_lin,3)
	plot_3(data3, labels3, labeled_data3_lin3, clusters3_lin, '3-linear-kernel')
	labeled_data3_poly, clusters3_poly = kernel(data3,labels3,3,'poly')
	labeled_data3_poly3 = ind_to_pt(data3,labeled_data3_poly,3)
	plot_3(data3, labels3, labeled_data3_poly3, clusters3_poly, '3-polynomial-kernel')
	labeled_data3_rbf, clusters3_rbf = kernel(data3,labels3,3,'rbf')
	labeled_data3_rbf3 = ind_to_pt(data3,labeled_data3_rbf,3)
	plot_3(data3, labels3, labeled_data3_rbf3, clusters3_rbf, '3-rbf-kernel')
