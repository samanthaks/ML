import scipy.io as spio
import numpy as np
import pickle
import random
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import time
import csv
import warnings
warnings.filterwarnings("ignore")

class Regressor():
	'''
	Musical song regressor
	'''

	def __init__(self):
		'''
		Initializes the data
		'''
		self.k = 5

		self.train_X,self.train_Y,self.test_Xpreprocessing.scale(X_train) = self.read_file("MSdata.mat")
		# pickle.dump(self.train_X, open("train_X.pkl", "wb"))
		# pickle.dump(self.train_Y, open("train_Y.pkl", "wb"))
		# pickle.dump(self.test_X, open("test_X.pkl", "wb"))
		# self.train_X = pickle.load(open("train_X.pkl", "rb"))
		# self.train_Y = pickle.load(open("train_Y.pkl", "rb"))
		# self.test_X = pickle.load(open("test_X.pkl", "rb"))
		print("Finished loading training and testing data")

		self.train_folds,self.test_folds = self.create_folds()
		# pickle.dump(self.train_folds, open("train_folds.pkl", "wb"))
		# pickle.dump(self.test_folds, open("test_folds.pkl", "wb"))
		# self.train_folds = pickle.load(open("train_folds.pkl", "rb"))
		# self.test_folds = pickle.load(open("test_folds.pkl", "rb"))
		print("Finished creating k folds")

		self.reg = self.build_reg()
		print("Finished building regressor")

		self.mae = self.evaluate()
		print("Accuracy of regressor: " + str(self.mae))

		self.predict()

	def read_file(self,filename):
		'''
		Returns the variables from the data file
		'''
		data = spio.loadmat(filename)
		train_X = np.asarray(data['trainx'], dtype=np.int32) # 463715×90 matrix
		train_Y = np.asarray(data['trainy'], dtype=np.int32) # 463715×1 vector
		test_X = np.asarray(data['testx'], dtype=np.int32) # 463715×90 matrix
		return train_X,train_Y,test_X

	def create_folds(self):
		'''
		Returns k folds
		'''
		# Create folds
		X_and_Y = np.append(self.train_X,self.train_Y,axis=1)
		random.shuffle(X_and_Y)
		train = []
		test = []
		fold_size = int(len(self.train_X)/self.k)
		curr_ind = 0
		for fold in range(self.k):
			test_inds = np.arange(curr_ind,curr_ind+fold_size,1)
			train_inds = np.delete(range(len(X_and_Y)),test_inds)
			test.append(X_and_Y[test_inds,:]) # list of length k, testing folds w/ X and Y
			train.append(X_and_Y[train_inds,:]) # list of length k, training folds w/ X and Y
			curr_ind += fold_size
		return train,test

	def build_reg(self):
		'''
		Return regressor
		'''
		reg = linear_model.LinearRegression()
		# svr_rbf = SVR(kernel='rbf')
		# return svr_rbf
		return reg

	def evaluate(self):
		'''
		Return mean absolute error for model on k folds
		'''
		errors = []
		for fold_ind in range(self.k):
			fold_reg = self.reg
			fold_reg.fit(self.train_folds[fold_ind][:,:-1], self.train_folds[fold_ind][:,-1])
			y_pred = fold_reg.predict(self.test_folds[fold_ind][:,:-1])
			fold_err = mean_absolute_error(self.test_folds[fold_ind][:,-1], y_pred)
			errors.append(fold_err)
			print(fold_err)
		final_err = sum(errors)/self.k
		return final_err

	def predict(self):
		'''
		Predict on test samples, write to CSV file
		'''
		y_pred = self.reg.predict(self.test_X)
		ts = time.time()
		filename = str(ts) + "-" + str(self.mae) + ".csv"
		with open(filename, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			csvwriter.writerow(["dataid","prediction"])
			id = 1
			for pred in y_pred:
				csvwriter.writerow([str(id), str(pred)])
				id += 1

if __name__ == "__main__":
	reg = Regressor()