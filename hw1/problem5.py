import scipy.io as spio
import numpy as np

def read_file(filename):
	'''
	Returns the images matrix and labels matrix
	'''
	data = spio.loadmat(filename)
	X = np.asarray(data['X'], dtype=np.int32)
	Y = np.asarray(data['Y'], dtype=np.int32)
	return X,Y

class MultiGauss():
	'''
	Multivariate Gaussian Classifier
	'''

	def __init__(self):
		'''
		Initializes the data
		'''
		self.X,self.Y = read_file('hw1data.mat')
		self.separated_data = self.separate_by_label()
		self.means,self.covs = self.get_mles()

	def separate_by_label(self):
		'''
		Creates list of lists, each containing the data for one label
		'''
		# Preprocessing - choose the top 200 features with the highest variance
		vars = []
		for feature in self.X.T:
	   		vars.append(np.var(feature))
		inds = np.argpartition(vars, -200)[-200:]
		new_X = self.X[:,inds]

		# Preprocessing - normalize each of the features to have mean zero and variance one.
		new_X = (new_X - new_X.mean(axis=0)) / new_X.std(axis=0)

		# Separate data by label
		separated_data = [[] for i in range(10)] # images for the label that corresponds to that index, i.e. [[images for label 0],[images with label 1],...]
		for vector,label in zip(new_X,self.Y):
			separated_data[int(label)].append(vector)
		return separated_data


	def get_mles(self):
		'''
		Creates list of means and covariance matrices, for all labels in the data
		'''
		# Get means, covs for each label
		means = []
		covs = []
		offset = 0 * np.identity(200)
		for label_data in self.separated_data:
			label_data = np.asarray(label_data)
			means.append(label_data.mean(axis=0))
			covs.append(np.cov(label_data,rowvar=False)+offset)
		return means,covs

	def get_probabilities(label):
		pass
		#return prior, conditional

	def calc_multigauss(label):
		pass
		#return probability

	def predict(self, image):
		'''
		Predicts label for image
		'''
		max_probability = 0
		max_label = -1
		for label in range(10):
			probability = calc_multigauss(label)
			if probability > max_probability:
				max_probability = probability
				max_label = label
		return label

	def evaluate():
		pass

if __name__ == "__main__":
	mg_classifier = MultiGauss()
