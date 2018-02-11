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
		self.preprocess_data() 
		self.separated_data = self.separate_by_label()
		self.means,self.covs = self.get_mles()

	def preprocess_data(self):
		'''
		Preprocess data to avoid underflow or overflow
		'''
		# Choose the top 200 features with the highest variance
		vars = []
		for feature in self.X.T:
	   		vars.append(np.var(feature))
		inds = np.argpartition(vars, -200)[-200:]
		self.X = self.X[:,inds]
		self.Y = self.Y[inds]

		# Normalize each of the features to have mean zero and variance one.
		self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

	def separate_by_label(self):
		'''
		Creates list of lists, each containing the data for one label
		'''
		# Separate data by label
		separated_data = [[] for i in range(10)] # images for the label that corresponds to that index, i.e. [[images for label 0],[images with label 1],...]
		for vector,label in zip(self.X,self.Y):
			separated_data[int(label)].append(vector)
		return separated_data

	def get_mles(self):
		'''
		Creates list of means and covariance matrices, for all labels in the data
		'''
		# Get means, covs for each label
		means = []
		covs = []
		offset = 0.1 * np.identity(200)
		for label_data in self.separated_data:
			label_data = np.asarray(label_data)
			means.append(label_data.mean(axis=0))
			covs.append(np.cov(label_data,rowvar=False)+offset)
		return means,covs

	def get_prior(self,label):
		'''
		Calculates prior which will be used to determine probability of label
		'''
		prior = len(self.separated_data[label])/len(self.X)
		return prior

	def get_conditional(self,label,image):
		'''
		Calculates conditional using Multivariate Gaussian formula, used to determine probability of label
		'''
		frac = 1/np.linalg.det(self.covs[label])
		diff = np.subtract(image,self.means[label])
		inverse = np.linalg.inv(self.covs[label])
		power = np.matmul(diff.T,np.matmul(inverse,diff))
		power = -0.5 * power
		conditional = frac * np.exp(power)
		return conditional

	def predict(self, image):
		'''
		Predicts label for image
		'''
		max_probability = 0
		max_label = -1
		for label in range(10):
			prior = self.get_prior(label)
			conditional = self.get_conditional(label,image)
			probability = prior*conditional
			if probability > max_probability:
				max_probability = probability
				max_label = label
		return max_label

	def evaluate(self):
		'''
		Predicts on testing data, compares to true labels
		'''
		predictions = []
		for image in self.X[1:200,]:
			label = self.predict(image)
			predictions.append(label)

		count = 0
		for pred,act in zip(predictions,self.Y[1:200,]):
			if pred == act:
				count += 1
		return count/len(predictions)
		
if __name__ == "__main__":
	mg_classifier = MultiGauss()
	print(mg_classifier.evaluate())
