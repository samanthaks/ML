import scipy.io as spio
import numpy as np

def read_file(filename):
	data = spio.loadmat(filename)
	imagedata = np.asarray(data['X'], dtype=np.int32)
	labeldata = np.asarray(data['Y'], dtype=np.int32)
	return imagedata, labeldata

if __name__ == "__main__":
	imagedata, labeldata = read_file('hw1data.mat')