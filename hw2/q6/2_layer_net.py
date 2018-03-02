import pickle
import scipy.io as spio
import random
import numpy as np


data_file = 'hw2data_2.mat'

class NeuralNet_2L():
    '''
        2 layer neural network for 1-d functions
    '''

    def __init__(self, train_ratio):
        '''
            Initialize
        '''
        # self.k = 4
        self.X, self.Y = self.read_data()
        self.train_ratio = train_ratio
        self.X_train,self.X_test,self.Y_train,self.Y_test = self.separate_data()
        self.weights = self.gradient_descent(1)
        if self.train_ratio < 1:
            self.accuracy = self.evaluate()


    def read_data(self, filename):
        '''
            reads file
        '''
        data = spio.loadmat(data_file)
        X = np.asarray(data['X'], dtype=np.int32)
        Y = np.asarray(data['Y'], dtype=np.int32)
        print('Finished reading in the file')
        return X,Y

    def separate_data(self):
        '''
            Create training and testing sets
        '''
        inds = sample(range(len(self.X)),int(self.train_ratio*len(self.X)))
        X_train = self.X[inds,]
        X_test = np.delete(self.X,inds,axis=0)
        Y_train = self.Y[inds]
        Y_test = np.delete(self.Y,inds,axis=0)
        return X_train,Y_train,X_test,Y_test


    def gradient_descent(self, passes):
        ''' 
            Learn the parameters of the 2-layer ff nn.
        '''


    # where k is the size of the immediate layer
    def evaluate(self, x_data, y_data, k):
        ''' 
            Evaluate the current layer of the Network
        '''

        #initialize weight parameters randomly
        self.w1 = random.uniform(-1,1)
        self.w2 = random.uniform(-1,1)
        self.b1 = random.uniform(-1,1)
        self.b2 = random.uniform(-1,1) 


        # reaches congergence when the difference between the previous and 
        # the current loss is small
        # change loop to run until convergence. 
        # for now just leave on set epochs so we can observe
        epochs = 4
        for x in range(epochs):
            for idx in range(len(x_data)):
                # compute network output




                # calculate loss
                x_1 = X.dot(self.w1) + self.b1
                # apply activation function
                a_1 = 1 / (1 + np.exp(-x_1))





