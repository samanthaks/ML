import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet_2L():
    '''
        2 layer neural network for 1-d functions
    '''

    def __init__(self,data_file,k):
        '''
            Initialize the data
        '''
        self.X,self.Y = self.read_data(data_file)
        self.k = k
        self.W1,self.b1,self.W2,self.b2 = self.gradient_descent()

    def read_data(self,data_file):
        '''
            Gets x and y from file
        '''
        data = spio.loadmat(data_file)
        X = np.asarray(data['X'], dtype=np.int32)
        Y = np.asarray(data['Y'], dtype=np.int32)
        print('Finished getting X and Y from the file')
        return X,Y

    def gradient_descent(self):
        ''' 
            Learn the network parameters
        '''
        # initialize weights randomly
        learn = 0.01 
        convergence = False 
        epoch = 1
        W1 = np.random.rand(1,self.k)
        b1 = np.random.rand(self.k,1)
        W2 = np.random.rand(self.k,1)
        b2 = np.random.rand(1,1)
        v1_arr = []
        v2_arr = []
        while not convergence:
            print("epoch: ", epoch)
            for x,y in zip(self.X,self.Y):
                v1 = self.f_layer(W1,b1,x) # k x 1 hidden layer
                v2 = self.f_layer(W2,b2,v1) # 1 x 1 y output
                v1_arr.append(v1)
                v2_arr.append(v2)
            # update weight parameters
            W1_change = learn * self.W1_deriv()
            b1_change = learn * self.b1_deriv()
            W2_change = learn * self.W2_deriv()
            b2_change = learn * self.b2_deriv()
            W1 = np.subtract(W1,W1_change)
            b1 = np.subtract(b1,b1_change)
            W2 = np.subtract(W2,w2_change)
            b2 = np.subtract(b2,b2_change)
            epoch += 1
        return W1,b1,W2,b2

    def f_layer(self,W,b,v):
        '''
            x to k x 1 hidden layer, then k x 1 hidden layer to y
        '''
        param = np.add(np.matmul(W.T,v),b)
        return self.activation(param)

    def activation(self,param):
        '''
            Activation function
        '''
        return 1/(1+np.exp(-param))

    def W1_deriv():
        '''
            Partial derivative of E w.r.t W1
        '''
        pass

    def b1_deriv():
        '''
            Partial derivative of E w.r.t b1
        '''
        pass

    def W2_deriv():
        '''
            Partial derivative of E w.r.t. W2
        '''
        pass

    def b2_deriv():
        '''
            Partial derivative of E w.r.t. b2
        '''
        pass

    def evaluate():
        '''
            Plot network output along with given Y values for each input value X
        '''
        y_hats = []
        for x,y in zip(self.X,self.Y):
            v1 = f_layer(W1,b1,x)
            v2 = f_layer(W2,b2,v1)
            y_hats.append(v2)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.X,y_hats,c='r',label='Network output')
        ax1.scatter(self.X,self.Y,c='b',label='Given Y Values')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Network output and given Y values vs. X')
        plt.legend(loc='lower left')
        plt.savefig('problem6.png')
        plt.show()

if __name__ == "__main__":
    nn = NeuralNet_2L(data_file='hw2data_2.mat',k=5)