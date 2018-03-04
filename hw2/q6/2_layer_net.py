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
        self.W1,self.b1,self.W2,self.b2 = self.gradient_descent(1)
        self.evaluate()

    def read_data(self,data_file):
        '''
            Gets x and y from file
        '''
        data = spio.loadmat(data_file)
        X = np.asarray(data['X'], dtype=np.float32)
        Y = np.asarray(data['Y'], dtype=np.float32)
        print('Finished getting X and Y from the file')
        return X,Y

    def gradient_descent(self, max_epoch):
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
        while (not convergence) or epoch < max_epoch:
            print("epoch: ", epoch)
            for x,y in zip(self.X,self.Y):
                print("x = %s" % x)
                print("w1 = %s" % W1)
                print("b1 = %s" % b1)
                v1 = self.f_layer(W1,b1,x) # k x 1 hidden layer
                v2 = self.f_layer(W2,b2,v1) # 1 x 1 y output
                v1_arr.append(v1)
                v2_arr.append(v2)
            #initialize weight parameters
            W1_change = 0
            b1_change = 0
            W2_change = 0
            b2_change = 0
            # update weight parameters
            for x,y in zip(self.X,self.Y):
                print("x is %s" % x)
                #print(self.W1_deriv( W1, W2, b1, b2, x, y))
                W1_change +=  self.W1_deriv( W1, W2, b1, b2, x, y)
                b1_change +=  self.b1_deriv( W1, W2, b1, b2, x, y)
                W2_change +=  self.W2_deriv( W1, W2, b1, b2, x, y)
                b2_change +=  self.b2_deriv( W1, W2, b1, b2, x, y)
           
            
            n = self.X.size
            #divide each by n and then multiply by the learning ratio
            W1_change *= (1/n) * learn
            b2_change *= (1/n) * learn
            W2_change *= (1/n) * learn
            b2_change *= (1/n) * learn
            print("W1_change is %s" % W1_change)
            W1 = np.subtract(W1,W1_change)
            b1 = np.subtract(b1,b1_change)
            W2 = np.subtract(W2,W2_change)
            b2 = np.subtract(b2,b2_change)
            print("W1 is %s" % W1)
            if np.where(W1_change < .001): #this is an arbitrary learning factor
                convergence = True
            epoch += 1
        


        return W1,b1,W2,b2

    def f_layer(self,W,b,v):
        '''
            x to k x 1 hidden layer, then k x 1 hidden layer to y
        '''
        #print("W.T shape = %s" % W.shape)
       # print("v.shape = %s" % v.shape)
       # print("b.shape = %s" % b.shape)
        param = np.add(np.matmul(W.T,v),b)
        
        print("W.T is {} v is {} b is {}".format(W.T.shape, v.shape, b.shape))
        print("param is %s" % param)

        return self.activation(param)

    def activation(self,param):
        '''
            Activation function
        '''
        act = 1/(1+np.exp(-param))

       # print("The activation result is %s" % act)
        return act
    def W1_deriv(self, W1, W2, b1, b2, x, y):
        '''
            Partial derivative of E w.r.t W1
        '''
        v1 = self.f_layer(W1, b1, x)
        n2 = self.f_layer(W2, b2, v1) 
        print("v1 is %s" % v1)
        print("n2 is %s" % n2)
        deriv = (n2-y) * n2 * (1 - n2) * v1 * (1 - v1) * W2 * x
        print("the derivative is %s" % deriv)
        return deriv

    def b1_deriv(self, W1, W2, b1, b2, x, y):
        '''
            Partial derivative of E w.r.t b1
        '''
        v1 = self.f_layer(W1, b1, x)
        n2 = self.f_layer(W2, b2, v1) 
        deriv = (n2-y) * n2 * (1 - n2) * v1 * (1 - v1) * W2 

        return deriv


    def W2_deriv(self, W1, W2, b1, b2, x, y):
        '''
            Partial derivative of E w.r.t. W2
        '''

        v1 = self.f_layer(W1, b1, x)
        n2 = self.f_layer(W2, b2, v1) 
        deriv = (n2-y) * n2 * (1 - n2) * v1 
        return deriv

    def b2_deriv(self, W1, W2, b1, b2, x, y):
        '''
            Partial derivative of E w.r.t. b2
        '''
        v1 = self.f_layer(W1, b1, x)
        n2 = self.f_layer(W2, b2, v1) 
        deriv = (n2-y) * n2 * (1 - n2) 
        return deriv

    def evaluate(self):
        '''
            Plot network output along with given Y values for each input value X
        '''

        print(self.W1)
        print(self.b1)
    
        y_hats = []
        for x,y in zip(self.X,self.Y):
            print(x)
            v1 = self.f_layer(self.W1,self.b1,x)
            v2 = self.f_layer(self.W2,self.b2,v1)
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