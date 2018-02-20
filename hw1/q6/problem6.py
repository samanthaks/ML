
import scipy.io as spio
import numpy as np
import sys
from random import sample
import statistics
from collections import Counter


class Node(object):
    left = None
    right = None
    data = None
    threshold = None
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


def readfile(filename):
    '''
    Returns the images matrix and the labels matrix
    '''
    data = spio.loadmat(filename)
    X = np.asarray(data['X'], dtype=np.int32)
    Y = np.asarray(data['Y'], dtype=np.int32)
    return X,Y

def separate_data(X, Y, ratio):
    '''
    Create training and testing sets
    '''
    inds = sample(range(len(X)),int(ratio*len(X)))
    X_train = X[inds,]
    X_test = np.delete(X,inds,axis=0)
    Y_train = Y[inds]
    Y_test = np.delete(Y,inds,axis=0)
    return X_train,Y_train,X_test,Y_test


def build_decision_tree(examples, labels, threshold, depth, row=0):
    root = Node()
    
    #examples - a matrix of training data
    #target - the lables of the training data
    #attributes - ???
    #depth - the adjustable hyperparameter K that determines depth of tree
    """ This was ID3 but now we have to do something else
    set_all = set(exmaples)
    if len(set_all) == 1:
        root.data = list(set_all)[0]
        return root
    
    if attributes = None:
        val, counts = np.unique(data, return_counts=True)
        root.data = val[argmax(counts)]
        return root
        
    a = best_att(examples, attributes)
    root.data = a
    
    root.left = Node()
    
    root.right = Node()
    attributes = remove_att(attributes, a)
    """
    #Base case: There is only one element in the set OR max depth (K) has been reached
    if examples.size == 0:#is empty
        #make something up!
        root.data = random.randrange(0,9)
        return root
    elif depth == 0:
        #Take the mode of the labels of the remaining data    
        #Return the tree model
        root.data = Counter(labels.flatten()).most_common(1)
        return root
      
    
    #Split The Data Into Two Categories at threshold
    left_inds = []
    right_inds = []
    for i in range(examples.shape[1]):
        d_col = examples[:,i] # image
        d_label = labels[i] # number

        if examples[row][i] < threshold:
            left_inds.append(i)
        else:
            right_inds.append(i)
    left_examples = examples[:,left_inds]
    left_labels = labels[left_inds]
    right_examples = examples[:,right_inds]
    right_labels = labels[right_inds]
    
    left_child = Node()
    right_child = Node()
    #Run Decision Tree Builder on X < T, X >=T
    left_child = build_decision_tree(left_examples, left_labels, threshold, depth-1, row+1)    
    right_child = build_decision_tree(right_examples, right_labels, threshold, depth-1, row+1)
    
    #Aribitrary Threshold
    left_child.threshold = 128
    right_child.threshold = 128
    
    
    root.left = left_child 
    root.right = right_child
    
    return root


def best_att(examples, attributes):
    entropies = {}
    for a in attributes:
        entropies[a] = entropy(a)
    
    best = sorted(entropies, key=entropies.__getitem__, reverse=True)[0]
    return best
    
def entropy(data):
    entropy = 0.0
    
    val, counts = np.unique(data, return_counts=True)
    
    freqs = []
    for count in counts:
        freqs.append(float(count)/len(data))
    
    for p_y in freqs:
        if p_y != 0.0:
            entropy += p_y * np.log(1/p_y)

    return entropy



def run_tree_classifier(root, data, threshold):
    idx = 0
    """
    while threshold != None:
        if data[idx] < root.threshold:
            root = root.left
        elif root == root.right:
            idx += 1
    """
    return root.data



def test_tree_classifier(root, test_data, test_labels):
    passed = 0
    for i in range(test_data.shape[1]):
        if test_labels[i] == run_tree_classifier(root, test_data[:,i],128):
            passed += 1
    accuracy = passed/test_data.shape[1]
    return accuracy        



def main():
    if len(sys.argv) < 3:
        print("Usage: run_decision_classifier <data_file> <K>")
        return
    
    data_file = sys.argv[1]
    K = int(sys.argv[2])
    
    if K < 1:
        print("K must be greater than 0")
        return
    X, Y = readfile(data_file)
    X_train,Y_train,X_test,Y_test = separate_data(X,Y,.80)
    
    tree_classifier = build_decision_tree(X_train, Y_train, 128, K)
    accuracy = test_tree_classifier(tree_classifier, X_test, Y_test)
    
    print("The accuracy of the classifer generated from the data is: %f" % accuracy)


if __name__ == "__main__":
    main()

