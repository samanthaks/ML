%matplotlib inline
#Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def makeXY(point_array):
    x = []
    y = []
    for i in range(len(point_array)):
        x.append(point_array[i][0])
        y.append(point_array[i][1])
    return x,y

def main(learning_rate):
    #Initialize all city longitudes and latitudes randomly
    cities = []
    for i in range(9):
        city = np.random.rand(2,) * 4000
        cities.append(city)
    
    #Plot results
    labels = ['BOS','NYC','DC','MIA','CHI','SEA','SF','LA','DEN']
   
    #Define distance matrix
    D = np.asarray([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
        [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
        [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
        [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
        [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
        [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
        [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
        [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
        [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])
    #print(D)
    #Do gradient decent for a certain number to cycles
        #Array of derivatives
    derivative = np.ones((9,2)) * 100

    for n in range(5000):
        for i in range(9):
            for j in range(9):
                if i != j:
                    norm  = np.linalg.norm(cities[i]-cities[j])
                    #print(norm)
                    first = 2*((norm - D[i][j])) * (1/norm)
                    second = cities[i] - cities[j]
                    #print("first: %s" % first)
                    #print("Printing second: %s" % second)
                    #print("Printing derivative: %s" % derivative)
                    derivative[i] =  first * second
                    #print(derivative)
                    cities[i] = cities[i] - (derivative[i] * learning_rate)
                    #print("Cities %d, is %s" % (i, cities[i]))
        if n % 100 == 0:
            print("Running test n=%d" % n)
            x,y = makeXY(cities)
            plt.scatter(x,y)
            for label, x, y in zip(labels, x, y):
                plt.annotate(label, xy=(x,y))
            plt.show()
            print(derivative) 
if __name__ == "__main__":
    main(.1)