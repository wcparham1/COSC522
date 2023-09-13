import numpy as np
import pandas as pd
#William Connor Parham
#Dr. Hairong Qi
#COSC 522
#Project 1
#9/19/2023

training_set = pd.read_csv('synth.tr')
testing_set = pd.read_csv('synth.te')
print(training_set)
print(testing_set)

#KNN Implementation
class KNN_Classifier:
    def __init__(self, k=1):
        self.n_neighbors = k

    def euclidian_distance(self, a, b):
        eucl_distance = 0.0

        for index in range(len(a)):
            eucl_distance += (a[index] - b[index] ** 2)
            euclidian_distance = np.sqrt(eucl_distance)
        
        return euclidian_distance


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict_knn(self, X): 
        #initialize prediction_knn as empty list 
        prediction_knn = []

        for index in range(len(X)):
            #initialize euclidian distances as empty list
            euclidian_distances = []

            for row in self.X_train:
                #for every row in X_train, find eucl_distance to X using
                #euclidian_distance() and append to euclidian_distances list
                eucl_distance = self.euclidian_distance(row, X[index])
                euclidian_distances.append(eucl_distance)
            
            #sort euclidian_distances in ascending order, and retain only k
            #neighbors as specified in n_neighbors(n_neighbors=k)
            neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

            #initialize dict to count class occurrences in y_train
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_tain[val]] = 1
                
            #max count labels to prediction_knn
            prediction_knn.append(max(count_neighbors, key=count_neighbors.get))
        
        return prediction_knn
    
    def display_knn(self, x):

        #initialize euclidian_distances as empty list
        euclidian_distances = []

        #for every row in X_train, find eucl_distance to x
        #using euclidian_distance() and append to euclidian_distances list
        for row in self.X_train:
            eucl_distance = self.euclidian_distance(row,x)
            euclidian_distances.append(eucl_distance)
        
        #sort euclidian-distances in ascneding order, and retain only k
        #neighbors as specified in n_neighbors (n_neighbors = k)
        neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

        #initiate empty display_knn_values list
        display_knn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_distances = euclidian_distances[index]
            display_knn_values.append((neighbor_index, e_distances))
        
        return display_knn_values
