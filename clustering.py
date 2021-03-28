# clustering.py

# Brianna Drew
# March 27, 2021
# ID: #0622446
# Lab #9

# import required libraries and modules
import sklearn.datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy

data = sklearn.datasets.load_diabetes() # load diabetes dataset
numData = data['data'] # get data excluding class labels
scaledData = preprocessing.scale(numData) # scale the data

SSE = [] # create array to hold Sum of Squared Errors calculated for each number of clusters

for i in range (1,9): # comparing results from 1 to 9 clusters
    kmeans = KMeans(n_clusters = i, max_iter = 300) # create kMeans clustering for current iteration
    kmeans.fit(scaledData) # apply kMeans clustering to scaled data
    SSE.append(kmeans.inertia_) # add the SSE to the SSE array

# create and show plot
plt.plot(range(1,9), SSE)
plt.title("Determining Number of Clusters")
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.show()