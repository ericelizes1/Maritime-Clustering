# -*- coding: utf-8 -*-
"""
This file is a copy of predictVessel.py, but adjusted to test the performance
of approaches using a dynamic k number of clusters and different scalers.

@author: Eric Elizes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from utils import loadData, plotVesselTracks

def predictWithK(testFeatures, numVessels, scaler, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    scaledFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
    predVessels = km.fit_predict(scaledFeatures)
    return predVessels

def evaluate_clustering(predict_function, features, labels, numVessels, scaler, method_name):
    # Run the prediction function to get cluster labels
    predicted_labels = predict_function(features, numVessels, scaler)

    # Calculate the Adjusted Rand Index
    ari_score = adjusted_rand_score(labels, predicted_labels)

    # Print the results
    print(f'Adjusted Rand Index for {method_name} with {scaler.__class__.__name__}: {ari_score}')

if __name__ == "__main__":
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    numVessels = np.unique(labels).size

    for scaler in scalers:
        evaluate_clustering(predictWithK, features, labels, numVessels, scaler, "Dynamic K")

