# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Eric Elizes, Kiara Cleveland
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    
    # Using Gaussian Mixture model for clustering
    gmm = GaussianMixture(n_components=numVessels, random_state=100)
    predVessels = gmm.fit_predict(testFeatures)
    
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    scaler = StandardScaler()
    max_clusters = 30
    testFeatures = scaler.fit_transform(testFeatures)

    sum_of_squared_distances = []
    K = range(1, max_clusters + 1)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=100)
        km = km.fit(testFeatures)
        sum_of_squared_distances.append(km.inertia_)

    # Calculate the rate of change (first derivative)
    first_derivative = np.diff(sum_of_squared_distances)

    # Calculate the second derivative
    second_derivative = np.diff(first_derivative)

    # The elbow point is where the second derivative is maximum (most negative)
    elbow_point = np.argmin(second_derivative) + 2  # +2 because we lose 2 points in two np.diff()

    return predictWithK(testFeatures, elbow_point)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    