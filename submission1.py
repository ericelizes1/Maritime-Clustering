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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None, return_bic=False, skip_tuning=False):
    # Define scaler
    scaler = StandardScaler()
    
    # Initialize variables for best model and parameters
    best_gmm = None
    lowest_bic = np.infty
    best_params = {}

    # Scale the features
    scaled_features = scaler.fit_transform(testFeatures)

    # Check if parameter tuning is to be skipped
    if skip_tuning:
        # Use GaussianMixture with default settings
        gmm = GaussianMixture(n_components=numVessels, random_state=100)
        gmm.fit(scaled_features)
        predVessels = gmm.predict(scaled_features)
        bic = gmm.bic(scaled_features)

        if return_bic:
            return predVessels, bic
        else:
            return predVessels
        
    # Define additional hyperparameters for GMM
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    init_params_options = ['kmeans', 'random']
    max_iter_options = [100, 200, 300]

    # Iterate over GMM hyperparameters
    for cov_type in covariance_types:
        for init_param in init_params_options:
            for max_iter in max_iter_options:
                gmm = GaussianMixture(n_components=numVessels, 
                    covariance_type=cov_type, 
                    init_params=init_param, 
                    max_iter=max_iter, 
                    random_state=100)
                gmm.fit(scaled_features)
                bic = gmm.bic(scaled_features)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    best_params = {
                        'covariance_type': cov_type,
                        'init_params': init_param,
                        'max_iter': max_iter
                    }

    # Predict using the best GMM model
    predVessels = best_gmm.predict(scaled_features)

    # Print the best parameters
    print("Best GMM Parameters:")
    print(f"Covariance Type: {best_params['covariance_type']}")
    print(f"Init Params: {best_params['init_params']}")
    print(f"Max Iterations: {best_params['max_iter']}")

    if return_bic:
        return predVessels, best_gmm.bic(scaled_features)
    else:
        return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    max_clusters = 30
    metrics = []

    # Evaluate each number of clusters using default settings for GMM
    for k in range(1, max_clusters + 1):
        _, bic = predictWithK(testFeatures, k, return_bic=True, skip_tuning=True)
        metrics.append(bic)

    # Find the elbow point using the BIC
    first_derivative = np.diff(metrics)
    second_derivative = np.diff(first_derivative)
    elbow_point = np.argmin(second_derivative) + 2

    # Now use predictWithK with parameter tuning
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
    