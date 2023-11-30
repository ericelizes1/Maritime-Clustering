# -*- coding: utf-8 -*-
"""
This file is a copy of predictVessel.py, but adjusted to test the performance
of approaches using a dynamic k number of clusters.

@author: Eric Elizes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import metric
from functools import partial

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
                random_state=100)
    predVessels = km.fit_predict(testFeatures)
    
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

def predictWithoutK_Silhouette(testFeatures, max_clusters=30):
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    silhouette_scores = []
    K = range(2, max_clusters + 1)  # Silhouette score is not defined for k=1
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=100)
        labels = km.fit_predict(testFeatures)
        score = silhouette_score(testFeatures, labels)
        silhouette_scores.append(score)
        
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    return predictWithK(testFeatures, optimal_k)

def predictWithoutK_Elbow(testFeatures, max_clusters=30):
    scaler = StandardScaler()
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

def predictWithoutK_XMeans(testFeatures, trainFeatures=None, trainLabels=None, metrictype=metric.type_metric.EUCLIDEAN):
    # Unsupervised prediction, so training data is unused

    # Initialize centers using KMeans++ method
    initial_centers = kmeans_plusplus_initializer(testFeatures, 2).initialize()

    # Create instance of XMeans algorithm. The 'criterion' parameter can be adjusted.
    xmeans_instance = xmeans(testFeatures, initial_centers, ccore=True, 
                            criterion=metrictype)

    # Run clustering
    xmeans_instance.process()
    
    # Extracting the optimal number of clusters
    clusters = xmeans_instance.get_clusters()
    optimal_number_of_clusters = len(clusters)

    # Now use this optimal number in your original clustering method
    return predictWithK(testFeatures, optimal_number_of_clusters, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    #plotVesselTracks(features[:,[2,1]])
    #plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)


    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    
    def evaluate_clustering(predict_function, features, labels, method_name):
        # Run the prediction function to get cluster labels
        predicted_labels = predict_function(features)

        # Calculate the number of unique clusters
        num_clusters = np.unique(predicted_labels).size

        # Calculate the Adjusted Rand Index
        ari_score = adjusted_rand_score(labels, predicted_labels)

        # Print the results
        print(f'Adjusted Rand Index for {method_name} (K = {num_clusters}): {ari_score}')

    # Example usage
    evaluate_clustering(predictWithoutK, features, labels, "Arbitrary K")
    evaluate_clustering(predictWithoutK_Elbow, features, labels, "Elbow Method")
    evaluate_clustering(predictWithoutK_Silhouette, features, labels, "Silhouette Method")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.EUCLIDEAN), features, labels, "XMeans (Euclidean)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.EUCLIDEAN_SQUARE), features, labels, "XMeans (Euclidean Square)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.MANHATTAN), features, labels, "XMeans (Manhattan)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.CHEBYSHEV), features, labels, "XMeans (CHEBYSHEV)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.MINKOWSKI), features, labels, "XMeans (MINKOWSKI)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.CANBERRA), features, labels, "XMeans (CANBERRA)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.CHI_SQUARE), features, labels, "XMeans (CHI_SQUARE)")
    evaluate_clustering(partial(predictWithoutK_XMeans, metrictype=metric.type_metric.GOWER), features, labels, "XMeans (GOWER)")




    
    # Plotting Function
    #def plotVesselTracks(coordinates, cluster_labels, title):
    #    plt.figure()
    #    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, cmap='viridis', marker='.')
    #    plt.xlabel('Longitude')
    #    plt.ylabel('Latitude')
    #    plt.title(title)
    #    plt.show()

    # Vessel tracks by cluster with original K
    #plotVesselTracks(features[:,[2,1]], predVesselsWithK, 'Vessel tracks by cluster with K')

    # Vessel tracks by cluster with arbitrary K=20
    #plotVesselTracks(features[:,[2,1]], predVesselsWithoutK_Arbitrary, 'Vessel tracks by cluster without K (Arbitrary)')

    # Vessel tracks by cluster with Elbow Method
    #plotVesselTracks(features[:,[2,1]], predVesselsWithoutK_Elbow, 'Vessel tracks by cluster without K (Elbow Method)')

    # Vessel tracks by cluster with Silhouette Method
    #plotVesselTracks(features[:,[2,1]], predVesselsWithoutK_Silhouette, 'Vessel tracks by cluster without K (Silhouette Method)')

    # Vessel tracks by cluster with XMeans
    #plotVesselTracks(features[:,[2,1]], predVesselsWithoutK_XMeans, 'Vessel tracks by cluster without K (XMeans)')
    
    # Vessel tracks by actual labels
    #plotVesselTracks(features[:,[2,1]], labels, 'Vessel tracks by label')

    