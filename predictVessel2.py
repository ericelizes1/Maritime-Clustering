# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.
@author: Kiara
This is for part one with given number of vessels
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import loadData, plotVesselTracks

def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
    predVessels = km.fit_predict(testFeatures)
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

def evaluatePrediction(testFeatures, testLabels, numVessels, trainFeatures=None, trainLabels=None):
    # Run prediction algorithms and check accuracy
    # Prediction with specified number of vessels
    predVesselsWithK = predictWithK(testFeatures, numVessels, trainFeatures, trainLabels)
    ariWithK = adjusted_rand_score(testLabels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(testFeatures, trainFeatures, trainLabels)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(testLabels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: {ariWithoutK}')
    
    return predVesselsWithK, predVesselsWithoutK

if __name__ == "__main__":
    # Load test data
    testData = loadData('set2noVID.csv')
    testFeatures = testData[:, 2:]
    testLabels = testData[:, 1]
    
    # Load training data (may not necessarily be used)
    trainData = loadData('set1.csv')
    trainFeatures = trainData[:, 2:]
    trainLabels = trainData[:, 1]
    
    # Run evaluation
    numVessels = np.unique(testLabels).size
    predVesselsWithK, predVesselsWithoutK = evaluatePrediction(testFeatures, testLabels, numVessels, trainFeatures, trainLabels)
    
    # Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(testFeatures[:, [2, 1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    
    plotVesselTracks(testFeatures[:, [2, 1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    
    plotVesselTracks(testFeatures[:, [2, 1]], testLabels)
    plt.title('Vessel tracks by label')
    plt.show()

