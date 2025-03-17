import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import json
import os
import math
from numpy.linalg import pinv # pseudo-inverse of a matrix

class Array_similarity:
    def __init__(self, arrays):
        self.arrays = arrays

    def Minkowski_distance(self, p = 2):
     distance_matrix = np.zeros((len(self.arrays), len(self.arrays)))
     for i in range(len(self.arrays)):
         for j in range(len(self.arrays)):
             diff = np.abs(self.arrays[i] - self.arrays[j]) ** p
             distance_matrix[i][j] = np.sum(diff) ** (1 / p)

     return distance_matrix

    def cosine_similarity(self):
        norms = np.linalg.norm(self.arrays, axis=1, keepdims=True)
        normalized_vectors = self.arrays / norms
        cosine_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        return cosine_matrix

    def Pearson_correlation(self):
        correlation_matrix = np.corrcoef(vectors)
        return correlation_matrix

    def RMSD(self):
        '''
        Root Mean Square Deviation
        '''
        rmsd_matrix = np.zeros((len(self.arrays), len(self.arrays)))
        for i in range(len(self.arrays)):
            diff = (self.arrays[i] - self.arrays) ** 2
            rmsd = np.sqrt(np.sum(diff)/len(self.arrays))
            rmsd_matrix[i] = rmsd
        return rmsd_matrix

    def cluster_similarity(self):
        # from github : https://github.com/smutaogroup/UMAP_analyses/blob/main/cluster_similarity.py
        pass

    def Silhouette_coefficient(slef):
        # from https://scikit-learn.org/stable/modules/clustering.html
        pass


def standardize_array(array):
    '''
    variables will be standardized to have a mean of 0 and a standard deviation of 1.
    '''
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    std[std == 0] = 1
    standardized_array = (array- mean) / std

    return standardized_array

def PCA_array(array):
    X = standardize_array(array)
    covariance_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    return eigen_values, eigen_vectors
