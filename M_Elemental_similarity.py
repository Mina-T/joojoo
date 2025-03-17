import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import json
import os
import math


class array_similarity:
    def __init__(self, arrays):
        self.arrays = arrays

    def Minkowski_distance(self, p = 2):
     distance_matrix = np.zeros((len(self.arrays), len(self.arrays)))
     for i in range(len(self.arrays)):
         diff = np.abs(self.arrays[i] - self.arrays) ** p
         distance_matrix[i] = np.sum(diff, axis=1) ** (1 / p)

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

