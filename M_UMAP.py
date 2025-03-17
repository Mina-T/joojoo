import pickle
import numpy as np
import umap
import json
import os
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import umap
import math

def get_drautz_label(key):
    if 'cluster' in key:
        label = 1
    elif 'sp2' in key:
        label = 2
    elif 'sp3' in key:
        label = 3
    elif 'bulk' in key:
        label = 4
    elif 'amorph' in key:
        label = 5
    else:
        label = None
    return label

def descr_to_array(path, _file):
    desc_file = pickle.load(open(os.path.join(path , _file), 'rb'))
    array_lst = []
    for key, sys in desc_file.items():
        try:
            _label = get_drautz_label(key)
            for atom in sys:
                labeled_atom = np.append(atom, _label)
                array_lst.append(labeled_atom)
        except NameError:
            for atom in sys:
                array_lst.append(atom)
    array = np.array(array_lst)
    print('number of atoms in data set: ', len(array))
    np.save(open(os.path.join(path, 'flattened_descr.npy'), 'wb'), array)

def pearson_corr(X, Y):
    """Calculates the Pearson correlation coefficient between two datasets."""
    corr_matrix = np.corrcoef(X.T, Y.T)  
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix[:X.shape[1], X.shape[1]:].mean()


def perform_umap(path, metric, metric_kwds=None):
    array = np.load(os.path.join(path, 'flattened_descr.npy'), allow_pickle = True)
    array = array[:, :-1]
    best_score = -1
    best_params = {}
    n_range = range(10, 110,10)
    min_d_range = np.arange(0.0, 0.75, 0.25)

    for n in n_range:
        for m in min_d_range:
          fit = umap.UMAP(n_neighbors = n, min_dist = m, n_components = 2, metric = metric, random_state=42,  metric_kwds=metric_kwds)
          u = fit.fit_transform(array)
#          score = pearson_corr(array, u)
#          print(f"n_neighbors={n}, min_dist={m}=> Pearson correlation: {score}")
#          if score > best_score:
#                best_score = score
#                best_params = {'n_neighbors': n, 'min_dist': m}
          if metric_kwds:
              np.save(open(f"{metric}_{metric_kwds['p']}_n{n}_dist{m}.npy",  'wb'), u)
          else:
              np.save(open(f'{metric}_n{n}_dist{m}.npy',  'wb'), u)
          print(f'{metric}_n{n}_dist{m}.npy', flush = True)
#    print('best_params: ', best_params, flush = True)
    print('Done', flush = True)

def kmeans(n, umap_data):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n, random_state=42)
    clusters = kmeans.fit_predict(umap_data)
    return clusters

def plot_umap(path, n, min_d, metric = 'euclidean'):
    u = np.load(os.path.join(path, f'{metric}_n{n}_dist{min_d}.npy'), allow_pickle = True)
    scatter = plt.scatter(u[:,0], u[:,1], s = 4, color = 'g', alpha = 0.2, zorder=2)
    plt.legend(loc = (1,0.5))
    plt.title( f'{metric:} u_{n}_dist{min_d}')
    plt.show()

