import os
import torch
import time
import numpy as np
import ml_collections

def OCS_config():
    config = ml_collections.ConfigDict()
    config.kernel='rbf'
    #config.nu = 0.8
    config.tol = 1e-5
    #config.coef0=0.1
    config.gamma='auto'
    return config
def opticsClustering_config():
    config = ml_collections.ConfigDict()
    config.min_samples= 2
    config.max_eps = np.inf
    config.metric = 'euclidean'
    config.leaf_size = 2
    return config
def kmeans_config():
    config = ml_collections.ConfigDict()
    config.n_clusters= 2
    config.init = 'k-means++'
    config.algorithm = 'lloyd'
    config.tol = 5e-4
    return config
def agglomerativeClustering_config():
    config = ml_collections.ConfigDict()
    config.n_clusters= 2
    config.metric = 'euclidean'
    config.linkage = 'ward'
    config.compute_distances = True
    return config
def birch_config():
    config = ml_collections.ConfigDict()
    config.n_clusters= 2
    config.threshold= 0.7819798519436253
    config.branching_factor = 40
    return config
def spectral_config():
    config = ml_collections.ConfigDict()
    config.n_clusters= 2
    config.assign_labels = 'kmeans'
    config.eigen_solver = 'arpack'
    config.affinity = 'nearest_neighbors'
    return config