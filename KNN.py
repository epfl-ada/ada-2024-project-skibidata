import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import heapq


class KNN_based_recommender:
    def __init__(self, k_neighbors_reco=20, k_neighbors_fit=20,
                 metric='cosine', normalization='baseline', reco_type='user_based', sparse_train, sparse_test,
                 df_ratings):
        """
        Initialize the recommender system.
        """
        self.k_neighbors_reco = k_neighbors_reco
        self.k_neighbors_fit = k_neighbors_fit
        self.metric = metric
        self.normalization = normalization
        self.type = reco_type
        self.global_mean = df_ratings['rating'].mean()
        self.sparse_train = sparse_train
        self.sparse_test = sparse_test


    def compute_similarities(self):
        print("=== Compute similarities with KNN ===")
        if self.type == 'user_based':
            knn = NearestNeighbors(metric=self.metric, algorithm='brute', n_jobs=-1, n_neighbors=self.k_neighbors_fit)
            knn.fit(self.sparse_train)

        elif self.type == 'item_based':
            knn = NearestNeighbors(metric=self.metric, algorithm='brute', n_jobs=-1, n_neighbors=self.k_neighbors_fit)
            knn.fit(self.sparse_train.T)

        return knn

    def get_neighbors(self):
        
        return None

    def default_prediction(self):
        return self.global_mean


