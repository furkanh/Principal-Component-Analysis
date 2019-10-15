# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:41:05 2019

@author: User
"""

import numpy as np

class PCA:
    def __init__(self, k=None):
        self.k = k
    
    def fit(self, X):
        if self.k is None or self.k > X.shape[0]:
            self.k = X.shape[0]
        X_mean = X.sum(axis=0)/X.shape[0]
        self.X_mean = X_mean
        X = (X-X_mean)
        E = np.matmul(X.transpose(), X)/(X.shape[0]-1)
        eig_values, eig_vectors = np.linalg.eig(E)
        eig_vectors = eig_vectors[:, eig_values.argsort()[::-1]]
        eig_values.sort()
        eig_values = eig_values[::-1]
        self.eig_values = np.real(eig_values)
        self.eig_vectors = np.real(eig_vectors)
        self.pve_list = []
        eig_sum = np.sum(self.eig_values)
        pve_sum = 0
        for i in range(self.eig_values.shape[0]):
          pve_sum = pve_sum + self.eig_values[i]
          self.pve_list.append(pve_sum/eig_sum)
    
    def transform(self, X):
        X = X-self.X_mean
        Z = np.matmul(X, self.eig_vectors)[:,:self.k]
        return Z