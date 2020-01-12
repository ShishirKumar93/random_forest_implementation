#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:19:33 2019

@author: MrMndFkr
"""

import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import resample
from dtree import *

class RandomForest621:
    
    
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.oob_index = []
        self.trees = []
        self.nunique = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.nunique = len(np.unique(y))
        self.trees = []
        for i in range(0,self.n_estimators):
            boot_index = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot, y_boot = X[boot_index], y[boot_index]
            oob_indexes = np.array(list(set(range(X.shape[0])) - set(boot_index)))
            tree = self.tree_type
            tree.fit(X_boot, y_boot)
            self.trees.append(tree.root)
            self.oob_index.append(oob_indexes)
        if self.oob_score:  self.oob_score_ = self.compute_oob_score(X,y)


class RandomForestRegressor621(RandomForest621):


    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_type = RegressionTree621()

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        n_obs = np.zeros(X_test.shape[0])
        n_preds = np.zeros(X_test.shape[0])
        for tree in self.trees:
            array_leaves = np.apply_along_axis(tree.leaf, axis=1, arr=X_test)
            obs = np.array([leaf.n for leaf in array_leaves])
            n_obs += obs
            preds = np.array([leaf.prediction for leaf in array_leaves])
            n_preds += np.multiply(obs,preds)
        return n_preds / n_obs
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        preds = self.predict(X_test)
        return r2_score(y_test, preds)
    
    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_preds = np.zeros(X.shape[0])
        for index, tree in enumerate(self.trees):
            X_oob = X[self.oob_index[index]]
            array_leaves = np.apply_along_axis(tree.leaf, axis=1, arr=X_oob)
            array_counts = np.array([leaf.n for leaf in array_leaves])
            oob_counts[self.oob_index[index]] += array_counts
            oob_preds[self.oob_index[index]] += np.multiply(array_counts, 
                     np.array([leaf.prediction for leaf in array_leaves]))
        oob_avg_preds = oob_preds[oob_counts > 0] / oob_counts[oob_counts > 0]
        return r2_score(y[oob_counts>0], oob_avg_preds)

    
class RandomForestClassifier621(RandomForest621):
    
    
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_type = ClassifierTree621()

    def predict(self, X_test) -> np.ndarray:
        preds_classes = np.zeros((X_test.shape[0],self.nunique))
        for tree in self.trees:
            array_leaves = np.apply_along_axis(tree.leaf, axis=1, arr=X_test)
            n_preds = np.array([leaf.prediction for leaf in array_leaves])
            index = np.array(range(len(n_preds)))
            preds_classes[index,n_preds]+= 1
        return np.argmax(preds_classes, axis=1)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)
    
    def compute_oob_score(self, X, y):
        oob_counts = np.zeros(X.shape[0])
        oob_preds = np.zeros((X.shape[0],self.nunique))
        for index, tree in enumerate(self.trees):
            X_oob = X[self.oob_index[index]]
            array_leaves = np.apply_along_axis(tree.leaf, axis=1, arr=X_oob)
            array_counts = np.array([leaf.n for leaf in array_leaves])
            oob_counts[self.oob_index[index]] += 1
            preds = np.array([leaf.prediction for leaf in array_leaves])
            oob_preds[self.oob_index[index] , preds] += array_counts
        oob_votes = np.argmax(oob_preds[oob_counts > 0], axis=1)
        return accuracy_score(y[oob_counts>0], oob_votes)