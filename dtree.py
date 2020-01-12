#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:19:33 2019

@author: MrMndFkr
"""

import numpy as np
#from dtreeviz.trees import *
#from lolviz import *
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import math


def gini(y):
    "Return the gini impurity score for values in y"
    y_unique = np.unique(y)
    prob_sum = 0
    for val in y_unique:
        prob_y = np.sum(y==val) / len(y)
        prob_sum += prob_y**2
    return 1 - prob_sum

# =============================================================================
# y = [0,0,0,1,1,1]
# np.unique(y)
# gini(y)
# =============================================================================


class DecisionNode:
    
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
    
    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)
    
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if isinstance(self, LeafNode): return self
        elif x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)


class LeafNode:
    
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
    
    def predict(self, x_test):
        # return prediction
        return self.prediction
    
    def leaf(self, x_test):
        return self
        
    

# =============================================================================
# x_test = np.array([[1,2,3],[4,3,1],[1,3,4]])
# dec_node = DecisionNode(2, 5, LeafNode(np.array([3,7,3]),3), LeafNode(np.array([7,7,4,5]),7))
# treeviz(dec_node)
# for i in [-9,0,3,5,9,100]:
#     if i<=5: assert dec_node.predict(np.array([1,2,i,4,5])) == 3
#     else: assert dec_node.predict(np.array([1,2,i,4,5])) == 7
# =============================================================================

def RFbestsplit(X,y, max_features, min_samples_leaf, loss=None):
    best = {'col':-1,'split':-1,'loss':loss(y)}
    vars_index = np.random.choice(range(0,X.shape[1]), max(round(max_features * X.shape[1]),1), replace=False)
    for var in vars_index:
        if len(X[:,0]) > 11:
            split_values = np.random.choice(X[:, var], 11, replace=False) ## take 11 if observations > 11
        else:   split_values = list(X[:, var]) ## else take all
        for split_value in split_values:
            y_left = y[np.where(X[:,var] <= split_value)]
            y_right = y[np.where(X[:,var] > split_value)]
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf: continue
            weighted_loss = (len(y_left) * loss(y_left) + len(y_right) * loss(y_right)) / len(y)
            if weighted_loss==0: return var, split_value
            if weighted_loss < best['loss']: best = {'col':var, 'split':split_value, 'loss':weighted_loss} 
    return best['col'], best['split']

# =============================================================================
# x = np.array([[1,2,3],[4,5,6],[3,4,3]])
# y_reg = np.array([5,-9,3])
# y_class = np.array([1,0,1])
# RFbestsplit(x,y_reg, 0.1, 1, loss=np.std)
# RFbestsplit(x,y_class, 0.1, 1, loss=gini)
# =============================================================================

class DecisionTree621:
    
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        
    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:  return self.create_leaf(y)
        col, split = RFbestsplit(X, y, self.max_features, self.min_samples_leaf, self.loss)
        if col == -1:   return self.create_leaf(y)
        left_child = self.fit_(X[X[:,col] <= split], y[X[:,col] <= split])
        right_child = self.fit_(X[X[:,col] > split], y[X[:,col] > split])
        return DecisionNode(col, split, left_child, right_child)
        
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        preds = np.zeros((1,len(X_test[:,0])))
        return np.apply_along_axis( self.root.predict, axis=1, arr=X_test )
    

class RegressionTree621(DecisionTree621):
    
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, loss=np.std)
        self.max_features = max_features
    
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))
    
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))

class ClassifierTree621(DecisionTree621):
    
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, loss=gini)
        self.max_features = max_features
    
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))
        
    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, int(mode(y)[0]))
    
