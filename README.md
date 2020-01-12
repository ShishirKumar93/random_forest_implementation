# random_forest_implementation
Python implementation from scratch - Random Forest

Here, we implement random forests with boostrapping. We create random forests by calling upon DecisionTree class built in one of the previous projects. In the file dtree.py, we have the following functionalities: 

1) DecisionNode : class that defines a decision node.
2) LeafNode : class that defines leaves of the tree.
3) bestsplit : function that gives the best split point using gini / MSE criteria. We select a subset of features and randomly choose 11 datapoints from these features to find the best split point. This increases the speed of our algorithm greatly.
4) DecisionTree : class that constructs the decision tree recursively, using the best split point obtained from bestsplit function.
5) RegressionTree621 / ClassifierTree621 : classes built upon DecisionTree class that encapsulate functionalities like fit(), predict() etc.

The file rf.py has the following functionality:

1. RandomForest621: creates a forest of decision trees with bootstrapped data. Also has the option to calculate out of bag score.
2. RandomForestRegressor621 / RandomForestClassifier621: inherits from RandomForest621 and has additional methods for prediction and scoring.

File test_rf.py has tests designed to check performance of our implementation vs scikit-learn.

Thanks to Prof [Terence Parr](https://github.com/parrt) for his guidance and support in this school project.

