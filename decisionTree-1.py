import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # leaf node 
    
    def if_leaf_node(self):
        return self.value is not None
    
class MyDecisionTree:
    
    def __init__(self, min_sample_split=2, max_depth=10, n_features=None):
        
        self.min_sample_split = min_sample_split 
        
        self.max_depth = max_depth 
        
        self.n_features = n_features 
        self.root = None
        
    def fit(self, X, y):
                
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        
        self.root = self._build_tree(X, y)
        
        # self.print_tree()  
        
    def _build_tree(self, X, y, depth=0):
        
        n_samples, n_features = X.shape   
        n_labels = len(np.unique(y))        
        
        # check the stopping criteria 
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            
            if len(y) > 0:
                leaf_node_value = self.calculate_leaf_value(y) 
                return Node(value=leaf_node_value)
            else: 
                return Node(value=0)
        
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
                
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        
        left_indices, right_indices = self._split_node(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_indices, :], y[left_indices], depth+1)
        right = self._build_tree(X[right_indices, :], y[right_indices], depth+1)
        return Node(best_feature, best_threshold, left, right)
        
    def calculate_leaf_value(self, y):
          
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        
        return value 
    
    def _find_best_split(self, X, y, feature_indices):
        
        best_gain = -1 
        split_index, split_threshold = None, None
        
        for feature_index in feature_indices:
            
            X_column = X[:, feature_index]  
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain 
                    split_index = feature_index
                    split_threshold = threshold
        
        return split_index, split_threshold     
        
    def _information_gain(self, y, X_column, threshold):
        
        parent_entropy = self._entropy(y)
        
        left_indices, right_indices = self._split_node(X_column, threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        total_samples = len(y)
        
        left_subtree_samples, right_subtree_samples = len(left_indices), len(right_indices)
        entropy_left_subtree, entropy_right_subtree = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        
        child_entropy = (left_subtree_samples/total_samples) * entropy_left_subtree + (right_subtree_samples/total_samples) * entropy_right_subtree
        
        information_gain = parent_entropy - child_entropy
        return information_gain
        
    def _entropy(self, y):
        
        hist = np.bincount(y) 
        ps = hist/len(y)
        
        return -np.sum([p * np.log(p) for p in ps if p > 0])
        
    def _split_node(self, X_column, threshold):
        
       
        
        left_indices = np.argwhere(X_column <= threshold).flatten()
        right_indices = np.argwhere(X_column > threshold).flatten()
        return left_indices, right_indices
            
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
            
    def _traverse_tree(self, x, node):
        
        if node.if_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:  
            return self._traverse_tree(x, node.left)
        
        else:
            return self._traverse_tree(x, node.right)
        
    def print_tree(self):
        self._print_node(self.root)
        
    def _print_node(self, node, indent=""):
        if node is None:
            return
        if node.if_leaf_node():
            print(indent + "Leaf Node: " + str(node.value))
        else:
            print(indent + "[LEFT] Feature " + str(node.feature) + " <= " + str(node.threshold) + "?")
            self._print_node(node.left, indent + "  ")
            print(indent + "[RIGHT] Feature " + str(node.feature) + " > " + str(node.threshold) + "?")
            self._print_node(node.right, indent + "  ")

import pandas as pd

data = pd.read_csv('Iris.csv')

le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

X = data.drop('Species', axis=1).values
y = data['Species'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = MyDecisionTree(min_sample_split=2, max_depth=5, n_features=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calculate accuracy for each prediction
accuracies = (y_test == y_pred)

# Print accuracy for each prediction
for idx, acc in enumerate(accuracies):
    if(idx==10):
     break
    print(f"Prediction {idx+1} - accuracy: {'correct' if acc else 'Incorrect'} { acc }")

# Print Confusion Matrix
from sklearn.metrics import confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate overall accuracy
accuracy = np.mean(accuracies)
print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
