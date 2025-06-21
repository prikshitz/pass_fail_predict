import numpy as np

# 1. Calculate Gini Impurity
def gini(y):
    classes = np.unique(y)
    impurity = 1.0
    for cls in classes:
        prob = np.sum(y == cls) / len(y)
        impurity -= prob ** 2
    return impurity

# 2. Split dataset based on feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# 3. Find best split
def best_split(X, y):
    best_gini = float('inf')
    best_index, best_threshold = None, None
    n_features = X.shape[1]
    
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_index = feature_index
                best_threshold = threshold
    
    return best_index, best_threshold

# 4. Create tree node
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# 5. Build the tree recursively
def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1 or depth == max_depth:
        leaf_value = max(set(y), key=list(y).count)
        return Node(value=leaf_value)
    
    feature_index, threshold = best_split(X, y)
    if feature_index is None:
        return Node(value=max(set(y), key=list(y).count))
    
    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
    left_child = build_tree(X_left, y_left, depth + 1, max_depth)
    right_child = build_tree(X_right, y_right, depth + 1, max_depth)
    
    return Node(feature_index, threshold, left_child, right_child)

# 6. Predict a single sample
def predict_sample(node, x):
    if node.is_leaf():
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict_sample(node.left, x)
    else:
        return predict_sample(node.right, x)

# 7. Predict a batch
def predict(tree, X):
    return np.array([predict_sample(tree, x) for x in X])
