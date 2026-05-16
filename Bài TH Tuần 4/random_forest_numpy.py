import numpy as np
from collections import Counter


# ======================================
# NODE
# ======================================

class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):

        return self.value is not None


# ======================================
# DECISION TREE
# ======================================

class DecisionTree:

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        n_features=None
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    # ======================================
    # TRAIN
    # ======================================

    def fit(self, X, y):

        self.n_features = (
            X.shape[1]
            if not self.n_features
            else min(self.n_features, X.shape[1])
        )

        self.root = self._grow_tree(X, y)

    # ======================================
    # BUILD TREE
    # ======================================

    def _grow_tree(self, X, y, depth=0):

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        print(
            f"Tree Depth = {depth} | "
            f"Samples = {n_samples}"
        )

        # Điều kiện dừng
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        # Random feature selection
        feat_idxs = np.random.choice(
            n_feats,
            self.n_features,
            replace=False
        )

        best_feature, best_thresh = self._best_split(
            X,
            y,
            feat_idxs
        )

        if best_feature is None:

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(
            X[:, best_feature],
            best_thresh
        )

        left = self._grow_tree(
            X[left_idxs, :],
            y[left_idxs],
            depth + 1
        )

        right = self._grow_tree(
            X[right_idxs, :],
            y[right_idxs],
            depth + 1
        )

        return Node(
            best_feature,
            best_thresh,
            left,
            right
        )

    # ======================================
    # BEST SPLIT
    # ======================================

    def _best_split(self, X, y, feat_idxs):

        best_gain = -1

        split_idx = None
        split_thresh = None

        for feat_idx in feat_idxs:

            X_column = X[:, feat_idx]

            thresholds = np.unique(X_column)

            for threshold in thresholds:

                gain = self._information_gain(
                    y,
                    X_column,
                    threshold
                )

                if gain > best_gain:

                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    # ======================================
    # INFORMATION GAIN
    # ======================================

    def _information_gain(
        self,
        y,
        X_column,
        threshold
    ):

        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(
            X_column,
            threshold
        )

        if len(left_idxs) == 0 or len(right_idxs) == 0:

            return 0

        n = len(y)

        n_l = len(left_idxs)
        n_r = len(right_idxs)

        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])

        child_entropy = (
            (n_l / n) * e_l
            + (n_r / n) * e_r
        )

        ig = parent_entropy - child_entropy

        return ig

    # ======================================
    # SPLIT
    # ======================================

    def _split(self, X_column, threshold):

        left_idxs = np.argwhere(
            X_column <= threshold
        ).flatten()

        right_idxs = np.argwhere(
            X_column > threshold
        ).flatten()

        return left_idxs, right_idxs

    # ======================================
    # ENTROPY
    # ======================================

    def _entropy(self, y):

        hist = np.bincount(y)

        ps = hist / len(y)

        return -np.sum(
            [p * np.log2(p) for p in ps if p > 0]
        )

    # ======================================
    # MOST COMMON LABEL
    # ======================================

    def _most_common_label(self, y):

        counter = Counter(y)

        return counter.most_common(1)[0][0]

    # ======================================
    # PREDICT
    # ======================================

    def predict(self, X):

        return np.array([
            self._traverse_tree(x, self.root)
            for x in X
        ])

    def _traverse_tree(self, x, node):

        if node.is_leaf():

            return node.value

        if x[node.feature] <= node.threshold:

            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)


# ======================================
# RANDOM FOREST
# ======================================

class RandomForestNumpy:

    def __init__(
        self,
        n_trees=5,
        max_depth=10,
        min_samples_split=2,
        n_features=None
    ):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.trees = []

    # ======================================
    # TRAIN
    # ======================================

    def fit(self, X, y):

        print("\nBắt đầu train Random Forest NumPy...\n")

        self.trees = []

        for i in range(self.n_trees):

            print(f"\nĐang train cây thứ {i+1}/{self.n_trees}")

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )

            X_sample, y_sample = self._bootstrap_samples(
                X,
                y
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        print("\nTrain Random Forest NumPy hoàn tất!")

    # ======================================
    # BOOTSTRAP
    # ======================================

    def _bootstrap_samples(self, X, y):

        n_samples = X.shape[0]

        idxs = np.random.choice(
            n_samples,
            n_samples,
            replace=True
        )

        return X[idxs], y[idxs]

    # ======================================
    # PREDICT
    # ======================================

    def predict(self, X):

        tree_preds = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        tree_preds = np.swapaxes(tree_preds, 0, 1)

        predictions = np.array([
            self._most_common_label(pred)
            for pred in tree_preds
        ])

        return predictions

    def _most_common_label(self, y):

        counter = Counter(y)

        return counter.most_common(1)[0][0]