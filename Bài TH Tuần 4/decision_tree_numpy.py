import numpy as np
from collections import Counter


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


class DecisionTreeNumpy:

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ======================================
    # TRAIN MODEL
    # ======================================

    def fit(self, X, y):

        print("\nBắt đầu huấn luyện Decision Tree NumPy...\n")

        self.root = self._grow_tree(X, y, depth=0)

        print("\nHuấn luyện hoàn tất!\n")

    # ======================================
    # XÂY DỰNG CÂY
    # ======================================

    def _grow_tree(self, X, y, depth):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        print(
            f"Depth = {depth} | "
            f"Số samples = {n_samples} | "
            f"Số classes = {n_labels}"
        )

        # Điều kiện dừng
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        # Tìm split tốt nhất
        best_feature, best_threshold = self._best_split(X, y)

        # Nếu không split được
        if best_feature is None:

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        # Chia dữ liệu
        left_idxs, right_idxs = self._split(
            X[:, best_feature],
            best_threshold
        )

        # Tạo node trái
        left = self._grow_tree(
            X[left_idxs, :],
            y[left_idxs],
            depth + 1
        )

        # Tạo node phải
        right = self._grow_tree(
            X[right_idxs, :],
            y[right_idxs],
            depth + 1
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    # ======================================
    # TÌM SPLIT TỐT NHẤT
    # ======================================

    def _best_split(self, X, y):

        best_gain = -1

        split_idx = None
        split_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):

            X_column = X[:, feature_idx]

            thresholds = np.unique(X_column)

            for threshold in thresholds:

                gain = self._information_gain(
                    y,
                    X_column,
                    threshold
                )

                if gain > best_gain:

                    best_gain = gain

                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    # ======================================
    # INFORMATION GAIN
    # ======================================

    def _information_gain(self, y, X_column, threshold):

        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(
            X_column,
            threshold
        )

        # Nếu split lỗi
        if len(left_idxs) == 0 or len(right_idxs) == 0:

            return 0

        n = len(y)

        n_left = len(left_idxs)
        n_right = len(right_idxs)

        e_left = self._entropy(y[left_idxs])
        e_right = self._entropy(y[right_idxs])

        child_entropy = (
            (n_left / n) * e_left
            + (n_right / n) * e_right
        )

        ig = parent_entropy - child_entropy

        return ig

    # ======================================
    # SPLIT DATA
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
    # TÍNH ENTROPY
    # ======================================

    def _entropy(self, y):

        hist = np.bincount(y)

        ps = hist / len(y)

        entropy = -np.sum(
            [p * np.log2(p) for p in ps if p > 0]
        )

        return entropy

    # ======================================
    # LABEL PHỔ BIẾN NHẤT
    # ======================================

    def _most_common_label(self, y):

        counter = Counter(y)

        value = counter.most_common(1)[0][0]

        return value

    # ======================================
    # DỰ ĐOÁN
    # ======================================

    def predict(self, X):

        predictions = np.array([
            self._traverse_tree(x, self.root)
            for x in X
        ])

        return predictions

    # ======================================
    # DUYỆT CÂY
    # ======================================

    def _traverse_tree(self, x, node):

        if node.is_leaf():

            return node.value

        if x[node.feature] <= node.threshold:

            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)