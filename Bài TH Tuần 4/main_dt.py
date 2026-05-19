from ucimlrepo import fetch_ucirepo 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from decision_tree_numpy import DecisionTreeNumpy
from decision_tree_sklearn import DecisionTreeSKLearn


# ======================================
# LOAD DATASET
# ======================================

print("=" * 60)
print("ĐANG TẢI DATASET WINE QUALITY...")
print("=" * 60)

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  

print("\nDataset loaded thành công!")

# ======================================
# CHUYỂN ĐỔI DỮ LIỆU
# ======================================

# Nếu y là dataframe
if isinstance(y, pd.DataFrame):

    y = y.iloc[:, 0]

# Encode label
encoder = LabelEncoder()

y = encoder.fit_transform(y)

# Chuyển sang NumPy
X = X.to_numpy()
y = np.array(y)

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

# ======================================
# CHIA TRAIN TEST
# ======================================

print("\n" + "=" * 60)
print("CHIA TRAIN / TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ======================================
# TÍNH F1 SCORE BẰNG TAY
# ======================================

def manual_f1_score(y_true, y_pred):

    classes = np.unique(y_true)

    f1_scores = []
    weights = []

    for cls in classes:

        # TP, FP, FN
        tp = np.sum((y_true == cls) & (y_pred == cls))

        fp = np.sum((y_true != cls) & (y_pred == cls))

        fn = np.sum((y_true == cls) & (y_pred != cls))

        # Precision
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        # Recall
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        # F1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

        # trọng số của class
        weights.append(np.sum(y_true == cls))

        print(f"\nClass {cls}")
        print(f"TP = {tp}, FP = {fp}, FN = {fn}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall    = {recall:.4f}")
        print(f"F1 Score  = {f1:.4f}")

    # Weighted F1
    weights = np.array(weights)

    weighted_f1 = np.sum(
        np.array(f1_scores) * weights
    ) / np.sum(weights)

    return weighted_f1

# ======================================
# NUMPY DECISION TREE
# ======================================

print("\n" + "=" * 60)
print("DECISION TREE - NUMPY")
print("=" * 60)

tree_numpy = DecisionTreeNumpy(
    max_depth=10
)

tree_numpy.fit(X_train, y_train)

y_pred_numpy = tree_numpy.predict(X_test)

f1_numpy = f1_score(
    y_test,
    y_pred_numpy,
    average='weighted'
)

f1_numpy = manual_f1_score(
    y_test,
    y_pred_numpy
)

print("\nF1 Score (NumPy):", round(f1_numpy, 4))

# ======================================
# SCIKIT-LEARN DECISION TREE
# ======================================

print("\n" + "=" * 60)
print("DECISION TREE - SCIKIT-LEARN")
print("=" * 60)

tree_sklearn = DecisionTreeSKLearn(
    max_depth=10
)

tree_sklearn.fit(X_train, y_train)

y_pred_sklearn = tree_sklearn.predict(X_test)

f1_sklearn = tree_sklearn.evaluate(
    y_test,
    y_pred_sklearn
)

print("F1 Score (Scikit-Learn):", round(f1_sklearn, 4))

# ======================================
# SO SÁNH KẾT QUẢ
# ======================================

print("\n" + "=" * 60)
print("SO SÁNH KẾT QUẢ")
print("=" * 60)

print(f"NumPy Decision Tree      : {f1_numpy:.4f}")
print(f"Scikit-Learn DecisionTree: {f1_sklearn:.4f}")
