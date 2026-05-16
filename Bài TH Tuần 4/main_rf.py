from ucimlrepo import fetch_ucirepo 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from random_forest_numpy import RandomForestNumpy
from random_forest_sklearn import RandomForestSKLearn


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
# RANDOM FOREST NUMPY
# ======================================

print("\n" + "=" * 60)
print("RANDOM FOREST - NUMPY")
print("=" * 60)

rf_numpy = RandomForestNumpy(
    n_trees=5,
    max_depth=10,
    n_features=5
)

rf_numpy.fit(X_train, y_train)

y_pred_numpy = rf_numpy.predict(X_test)

f1_numpy = f1_score(
    y_test,
    y_pred_numpy,
    average='weighted'
)

print("\nF1 Score (NumPy):", round(f1_numpy, 4))

# ======================================
# RANDOM FOREST SKLEARN
# ======================================

print("\n" + "=" * 60)
print("RANDOM FOREST - SCIKIT-LEARN")
print("=" * 60)

rf_sklearn = RandomForestSKLearn(
    n_estimators=5,
    max_depth=10
)

rf_sklearn.fit(X_train, y_train)

y_pred_sklearn = rf_sklearn.predict(X_test)

f1_sklearn = rf_sklearn.evaluate(
    y_test,
    y_pred_sklearn
)

print("\nF1 Score (Scikit-Learn):", round(f1_sklearn, 4))

# ======================================
# SO SÁNH
# ======================================

print("\n" + "=" * 60)
print("SO SÁNH KẾT QUẢ")
print("=" * 60)

print(f"Random Forest NumPy      : {f1_numpy:.4f}")
print(f"Random Forest ScikitLearn: {f1_sklearn:.4f}")