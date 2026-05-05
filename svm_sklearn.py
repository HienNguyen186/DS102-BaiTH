import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

class SVMSklearn:
    """
    Wrapper class cho sklearn SVC, phục vụ Assignment 2.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0, n_components: int = None):
        self.kernel = kernel
        self.C = C
        self.n_components = n_components
        self.model = SVC(kernel=kernel, C=C)
        self.pca = PCA(n_components=n_components) if n_components else None

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Xử lý PCA nếu có yêu cầu."""
        if self.pca is not None:
            return self.pca.fit_transform(X) if fit else self.pca.transform(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"Training SVM (kernel='{self.kernel}', C={self.C}" 
              + (f", PCA={self.n_components}" if self.pca else "") + ")...")
        
        X_transformed = self._transform(X, fit=True)
        self.model.fit(X_transformed, y)
        print("Huấn luyện xong!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn."""
        X_transformed = self._transform(X, fit=False)
        return self.model.predict(X_transformed)

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Đánh giá model bằng Precision, Recall, F1."""
        y_pred = self.predict(X)
        return {
            "Precision": precision_score(y, y_pred, pos_label=1),
            "Recall": recall_score(y, y_pred, pos_label=1),
            "F1": f1_score(y, y_pred, pos_label=1)
        }