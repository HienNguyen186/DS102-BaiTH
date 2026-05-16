import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


class SVM:
    def __init__(self, C: float = 1.0, lr: float = 1e-5, n_iterations: int = 100):
        self.C = C
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = 0
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape

        # Khởi tạo tham số
        self.W = np.zeros((dim,))
        self.b = 0
        self.losses = []

        for epoch in range(self.n_iterations):
            pbar = tqdm(range(N), desc=f"Epoch {epoch} - Training")
            for ith in pbar:
                x = X[ith]  
                a = y[ith]  

                y_pred = self.predict(np.expand_dims(x, axis=0))  
                loss = self.loss_fn(a, y_pred)
                self.losses.append(loss)

                pbar.set_postfix({"Loss": loss})

                if a * y_pred >= 1:
                    dW = self.W
                    db = 0
                else:
                    dW = self.W + self.C * (-a * x)
                    db = self.C * (-a)

                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        regularization = 0.5 * np.dot(self.W.T, self.W)
        hinge = self.C * np.where(1 - y * y_hat < 0, 0, 1 - y * y_hat).sum()
        return regularization + hinge

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0, 1, -1)

        P  = precision_score(y, y_pred, pos_label=1)
        R  = recall_score(y, y_pred, pos_label=1)
        f1 = f1_score(y, y_pred, pos_label=1)

        return {
            "Precision": P,
            "Recall"   : R,
            "F1"       : f1
        }


if __name__ == "__main__":
    from main import collect_data

    np.random.seed(42)

    print("Loading data...")
    X_train, y_train = collect_data("train")
    X_test, y_test   = collect_data("test")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Khởi tạo và train model
    model = SVM(
        C=1,
        lr=1e-5,
        n_iterations=100
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    metrics = model.get_metrics(X_test, y_test)
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
