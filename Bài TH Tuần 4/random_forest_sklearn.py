from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class RandomForestSKLearn:

    def __init__(
        self,
        n_estimators=5,
        max_depth=10,
        random_state=42
    ):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    # ======================================
    # TRAIN
    # ======================================

    def fit(self, X_train, y_train):

        print("\nBắt đầu train Random Forest Scikit-Learn...\n")

        self.model.fit(X_train, y_train)

        print("\nTrain Scikit-Learn hoàn tất!")

    # ======================================
    # PREDICT
    # ======================================

    def predict(self, X_test):

        return self.model.predict(X_test)

    # ======================================
    # EVALUATE
    # ======================================

    def evaluate(self, y_test, y_pred):

        score = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        return score