from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


class DecisionTreeSKLearn:

    def __init__(
        self,
        max_depth=10,
        random_state=42
    ):

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )

    # ======================================
    # TRAIN MODEL
    # ======================================

    def fit(self, X_train, y_train):

        print("\nBắt đầu huấn luyện Decision Tree Scikit-Learn...\n")

        self.model.fit(X_train, y_train)

        print("\nHuấn luyện Scikit-Learn hoàn tất!\n")

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