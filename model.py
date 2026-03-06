# model.py

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


def train_and_evaluate(X, y, model_type="Gaussian", test_size=0.2):

    # Convert y to categorical if possible
    y = pd.Series(y)

    # Check if y has enough samples per class
    class_counts = y.value_counts()

    use_stratify = True

    if class_counts.min() < 2:
        use_stratify = False

    # Safe train test split
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

    # Model selection
    if model_type == "Gaussian":
        model = GaussianNB()
    elif model_type == "Multinomial":
        model = MultinomialNB()
    else:
        model = BernoulliNB()

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    return model, train_acc, train_cm, test_acc, test_cm