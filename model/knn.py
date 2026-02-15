from sklearn.neighbors import KNeighborsClassifier
from model.preprocessing import load_and_preprocess_data
from model.metrics import evaluate_model


def train_knn(X_external=None, y_external=None):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        scale_features=True
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    if X_external is not None and y_external is not None:
        y_pred = model.predict(X_external)
        y_prob = model.predict_proba(X_external)[:, 1]
        y_test_final = y_external
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_test_final = y_test

    metrics = evaluate_model(y_test_final, y_pred, y_prob)
    return metrics, y_test_final, y_pred
