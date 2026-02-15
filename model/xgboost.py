from xgboost import XGBClassifier
from model.preprocessing import load_and_preprocess_data
from model.metrics import evaluate_model


def train_xgboost(X_external=None, y_external=None):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        scale_features=False
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
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
