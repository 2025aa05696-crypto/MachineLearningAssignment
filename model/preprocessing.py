import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(
    csv_path="dataset_phishing.csv",
    test_size=0.2,
    random_state=42,
    scale_features=True
):
    """
    Load and preprocess phishing dataset.

    Steps:
    1. Load CSV
    2. Drop URL column (text, not used for ML)
    3. Encode target label
    4. Train-test split
    5. Feature scaling (optional)

    Returns:
    X_train, X_test, y_train, y_test
    """

    # ===============================
    # Load dataset
    # ===============================
    data = pd.read_csv(csv_path)

    # ===============================
    # Drop non-numeric / unused column
    # ===============================
    data = data.drop(columns=["url"])

    # ===============================
    # Separate features & target
    # ===============================
    X = data.drop(columns=["status"])
    y = data["status"]

    # ===============================
    # Encode target label
    # legitimate -> 0, phishing -> 1
    # ===============================
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # ===============================
    # Train-test split
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ===============================
    # Feature scaling
    # ===============================
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
