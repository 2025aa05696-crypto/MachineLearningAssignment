import pandas as pd
from model.preprocessing import load_and_preprocess_data

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# ===============================
# Convert y to pandas Series
# ===============================
y_train_series = pd.Series(y_train, name="status")
y_test_series = pd.Series(y_test, name="status")

# ===============================
# Print shapes
# ===============================
print("Train feature shape:", X_train.shape)
print("Test feature shape:", X_test.shape)

# ===============================
# Class distribution
# ===============================
print("\nTraining label distribution:")
print(y_train_series.value_counts())

print("\nTest label distribution:")
print(y_test_series.value_counts())
