import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = "dataset"

FEATURES = []
LABELS = []

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features_from_file(filepath):
    df = pd.read_csv(filepath)

    # -------- Label --------
    label = df["Quality"].iloc[0].strip().lower()
    y = 1 if label == "perfect" else 0

    # -------- Sensor columns --------
    sensor_cols = [
        "Accel_X", "Accel_Y", "Accel_Z",
        "Gyro_X", "Gyro_Y", "Gyro_Z"
    ]
    df = df[sensor_cols].dropna()

    features = []

    # -------- Statistical features --------
    for col in sensor_cols:
        data = df[col].values
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.sqrt(np.mean(data**2))  # RMS
        ])

    # -------- Magnitudes --------
    accel_mag = np.sqrt(
        df["Accel_X"]**2 +
        df["Accel_Y"]**2 +
        df["Accel_Z"]**2
    )
    gyro_mag = np.sqrt(
        df["Gyro_X"]**2 +
        df["Gyro_Y"]**2 +
        df["Gyro_Z"]**2
    )

    features.extend([
        accel_mag.mean(), accel_mag.std(),
        gyro_mag.mean(), gyro_mag.std()
    ])

    # -------- Jerk (smoothness) --------
    for col in ["Accel_X", "Accel_Y", "Accel_Z"]:
        jerk = np.diff(df[col].values)
        features.extend([
            np.mean(np.abs(jerk)),
            np.std(jerk)
        ])

    # -------- Energy --------
    for col in sensor_cols:
        data = df[col].values
        energy = np.sum(data**2) / len(data)
        features.append(energy)

    # -------- Gyro stability ratio --------
    features.append(gyro_mag.max() / (gyro_mag.mean() + 1e-6))

    return features, y


# =====================================================
# LOAD DATA
# =====================================================
for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        x, y = extract_features_from_file(os.path.join(DATASET_PATH, file))
        FEATURES.append(x)
        LABELS.append(y)

X = np.array(FEATURES, dtype=float)
y = np.array(LABELS)

print("Dataset shape:", X.shape)
print("Perfect:", sum(y), "| Bad:", len(y) - sum(y))

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALING
# =====================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# RANDOM FOREST (TUNED)
# =====================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# -------- Feature importance --------
plt.figure(figsize=(10,4))
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title("RF Feature Importance")
plt.show()

# =====================================================
# FEATURE SELECTION FOR RF
# =====================================================
top_idx = np.argsort(rf.feature_importances_)[-25:]

X_train_rf = X_train_scaled[:, top_idx]
X_test_rf = X_test_scaled[:, top_idx]

rf2 = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced"
)

rf2.fit(X_train_rf, y_train)
y_pred_rf2 = rf2.predict(X_test_rf)

print("\n--- RF (Selected Features) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf2))
print("F1:", f1_score(y_test, y_pred_rf2))

# =====================================================
# MLP WITH EARLY STOPPING
# =====================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

mlp = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

mlp.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = mlp.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = mlp.evaluate(X_test_scaled, y_test, verbose=0)
print("\n--- MLP ---")
print("MLP Accuracy:", acc)

# -------- MLP training curve --------
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("MLP Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# =====================================================
# ENSEMBLE (RF + MLP) ⭐⭐⭐
# =====================================================
rf_prob = rf2.predict_proba(X_test_rf)[:, 1]
mlp_prob = mlp.predict(X_test_scaled).flatten()

ensemble_prob = 0.5 * rf_prob + 0.5 * mlp_prob
ensemble_pred = (ensemble_prob > 0.5).astype(int)

print("\n--- ENSEMBLE MODEL ---")
print("Accuracy:", accuracy_score(y_test, ensemble_pred))
print("F1:", f1_score(y_test, ensemble_pred))
print(confusion_matrix(y_test, ensemble_pred))
