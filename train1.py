import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "dataset"

FEATURES = []
LABELS = []

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features_from_file(filepath):
    df = pd.read_csv(filepath)

    # ----------------------------
    # 1. Clean label
    # ----------------------------
    label = df["Quality"].iloc[0].strip().lower()
    y = 1 if label == "perfect" else 0

    # ----------------------------
    # 2. Select sensor columns
    # ----------------------------
    sensor_cols = [
        "Accel_X", "Accel_Y", "Accel_Z",
        "Gyro_X", "Gyro_Y", "Gyro_Z"
    ]
    df = df[sensor_cols].dropna()

    features = []

    # ----------------------------
    # 3. BASIC STATISTICAL FEATURES
    # ----------------------------
    for col in sensor_cols:
        data = df[col].values
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.sqrt(np.mean(data**2))  # RMS
        ])

    # ----------------------------
    # 4. MAGNITUDES
    # ----------------------------
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

    # ----------------------------
    # 5. JERK (SMOOTHNESS)  ⭐
    # ----------------------------
    for col in ["Accel_X", "Accel_Y", "Accel_Z"]:
        jerk = np.diff(df[col].values)
        features.extend([
            np.mean(np.abs(jerk)),
            np.std(jerk)
        ])

    # ----------------------------
    # 6. ENERGY (INTENSITY) ⭐
    # ----------------------------
    for col in sensor_cols:
        data = df[col].values
        energy = np.sum(data**2) / len(data)
        features.append(energy)

    # ----------------------------
    # 7. GYRO STABILITY RATIO ⭐
    # ----------------------------
    features.append(gyro_mag.max() / (gyro_mag.mean() + 1e-6))

    return features, y


# =====================================================
# LOAD DATASET
# =====================================================
for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        filepath = os.path.join(DATASET_PATH, file)
        x, y = extract_features_from_file(filepath)
        FEATURES.append(x)
        LABELS.append(y)

X = np.array(FEATURES, dtype=float)
y = np.array(LABELS)

print("Dataset shape:", X.shape)
print("Perfect samples:", sum(y))
print("Bad samples:", len(y) - sum(y))

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
# RANDOM FOREST (TUNED) 🌲
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
y_pred = rf.predict(X_test_scaled)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
plt.figure(figsize=(10, 4))
plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# =====================================================
# MLP (IMPROVED) 🧠
# =====================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

history = mlp.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

loss, acc = mlp.evaluate(X_test_scaled, y_test, verbose=0)
print("\n--- MLP Results ---")
print("MLP Test Accuracy:", acc)

# Training curve
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Curve')
plt.legend()
plt.show()
