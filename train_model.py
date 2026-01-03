import os
import pandas as pd
import numpy as np

DATASET_PATH = "dataset"   

FEATURES = []
LABELS = []

def extract_features_from_file(filepath):
    df = pd.read_csv(filepath)   # panadas read csv 

    # ----------------------------
    # 1. Clean label
    """
    .strip(): Removes accidental spaces. If a human typed "perfect " 
    (with a space at the end) into the CSV, the computer would think "perfect " is NOT the same as
    """
    # ----------------------------
    label = df["Quality"].iloc[0].strip().lower()
    if label == "perfect":
        y = 1
    else:
        y = 0

    # ----------------------------
    # 2. Select sensor columns
    
    # ----------------------------
    sensor_cols = [
        "Accel_X", "Accel_Y", "Accel_Z",
        "Gyro_X", "Gyro_Y", "Gyro_Z"
    ]
    df = df[sensor_cols].dropna()     #we drop other columns ; we keep the important ones  ; dropna removes each row has a missing value 

    """
    df is like this now : 

    Accel_X	Accel_Y	Accel_Z	Gyro_X	Gyro_Y	Gyro_Z	Quality
0	0.5	    -0.1	 9.8	 0.0    -0.2	  0.0	perfect
1	NaN	    -0.1     9.8	 0.0	-0.1	  0.0	perfect

after

  Accel_X	Accel_Y	Accel_Z	Gyro_X	Gyro_Y	Gyro_Z	
0	0.5	    -0.1	 9.8	 0.0    -0.2	  0.0	

    """

    # ----------------------------
    # 3. Feature extraction
    # ----------------------------
    features = []

    for col in sensor_cols:
        data = df[col].values      #data = [0.5,0.1,.........] 1D array why , because its going to be faster  
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.sqrt(np.mean(data**2))  # RMS
        ])

    # Magnitudes (important!)
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

    return features, y

# ----------------------------
# Loop over all files
# ----------------------------
for file in os.listdir(DATASET_PATH):
    if file.endswith(".csv"):
        filepath = os.path.join(DATASET_PATH, file)
        x, y = extract_features_from_file(filepath)
        FEATURES.append(x)    #FEATURES[[0.5,1.5........34],[0.5,1.5........34],[0.5,1.5........34].................1000csv_file]
        LABELS.append(y)      #LABELS[0,0,0,0,1,1,0,1...................1000csv]

X = np.array(FEATURES)
y = np.array(LABELS)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Perfect samples:", sum(y))
print("Bad samples:", len(y) - sum(y))




from sklearn.model_selection import train_test_split  #import the library responsible for spliting our dataset into 80% of training and 20% of testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  #same random number 
    stratify=y   #our test data will contain the perfect and the bad samples with same % of the first time 
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


from sklearn.preprocessing import StandardScaler

X_train = X_train.astype(float)  #ensures all the numbers are floats 
X_test = X_test.astype(float)


# he calcuate the mean and std for all the 34 columns * 1000 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#applicate the formula 
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Train Random Forest
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=300,  #300 individual tree 
    random_state=42,
    class_weight="balanced"

)

rf.fit(X_train_scaled, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = rf.predict(X_test_scaled)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt

importances = rf.feature_importances_

plt.figure(figsize=(10, 4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ----------------------------
# MLP Model
# ----------------------------
mlp = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = mlp.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# Evaluation
# ----------------------------
loss, acc = mlp.evaluate(X_test_scaled, y_test, verbose=0)
print("\nMLP Test Accuracy:", acc)
