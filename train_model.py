import os
import json
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from app.config import Config


# -----------------------------
# PATHS
# -----------------------------
TRAIN_DIR = os.path.join(Config.DATASET_DIR, "train")
VAL_DIR = os.path.join(Config.DATASET_DIR, "val")
TEST_DIR = os.path.join(Config.DATASET_DIR, "test")

MODEL_DIR = "app/model"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset(root_dir):
    X = []
    y = []

    pose_root = os.path.join(root_dir, "pose")

    classes = sorted(
        [d for d in os.listdir(pose_root) if os.path.isdir(os.path.join(pose_root, d))]
    )

    for cls in classes:
        cls_path = os.path.join(pose_root, cls)

        for file in os.listdir(cls_path):
            if file.endswith(".npy"):
                file_path = os.path.join(cls_path, file)

                data = np.load(file_path)
                data = data.flatten()

                X.append(data)
                y.append(cls)

    return np.array(X), np.array(y), classes


print("Loading train data...")
X_train, y_train, classes = load_dataset(TRAIN_DIR)

print("Loading val data...")
X_val, y_val, _ = load_dataset(VAL_DIR)

print("Loading test data...")
X_test, y_test, _ = load_dataset(TEST_DIR)


# -----------------------------
# LABEL ENCODER
# -----------------------------
label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))


# -----------------------------
# SCALER
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


# -----------------------------
# FEATURE ORDER SAVE
# -----------------------------
feature_order = [f"f_{i}" for i in range(X_train.shape[1])]

with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
    json.dump(feature_order, f)


# -----------------------------
# RESHAPE FOR LSTM
# -----------------------------
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# -----------------------------
# ONE HOT LABELS
# -----------------------------
y_train_cat = to_categorical(y_train_enc)
y_val_cat = to_categorical(y_val_enc)
y_test_cat = to_categorical(y_test_enc)


# -----------------------------
# MODEL
# -----------------------------
model = Sequential([
    LSTM(128, input_shape=(1, X_train.shape[2])),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(classes), activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# TRAIN
# -----------------------------
print("Training started...")

model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=20,
    batch_size=16
)


# -----------------------------
# EVALUATE
# -----------------------------
loss, acc = model.evaluate(X_test, y_test_cat)

print(f"Test Accuracy: {acc:.4f}")


# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(os.path.join(MODEL_DIR, "sign_language_recognition_fixed.keras"))

print("Model saved successfully!")