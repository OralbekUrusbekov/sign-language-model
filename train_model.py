import os
import json
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# -----------------------------
# PATHS
# -----------------------------
TRAIN_DIR = "app/dataset/train"
VAL_DIR = "app/dataset/val"
TEST_DIR = "app/dataset/test"

MODEL_DIR = "app/model"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# LOAD DATASET (99 FEATURES, 5 TIMESTEPS)
# -----------------------------
def load_dataset(root_dir, sequence_length=5):
    """
    Load pose data from .npy files
    Each file: (n_frames, 99) - multiple frames
    Creates sequences of sequence_length frames
    """
    X = []
    y = []
    file_counts = {}

    pose_root = os.path.join(root_dir, "pose")

    if not os.path.exists(pose_root):
        print(f"⚠ Directory not found: {pose_root}")
        return np.array(X), np.array(y), []

    classes = sorted([d for d in os.listdir(pose_root) if os.path.isdir(os.path.join(pose_root, d))])
    print(f"📁 Found classes: {len(classes)}")

    for cls in classes:
        cls_path = os.path.join(pose_root, cls)
        files = sorted([f for f in os.listdir(cls_path) if f.endswith(".npy")])

        print(f"  📂 {cls}: {len(files)} files")
        file_counts[cls] = len(files)

        for file in files:
            file_path = os.path.join(cls_path, file)
            data = np.load(file_path)

            # Handle different shapes
            if len(data.shape) == 2 and data.shape[1] == 99:
                # Shape: (n_frames, 99)
                n_frames = data.shape[0]

                # Create sequences of sequence_length
                # Use overlapping windows for more data
                for i in range(0, n_frames - sequence_length + 1, 2):
                    sequence = data[i:i + sequence_length]
                    sequence = enhance_sequence(sequence)  # (5, 99)
                    X.append(sequence)
                    y.append(cls)

            elif len(data.shape) == 1 and data.shape[0] == 99:
                # Shape: (99,) - single frame
                # Duplicate to create sequence (data augmentation)
                sequence = np.array([data] * sequence_length)
                sequence = enhance_sequence(sequence)

                X.append(sequence)
                y.append(cls)

            else:
                print(f"⚠ Unexpected shape {data.shape} in {file_path}")
                continue

    print(f"\n📊 Total sequences: {len(X)}")
    return np.array(X), np.array(y), classes


def enhance_sequence(sequence):
    result = []

    for t in range(sequence.shape[0]):
        frame_features = []

        for i in range(33):
            for axis in range(3):
                idx = i * 3 + axis

                vals = sequence[:, idx]

                current_val = sequence[t, idx]
                std_val = np.std(vals)

                if t == 0:
                    delta = 0.0
                else:
                    delta = sequence[t, idx] - sequence[t - 1, idx]

                frame_features.append(current_val)
                frame_features.append(std_val)
                frame_features.append(delta)

        result.append(frame_features)

    return np.array(result)

print("=" * 60)
print("📊 LOADING TRAIN DATA...")
print("=" * 60)
X_train, y_train, classes = load_dataset(TRAIN_DIR)
print(f"✅ Train data shape: {X_train.shape}")
print(f"✅ Train labels: {len(y_train)}")

print("\n" + "=" * 60)
print("📊 LOADING VALIDATION DATA...")
print("=" * 60)
X_val, y_val, _ = load_dataset(VAL_DIR)
print(f"✅ Val data shape: {X_val.shape}")

print("\n" + "=" * 60)
print("📊 LOADING TEST DATA...")
print("=" * 60)
X_test, y_test, _ = load_dataset(TEST_DIR)
print(f"✅ Test data shape: {X_test.shape}")

# Check if data is empty
if len(X_train) == 0:
    print("\n❌ No training data found!")
    print("Please check your dataset structure:")
    print("  app/dataset/train/pose/[class_name]/*.npy")
    print("\nEach .npy file should be either:")
    print("  - Shape (n_frames, 99) - multiple frames")
    print("  - Shape (99,) - single frame")
    exit(1)

# -----------------------------
# LABEL ENCODER
# -----------------------------
print("\n" + "=" * 60)
print("🏷️  LABEL ENCODER")
print("=" * 60)

label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

print(f"✅ Classes: {label_encoder.classes_}")
print(f"✅ Number of classes: {len(label_encoder.classes_)}")

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"✅ Label encoder saved to: {MODEL_DIR}/label_encoder.pkl")

# -----------------------------
# SCALER (2D reshaping for scaling)
# -----------------------------
print("\n" + "=" * 60)
print("📏 SCALER")
print("=" * 60)

# Reshape to 2D for scaling: (n_samples * sequence_length, 99)
n_samples_train, seq_len, n_features = X_train.shape
print(f"📊 Training samples: {n_samples_train}")
print(f"📊 Sequence length: {seq_len}")
print(f"📊 Features per frame: {n_features}")

X_train_2d = X_train.reshape(-1, n_features)
X_val_2d = X_val.reshape(-1, n_features) if len(X_val) > 0 else np.array([])
X_test_2d = X_test.reshape(-1, n_features) if len(X_test) > 0 else np.array([])

print(f"📊 Scaling {X_train_2d.shape[0]} samples with {n_features} features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d)

if len(X_val_2d) > 0:
    X_val_scaled = scaler.transform(X_val_2d)
else:
    X_val_scaled = np.array([])

if len(X_test_2d) > 0:
    X_test_scaled = scaler.transform(X_test_2d)
else:
    X_test_scaled = np.array([])

print(f"✅ Scaler expects {scaler.n_features_in_} features")

# Reshape back to 3D: (n_samples, sequence_length, 99)
X_train = X_train_scaled.reshape(n_samples_train, seq_len, n_features)

if len(X_val) > 0:
    n_samples_val = X_val.shape[0]
    X_val = X_val_scaled.reshape(n_samples_val, seq_len, n_features)
else:
    X_val = np.array([])

if len(X_test) > 0:
    n_samples_test = X_test.shape[0]
    X_test = X_test_scaled.reshape(n_samples_test, seq_len, n_features)
else:
    X_test = np.array([])

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"✅ Scaler saved to: {MODEL_DIR}/scaler.pkl")

# -----------------------------
# FEATURE ORDER SAVE
# -----------------------------
print("\n" + "=" * 60)
print("📋 FEATURE ORDER")
print("=" * 60)

feature_order = []

for i in range(33):
    for axis in ['x', 'y', 'z']:
        feature_order.append(f"pose_{i}_{axis}")
        feature_order.append(f"pose_{i}_{axis}_std")
        feature_order.append(f"pose_{i}_{axis}_delta")


print(f"✅ Feature order: {len(feature_order)} features")

with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
    json.dump(feature_order, f, indent=2)
print(f"✅ Feature order saved to: {MODEL_DIR}/feature_order.json")

# -----------------------------
# ONE HOT LABELS
# -----------------------------
print("\n" + "=" * 60)
print("🔥 ONE-HOT ENCODING")
print("=" * 60)

num_classes = len(classes)
y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)

if len(y_val_enc) > 0:
    y_val_cat = to_categorical(y_val_enc, num_classes=num_classes)
else:
    y_val_cat = np.array([])

if len(y_test_enc) > 0:
    y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
else:
    y_test_cat = np.array([])

print(f"✅ Labels shape: {y_train_cat.shape}")

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
print("\n" + "=" * 60)
print("🧠 MODEL ARCHITECTURE")
print("=" * 60)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Model Summary:")
model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1,
        mode='min'
    )
]

print("\n" + "=" * 60)
print("🚀 TRAINING STARTED")
print("=" * 60)

# -----------------------------
# TRAIN
# -----------------------------
if len(X_val) == 0:
    print("\n⚠ No validation data, using 20% of training for validation")
    validation_split = 0.2
    validation_data = None
else:
    validation_split = 0
    validation_data = (X_val, y_val_cat)

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=validation_data,
    validation_split=validation_split,
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# EVALUATE
# -----------------------------
print("\n" + "=" * 60)
print("📊 EVALUATION")
print("=" * 60)

if len(X_test) > 0:
    print("\n📊 Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test Loss: {loss:.4f}")
else:
    print("\n⚠ No test data available")

# -----------------------------
# SAVE MODEL
# -----------------------------
print("\n" + "=" * 60)
print("💾 SAVING MODEL")
print("=" * 60)

model.save(os.path.join(MODEL_DIR, "sign_language_recognition_fixed.keras"))
model.save(os.path.join(MODEL_DIR, "sign_language_recognition.h5"), save_format='h5')

print(f"\n✅ Model saved to: {MODEL_DIR}")
print(f"   - sign_language_recognition_fixed.keras")
print(f"   - sign_language_recognition.h5")

# -----------------------------
# TEST SINGLE PREDICTION
# -----------------------------
print("\n" + "=" * 60)
print("🧪 TESTING SINGLE PREDICTION")
print("=" * 60)

if len(X_test) > 0:
    test_sample = X_test[0:1]
    pred = model.predict(test_sample, verbose=0)
    pred_class = label_encoder.inverse_transform([np.argmax(pred[0])])
    confidence = np.max(pred[0]) * 100
    print(f"✅ Sample prediction: {pred_class[0]} ({confidence:.2f}%)")
elif len(X_train) > 0:
    test_sample = X_train[0:1]
    pred = model.predict(test_sample, verbose=0)
    pred_class = label_encoder.inverse_transform([np.argmax(pred[0])])
    confidence = np.max(pred[0]) * 100
    print(f"✅ Sample prediction (from train): {pred_class[0]} ({confidence:.2f}%)")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 60)