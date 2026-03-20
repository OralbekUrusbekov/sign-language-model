import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIGURATION FOR WLASL100
# -----------------------------
CURRENT_FILE = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(CURRENT_FILE)
APP_DIR = os.path.join(BASE_DIR, "app")

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 APP_DIR: {APP_DIR}")

# Paths
TRAIN_DIR = os.path.join(APP_DIR, "dataset/train")
VAL_DIR = os.path.join(APP_DIR, "dataset/val")
TEST_DIR = os.path.join(APP_DIR, "dataset/test")
MODEL_DIR = os.path.join(APP_DIR, "model")

print(f"\n📂 TRAIN_DIR: {TRAIN_DIR}")
print(f"📂 VAL_DIR: {VAL_DIR}")
print(f"📂 TEST_DIR: {TEST_DIR}")
print(f"📂 MODEL_DIR: {MODEL_DIR}")

# Check directories
print("\n🔍 Checking directories:")
print(f"  Train exists: {os.path.exists(TRAIN_DIR)}")
print(f"  Val exists: {os.path.exists(VAL_DIR)}")
print(f"  Test exists: {os.path.exists(TEST_DIR)}")

if os.path.exists(TRAIN_DIR):
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    print(f"  Classes found: {len(classes)}")
    if classes:
        print(f"  First 5 classes: {classes[:5]}")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("🚀 WLASL100 TRAINING - IMPROVED VERSION")
print("=" * 70)


# -----------------------------
# DATASET LOADER
# -----------------------------
def load_wlasl100_dataset(root_dir, sequence_length=15, stride=3):
    """Load WLASL100 dataset with 225 features"""
    X = []
    y = []
    video_names = []

    print(f"\n📂 Loading from: {root_dir}")

    if not os.path.exists(root_dir):
        print(f"❌ Directory not found: {root_dir}")
        return np.array(X), np.array(y), [], []

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"📁 Found {len(classes)} classes")

    if len(classes) == 0:
        print(f"⚠️ No classes found in {root_dir}")
        return np.array(X), np.array(y), [], []

    total_files = 0
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]
        total_files += len(files)

    print(f"📊 Total .npy files: {total_files}")
    print("-" * 70)

    for cls_idx, cls in enumerate(classes):
        cls_path = os.path.join(root_dir, cls)
        files = sorted([f for f in os.listdir(cls_path) if f.endswith(".npy")])

        print(f"  📂 {cls}: {len(files)} files", end=" ")

        cls_count = 0
        for file in files:
            file_path = os.path.join(cls_path, file)
            try:
                data = np.load(file_path)
                n_frames = data.shape[0]

                # Check feature dimension
                if data.shape[1] != 225:
                    print(f"\n  ⚠ Warning: {file} has {data.shape[1]} features, expected 225")
                    if data.shape[1] < 225:
                        padding = np.zeros((n_frames, 225 - data.shape[1]))
                        data = np.hstack([data, padding])
                    else:
                        data = data[:, :225]

                # Interpolate if needed
                if n_frames < sequence_length:
                    repeats = sequence_length // n_frames + 1
                    data = np.tile(data, (repeats, 1))
                    n_frames = data.shape[0]

                # Create sequences with overlap
                for i in range(0, n_frames - sequence_length + 1, stride):
                    sequence = data[i:i + sequence_length]

                    # Data augmentation (70% probability)
                    if np.random.random() > 0.3:
                        sequence = augment_sequence_improved(sequence)

                    X.append(sequence)
                    y.append(cls)
                    video_names.append(file)
                    cls_count += 1

            except Exception as e:
                print(f"\n  ⚠ Error loading {file}: {e}")

        print(f"→ {cls_count} sequences")

    print("-" * 70)
    print(f"✅ Total sequences: {len(X)}")
    if len(X) > 0:
        print(f"✅ Sequence shape: {sequence_length} frames × {X[0].shape[1]} features")

    return np.array(X), np.array(y), classes, video_names


def augment_sequence_improved(sequence):
    """Stronger data augmentation for better generalization"""
    augmented = sequence.copy()

    # Random scale (0.85 - 1.15)
    scale = np.random.uniform(0.85, 1.15)
    augmented *= scale

    # Random shift
    shift_x = np.random.uniform(-0.08, 0.08)
    shift_y = np.random.uniform(-0.08, 0.08)
    shift_z = np.random.uniform(-0.05, 0.05)

    for i in range(75):  # 75 landmarks total
        idx_x = i * 3
        idx_y = i * 3 + 1
        idx_z = i * 3 + 2
        if idx_z < augmented.shape[1]:
            augmented[:, idx_x] += shift_x
            augmented[:, idx_y] += shift_y
            augmented[:, idx_z] += shift_z

    # Random rotation (2D)
    angle = np.random.uniform(-0.15, 0.15)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for i in range(75):
        idx_x = i * 3
        idx_y = i * 3 + 1
        if idx_y + 1 < augmented.shape[1]:
            x = augmented[:, idx_x]
            y = augmented[:, idx_y]
            augmented[:, idx_x] = x * cos_a - y * sin_a
            augmented[:, idx_y] = x * sin_a + y * cos_a

    # Random noise
    noise = np.random.normal(0, 0.015, size=augmented.shape)
    augmented += noise

    # Random temporal masking (drop some frames)
    if np.random.random() > 0.7:
        mask_start = np.random.randint(0, augmented.shape[0] // 2)
        mask_end = mask_start + np.random.randint(1, augmented.shape[0] // 3)
        augmented[mask_start:mask_end] = 0

    return augmented


# -----------------------------
# IMPROVED MODEL ARCHITECTURE
# -----------------------------
def create_model_improved(seq_len=15, n_features=225, num_classes=100):
    """Improved model with better regularization"""
    inputs = Input(shape=(seq_len, n_features))

    # Spatial feature extraction with strong regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Temporal feature extraction with fewer parameters
    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(0.01),
                           recurrent_regularizer=l2(0.01)))(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(32, return_sequences=False,
                           kernel_regularizer=l2(0.01),
                           recurrent_regularizer=l2(0.01)))(x)
    x = Dropout(0.5)(x)

    # Classification head
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------
# LOAD DATASET
# -----------------------------
print("\n" + "=" * 70)
print("📊 LOADING TRAIN DATA")
print("=" * 70)
X_train, y_train, classes, train_videos = load_wlasl100_dataset(TRAIN_DIR, sequence_length=15)

print("\n" + "=" * 70)
print("📊 LOADING VALIDATION DATA")
print("=" * 70)
X_val, y_val, _, val_videos = load_wlasl100_dataset(VAL_DIR, sequence_length=15)

print("\n" + "=" * 70)
print("📊 LOADING TEST DATA")
print("=" * 70)
X_test, y_test, _, test_videos = load_wlasl100_dataset(TEST_DIR, sequence_length=15)

if len(X_train) == 0:
    print("\n❌ No training data found!")
    print(f"Please check: {TRAIN_DIR}/[class_name]/*.npy")
    exit(1)

# -----------------------------
# LABEL ENCODER
# -----------------------------
print("\n" + "=" * 70)
print("🏷️  LABEL ENCODER")
print("=" * 70)

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)

if len(y_val) > 0:
    y_val_enc = label_encoder.transform(y_val)
else:
    y_val_enc = np.array([])

if len(y_test) > 0:
    y_test_enc = label_encoder.transform(y_test)
else:
    y_test_enc = np.array([])

print(f"✅ Number of classes: {len(label_encoder.classes_)}")

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder_225.pkl"))
print(f"✅ Label encoder saved")

# -----------------------------
# SCALER
# -----------------------------
print("\n" + "=" * 70)
print("📏 SCALER")
print("=" * 70)

n_samples, seq_len, n_features = X_train.shape
print(f"📊 Training samples: {n_samples}")
print(f"📊 Features per frame: {n_features}")

X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2d)
X_train = X_train_scaled.reshape(n_samples, seq_len, n_features)

if len(X_val) > 0:
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_2d)
    X_val = X_val_scaled.reshape(n_val, seq_len, n_features)

if len(X_test) > 0:
    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_2d)
    X_test = X_test_scaled.reshape(n_test, seq_len, n_features)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_225.pkl"))
print(f"✅ Scaler saved")

# -----------------------------
# FEATURE ORDER
# -----------------------------
print("\n" + "=" * 70)
print("📋 FEATURE ORDER")
print("=" * 70)

feature_order = []
for i in range(33):
    for axis in ['x', 'y', 'z']:
        feature_order.append(f"pose_{i}_{axis}")
for i in range(21):
    for axis in ['x', 'y', 'z']:
        feature_order.append(f"hand_left_{i}_{axis}")
for i in range(21):
    for axis in ['x', 'y', 'z']:
        feature_order.append(f"hand_right_{i}_{axis}")

with open(os.path.join(MODEL_DIR, "feature_order_225.json"), "w") as f:
    json.dump(feature_order, f, indent=2)

# -----------------------------
# ONE-HOT ENCODING
# -----------------------------
print("\n" + "=" * 70)
print("🔥 ONE-HOT ENCODING")
print("=" * 70)

num_classes = len(classes)
y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)

if len(y_val) > 0:
    y_val_cat = to_categorical(y_val_enc, num_classes=num_classes)
else:
    y_val_cat = np.array([])

if len(y_test) > 0:
    y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
else:
    y_test_cat = np.array([])

print(f"✅ Labels shape: {y_train_cat.shape}")

# -----------------------------
# CREATE MODEL
# -----------------------------
print("\n" + "=" * 70)
print("🧠 BUILDING IMPROVED MODEL")
print("=" * 70)

model = create_model_improved(
    seq_len=15,
    n_features=225,
    num_classes=num_classes
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1,
        mode='min'
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_wlasl100_225.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# -----------------------------
# TRAIN
# -----------------------------
print("\n" + "=" * 70)
print("🚀 TRAINING STARTED")
print("=" * 70)

if len(X_val) == 0:
    print("\n⚠ No validation data, using 20% training split")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
else:
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

# -----------------------------
# EVALUATE
# -----------------------------
print("\n" + "=" * 70)
print("📊 FINAL EVALUATION")
print("=" * 70)

if len(X_test) > 0:
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"✅ Test Loss: {loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)

    top1_acc = np.mean(y_pred_classes == y_true_classes)
    print(f"✅ Top-1 Accuracy: {top1_acc:.4f}")

    top3 = 0
    for i in range(len(y_true_classes)):
        if y_true_classes[i] in np.argsort(y_pred[i])[-3:]:
            top3 += 1
    top3_acc = top3 / len(y_true_classes)
    print(f"✅ Top-3 Accuracy: {top3_acc:.4f}")

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
print("\n" + "=" * 70)
print("💾 SAVING FINAL MODEL")
print("=" * 70)

model.save(os.path.join(MODEL_DIR, "sign_language_recognition_225.keras"))
model.save(os.path.join(MODEL_DIR, "sign_language_recognition_225.h5"))

print(f"\n✅ All models saved to: {MODEL_DIR}")
print(f"   - sign_language_recognition_225.keras")
print(f"   - sign_language_recognition_225.h5")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)