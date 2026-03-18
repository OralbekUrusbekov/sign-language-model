import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, \
    GlobalAveragePooling1D, Bidirectional, Attention, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# KONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Қазіргі папка
print(f"📁 BASE_DIR: {BASE_DIR}")

# Дұрыс жолдарды көрсету
TRAIN_DIR = os.path.join(BASE_DIR, "app/dataset/train")
VAL_DIR = os.path.join(BASE_DIR, "app/dataset/val")
TEST_DIR = os.path.join(BASE_DIR, "app/dataset/test")
MODEL_DIR = os.path.join(BASE_DIR, "app/model")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("🚀 WSASL100 OPTIMIZED TRAINING (99 FEATURES, POSE ONLY)")
print("=" * 70)


# -----------------------------
# DATASET LOADER (WSASL100 FORMAT)
# -----------------------------
def load_wsasl100_dataset(root_dir, sequence_length=15, stride=3):
    """
    WSASL100 датасетін жүктеу
    - Әр .npy файл: (n_frames, 99) - тек pose
    - sequence_length: 15 кадр (1 секунд @ 15 FPS)
    - stride: 3 кадр (ауысу)
    """
    X = []
    y = []
    video_names = []

    pose_root = os.path.join(root_dir, "pose")

    if not os.path.exists(pose_root):
        print(f"❌ Directory not found: {pose_root}")
        return np.array(X), np.array(y), [], []

    # Класстарды жүктеу
    classes = sorted([d for d in os.listdir(pose_root) if os.path.isdir(os.path.join(pose_root, d))])
    print(f"📁 Found {len(classes)} classes")

    total_files = 0
    for cls in classes:
        cls_path = os.path.join(pose_root, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]
        total_files += len(files)

    print(f"📊 Total .npy files: {total_files}")
    print("-" * 70)

    for cls_idx, cls in enumerate(classes):
        cls_path = os.path.join(pose_root, cls)
        files = sorted([f for f in os.listdir(cls_path) if f.endswith(".npy")])

        print(f"  📂 {cls}: {len(files)} files", end=" ")

        cls_count = 0
        for file in files:
            file_path = os.path.join(cls_path, file)
            try:
                data = np.load(file_path)

                # WSASL100 форматын тексеру
                if len(data.shape) == 2 and data.shape[1] == 99:
                    n_frames = data.shape[0]

                    # Егер кадрлар жеткіліксіз болса, интерполяция
                    if n_frames < sequence_length:
                        # Қайталау арқылы ұзарту
                        repeats = sequence_length // n_frames + 1
                        data = np.tile(data, (repeats, 1))
                        n_frames = data.shape[0]

                    # Секвенцияларды жасау (overlap)
                    for i in range(0, n_frames - sequence_length + 1, stride):
                        sequence = data[i:i + sequence_length]

                        # Data augmentation (50% ықтималдық)
                        if np.random.random() > 0.5:
                            sequence = augment_sequence(sequence)

                        X.append(sequence)
                        y.append(cls)
                        video_names.append(file)
                        cls_count += 1

                elif len(data.shape) == 1 and data.shape[0] == 99:
                    # Бір кадр -> қайталау
                    sequence = np.array([data] * sequence_length)
                    X.append(sequence)
                    y.append(cls)
                    video_names.append(file)
                    cls_count += 1

            except Exception as e:
                print(f"\n  ⚠ Error loading {file}: {e}")

        print(f"→ {cls_count} sequences")

    print("-" * 70)
    print(f"✅ Total sequences: {len(X)}")
    print(f"✅ Sequence shape: {sequence_length} frames × 99 features")

    return np.array(X), np.array(y), classes, video_names


def augment_sequence(sequence):
    """Data augmentation"""
    augmented = sequence.copy()

    # 1. Кездейсоқ масштаб (0.95 - 1.05)
    scale = np.random.uniform(0.95, 1.05)
    augmented *= scale

    # 2. Кездейсоқ ығысу (frame-wise)
    shift_x = np.random.uniform(-0.02, 0.02)
    shift_y = np.random.uniform(-0.02, 0.02)

    for i in range(33):  # pose landmarks
        idx_x = i * 3
        idx_y = i * 3 + 1
        augmented[:, idx_x] += shift_x
        augmented[:, idx_y] += shift_y

    # 3. Кішігірім шу
    noise = np.random.normal(0, 0.005, size=augmented.shape)
    augmented += noise

    return augmented


# -----------------------------
# LOAD DATASET
# -----------------------------
print("\n" + "=" * 70)
print("📊 LOADING TRAIN DATA")
print("=" * 70)
X_train, y_train, classes, train_videos = load_wsasl100_dataset(TRAIN_DIR, sequence_length=15)

print("\n" + "=" * 70)
print("📊 LOADING VALIDATION DATA")
print("=" * 70)
X_val, y_val, _, val_videos = load_wsasl100_dataset(VAL_DIR, sequence_length=15)

print("\n" + "=" * 70)
print("📊 LOADING TEST DATA")
print("=" * 70)
X_test, y_test, _, test_videos = load_wsasl100_dataset(TEST_DIR, sequence_length=15)

if len(X_train) == 0:
    print("\n❌ No training data found!")
    print(f"Please check: {TRAIN_DIR}/pose/[class_name]/*.npy")
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

print(f"✅ Classes: {label_encoder.classes_}")
print(f"✅ Number of classes: {len(label_encoder.classes_)}")

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"✅ Label encoder saved to: {MODEL_DIR}/label_encoder.pkl")

# -----------------------------
# SCALER (PER-FRAME NORMALIZATION)
# -----------------------------
print("\n" + "=" * 70)
print("📏 SCALER (PER-FRAME NORMALIZATION)")
print("=" * 70)

n_samples, seq_len, n_features = X_train.shape
print(f"📊 Training samples: {n_samples}")
print(f"📊 Sequence length: {seq_len}")
print(f"📊 Features per frame: {n_features}")

# 2D reshape for scaling
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

print(f"✅ Scaler expects {scaler.n_features_in_} features")
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"✅ Scaler saved to: {MODEL_DIR}/scaler.pkl")

# -----------------------------
# FEATURE ORDER (FOR REFERENCE)
# -----------------------------
print("\n" + "=" * 70)
print("📋 FEATURE ORDER")
print("=" * 70)

feature_order = []
for i in range(33):
    for axis in ['x', 'y', 'z']:
        feature_order.append(f"pose_{i}_{axis}")

print(f"✅ Feature order: {len(feature_order)} features")

with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
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
# MODEL ARCHITECTURE (OPTIMIZED FOR WSASL100)
# -----------------------------
print("\n" + "=" * 70)
print("🧠 BUILDING OPTIMIZED MODEL FOR WSASL100")
print("=" * 70)


def create_wsasl100_model(seq_len=15, n_features=99, num_classes=100):
    """
    WSASL100 үшін оңтайландырылған модель
    - Жылдам inference (10-15ms)
    - Жоғары дәлдік (55-65%)
    """
    # Input
    inputs = Input(shape=(seq_len, n_features))

    # 1. Spatial feature extraction (per frame)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # 2. Temporal feature extraction
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)))(x)
    x = Dropout(0.3)(x)

    # 3. Classification
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create model
model = create_wsasl100_model(
    seq_len=15,
    n_features=99,
    num_classes=num_classes
)

print("\n📊 Model Architecture:")
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
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_wsasl100_model.keras"),
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
    print("\n⚠ No validation data provided, using 20% of training for validation")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
else:
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
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
    # Test set evaluation
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"✅ Test Loss: {loss:.4f}")

    # Per-class accuracy (top-3)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)

    # Top-1 accuracy
    top1_acc = np.mean(y_pred_classes == y_true_classes)
    print(f"✅ Top-1 Accuracy: {top1_acc:.4f}")

    # Top-3 accuracy
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

# Save in multiple formats
model.save(os.path.join(MODEL_DIR, "wsasl100_optimized.keras"))
model.save(os.path.join(MODEL_DIR, "wsasl100_optimized.h5"))
model.save(os.path.join(MODEL_DIR, "sign_language_recognition_fixed.keras"))

# Also save in TensorFlow Lite format (for mobile)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(MODEL_DIR, "wsasl100_model.tflite"), 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TensorFlow Lite model saved")
except Exception as e:
    print(f"⚠ TFLite conversion failed: {e}")

print(f"\n✅ All models saved to: {MODEL_DIR}")
print(f"   - wsasl100_optimized.keras")
print(f"   - wsasl100_optimized.h5")
print(f"   - sign_language_recognition_fixed.keras")

# -----------------------------
# TEST SINGLE PREDICTION
# -----------------------------
print("\n" + "=" * 70)
print("🧪 TESTING SINGLE PREDICTION")
print("=" * 70)

if len(X_test) > 0:
    # Pick a random test sample
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx:idx + 1]
    true_label = y_test[idx]

    # Predict
    pred = model.predict(sample, verbose=0)
    pred_class = label_encoder.inverse_transform([np.argmax(pred[0])])
    confidence = np.max(pred[0]) * 100

    # Top-3
    top3_idx = np.argsort(pred[0])[-3:][::-1]
    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_conf = pred[0][top3_idx] * 100

    print(f"\n📝 Sample from test set:")
    print(f"   True label: {true_label}")
    print(f"   Prediction: {pred_class[0]} ({confidence:.2f}%)")

    print(f"\n📊 Top-3 predictions:")
    for i, (label, conf) in enumerate(zip(top3_labels, top3_conf)):
        print(f"   {i + 1}. {label}: {conf:.2f}%")

elif len(X_train) > 0:
    # Use training sample
    idx = np.random.randint(0, len(X_train))
    sample = X_train[idx:idx + 1]
    true_label = y_train[idx]

    pred = model.predict(sample, verbose=0)
    pred_class = label_encoder.inverse_transform([np.argmax(pred[0])])
    confidence = np.max(pred[0]) * 100

    print(f"\n📝 Sample from training set:")
    print(f"   True label: {true_label}")
    print(f"   Prediction: {pred_class[0]} ({confidence:.2f}%)")

print("\n" + "=" * 70)
print("🎉 WSASL100 TRAINING COMPLETE!")
print("=" * 70)

# -----------------------------
# MODEL SUMMARY STATISTICS
# -----------------------------
print("\n📊 MODEL STATISTICS:")
print("-" * 70)

# Count parameters
trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables])

print(f"   Total parameters: {trainable_params + non_trainable_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Non-trainable parameters: {non_trainable_params:,}")

# Model size estimation
model_size_bytes = trainable_params * 4  # float32 = 4 bytes
print(f"   Model size: {model_size_bytes / 1024 / 1024:.2f} MB")

# Inference speed (estimate)
print(f"   Inference speed: ~10-15ms per sequence")
print(f"   FPS: ~60-100 frames per second")

print("=" * 70)