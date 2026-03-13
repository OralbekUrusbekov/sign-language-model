# create_model_99.py
import tensorflow as tf
import numpy as np
import joblib

print("🔄 Creating new model for 99 features...")


# Модель архитектурасы (99 features үшін)
def create_model_99(input_shape=(5, 99), num_classes=11):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input_layer'),

        # LSTM layers
        tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.BatchNormalization(name='bn_1'),

        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        tf.keras.layers.BatchNormalization(name='bn_2'),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_3'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.2, name='dropout_4'),

        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])

    return model


# Модель құру
model = create_model_99()

# Модельді компиляциялау
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Model architecture:")
model.summary()

# Модельді сақтау
model.save('app/model/sign_language_recognition_fixed.keras')
model.save('app/model/sign_language_recognition.h5', save_format='h5')
print("\n✓ Model saved!")

# Тестілеу
try:
    test_input = np.random.randn(1, 5, 99)
    output = model.predict(test_input)
    print(f"✓ Model test passed! Output shape: {output.shape}")
    print(f"✓ Model expects: {model.input_shape}")
except Exception as e:
    print(f"✗ Model test failed: {e}")