# fix_model.py
import tensorflow as tf
import numpy as np
import joblib
import json

print("Модельді қайта құру...")


# 1. Модель архитектурасын анықтау
def create_model(input_shape=(5, 18), num_classes=11):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input_layer'),

        # LSTM layers
        tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1'),
        tf.keras.layers.Dropout(0.3, name='dropout'),
        tf.keras.layers.BatchNormalization(name='batch_normalization'),

        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_1'),

        # Dense layers
        tf.keras.layers.Dense(64, activation='relu', name='dense'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_1')
    ])

    return model


# 2. Жаңа модель құру
new_model = create_model()

# 3. Модельді компиляциялау
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Модель архитектурасын көрсету
print("\nЖаңа модель архитектурасы:")
new_model.summary()

# 5. Модельді сақтау
new_model.save('app/model/sign_language_recognition_fixed.keras')
new_model.save('app/model/sign_language_recognition.h5', save_format='h5')
print("\n✓ Модель сақталды!")

# 6. Сақталған модельді тексеру
try:
    test_input = np.random.randn(1, 5, 18)
    output = new_model.predict(test_input)
    print(f"✓ Модель дұрыс жұмыс істейді! Шығыс формасы: {output.shape}")
except Exception as e:
    print(f"✗ Модельді тестілеу қатесі: {e}")