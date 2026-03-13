# rebuild_model.py
import tensorflow as tf
import json
import joblib
import numpy as np

print("Модельді қайта құру...")


# 1. Модель архитектурасын анықтау
def create_model(input_shape=(5, 18), num_classes=11):
    model = tf.keras.Sequential([
        # Input layer
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


# 2. Ескі модельдің салмақтарын жүктеу (егер мүмкін болса)
try:
    # Алдымен ескі модельді жүктеуге тырысу
    old_model = tf.keras.models.load_model(
        'app/model/sign_language_recognition.keras',
        custom_objects={'Functional': tf.keras.Model}
    )
    print("Ескі модель жүктелді!")

    # Жаңа модель құру
    new_model = create_model()

    # Салмақтарды көшіру (егер мүмкін болса)
    for i, layer in enumerate(new_model.layers):
        if layer.name in [l.name for l in old_model.layers]:
            old_layer = old_model.get_layer(layer.name)
            try:
                layer.set_weights(old_layer.get_weights())
                print(f"  ✓ {layer.name} салмақтары көшірілді")
            except:
                print(f"  ✗ {layer.name} салмақтарын көшіру мүмкін емес")

except Exception as e:
    print(f"Ескі модельді жүктеу мүмкін емес: {e}")
    print("Жаңа модель құрылуда...")
    new_model = create_model()

# 3. Модельді компиляциялау
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Модель архитектурасын көрсету
new_model.summary()

# 5. Жаңа модельді сақтау
new_model.save('app/model/sign_language_recognition_fixed.keras')
new_model.save('app/model/sign_language_recognition.h5', save_format='h5')
print("\nЖаңа модель сақталды!")

# 6. Тестілеу
try:
    # Кездейсоқ деректермен тестілеу
    test_input = np.random.randn(1, 5, 18)
    output = new_model.predict(test_input)
    print(f"Модель дұрыс жұмыс істейді! Шығыс формасы: {output.shape}")
except Exception as e:
    print(f"Тестілеу қатесі: {e}")