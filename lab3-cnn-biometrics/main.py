
import tensorflow as tf

print("TensorFlow:", tf.__version__)

# 1. Загружаем данные MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Подготавливаем изображения
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. Преобразуем метки в One-Hot Encoding
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 4. Создаём свёрточную нейронную сеть
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

# 5. Настраиваем обучение
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Выводим структуру модели
model.summary()
# Обучение нейронной сети
history = model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test_cat)
)

# Оценка качества модели на тестовых данных
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\nРезультаты тестирования:")
print(f"Потери (Loss): {test_loss:.4f}")
print(f"Точность (Accuracy): {test_accuracy:.4f}")
print(f"Точность в процентах: {test_accuracy * 100:.2f}%")
# Сохранение обученной модели
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model.save(BASE_DIR / "cnn_mnist_model.keras")

print("\nОбученная модель сохранена: cnn_mnist_model.keras")
# Проверка предсказаний на первых пяти изображениях
predictions = model.predict(X_test[:5], verbose=0)

predicted_classes = tf.argmax(predictions, axis=1).numpy()

print("\nПроверка первых пяти изображений:")

for i in range(5):
    print(
        f"Изображение {i + 1}: "
        f"предсказание модели = {predicted_classes[i]}, "
        f"правильный ответ = {y_test[i]}"
    )
    # Загрузка сохранённой модели
loaded_model = tf.keras.models.load_model("cnn_mnist_model.keras")

print("\nСохранённая модель успешно загружена!")

# Проверка загруженной модели
loaded_predictions = loaded_model.predict(X_test[:5], verbose=0)
loaded_classes = tf.argmax(loaded_predictions, axis=1).numpy()

print("\nПредсказания загруженной модели:")

for i in range(5):
    print(
        f"Изображение {i + 1}: "
        f"предсказание = {loaded_classes[i]}, "
        f"правильный ответ = {y_test[i]}"
    )