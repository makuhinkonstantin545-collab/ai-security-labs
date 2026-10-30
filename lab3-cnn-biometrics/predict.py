import tensorflow as tf
from pathlib import Path

# Определяем путь к текущей папке
BASE_DIR = Path(__file__).resolve().parent

# Загружаем тестовые данные MNIST
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Предобработка изображений
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Загружаем обученную модель
model_path = BASE_DIR / "cnn_mnist_model.keras"
model = tf.keras.models.load_model(model_path)

print("Обученная модель успешно загружена!")

# Оцениваем качество модели
test_loss, test_accuracy = model.evaluate(
    X_test,
    tf.keras.utils.to_categorical(y_test, 10),
    verbose=0
)

print(f"\nТочность модели: {test_accuracy * 100:.2f}%")
print(f"Потери модели: {test_loss:.4f}")

# Выполняем предсказания для первых пяти изображений
predictions = model.predict(X_test[:5], verbose=0)
predicted_classes = tf.argmax(predictions, axis=1).numpy()

print("\nРезультаты распознавания:")

for i in range(5):
    print(
        f"Изображение {i + 1}: "
        f"предсказание = {predicted_classes[i]}, "
        f"правильный ответ = {y_test[i]}"
    )