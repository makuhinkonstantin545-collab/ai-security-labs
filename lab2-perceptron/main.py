# Лабораторная работа № 2
# Однослойный и многослойный персептрон
# в задаче бинарной классификации сетевого трафика

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Фиксируем случайность для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)

print("Библиотеки успешно подключены!")
print("Версия TensorFlow:", tf.__version__)

# 1. Генерация искусственных данных сетевого трафика

n_samples = 1000

# Генерируем примеры нормального трафика
normal = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(n_samples // 2, 5)
)

# Генерируем примеры условно вредоносного трафика
malicious = np.random.normal(
    loc=1.5,
    scale=1.0,
    size=(n_samples // 2, 5)
)

# Объединяем примеры двух классов
X = np.vstack([normal, malicious])

# Создаём правильные метки классов:
# 0 — нормальный трафик, 1 — вредоносный
y = np.hstack([
    np.zeros(n_samples // 2),
    np.ones(n_samples // 2)
])

# Перемешиваем примеры
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

print("\nДанные успешно созданы!")
print("Размер X:", X.shape)
print("Размер y:", y.shape)
print("Нормальных примеров:", np.sum(y == 0))
print("Вредоносных примеров:", np.sum(y == 1))

# 2. Разделение данных на обучающую и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nДанные разделены!")
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
print("Примеров для обучения:", len(y_train))
print("Примеров для тестирования:", len(y_test))

# 3. Масштабирование признаков
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nМасштабирование выполнено!")
print("Средние значения признаков:",
      np.round(X_train_scaled.mean(axis=0), 2))
print("Стандартные отклонения:",
      np.round(X_train_scaled.std(axis=0), 2))

# 4. Создание однослойного персептрона
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="sigmoid",
                          input_shape=(5,))
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nОднослойный персептрон создан!")
model.summary()

# 5. Обучение однослойного персептрона
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("\nОбучение персептрона завершено!")

# 6. Оценка однослойного персептрона на тестовых данных
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nРезультаты тестирования персептрона:")
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))
print("Точность в процентах:", round(test_accuracy * 100, 2), "%")

# 7. Подробный отчёт классификации
y_prob = model.predict(X_test_scaled, verbose=0)

# Преобразуем вероятности в классы 0 и 1
y_pred = (y_prob >= 0.5).astype(int).ravel()

print("\nОтчёт классификации:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Нормальный трафик", "Вредоносный трафик"],
    zero_division=0
))

# 8. Создание многослойного персептрона (MLP)

model_mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(5,)),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_mlp.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nМногослойный персептрон создан!")
model_mlp.summary()

# 9. Обучение многослойного персептрона

history_mlp = model_mlp.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("\nОбучение многослойного персептрона завершено!")

# 10. Проверка MLP на тестовой выборке

mlp_test_loss, mlp_test_accuracy = model_mlp.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nРезультаты тестирования MLP:")
print("Test loss:", round(mlp_test_loss, 4))
print("Test accuracy:", round(mlp_test_accuracy, 4))
print("Точность в процентах:", round(mlp_test_accuracy * 100, 2), "%")

# 11. Отчёт классификации для MLP

mlp_prob = model_mlp.predict(X_test_scaled, verbose=0)
mlp_pred = (mlp_prob >= 0.5).astype(int).ravel()

print("\nОтчёт классификации MLP:")
print(classification_report(
    y_test,
    mlp_pred,
    target_names=["Нормальный трафик", "Вредоносный трафик"],
    zero_division=0
))

# 12. Сравнение моделей

print("\nСравнение моделей:")
print("Однослойный персептрон:",
      round(test_accuracy * 100, 2), "%")
print("Многослойный персептрон:",
      round(mlp_test_accuracy * 100, 2), "%")

difference = (mlp_test_accuracy - test_accuracy) * 100

print("Разница в точности:",
      round(difference, 2), "процентных пункта")
