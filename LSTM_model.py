""" Модель для выявления раскаченности на последовательностьях n-ной длины """
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Считываем данные из датасета
data = pd.read_csv()


# Разделяем данные от меток
X = data["sequence"]
y = data["lables"]


# Разделяем данные на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создание модели
model = Sequential()

# Первый LSTM слой. Параметр return_sequences=True означает, что на каждом шаге будут возвращаться все последовательности.
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))

# Полносвязный слой с 1 нейроном для бинарной классификации
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Выводим информацию о модели
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# Оценка модели
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Точность на тестовых данных: {test_accuracy * 100:.2f}%")
