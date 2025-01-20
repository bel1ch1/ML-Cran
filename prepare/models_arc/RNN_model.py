import numpy as np

# Загрузка данных для RNN
data = np.load('rnn_crane_swing_dataset.npz')
X = data['X']  # Последовательности координат (x, y)
y = data['y']  # Метки раскачивания


from sklearn.model_selection import train_test_split

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Создание RNN модели
model = Sequential()

# Добавление слоя LSTM с 50 нейронами
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# Добавление выходного слоя с 1 нейроном для бинарной классификации
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Вывод структуры модели
model.summary()


# Обучение модели
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))


# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt

# Построение графика точности
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Построение графика потерь
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
