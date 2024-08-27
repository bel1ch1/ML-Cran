import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv('data.csv')

# Преобразование временных меток в числовой формат (например, количество секунд от начала)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

# Преобразование данных в массивы
times = df['time'].values
x_coords = df['x'].values
y_coords = df['y'].values

# Нормализация данных
scaler_time = MinMaxScaler()
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

times = scaler_time.fit_transform(times.reshape(-1, 1))
x_coords = scaler_x.fit_transform(x_coords.reshape(-1, 1))
y_coords = scaler_y.fit_transform(y_coords.reshape(-1, 1))

# Создание последовательностей для RNN
def create_sequences(time, x, y, seq_length):
    X, Y = [], []
    for i in range(len(time) - seq_length):
        X.append(np.hstack((time[i:i+seq_length], x[i:i+seq_length], y[i:i+seq_length])))
        Y.append(np.hstack((x[i+seq_length], y[i+seq_length])))
    return np.array(X), np.array(Y)

seq_length = 10  # длина последовательности
X, y = create_sequences(times, x_coords, y_coords, seq_length=3)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Определение модели
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(seq_length, 3)))  # 3 - это количество признаков (время, x и y)
model.add(Dense(2))  # 2 - это количество выходных признаков (x и y)

model.compile(optimizer='adam', loss='mean_squared_error')

# Обзор структуры модели
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, epochs=20000, batch_size=32, validation_split=0.1)

# Оценка модели
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Предсказание
y_pred = model.predict(X_test)

# Преобразование результатов обратно в исходный масштаб
y_pred = np.concatenate((scaler_x.inverse_transform(y_pred[:, :1]), scaler_y.inverse_transform(y_pred[:, 1:])), axis=1)


import matplotlib.pyplot as plt

# Визуализация предсказанных и фактических данных
plt.figure(figsize=(12, 6))
plt.plot(scaler_x.inverse_transform(y_test[:, :1]), scaler_y.inverse_transform(y_test[:, 1:]), label='Actual Positions', marker='o')
plt.plot(y_pred[:, 0], y_pred[:, 1], label='Predicted Positions', marker='x')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Actual vs Predicted Positions')
plt.legend()
plt.show()
