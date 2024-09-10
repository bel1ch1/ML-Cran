import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense


# Загрузка данных
data = pd.read_csv('cargo_trajectory_data.csv')

# Разделение данных на обучающую и тестовую выборки
train_data = data.loc[data['timestamp'] <= 500]
test_data = data.loc[data['timestamp'] > 500]

# Проверка разделения данных
print(train_data.shape)  # (500, 4)
print(test_data.shape)   # (500, 4)
print(test_data["timestamp", "deviation_x", "deviation_y"])

# # Build the RNN model
# inputs = Input(shape=(50, 2))
# x = LSTM(units=50, return_sequences=True)(inputs)
# x = LSTM(units=50)(x)
# outputs = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# model.fit(train_data["timestamp", "deviation_x", "deviation_y"], train_data["swinging"], epochs=50, batch_size=16, validation_data=(test_data["timestamp", "deviation_x", "deviation_y"], test_data["swinging"]))

# # Evaluate the model
# loss, accuracy = model.evaluate(test_data["timestamp", "deviation_x", "deviation_y"], test_data["swinging"])
# print(f'Test accuracy: {accuracy:.3f}')

# # Plot the accuracy curve
# import matplotlib.pyplot as plt

# history = model.history.history
# plt.plot(history['accuracy'], label='Training accuracy')
# plt.plot(history['val_accuracy'], label='Validation accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
