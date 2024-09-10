import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Загрузка данных
data = pd.read_csv('cargo_trajectory_data.csv')

data_array = data.values

# Split the data into training and testing sets
train_size = int(0.7 * len(data_array))
train_data = data_array[:train_size]
test_data = data_array[train_size:]

# Create sub-arrays from the training and testing sets
seq_length = 20
X_train = []
y_train = []
for i in range(len(train_data) - seq_length):
    seq = train_data[i:i+seq_length]
    label = train_data[i+seq_length-1, 3]  # assuming "swinging" is the 4th column
    X_train.append(seq)
    y_train.append(label)

X_test = []
y_test = []
for i in range(len(test_data) - seq_length):
    seq = test_data[i:i+seq_length]
    label = test_data[i+seq_length-1, 3]  # assuming "swinging" is the 4th column
    X_test.append(seq)
    y_test.append(label)

X_train = np.array(X_train).reshape(-1, seq_length, 4)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(-1, seq_length, 4)
y_test = np.array(y_test)


# Модель
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 4)))
model.add(LSTM(units=50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(X_train.shape)
print(y_train.shape)

history = model.fit(X_train, y_train, epochs=10000, batch_size=None, validation_data=(X_test, y_test))

predictions = model.predict(np.array(X_test))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
