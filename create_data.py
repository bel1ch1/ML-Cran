import pandas as pd
import numpy as np
import datetime

# Генерация данных
np.random.seed(0)  # Для воспроизводимости
num_samples = 1000  # Количество записей

timestamps = [datetime.datetime.now() + datetime.timedelta(seconds=i) for i in range(num_samples)]
x_coords = np.sin(np.linspace(0, 20, num_samples))  # Пример синусоидальных координат X
y_coords = np.cos(np.linspace(0, 20, num_samples))  # Пример косинусоидальных координат Y

# Создание DataFrame
data = {
    'timestamp': timestamps,
    'x': x_coords,
    'y': y_coords
}

df = pd.DataFrame(data)

# Сохранение в CSV
df.to_csv('data.csv', index=False)
