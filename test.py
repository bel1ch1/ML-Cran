import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv('CSV/my_trajectory.csv')

# Предполагается, что у вас есть столбцы 'x', 'y', 'timestamp', 'dx' и 'dy' в данных
# Вычисляем расстояния между точками по X и по Y
df['distance_x'] = np.sqrt(np.diff(df['dx'], prepend=df['dx'][0])**2)
df['distance_y'] = np.sqrt(np.diff(df['dy'], prepend=df['dy'][0])**2)

# Создаем маски для выделения расстояний, которые больше или равны 60 (по X) и 40 (по Y)
mask_x = df['distance_x'] >= 50
mask_y = df['distance_y'] >= 40

# График изменений X по времени
plt.figure(figsize=(12, 6))

# Основной график изменения по X
plt.plot(df['timestamp'], df['dx'], label='Изменение по X', color='r', marker='o')

# График изменений по Y
plt.plot(df['timestamp'], df['dy'], label='Изменение по Y', color='g', marker='o')

# Выделяем участки, где расстояние по X больше или равно 60
plt.scatter(df['timestamp'][mask_x], df['dx'][mask_x], color='b', label='Расстояние по X >= 50', zorder=5)

# Выделяем участки, где расстояние по Y больше или равно 40
plt.scatter(df['timestamp'][mask_y], df['dy'][mask_y], color='b', label='Расстояние по Y >= 40', zorder=5)

plt.xlabel('Время')
plt.ylabel('Изменение')
plt.title('График изменений по X и Y по времени')
plt.legend()
plt.grid(True)
plt.show()
