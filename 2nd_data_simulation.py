import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Устанавливаем начальное значение генератора случайных чисел
np.random.seed(42)

# Параметры
num_samples = 1000  # Количество образцов
noise_level = 0.05  # Уровень случайного шума (уменьшенный)
window_size = 20    # Размер окна для скользящего среднего
outlier_factor = 3  # Фактор увеличения отклонений для раскачивания
outlier_intervals = [(200, 300), (600, 700)]  # Промежутки времени для раскачивания

# Функция для генерации траекторий
def generate_trajectory(num_points, trajectory_type='linear'):
    t = np.linspace(0, 2 * np.pi, num_points)  # Временные метки или углы

    if trajectory_type == 'linear':
        x = t
        y = np.zeros(num_points)
    elif trajectory_type == 'circular':
        x = np.sin(t)
        y = np.cos(t)
    elif trajectory_type == 'elliptical':
        a, b = 1.0, 0.5
        x = a * np.sin(t)
        y = b * np.cos(t)
    else:
        raise ValueError("Unsupported trajectory type")

    return x, y

# Генерация данных
timestamps = np.arange(1, num_samples + 1)
deviation_x, deviation_y = generate_trajectory(num_samples, trajectory_type='circular')

# Добавление случайного шума
deviation_x += np.random.normal(0, noise_level, num_samples)
deviation_y += np.random.normal(0, noise_level, num_samples)

# Добавление сильных отклонений для раскачивания груза
for start, end in outlier_intervals:
    deviation_x[start:end] += np.random.normal(0, noise_level * outlier_factor, end - start)
    deviation_y[start:end] += np.random.normal(0, noise_level * outlier_factor, end - start)

# Применение сглаживания с помощью скользящего среднего
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Сглаживание данных
smoothed_deviation_x = moving_average(deviation_x, window_size)
smoothed_deviation_y = moving_average(deviation_y, window_size)
timestamps_smoothed = timestamps[(window_size - 1):]  # Скорректированные временные метки для сглаженных данных

# Симуляция метки раскачивания груза
swinging = np.zeros(num_samples)
for start, end in outlier_intervals:
    swinging[start:end] = 1

# Создание DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'deviation_x': deviation_x,
    'deviation_y': deviation_y,
    'swinging': swinging
})

# Сохранение датасета в файл CSV (если необходимо)
df.to_csv('cargo_trajectory_data.csv', index=False)

# Построение графиков
plt.figure(figsize=(14, 12))

# График траектории перемещения
plt.subplot(3, 1, 1)
plt.plot(smoothed_deviation_x, smoothed_deviation_y, label='Smoothed Trajectory Path', color='blue')
plt.xlabel('Deviation X')
plt.ylabel('Deviation Y')
plt.title('Smoothed Trajectory of Cargo')
plt.grid(True)
plt.legend()

# График отклонений по X и Y
plt.subplot(3, 1, 2)
plt.plot(timestamps, deviation_x, label='Deviation X (Noisy)', color='green', alpha=0.5)
plt.plot(timestamps, deviation_y, label='Deviation Y (Noisy)', color='red', alpha=0.5)
plt.plot(timestamps_smoothed, smoothed_deviation_x, label='Deviation X (Smoothed)', color='darkgreen')
plt.plot(timestamps_smoothed, smoothed_deviation_y, label='Deviation Y (Smoothed)', color='darkred')
plt.xlabel('Timestamp')
plt.ylabel('Deviation')
plt.title('Deviations Over Time')
plt.grid(True)
plt.legend()

# График раскачивания груза
plt.subplot(3, 1, 3)
plt.plot(timestamps, swinging, label='Swinging', color='orange', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Swinging')
plt.title('Cargo Swinging Over Time')
plt.grid(True)
plt.legend()

# Показываем графики
plt.tight_layout()
plt.show()
