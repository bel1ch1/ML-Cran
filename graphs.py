import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Устанавливаем начальное значение генератора случайных чисел
np.random.seed(42)

# Параметры
num_samples = 1000  # Количество образцов
noise_level = 0.1   # Уровень случайного шума

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

# Симуляция метки раскачивания груза (например, если отклонения превышают порог)
threshold = 0.2
swinging = (np.abs(deviation_x) > threshold) | (np.abs(deviation_y) > threshold)
swinging = swinging.astype(int)

# Создание DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'deviation_x': deviation_x,
    'deviation_y': deviation_y,
    'swinging': swinging
})

# Построение графиков
plt.figure(figsize=(14, 10))

# График траектории перемещения
plt.subplot(3, 1, 1)
plt.plot(deviation_x, deviation_y, label='Trajectory Path', color='blue')
plt.xlabel('Deviation X')
plt.ylabel('Deviation Y')
plt.title('Trajectory of Cargo')
plt.grid(True)
plt.legend()

# График отклонений по X и Y
plt.subplot(3, 1, 2)
plt.plot(timestamps, deviation_x, label='Deviation X', color='green', alpha=0.7)
plt.plot(timestamps, deviation_y, label='Deviation Y', color='red', alpha=0.7)
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
