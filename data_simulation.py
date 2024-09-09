import numpy as np
import pandas as pd

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

# Просмотр первых нескольких строк
print(df.head())

# Сохранение в CSV файл
df.to_csv('cargo_swinging_simulation.csv', index=False)

print("Датасет создан и сохранен в 'cargo_swinging_simulation.csv'.")
