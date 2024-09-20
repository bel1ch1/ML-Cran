import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Параметры симуляции
np.random.seed(42)
num_steps = 10000  # Количество шагов симуляции
time_step = 0.1  # Шаг по времени (в секундах)
time = np.arange(0, num_steps * time_step, time_step)  # Время
swing_threshold = 0.05  # Порог для маркировки раскачки
swing_period = 50  # Период для раскачки
damping = 0.98  # Коэффициент демпфирования

# Инициализация координат (груз под камерой в начальной точке)
x = np.zeros(num_steps)
y = np.zeros(num_steps)

# Начальные параметры для движения крана
velocity_x = 0.05  # Скорость движения по X
velocity_y = 0.03  # Скорость движения по Y

# Моделирование инерции и раскачки
for i in range(1, num_steps):
    # Инерция и смещение груза с случайными возмущениями
    x[i] = x[i-1] + velocity_x + np.random.normal(0, 0.005)
    y[i] = y[i-1] + velocity_y + np.random.normal(0, 0.005)

    # Моделирование раскачки
    swing_factor_x = np.sin(i / swing_period * 2 * np.pi) * np.exp(-i / num_steps) * damping
    swing_factor_y = np.cos(i / swing_period * 2 * np.pi) * np.exp(-i / num_steps) * damping
    x[i] += swing_factor_x
    y[i] += swing_factor_y

# Вычисление амплитуды раскачки и меток
swing_amplitude = np.sqrt(np.diff(x, prepend=0)**2 + np.diff(y, prepend=0)**2)
swing_label = (swing_amplitude > swing_threshold).astype(int)

# Создание DataFrame с метками
df = pd.DataFrame({
    'time': time,
    'x': x,
    'y': y,
    'swing_amplitude': swing_amplitude,
    'swing_label': swing_label
})


df.to_csv('CSV/better_dset.csv', mode='a', header=False, index=False)


# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(time, x, label='X coordinate')
plt.plot(time, y, label='Y coordinate')
plt.title('Simulated Movement of Object under Crane')
plt.xlabel('Time (s)')
plt.ylabel('Coordinates')
plt.legend()
plt.show()

# Вывод первых 5 строк датасета
df.head()
