import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Параметры симуляции с учетом привязки камеры к крану и инерции груза
stop_period = 100  # Период остановки крана для моделирования раскачивания
swing_damping_factor = 0.98  # Коэффициент затухания для раскачки

# Параметры симуляции
np.random.seed(42)
num_steps = 10000  # Количество шагов симуляции
time_step = 0.1  # Шаг по времени (в секундах)
time = np.arange(0, num_steps * time_step, time_step)  # Время
swing_threshold = 0.05  # Порог для маркировки раскачки
swing_period = 50  # Период для раскачки
damping = 0.98  # Коэффициент демпфирования
max_swing_strength = 1000

# Инициализация координат груза относительно камеры
x_crane = np.zeros(num_steps)
y_crane = np.zeros(num_steps)

for i in range(1, num_steps):
    # Определяем, движется ли кран или остановился
    is_moving = (i // stop_period) % 2 == 0

    if is_moving:
        # Плавное движение крана, груз слегка отклоняется от камеры из-за инерции
        x_crane[i] = x_crane[i-1] + np.random.normal(0, 0.01)  # небольшие случайные отклонения
        y_crane[i] = y_crane[i-1] + np.random.normal(0, 0.01)  # чтобы показать инерцию
    else:
        # При остановке груза — начинается раскачивание
        swing_factor_x = np.sin(i / swing_period * 2 * np.pi) * np.exp(-i / num_steps) * swing_damping_factor
        swing_factor_y = np.cos(i / swing_period * 2 * np.pi) * np.exp(-i / num_steps) * swing_damping_factor

        # Ограничение колебаний по амплитуде
        swing_factor_x = np.clip(swing_factor_x, -max_swing_strength, max_swing_strength)
        swing_factor_y = np.clip(swing_factor_y, -max_swing_strength, max_swing_strength)

        x_crane[i] = x_crane[i-1] + swing_factor_x
        y_crane[i] = y_crane[i-1] + swing_factor_y

# Вычисление амплитуды и меток раскачивания
swing_amplitude_crane = np.sqrt(np.diff(x_crane, prepend=0)**2 + np.diff(y_crane, prepend=0)**2)
swing_label_crane = (swing_amplitude_crane > swing_threshold).astype(int)

# Создание DataFrame для моделирования с раскачкой при остановке
df_crane = pd.DataFrame({
    'time': time,
    'x': x_crane,
    'y': y_crane,
    'swing_amplitude': swing_amplitude_crane,
    'swing_label': swing_label_crane
})


df_crane.to_csv('CSV/better_dset.csv', mode='a', header=False, index=False)


# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(time, x_crane, label='X coordinate (crane)')
plt.plot(time, y_crane, label='Y coordinate (crane)')
plt.title('Simulated Object Movement with Inertia and Swing on Crane Stop')
plt.xlabel('Time (s)')
plt.ylabel('Deviation from Camera Center')
plt.legend()
plt.show()

# Вывод первых 5 строк нового датасета
df_crane.head()
