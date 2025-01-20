import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Чтение данных из CSV-файла
df = pd.read_csv('datas/CSV/right_data.csv', header=None)

# Преобразуем данные в списки (например, строки содержат последовательности данных)
data_sequences = df[0].apply(eval).tolist()  # Преобразуем строковые списки в Python-списки

# Цикл для отображения каждой последовательности на отдельных графиках
for i, data in enumerate(data_sequences):
    # Разделяем данные на X, Y и скорость (spd)
    x_data = [point[0] for point in data]
    y_data = [point[1] for point in data]
    speed_data = [point[2] for point in data]

    # Создание итераций для графика
    iterations = list(range(len(data)))  # Индексы итераций

    # Создание фигуры с двумя подграфиками (2 строки, 1 столбец)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # 2 строки, 1 столбец

    # --- График зависимости x от итерации ---
    ax1.plot(iterations, x_data, color='blue', linestyle='-', marker='o', label='X values')
    ax1.set_title(f'X vs Iteration for Data Sequence {i + 1}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('X values')
    ax1.grid(True)
    ax1.legend()

    # --- График зависимости y от итерации ---
    ax2.plot(iterations, y_data, color='green', linestyle='-', marker='o', label='Y values')
    ax2.set_title(f'Y vs Iteration for Data Sequence {i + 1}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Y values')
    ax2.grid(True)
    ax2.legend()

    # Настройка общего заголовка и отображение графиков
    plt.tight_layout()  # Улучшает компоновку, чтобы графики не накладывались

    # Перевод окна графика в полноэкранный режим
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')


    plt.show()
    # Следующий набор графиков появится после закрытия текущего
