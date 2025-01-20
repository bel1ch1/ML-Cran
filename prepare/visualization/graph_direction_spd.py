import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

df = pd.read_csv('datas/CSV/right_data.csv', header=None)
data_sequences = df[0].apply(eval).tolist()  # Преобразуем строковые списки в Python-списки

for i, data in enumerate(data_sequences):
    x_data = [point[0] for point in data]
    y_data = [point[1] for point in data]

    counts = Counter(x_data)
    num_points = len(x_data)
    colors = np.linspace(0, 1, num_points)

    plt.figure(figsize=(8,4)) # чтобы рисовать в отдельном окне
    #plt.scatter(x_data, y_data, color='blue', label='Data points')  # Отображаем точки
    #plt.plot(x_data, y_data, color='green', linestyle='--')  # Линия, соединяющая точки
    scatter = plt.scatter(x_data, y_data, c=colors, cmap='viridis', label='Data points', s=100)

    # Отображаем количество вхождений на графике
    for x_val, count in counts.items():
        plt.text(x_val, max(y_data) * 1.05, f'{count}',  # Позиционируем текст
                 ha='center', color='red', fontsize=12)

    # Добавляем подписи
    plt.title(f'Plot of Data Sequence {i + 1}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.show() # чтобы рисовался следующий
