import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np

# Функция для чтения и обработки данных из CSV
def read_data(csv_file):
    df = pd.read_csv(csv_file, header=None)
    # Конвертируем строковые списки в действительные Python-списки
    df[0] = df[0].apply(ast.literal_eval)
    return df

# Функция для отрисовки графиков
def plot_sequences(df, batch_size=10):
    num_batches = len(df) // batch_size
    for batch_num in range(num_batches + 1):
        batch_start = batch_num * batch_size
        batch_end = batch_start + batch_size
        # Получаем текущие 10 последовательностей
        batch_data = df.iloc[batch_start:batch_end, 0].values

        plt.figure(figsize=(10, 6))

        for i, sequence in enumerate(batch_data):
            sequence = list(zip(*sequence))  # Транспонируем последовательность
            x = sequence[0]
            y = sequence[1]
            speed = sequence[2]  # Возьмем скорость из третьего столбца
            plt.plot(x, y, label=f'Sequence {batch_start + i + 1}')
            plt.scatter(x, y, c=speed, cmap='viridis', s=50, label=f'Speed {batch_start + i + 1}')

        plt.colorbar(label="Speed")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Sequences from {batch_start + 1} to {min(batch_end, len(df))}")
        plt.legend()
        plt.show()

# Чтение данных
csv_file = "datas\CSV\right_data.csv"  # Замените на путь к вашему файлу
df = read_data(csv_file)

# Построение графиков по 10 последовательностей за раз
plot_sequences(df, batch_size=10)
