import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Создание DataFrame из ваших данных
data = pd.read_csv('CSV/better_dset.csv')

df = pd.DataFrame(data)


# # График траектории (изменение по X и Y)
# plt.figure(figsize=(10, 6))
# plt.plot(df['dx'], df['dy'], marker='o', linestyle='-', color='b')
# plt.xlabel('Изменение по X')
# plt.ylabel('Изменение по Y')
# plt.title('График траектории объекта')
# plt.grid(True)
# plt.show()

# График изменений X по времени
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['x'], label='Изменение по X', color='r', marker='o')
plt.plot(df['timestamp'], df['y'], label='Изменение по Y', color='g', marker='o')
plt.xlabel('Время')
plt.ylabel('Изменение')
plt.title('График изменений по X по времени')
plt.legend()
plt.grid(True)
plt.show()

# График изменений Y по времени
# plt.figure(figsize=(12, 6))
# #plt.plot(df['timestamp'], df['dx'], label='Изменение по X', color='r', marker='o')
# plt.plot(df['timestamp'], df['dy'], label='Изменение по Y', color='g', marker='o')
# plt.xlabel('Время')
# plt.ylabel('Изменение')
# plt.title('График изменений по Y по времени')
# plt.legend()
# plt.grid(True)
# plt.show()


# График колебаний (поскольку все значения равны 0, он будет плоским)
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['swing_amplitude'], label='Колебания', color='c', marker='o')
plt.xlabel('Время')
plt.ylabel('Колебания')
plt.title('График колебаний по времени')
plt.legend()
plt.grid(True)
plt.show()
