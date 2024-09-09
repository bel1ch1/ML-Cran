import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report

# Пример данных
data = pd.read_csv('cargo_trajectory_data.csv')

# Создание DataFrame
df = pd.DataFrame(data)

# Подготовка данных
X = df[['deviation_x', 'deviation_y']]
y = df['swinging']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Предсказание вероятностей на тестовой выборке
y_prob = model.predict_proba(X_test)[:, 1]

# Оценка модели
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# График ROC-кривой
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (площадь = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Ложноположительная скорость')
plt.ylabel('Истинноположительная скорость')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Вывод отчета о классификации
print(classification_report(y_test, model.predict(X_test)))

# Подготовка данных
X = df['deviation_x']
Y = df['deviation_y']
swinging = df['swinging']

# График
plt.figure(figsize=(10, 6))

# Точки, где объект раскачивается
plt.scatter(X[swinging == 1], y[swinging == 1], color='red', label='Раскачивается', marker='o', edgecolor='k')

# Точки, где объект не раскачивается
plt.scatter(X[swinging == 0], y[swinging == 0], color='blue', label='Не раскачивается', marker='x', edgecolor='k')

plt.xlabel('Отклонение X')
plt.ylabel('Отклонение Y')
plt.title('Распределение объектов по признакам X и Y')
plt.legend()
plt.grid(True)
plt.show()
