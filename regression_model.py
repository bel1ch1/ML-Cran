import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns

# Загрузка данных
df = pd.read_csv('cargo_trajectory_data.csv')

# Подготовка данных
X = df[['deviation_x', 'deviation_y']]  # Признаки
y = df['swinging']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Вероятности положительного класса

# Оценка модели
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Печать результатов
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(f"\nROC AUC Score: {roc_auc:.2f}")

# График матрицы ошибок
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Swing', 'Swing'],
            yticklabels=['No Swing', 'Swing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, marker='o', color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# from sklearn.model_selection import GridSearchCV

# # Определение параметров для поиска по сетке
# param_grid = {
#     'penalty': ['l1', 'l2', 'none'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga'],
#     'max_iter': [100, 200, 300]
# }

# # Создание объекта GridSearchCV
# grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# # Обучение GridSearchCV
# grid_search.fit(X_train, y_train)

# # Лучшие параметры и оценка
# print("Лучшие параметры:", grid_search.best_params_)
# print("Лучший результат кросс-валидации:", grid_search.best_score_)

# # Прогнозирование и оценка на тестовой выборке с лучшими параметрами
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# y_prob = best_model.predict_proba(X_test)[:, 1]

# # Оценка модели
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_prob)

# # Печать результатов
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(class_report)
# print(f"\nROC AUC Score: {roc_auc:.2f}")

# # Графики (тот же код для графиков, что и выше)
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['No Swing', 'Swing'],
#             yticklabels=['No Swing', 'Swing'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# plt.figure(figsize=(10, 7))
# plt.plot(fpr, tpr, marker='o', color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True)
# plt.show()
